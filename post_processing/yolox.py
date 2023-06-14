from typing import List

import cv2
import numpy as np
import torch
import torchvision

from infer.data_class import DetectResult, BBox, ModelConfig
from infer.inferencer import TensorRTInferencer


class YOLOXDetector(TensorRTInferencer):

    def __init__(self, model_config: ModelConfig):
        # noinspection PyUnresolvedReferences
        self.input_size = self.input_size  # 输入图像尺寸
        # noinspection PyUnresolvedReferences
        self.num_classes = self.num_classes if hasattr(self, "num_classes") else 1  # 检测种类
        # byte track的yolox需要预处理，官方的yolox不需要预处理
        self.need_normalization = self.need_normalization if hasattr(self, "need_normalization") else False

        self.infer_image_shape = None
        super().__init__(model_config=model_config)

    def decode_outputs(self, outputs):
        """
        trt输出解码
        :return:
        """
        grids = []
        strides = []
        dtype = torch.FloatTensor
        for (hsize, wsize), stride in zip([torch.Size([int(self.input_size[0] // 8), int(self.input_size[1] // 8)]),
                                           torch.Size([int(self.input_size[0] // 16), int(self.input_size[1] // 16)]),
                                           torch.Size([int(self.input_size[0] // 32), int(self.input_size[1] // 32)])],
                                          [8, 16, 32]):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))
        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)
        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs

    def pre_processing(self, img: np.ndarray):
        """
        图像预处理
        :param img:
        :return:
        """
        padded_img = np.ones((self.input_size[0], self.input_size[1], 3)) * 114.0
        r = min(self.input_size[0] / img.shape[0], self.input_size[1] / img.shape[1])

        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
        if self.need_normalization:
            padded_img = padded_img[:, :, ::-1]
            padded_img /= 255.0
            padded_img -= (0.485, 0.456, 0.406)
            padded_img /= (0.229, 0.224, 0.225)
        padded_img = padded_img.transpose((2, 0, 1))
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img

    @staticmethod
    def post_processing(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]
        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):
            # If none are remaining => process next image
            if not image_pred.size(0):
                continue
            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
            conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
            detections = detections[conf_mask]
            if not detections.size(0):
                continue
            if class_agnostic:
                nms_out_index = torchvision.ops.nms(
                    detections[:, :4],
                    detections[:, 4] * detections[:, 5],
                    nms_thre,
                )
            else:
                nms_out_index = torchvision.ops.batched_nms(
                    detections[:, :4],
                    detections[:, 4] * detections[:, 5],
                    detections[:, 6],
                    nms_thre,
                )
            detections = detections[nms_out_index]
            if output[i] is None:
                output[i] = detections
            else:
                # noinspection PyTypeChecker
                output[i] = torch.cat((output[i], detections))
        return output

    def infer(self, image: np.ndarray = None) -> List[DetectResult]:
        super().prepare()
        self.infer_image_shape = image.shape
        image = self.pre_processing(image)
        np.copyto(self.inputs[0].host, image.flatten())
        trt_outputs = self.do_inference()
        trt_outputs = np.concatenate(trt_outputs)
        trt_outputs = torch.from_numpy(trt_outputs)
        trt_outputs.resize_(1, int(trt_outputs.shape.numel() / (5 + self.num_classes)), 5 + self.num_classes)
        trt_outputs = self.decode_outputs(trt_outputs)
        trt_outputs = self.post_processing(prediction=trt_outputs,
                                           num_classes=self.num_classes,
                                           conf_thre=0.3,
                                           nms_thre=0.3,
                                           class_agnostic=True)
        if trt_outputs[0] is None:
            return []
        # noinspection PyUnresolvedReferences
        results = trt_outputs[0].numpy()
        ratio = min(self.input_size[0] / self.infer_image_shape[0], self.input_size[1] / self.infer_image_shape[1])
        detect_results = []
        for result in results:
            bbox = list(map(int, result[:4] / ratio))
            score = float(result[4] * result[5])
            detect_results.append(
                DetectResult(bbox=BBox(ltx=bbox[0], lty=bbox[1], rbx=bbox[2], rby=bbox[3]), score=score,
                             category=int(result[6])))
        return detect_results
