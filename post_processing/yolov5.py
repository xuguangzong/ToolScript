from typing import List

import cv2
import numpy as np

from infer.data_class import DetectResult, BBox, ModelConfig
from infer.inferencer import TensorRTInferencer


class YOLOV5Detector(TensorRTInferencer):
    def __init__(self, model_config: ModelConfig):
        # noinspection PyUnresolvedReferences
        self.num_classes = self.num_classes  # 检测种类

        filters = (self.num_classes + 5) * 3
        self.output_shapes = [
            (1, 3, 80, 80, self.num_classes+5),
            (1, 3, 40, 40, self.num_classes+5),
            (1, 3, 20, 20, self.num_classes+5)
        ]
        self.strides = np.array([8., 16., 32.])
        anchors = np.array([
            [[10, 13], [16, 30], [33, 23]],
            [[30, 61], [62, 45], [59, 119]],
            [[116, 90], [156, 198], [373, 326]],
        ])
        self.nl = len(anchors)
        self.no = self.num_classes + 5  # outputs per anchor
        self.na = len(anchors[0])
        a = anchors.copy().astype(np.float32)
        a = a.reshape(self.nl, -1, 2)
        self.anchors = a.copy()
        self.anchor_grid = a.copy().reshape(self.nl, 1, -1, 1, 1, 2)

        super().__init__(model_config=model_config)

    def pre_process(self, img):
        img = cv2.resize(img, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = img.transpose((2, 0, 1)).astype(np.float16)
        img = img.transpose((2, 0, 1)).astype(np.float32)
        img /= 255.0
        return img

    def sigmoid_v(self, array):
        return np.reciprocal(np.exp(-array) + 1.0)

    def make_grid(self, nx, ny):
        """
        Create scaling tensor based on box location
        Source: https://github.com/ultralytics/yolov5/blob/master/models/yolo.py
        Arguments
            nx: x-axis num boxes
            ny: y-axis num boxes
        Returns
            grid: tensor of shape (1, 1, nx, ny, 80)
        """
        nx_vec = np.arange(nx)
        ny_vec = np.arange(ny)
        yv, xv = np.meshgrid(ny_vec, nx_vec)
        grid = np.stack((yv, xv), axis=2)
        grid = grid.reshape(1, 1, ny, nx, 2)
        return grid

    def post_process(self, outputs, conf_thres=0.001):
        """
        Transforms raw output into boxes, confs, classes
        Applies NMS thresholding on bounding boxes and confs
        Parameters:
            output: raw output tensor
        Returns:
            boxes: x1,y1,x2,y2 tensor (dets, 4)
            confs: class * obj prob tensor (dets, 1)
            classes: class type tensor (dets, 1)
        """
        scaled = []
        grids = []
        for out in outputs:
            out = self.sigmoid_v(out)
            _, _, width, height, _ = out.shape
            grid = self.make_grid(width, height)
            grids.append(grid)
            scaled.append(out)
        z = []
        for out, grid, stride, anchor in zip(scaled, grids, self.strides, self.anchor_grid):
            _, _, width, height, _ = out.shape
            out[..., 0:2] = (out[..., 0:2] * 2. - 0.5 + grid) * stride
            out[..., 2:4] = (out[..., 2:4] * 2) ** 2 * anchor

            out = out.reshape((1, 3 * width * height, self.num_classes+5))
            z.append(out)
        pred = np.concatenate(z, 1)
        xc = pred[..., 4] > conf_thres
        pred = pred[xc]
        return self.nms(pred)

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def non_max_suppression(self, boxes, confs, classes, iou_thres=0.6):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = confs.flatten().argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_thres)[0]
            order = order[inds + 1]
        boxes = boxes[keep]
        confs = confs[keep]
        classes = classes[keep]
        return boxes, confs, classes

    def nms(self, pred, iou_thres=0.6):
        boxes = self.xywh2xyxy(pred[..., 0:4])
        # 原仓库https://github.com/SeanAvery/yolov5-tensorrt/blob/master/python/lib/Processor.py
        # 没有下面这一行，置信度取得是class的置信度，这里给乘上obj的置信度，防止置信度都是1
        pred[:, 5:] *= pred[:, 4:5]
        confs = np.amax(pred[:, 5:], 1, keepdims=True)
        classes = np.argmax(pred[:, 5:], axis=-1)
        return self.non_max_suppression(boxes, confs, classes)

    def infer(self, image: np.ndarray = None) -> List[DetectResult]:
        super().prepare()
        shape_orig_WH = (image.shape[1], image.shape[0])
        resized = self.pre_process(image)
        # outputs = self.inference(resized)
        np.copyto(self.inputs[0].host, resized.flatten())
        outputs = self.do_inference()
        # reshape from flat to (1, 3, x, y, 85)
        reshaped = []
        for output, shape in zip(outputs, self.output_shapes):
            reshaped.append(output.reshape(shape))
        boxes, confs, classes = self.post_process(reshaped)
        detect_results = []
        for box, conf, category in zip(boxes, confs, classes):
            x_scale, y_scale = image.shape[1] / 640, image.shape[0] / 640
            detect_results.append(DetectResult(bbox=BBox(ltx=round(box[0]*x_scale),
                                                         lty=round(box[1]*y_scale),
                                                         rbx=round(box[2]*x_scale),
                                                         rby=round(box[3]*y_scale)),
                                               score=float(conf),
                                               category=int(category)))
        return detect_results

