from pathlib import Path
import base64
from ctypes import cdll, c_char_p, POINTER, byref, string_at, cast, create_string_buffer
from typing import List
from datetime import datetime

import cv2

from infer.arc_face_pro_3.constant import MOK, MERR_ASF_ALREADY_ACTIVATED, ASF_DETECT_MODE_IMAGE, ASF_OP_0_ONLY, \
    ASF_FACE_DETECT, ASF_FACERECOGNITION, ASVL_PAF_RGB24_B8G8R8, ASF_DETECT_MODEL_RGB, ASF_LIFE_PHOTO, ASF_ID_PHOTO, \
    ASF_LIVENESS, ASF_AGE, ASF_GENDER, ASF_FACE3DANGLE, ASF_IR_LIVENESS, ASF_IMAGEQUALITY, \
    MERR_FSDK_FACEFEATURE_LOW_CONFIDENCE_LEVEL, MERR_INVALID_PARAM
from infer.arc_face_pro_3.data_class import Face
from infer.arc_face_pro_3.structure import ASF_ActiveFileInfo, ASF_MultiFaceInfo, ASF_SingleFaceInfo, ASF_FaceFeature, \
    ASF_ImageData, ASF_ImageQualityInfo
from infer.arc_face_pro_3.type_map import MPChar, MRESULT, ASF_DetectMode, ASF_OrientPriority, MInt32, MHandle, MUInt8, \
    ASF_DetectModel, MFloat, ASF_RegisterOrNot, ASF_CompareModel, MByte, MPVoid
from infer.data_class import BBox, Image, FaceFeature
from infer.utils.path import get_model


class ArcFacePro3:
    """
    虹软人脸SDK Pro
    """

    def __init__(self):
        self.already_initialized = False  # 是否已经初始化，第一次推理的时候才会进行初始化，初始化调用init方法
        self.encoding = "utf-8"
        self.lib = None
        self.handle = None

    def __del__(self):
        if self.already_initialized:
            self.lib.ASFUninitEngine.argtypes = [MHandle]
            self.lib.ASFUninitEngine.restype = MRESULT
            self.lib.ASFUninitEngine(self.handle)

    def init(self):
        """
        初始化引擎
        最多检测5个人脸
        :return:
        """
        cdll.LoadLibrary(str(get_model("libarcsoft_face_3.1.so")))
        self.lib = cdll.LoadLibrary(str(get_model("libarcsoft_face_engine_hack.so")))
        self._offline_activate(get_model("ArcFace.dat"))
        self.lib.ASFInitEngine.argtypes = [ASF_DetectMode, ASF_OrientPriority, MInt32, MInt32, MInt32, POINTER(MHandle)]
        self.lib.ASFInitEngine.restype = MRESULT
        mask = ASF_FACE_DETECT | ASF_FACERECOGNITION | ASF_IMAGEQUALITY
        self.handle = MHandle()
        res = self.lib.ASFInitEngine(ASF_DETECT_MODE_IMAGE, ASF_OP_0_ONLY, 32, 5, mask, byref(self.handle))
        if res != MOK:
            raise Exception(res)
        self.already_initialized = True

    def _offline_activate(self, active_file: Path):
        """
        离线激活
        :return:
        """
        self.lib.ASFOfflineActivation.argtypes = [MPChar]
        self.lib.ASFOfflineActivation.restype = MRESULT
        res = self.lib.ASFOfflineActivation(MPChar(str(active_file).encode(self.encoding)))
        if res not in [MOK, MERR_ASF_ALREADY_ACTIVATED]:
            raise Exception(res)

    def _online_activate(self, app_id: str, sdk_key: str, active_key: str):
        """
        在线激活
        :param app_id:
        :param sdk_key:
        :param active_key:
        :return:
        """
        self.lib.ASFOnlineActivation.argtypes = [MPChar, MPChar, MPChar]
        self.lib.ASFOnlineActivation.restype = MRESULT
        res = self.lib.ASFOnlineActivation(c_char_p(app_id.encode(self.encoding)),
                                           c_char_p(sdk_key.encode(self.encoding)),
                                           c_char_p(active_key.encode(self.encoding)))
        if res not in [MOK, MERR_ASF_ALREADY_ACTIVATED]:
            raise Exception(res)

    def get_active_info(self) -> ASF_ActiveFileInfo:
        """
        获取激活文件信息
        :return:
        """
        if not self.already_initialized:
            self.init()
        self.lib.ASFGetActiveFileInfo.argtypes = [POINTER(ASF_ActiveFileInfo)]
        self.lib.ASFGetActiveFileInfo.restype = MRESULT
        activeFileInfo = ASF_ActiveFileInfo()
        res = self.lib.ASFGetActiveFileInfo(byref(activeFileInfo))
        if res != MOK:
            raise Exception(res)
        return activeFileInfo

    def get_device_info(self):
        if not self.already_initialized:
            self.init()
        self.lib.ASFGetActiveDeviceInfo.argtypes = [POINTER(MPChar)]
        self.lib.ASFGetActiveDeviceInfo.restype = MRESULT
        device_info = MPChar()
        res = self.lib.ASFGetActiveDeviceInfo(byref(device_info))
        if res != MOK:
            raise Exception(res)
        return device_info.value.decode(self.encoding)

    def get_sdk_end_time(self) -> datetime:
        """
        获取SDK的有效期截至时间
        :return:
        """
        if not self.already_initialized:
            self.init()
        active_info = self.get_active_info()
        end_time = active_info.endTime.decode(self.encoding)
        return datetime.fromtimestamp(int(end_time))

    def _bytes2string(self, feature: bytes) -> str:
        """
        人脸特征bytes转字符串
        :param feature:
        :return:
        """
        return base64.b64encode(feature).decode(self.encoding)

    def _string2bytes(self, feature: str) -> bytes:
        """
        人脸特征string转bytes
        :param feature:
        :return:
        """
        return base64.b64decode(feature.encode(self.encoding))

    def extract_feature(self, image: Image, face: Face) -> FaceFeature:
        """
        提取特征
        :param image:
        :param face:
        :return:
        """
        if not self.already_initialized:
            self.init()
        image = self._preprocess_image(image)
        self.lib.ASFFaceFeatureExtract.argtypes = [MHandle, MInt32, MInt32, MInt32, POINTER(MUInt8), POINTER(ASF_SingleFaceInfo), POINTER(ASF_FaceFeature)]
        self.lib.ASFFaceFeatureExtract.restype = MRESULT
        feature = ASF_FaceFeature()
        res = self.lib.ASFFaceFeatureExtract(self.handle,
                                             image.shape[1],
                                             image.shape[0],
                                             ASVL_PAF_RGB24_B8G8R8,
                                             image.ctypes.data_as(POINTER(MUInt8)),
                                             byref(face.faceInfo),
                                             byref(feature))
        if res not in [MOK, MERR_FSDK_FACEFEATURE_LOW_CONFIDENCE_LEVEL]:
            raise Exception(res)
        feature_bytes = string_at(feature.feature, feature.featureSize)
        return self._bytes2string(feature_bytes)

    def compare_feature(self, feature1: FaceFeature, feature2: FaceFeature) -> float:
        """
        比较两个人脸特征的相似程度
        :param feature1:
        :param feature2:
        :return:
        """
        if not self.already_initialized:
            self.init()
        self.lib.ASFFaceFeatureCompare.argtypes = [MHandle, POINTER(ASF_FaceFeature), POINTER(ASF_FaceFeature), POINTER(MFloat), ASF_CompareModel]
        self.lib.ASFFaceFeatureCompare.restype = MRESULT
        feature_bytes1 = self._string2bytes(feature1)
        face_feature1 = ASF_FaceFeature()
        face_feature1.feature = cast(create_string_buffer(feature_bytes1), POINTER(MByte))
        face_feature1.featureSize = len(feature_bytes1)
        feature_bytes2 = self._string2bytes(feature2)
        face_feature2 = ASF_FaceFeature()
        face_feature2.feature = cast(create_string_buffer(feature_bytes2), POINTER(MByte))
        face_feature2.featureSize = len(feature_bytes2)
        confidenceLevel = MFloat(0)
        res = self.lib.ASFFaceFeatureCompare(self.handle, byref(face_feature1), byref(face_feature2), byref(confidenceLevel), ASF_ID_PHOTO)
        if res not in [MOK, MERR_INVALID_PARAM]:
            raise Exception(res)
        return confidenceLevel.value

    @staticmethod
    def _preprocess_image(image: Image):
        assert image is not None, "图片不能为空"
        assert image.shape[1] % 4 == 0, "图片的宽度必须是4的倍数"
        assert min(image.shape[0], image.shape[1]) > 80, "待检测的图片中人脸至少80px"
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    @staticmethod
    def _get_image_data(image: Image) -> ASF_ImageData:
        """
        图片构建ASF_ImageData结构体
        :param image:
        :return:
        """
        assert image.shape[1] % 4 == 0
        image_data = ASF_ImageData()
        image_data.u32PixelArrayFormat = ASVL_PAF_RGB24_B8G8R8
        image_data.i32Width = image.shape[1]
        image_data.i32Height = image.shape[0]
        image_data.ppu8Plane[0] = image.ctypes.data_as(POINTER(MUInt8))
        image_data.pi32Pitch[0] = image.shape[1] * 3
        return image_data

    def detect_faces(self, image: Image) -> List[Face]:
        """
        人脸检测+人脸质量
        要求图片宽度是4的倍数，如果不是的话会进行裁剪，导致图片右侧的一条（1-3像素）被裁掉
        人脸至少要大于80像素
        :param image:
        :return: 检测到的人脸，同时返回人脸质量
        """
        if not self.already_initialized:
            self.init()
        image = self._preprocess_image(image)
        self.lib.ASFDetectFaces.argtypes = [MHandle, MInt32, MInt32, MInt32, POINTER(MUInt8), POINTER(ASF_MultiFaceInfo)]
        self.lib.ASFDetectFaces.restype = MRESULT
        detectedFaces = ASF_MultiFaceInfo()
        res = self.lib.ASFDetectFaces(self.handle,
                                      image.shape[1],
                                      image.shape[0],
                                      ASVL_PAF_RGB24_B8G8R8,
                                      image.ctypes.data_as(POINTER(MUInt8)),
                                      byref(detectedFaces))
        if res != MOK:
            raise Exception(res)
        self.lib.ASFImageQualityDetectEx.argtypes = [MHandle,
                                                     POINTER(ASF_ImageData),
                                                     POINTER(ASF_MultiFaceInfo),
                                                     POINTER(ASF_ImageQualityInfo),
                                                     ASF_DetectModel]
        self.lib.ASFImageQualityDetectEx.restype = MRESULT
        image_quality_info = ASF_ImageQualityInfo()
        res = self.lib.ASFImageQualityDetectEx(self.handle,
                                               byref(self._get_image_data(image)),
                                               byref(detectedFaces),
                                               byref(image_quality_info),
                                               ASF_DETECT_MODEL_RGB)
        if res not in [MOK, MERR_FSDK_FACEFEATURE_LOW_CONFIDENCE_LEVEL]:
            raise Exception(res)
        results = []
        for index in range(detectedFaces.faceNum):
            faceInfo = ASF_SingleFaceInfo()
            faceInfo.faceRect = detectedFaces.faceRect[index]
            faceInfo.faceOrient = detectedFaces.faceOrient[index]
            rect = detectedFaces.faceRect[index]
            try:
                score = image_quality_info.faceQualityValue[index]
            except ValueError:
                score = 0
            results.append(Face(bbox=BBox(rect.left, rect.top, rect.right, rect.bottom),
                                score=score,
                                faceInfo=faceInfo,
                                label="face"))
        return results


arc_face_pro_3 = ArcFacePro3()

if __name__ == '__main__':
    from tests import get_test_data
    from infer import vis_detect_results, COLOR, ArcFacePro3
    from scripts import get_scripts_output

    # 实例化sdk，需要参数是 app_id: str, sdk_key: str, active_key: str
    # arc_face_pro = ArcFacePro3("HCy6M18zCA5CVFNpFLeMR26Ttc1Abx48eaZW8dw2AvhE",
    #                            "26Eg8MbBGZfTg1eZrb6y4HZwocqS5AD6EkaMST1MsNKk",
    #                            "82G1-11B4-L12M-D45Z")
    arc_face_pro_3.get_device_info()
    print(arc_face_pro_3.get_active_info())
    # exit()
    print(f"SDK有效期截止到{arc_face_pro_3.get_sdk_end_time()}")
    # 读取测试图片，要求图片宽度是4的倍数，人脸所占画面比例不低于1/32
    image1 = cv2.imread(str(get_test_data("f1.jpg")))
    image2 = cv2.imread(str(get_test_data("f2.png")))
    image2 = cv2.resize(image2, (image2.shape[1] * 5, image2.shape[0] * 5))
    # 人脸检测
    faces1 = arc_face_pro_3.detect_faces(image1)
    faces2 = arc_face_pro_3.detect_faces(image2)
    print(f"画面image1检测到人脸{len(faces1)}个")
    face1 = faces1[0]
    face2 = faces2[0]
    # 人脸
    print(f"画面image1中的检测到人脸的bbox是{face1.bbox},人脸质量是{face1.score}")
    print(f"画面image2中的检测到人脸的bbox是{face2.bbox},人脸质量是{face2.score}")
    # 提取两个人脸的特征
    feature1 = arc_face_pro_3.extract_feature(image1, face1)
    feature2 = arc_face_pro_3.extract_feature(image2, face2)
    print(f"人脸特征是长度为{len(feature1)}的字符串")
    print(f"人脸1和人脸2的相似度是{arc_face_pro_3.compare_feature(feature1, feature2)}")
    # 把人脸检测结果画上去
    image = vis_detect_results(image2, [face2], bbox_color=COLOR.red)
    cv2.imwrite(str(get_scripts_output("output.jpg")), image)
