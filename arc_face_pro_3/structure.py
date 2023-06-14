from ctypes import Structure, POINTER, string_at

from infer.arc_face_pro_3.type_map import MPChar, MInt32, MPVoid, MFloat, MByte, MUInt32, MUInt8


class ASF_ActiveFileInfo(Structure):
    """
    激活文件信息
    """
    _fields_ = [
        ("startTime", MPChar),
        ("endTime", MPChar),
        ("activeKey", MPChar),
        ("platform", MPChar),
        ("sdkType", MPChar),
        ("appId", MPChar),
        ("sdkKey", MPChar),
        ("sdkVersion", MPChar),
        ("fileVersion", MPChar)
    ]


# noinspection SpellCheckingInspection
class MRECT(Structure):
    _fields_ = [
        ("left", MInt32),
        ("top", MInt32),
        ("right", MInt32),
        ("bottom", MInt32)
    ]


class ASF_SingleFaceInfo(Structure):
    """
    单人脸信息
    """
    _fields_ = [
        ("faceRect", MRECT),
        ("faceOrient", MInt32),
    ]


class ASF_MultiFaceInfo(Structure):
    """
    多人脸信息
    """
    _fields_ = [
        ("faceRect", POINTER(MRECT)),
        ("faceOrient", POINTER(MInt32)),
        ("faceNum", MInt32),
        ("faceID", POINTER(MInt32)),
    ]


class ASF_ImageQualityInfo(Structure):
    """
    图像质量
    """
    _fields_ = [
        ("faceQualityValue", POINTER(MFloat)),
        ("num", MInt32)
    ]


class ASF_FaceFeature(Structure):
    """
    人脸特征
    """
    _fields_ = [
        ("feature", POINTER(MByte)),
        ("featureSize", MInt32)
    ]


class ASVLOFFSCREEN(Structure):
    _fields_ = [
        ("u32PixelArrayFormat", MUInt32),
        ("i32Width", MInt32),
        ("i32Height", MInt32),
        ("ppu8Plane", POINTER(MUInt8) * 4),
        ("pi32Pitch", MInt32 * 4)
    ]


ASF_ImageData = ASVLOFFSCREEN
