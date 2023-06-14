# 函数返回值枚举
MOK = 0
MERR_INVALID_PARAM = 2
MERR_ASF_ALREADY_ACTIVATED = 90114
MERR_FSDK_FACEFEATURE_LOW_CONFIDENCE_LEVEL = 81925

# 检测模式 ASF_DetectMode
ASF_DETECT_MODE_VIDEO = 0x00000000  # VIDEO模式，一般用于多帧连续检测
ASF_DETECT_MODE_IMAGE = 0xFFFFFFFF  # IMAGE模式，一般用于静态图的单次检测

# 人脸检测方向 ArcSoftFace_OrientPriority
ASF_OP_0_ONLY = 0x1  # 常规预览下正方向
ASF_OP_90_ONLY = 0x2  # 基于0°逆时针旋转90°的方向
ASF_OP_270_ONLY = 0x3  # 基于0°逆时针旋转270°的方向
ASF_OP_180_ONLY = 0x4  # 基于0°旋转180°的方向（逆时针、顺时针效果一样）
ASF_OP_ALL_OUT = 0x5  # 全角度

# 引擎能力
ASF_FACE_DETECT = 0x00000001  # 此处detect可以是tracking或者detection两个引擎之一，具体的选择由detect mode 确定
ASF_FACERECOGNITION = 0x00000004  # 人脸特征
ASF_AGE = 0x00000008  # 年龄
ASF_GENDER = 0x00000010  # 性别
ASF_FACE3DANGLE = 0x00000020  # 3D角度
ASF_LIVENESS = 0x00000080  # RGB活体
ASF_IMAGEQUALITY = 0x00000200  # 图像质量检测
ASF_IR_LIVENESS = 0x00000400  # IR活体

# 颜色空间
ASVL_PAF_NV21 = 2050  # 8-bit Y 通道，8-bit 2x2 采样 V 与 U 分量交织通道
ASVL_PAF_NV12 = 2049  # 8-bit Y 通道，8-bit 2x2 采样 U 与 V 分量交织通道
ASVL_PAF_RGB24_B8G8R8 = 513  # RGB 分量交织，按 B, G, R, B 字节序排布
ASVL_PAF_I420 = 1537  # 8-bit Y 通道， 8-bit 2x2 采样 U 通道， 8-bit 2x2 采样 V 通道
ASVL_PAF_YUYV = 1289  # YUV 分量交织， V 与 U 分量 2x1 采样，按 Y0, U0, Y1, V0 字节序排布
ASVL_PAF_GRAY = 1793  # 8-bit IR图像
ASVL_PAF_DEPTH_U16 = 3074  # 16-bit IR图像

ASF_DETECT_MODEL_RGB = 0x1  # RGB图像检测模型

# 人脸比对可选的模型 ASF_CompareModel
ASF_LIFE_PHOTO = 0x1  # 用于生活照之间的特征比对，推荐阈值0.80
ASF_ID_PHOTO = 0x2  # 用于证件照或生活照与证件照之间的特征比对，推荐阈值0.82
