from dataclasses import dataclass

from infer.data_class import DetectResult

from infer.arc_face_pro_3.structure import ASF_SingleFaceInfo


@dataclass
class Face(DetectResult):
    """
    人脸
    """
    faceInfo: ASF_SingleFaceInfo = None
