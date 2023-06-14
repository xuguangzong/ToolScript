import cv2
import numpy as np
from pathlib import Path


class VideoWriter:
    """
    视频写入器
    """

    def __init__(self, output_video: Path, fps: int = 25):
        """
        初始化视频写入器
        :param output_video:
        :param fps: 帧率
        """
        self.output_video = output_video
        self.fps = fps
        self.video_writer = None

    def write(self, frame: np.ndarray):
        """
        写入一帧图像
        :param frame:
        :return:
        """

        if not self.video_writer:
            self.video_writer = cv2.VideoWriter(str(self.output_video),
                                                cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), self.fps,
                                                (int(frame.shape[1]), int(frame.shape[0])))
        self.video_writer.write(frame)

    def release(self):
        """
        结束写录像，并进行后处理
        :return:
        """
        if not self.video_writer:
            return
        self.video_writer.release()
