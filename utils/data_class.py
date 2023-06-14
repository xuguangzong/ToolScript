import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
from dataclasses import dataclass

Image = np.ndarray


@dataclass
class Color:
    """
    OpenCV颜色 BGR
    """
    blue: int
    green: int
    red: int

    def to_tuple(self) -> Tuple[int, int, int]:
        return self.blue, self.green, self.red

    def __hash__(self):
        """
        哈希，用于把Color当作字典的key
        :return:
        """
        return hash(self.to_tuple())


class COLOR:
    """
    颜色常量枚举
    """
    red = Color(0, 0, 255)  # 红色
    blue = Color(255, 0, 0)  # 蓝色
    black = Color(0, 0, 0)  # 黑色
    white = Color(255, 255, 255)  # 白色
    yellow = Color(0, 255, 255)  # 黄色
    green = Color(0, 255, 0)  # 绿色
    gray = Color(192, 192, 192)  # 灰色
    orange = Color(0, 97, 255)  # 橙色
    cyan = Color(255, 255, 0)  # 青色
    purple = Color(240, 32, 160)  # 紫色

    @classmethod
    def all(cls):
        """
        获取所有颜色
        :return:
        """
        colors = []
        for attr in dir(cls):
            if isinstance(getattr(cls, attr), Color):
                colors.append(getattr(cls, attr))

        return colors


@dataclass
class Point:
    """
    点
    """
    x: int
    y: int

    def to_tuple(self) -> Tuple:
        return self.x, self.y


@dataclass
class Segment:
    """
    线段
    """
    point1: Point
    point2: Point

    @property
    def center(self) -> Point:
        """
        线段中点
        :return:
        """
        return Point((self.point1.x + self.point2.x) // 2, (self.point1.y + self.point2.y) // 2)


@dataclass
class BBox:
    """
    bbox
    """
    ltx: int
    lty: int
    rbx: int
    rby: int

    @property
    def center(self) -> Point:
        """
        bbox的中心
        :return:
        """
        return Point(int((self.ltx + self.rbx) / 2), int((self.lty + self.rby) / 2))

    @property
    def width(self) -> int:
        return self.rbx - self.ltx

    @property
    def height(self) -> int:
        return self.rby - self.lty

    @property
    def size(self) -> int:
        return self.width * self.height

    @property
    def lt(self) -> Point:
        return Point(x=self.ltx, y=self.lty)

    @property
    def rt(self) -> Point:
        return Point(x=self.rbx, y=self.lty)

    @property
    def lb(self) -> Point:
        return Point(x=self.ltx, y=self.rby)

    @property
    def rb(self) -> Point:
        return Point(x=self.rbx, y=self.rby)

    # noinspection SpellCheckingInspection
    def to_xywh(self) -> List[int]:
        return [self.center.x, self.center.y, self.width, self.height]

    # noinspection SpellCheckingInspection
    def to_xyxy(self) -> List[int]:
        return [self.ltx, self.lty, self.rbx, self.rby]

    @property
    def left(self) -> Segment:
        return Segment(self.lt, self.lb)

    @property
    def top(self) -> Segment:
        return Segment(self.lt, self.rt)

    @property
    def right(self) -> Segment:
        return Segment(self.rt, self.rb)

    @property
    def bottom(self) -> Segment:
        return Segment(self.lb, self.rb)


@dataclass()
class ModelConfig:
    """
    模型配置
    """
    model_path: Path  # 模型文件所在路径
    model_name: str  # 模型文件名称，不带后缀
    device: int = 0  # 模型使用的GPU

    @property
    def onnx_file(self) -> Path:
        """
        onnx文件
        :return:
        """
        return self.model_path.joinpath(f"{self.model_name}.onnx")

    @property
    def engine_file(self) -> Path:
        """
        trt文件
        :return:
        """
        return self.model_path.joinpath(f"{self.model_name}.trt")
