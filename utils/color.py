import cv2
import numpy as np
from typing import Dict

from data_class import Image, Color, COLOR


def get_color_palette(image: Image) -> Dict[Color, float]:
    """
    获取图片的调色盘，返回每一种颜色在图片中的比例
    具体颜色类别包括黑色、灰色、白色、红色、橙色、黄色、绿色、青色、蓝色、紫色
    颜色分类参考 https://blog.csdn.net/huang_nansen/article/details/102793744   OpenCV HSV颜色对照表
    :param image: 图片
    :return: 调色盘
    """
    black_threshold = (0, 180, 0, 255, 0, 46)
    gray_threshold = (0, 180, 0, 43, 46, 220)
    white_threshold = (0, 180, 0, 30, 221, 255)
    red_threshold_1 = (0, 10, 43, 255, 46, 255)
    red_threshold_2 = (156, 180, 43, 255, 46, 255)
    orange_threshold = (11, 25, 43, 255, 46, 255)
    yellow_threshold = (26, 34, 43, 255, 46, 255)
    green_threshold = (35, 77, 43, 255, 46, 255)
    cyan_threshold = (78, 99, 43, 255, 46, 255)
    blue_threshold = (100, 124, 43, 255, 46, 255)
    purple_threshold = (125, 155, 43, 255, 46, 255)
    palette = {}
    # 将图片转为HSV格式的
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    for color, threshold in zip(
            [COLOR.black, COLOR.gray, COLOR.white, COLOR.red, COLOR.red, COLOR.orange, COLOR.yellow, COLOR.green,
             COLOR.cyan, COLOR.blue, COLOR.purple],
            [black_threshold, gray_threshold, white_threshold, red_threshold_1, red_threshold_2, orange_threshold,
             yellow_threshold, green_threshold, cyan_threshold,
             blue_threshold, purple_threshold]):
        # 像素个数
        pixel_count = cv2.inRange(image_hsv, np.array(threshold[::2], dtype=np.uint8),
                                  np.array(threshold[1::2], dtype=np.uint8))
        palette[color] = palette.get(color, 0) + pixel_count.sum()
        total = sum(palette.values())
        for color in palette.keys():
            palette[color] /= total

    return palette


if __name__ == "__main__":
    img = cv2.imread("./test/data/1.jpg")

    print(get_color_palette(img))
