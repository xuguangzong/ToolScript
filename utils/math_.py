from typing import List, Tuple

import numpy as np

from data_class import Point, Segment, BBox


# 数学相关函数


def get_point_to_point_distance(point1: Point, point2: Point) -> float:
    """
    计算点到点距离
    :param point1:
    :param point2:
    :return:
    """
    v1 = np.array(point1.to_tuple())
    v2 = np.array(point2.to_tuple())
    return np.linalg.norm(v1 - v2)


def get_segment_length(segment: Segment) -> float:
    """
    计算线段长度
    :param segment:
    :return:
    """
    return get_point_to_point_distance(segment.point1, segment.point2)


def get_point_to_segment_distance(point: Point, segment: Segment) -> float:
    """
    计算点到线段距离
    由于Point类型的成员是int类型的，中间结果的float进行了round处理，有不超过0.5的误差
    https://www.codeleading.com/article/99965333231/
    :param point:
    :param segment:
    :return:
    """
    if segment.point1 == segment.point2:
        return get_point_to_point_distance(point, segment.point1)
    line_magnitude = get_segment_length(segment)
    u1 = (((point.x - segment.point1.x) * (segment.point2.x - segment.point1.x)) + (
            (point.y - segment.point1.y) * (segment.point2.y - segment.point1.y)))
    u = u1 / (line_magnitude * line_magnitude)
    if (u < 0.00001) or (u > 1):
        # 点到直线的投影不在线段内, 计算点到两个端点距离的最小值即为"点到线段最小距离"
        ix = get_point_to_point_distance(point, segment.point1)
        iy = get_point_to_point_distance(point, segment.point2)
        return min(ix, iy)
    else:
        # 投影点在线段内部, 计算方式同点到直线距离, u 为投影点距离x1在x1x2上的比例, 以此计算出投影点的坐标
        ix = segment.point1.x + u * (segment.point2.x - segment.point1.x)
        iy = segment.point1.y + u * (segment.point2.y - segment.point1.y)
        # noinspection PyTypeChecker
        # 为了结果准确，没有做round处理
        return get_point_to_point_distance(point, Point(ix, iy))


def is_point_in_bbox(point: Point, bbox: BBox) -> bool:
    """
    判断一个点是否在bbox内
    :param point:
    :param bbox:
    :return:
    """
    return (bbox.ltx <= point.x <= bbox.rbx) and (bbox.lty <= point.y <= bbox.rby)


def get_point_to_bbox_distance(point: Point, bbox: BBox) -> float:
    """
    计算一个点到bbox的距离，如果点在bbox内距离定义为0
    :param point:
    :param bbox:
    :return:
    """
    if is_point_in_bbox(point, bbox):
        return 0
    left_distance = get_point_to_segment_distance(point, bbox.left)
    top_distance = get_point_to_segment_distance(point, bbox.top)
    right_distance = get_point_to_segment_distance(point, bbox.right)
    bottom_distance = get_point_to_segment_distance(point, bbox.bottom)
    return min(left_distance, top_distance, right_distance, bottom_distance)


def get_nearest_point_to_point(point: Point, points: List[Point]) -> Tuple[int, Point, float]:
    """
    给定一个点和一组点，计算一组点中距离给定点最近的点
    """
    assert points, f"点为空：{points}"
    target_index, target_point, target_distance = None, None, None
    for index, p in enumerate(points):
        distance = get_point_to_point_distance(point, p)
        if target_index is None or distance < target_distance:
            target_index = index
            target_point = p
            target_distance = distance
    return target_index, target_point, target_distance


def get_nearest_bbox_to_point(point: Point, bboxes: List[BBox]) -> Tuple[int, BBox, float]:
    """
    给定一个点和一组bbox，计算一组bbox中距离给定点最近的bbox
    """
    assert bboxes, f"bbox为空：{bboxes}"
    target_index, target_bbox, target_distance = None, None, None
    for index, bbox in enumerate(bboxes):
        distance = get_point_to_bbox_distance(point, bbox)
        if target_index is None or distance < target_distance:
            target_index = index
            target_bbox = bbox
            target_distance = distance
    return target_index, target_bbox, target_distance


def get_nearest_point_to_bbox(bbox: BBox, points: List[Point]) -> Tuple[int, Point, float]:
    """
    给定一个bbox和一组点，计算返回距离该bbox最近的点的信息
    :param bbox:
    :param points:
    :return:
    """
    assert points, "点不能为空"
    target_index, target_point, target_distance = None, None, None
    for index, point in enumerate(points):
        distance = get_point_to_bbox_distance(point, bbox)
        if target_index is None or distance < target_distance:
            target_index = index
            target_point = point
            target_distance = distance
    return target_index, target_point, target_distance


def is_two_bbox_cross(bbox1: BBox, bbox2: BBox) -> bool:
    """
    判断两个bbox是否相交（或一个bbox包含另外一个bbox）
    https://blog.csdn.net/s0rose/article/details/78831570
    :param bbox1:
    :param bbox2:
    :return:
    """
    return bbox1.rbx >= bbox2.ltx and bbox2.rbx >= bbox1.ltx and bbox1.rby >= bbox2.lty and bbox2.rby >= bbox1.lty


def cross(point1: Point, point2: Point, point3: Point) -> float:
    """
    跨立实验
    https://blog.csdn.net/s0rose/article/details/78831570
    :param point1:
    :param point2:
    :param point3:
    :return:
    """
    x1 = point2.x - point1.x
    y1 = point2.y - point1.y
    x2 = point3.x - point1.x
    y2 = point3.y - point1.y
    return x1 * y2 - x2 * y1


def get_bbox_from_segment(segment: Segment) -> BBox:
    """
    以一条线段为对角线构造bbox
    :param segment:
    :return:
    """
    ltx = min(segment.point1.x, segment.point2.x)
    rbx = max(segment.point1.x, segment.point2.x)
    lty = min(segment.point1.y, segment.point2.y)
    rby = max(segment.point1.y, segment.point2.y)
    return BBox(ltx=ltx, lty=lty, rbx=rbx, rby=rby)


def is_two_segment_cross(segment1: Segment, segment2: Segment) -> bool:
    """
    判断两个线段是否相交
    https://blog.csdn.net/s0rose/article/details/78831570
    :param segment1:
    :param segment2:
    :return:
    """
    bbox1 = get_bbox_from_segment(segment1)
    bbox2 = get_bbox_from_segment(segment2)
    if is_two_bbox_cross(bbox1, bbox2):
        if cross(segment1.point1, segment1.point2, segment2.point1) * cross(segment1.point1, segment1.point2,
                                                                            segment2.point2) <= 0 and \
                cross(segment2.point1, segment2.point2, segment1.point1) * cross(segment2.point1, segment2.point2,
                                                                                 segment1.point2) <= 0:
            return True
        else:
            return False
    else:
        return False


def is_segment_bbox_cross(segment: Segment, bbox: BBox) -> bool:
    """
    判断一个线段和bbox是否相交，即线段与bbox的任意一条边相交或线段在bbox内
    :param segment:
    :param bbox:
    :return:
    """
    if is_point_in_bbox(segment.point1, bbox):
        return True
    if is_point_in_bbox(segment.point2, bbox):
        return True
    if is_two_segment_cross(segment, bbox.left) or is_two_segment_cross(segment, bbox.top) or is_two_segment_cross(
            segment, bbox.right) or is_two_segment_cross(segment, bbox.bottom):
        return True
    return False


def get_bbox_inter_size(bbox1: BBox, bbox2: BBox) -> float:
    """
    计算俩bbox的交集面积
    """
    x_min = max(bbox1.ltx, bbox2.ltx)
    y_min = max(bbox1.lty, bbox2.lty)
    x_max = min(bbox1.rbx, bbox2.rbx)
    y_max = min(bbox1.rby, bbox2.rby)
    if (x_max - x_min) < 0 or (y_max - y_min) < 0:
        return 0
    return (x_max - x_min) * (y_max - y_min)
