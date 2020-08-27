import math
import numpy as np


def _get_rotate_info(angle, cx, cy, h, w):
    import math
    import numpy as np
    angle = math.radians(angle)
    cos = math.cos(angle)
    sin = math.sin(angle)
    M = np.array([[cos, sin], [-sin, cos]])
    cos = abs(cos)
    sin = abs(sin)
    nw = w * cos + h * sin
    nh = w * sin + h * cos
    return M, nw, nh


def rotate_points(points, angle, cx, cy, h, w):
    import numpy as np
    M, nw, nh = _get_rotate_info(angle, cx, cy, h, w)
    points = np.array(points)
    original_shape = points.shape
    points = points.reshape((-1, 2))
    points = M.dot((points - np.array([cx, cy])).T).T + np.array([nw // 2, nh // 2])
    points = points.reshape(original_shape)
    return points


def polar2cartesian(points, from_degree=False):
    points = np.array(points)
    shape = points.shape
    points = points.reshape((-1, 2))
    batch_angle = points[:, 1:2]
    if from_degree:
        batch_angle = np.radians(batch_angle)
    batch_angle -= np.pi
    batch_x = points[:, 0:1] * np.cos(batch_angle)
    batch_y = points[:, 0:1] * np.sin(batch_angle)
    points = np.concatenate((batch_x, batch_y), axis=1)
    points = points.reshape(shape)
    return points


def cartesian2polar(points, to_degree=False):
    points = np.array(points)
    shape = points.shape
    points = points.reshape((-1, 2))
    batch_radius = np.expand_dims(np.sqrt(np.square(points).sum(axis=1)), axis=-1)
    batch_angle = np.expand_dims(np.arctan2(points[:, 1], points[:, 0]), axis=-1) + np.pi
    if to_degree:
        batch_angle = np.degrees(batch_angle)
    points = np.concatenate((batch_radius, batch_angle), axis=1)
    points = points.reshape(shape)
    return points


def bounding_rect(points):
    points = np.array(points)
    l = np.min(points[:, 0])
    r = np.max(points[:, 0])
    t = np.min(points[:, 1])
    b = np.max(points[:, 1])
    return np.array([l, t, r, b])


def pad_quad(quad, pad_size=5, pad_ratio=None, limits=None):
    import math
    from math import inf
    limits = limits or (-inf, -inf, inf, inf)

    def dist(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def mean(li):
        return sum(li) / len(li)

    def minus(la, lb):
        return [x - y for x, y in zip(la, lb)]

    def add(la, lb):
        return [x + y for x, y in zip(la, lb)]

    def get_pad_params(pad_size, pad_ratio, quad):
        # l, t, r, b = box
        p0, p1, p2, p3 = quad
        bh = mean([dist(p0, p3), dist(p1, p2)])
        bw = mean([dist(p0, p1), dist(p3, p2)])
        if pad_ratio is not None:
            if isinstance(pad_ratio, (tuple, list)):
                if len(pad_ratio) == 2:
                    pad_l = pad_r = int(min(bh, bw) * pad_ratio[0])
                    pad_t = pad_b = int(min(bh, bw) * pad_ratio[1])
                else:
                    assert len(pad_ratio) == 4
                    pad_l = int(min(bh, bw) * pad_ratio[0])
                    pad_t = int(min(bh, bw) * pad_ratio[1])
                    pad_r = int(min(bh, bw) * pad_ratio[2])
                    pad_b = int(min(bh, bw) * pad_ratio[3])
            else:
                pad_l = pad_t = pad_r = pad_b = int(min(bh, bw) * pad_ratio)
        else:
            if isinstance(pad_size, (tuple, list)):
                if len(pad_size) == 2:
                    pad_l = pad_r = pad_size[0]
                    pad_t = pad_b = pad_size[1]
                else:
                    assert len(pad_size) == 4
                    pad_l, pad_t, pad_r, pad_b = pad_size
            else:
                pad_l = pad_t = pad_r = pad_b = pad_size
        return pad_l, pad_t, pad_r, pad_b

    pad_l, pad_t, pad_r, pad_b = get_pad_params(pad_size, pad_ratio, quad)
    p0, p1, p2, p3 = quad
    p0 = add(p0, [-pad_l, -pad_t])
    p1 = add(p1, [pad_r, -pad_t])
    p2 = add(p2, [pad_r, pad_b])
    p3 = add(p3, [-pad_l, pad_b])

    return limit_quad([p0, p1, p2, p3], limits=limits)

def points_distance(p0,p1):
    return np.sqrt(np.sum(np.square(np.array(p0)-np.array(p1))))
def points_center(points):
    return np.array(points).mean(axis=0)
def limit_point(p, limits):
    if limits is None: return p
    if len(limits) == 2:
        ml, mt = 0, 0
        mr, mb = limits
    else:
        assert len(limits) == 4
        ml, mt, mr, mb = limits
    x, y = p
    x = max(ml, min(mr, x))
    y = max(mt, min(mb, y))
    return [x, y]


def limit_quad(quad, limits=None):
    quad = [limit_point(p, limits) for p in quad]
    return quad


def organize_quad_points(points):
    points = np.array(points)
    center = points.mean(axis=0)
    angles = np.arctan2(center[1] - points[:, 1], points[:, 0] - center[0]) * 180 / np.pi
    ids = np.argsort(angles)[::-1]
    polygon = points[ids]
    return polygon


def organize_polygon_points(points):
    points = np.array(points)
    center = points.mean(axis=0)
    angles = np.arctan2(center[1] - points[:, 1], points[:, 0] - center[0]) * 180 / np.pi
    ids = np.argsort(angles)[::-1]
    polygon = points[ids]
    return polygon


def polygon_length(points):
    assert len(points) >= 3
    points=list(points)
    points+=[points[0]]
    dist=0
    for i in range(len(points)-1):
        p1=points[i]
        p2=points[i+1]
        dist+=np.sqrt(np.sum(np.square(np.array(p1)-np.array(p2))))
    return dist


def calc_polygon_area(points):
    area = 0
    q = points[-1]
    for p in points:
        area += p[0] * q[1] - p[1] * q[0]
        q = p
    return abs(area / 2)


def calc_quad_angle(quad):
    '''cal angle of a quad'''
    p0, p1, p2, p3 = quad

    def center(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return (x1 + x2) / 2, (y1 + y2) / 2

    x1, y1 = center(p0, p3)
    x2, y2 = center(p1, p2)
    angle = math.atan2(y1 - y2, x2 - x1) * 180 / math.pi
    return angle
