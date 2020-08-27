import torch.nn.functional as F
import torch
import numpy as np
from .polygon_utils import cartesian2polar, polar2cartesian


def heatmap_nms(heat, kernel=3):
    hmax = F.max_pool2d(heat, kernel, stride=1, padding=(kernel - 1) // 2)
    keep = (hmax == heat).float()
    return heat * keep


def get_reshaped_index(index, shape, coords=[]):
    dim = shape[-1]
    coord = index % dim
    index = index // dim
    coords = [coord] + coords
    shape = shape[:-1]
    if not len(shape):
        return coords
    else:
        return get_reshaped_index(index, shape, coords)


def heatmap_topk_with_distance(heatmap, distance, k):
    shape = heatmap.shape
    num_classes = shape[0]
    flat = heatmap.view((num_classes, -1))
    values, indexes = torch.topk(flat, k=k, dim=1)
    num_maps = distance.shape[0]
    dists = torch.transpose(distance, 0, 1)
    dists = torch.transpose(dists, 1, 2)
    dists = dists.reshape((-1, num_maps))
    dists = dists[indexes]
    indexes = indexes.numpy()
    indexes = [[get_reshaped_index(index, shape[1:]) for index in cls_indexes] for cls_indexes in indexes]
    indexes = np.array(indexes)
    values = values.numpy()
    dists = dists.numpy()
    return indexes, values, dists

def heatmap_topk_with_feature(heatmap, feature, k):
    shape = heatmap.shape
    num_classes = shape[0]
    flat = heatmap.view((num_classes, -1))
    values, indexes = torch.topk(flat, k=k, dim=1)
    num_maps = feature.shape[0]
    dists = torch.transpose(feature, 0, 1)
    dists = torch.transpose(dists, 1, 2)
    dists = dists.reshape((-1, num_maps))
    dists = dists[indexes]
    indexes = indexes.numpy()
    indexes = [[get_reshaped_index(index, shape[1:]) for index in cls_indexes] for cls_indexes in indexes]
    indexes = np.array(indexes)
    values = values.numpy()
    dists = dists.numpy()
    return indexes, values, dists


def heatmap_topk(heatmap, k):
    shape = heatmap.shape
    num_classes = shape[0]
    flat = heatmap.view((num_classes, -1))
    values, indexes = torch.topk(flat, k=k, dim=1)
    indexes = indexes.numpy()
    indexes = [[get_reshaped_index(index, shape[1:]) for index in cls_indexes] for cls_indexes in indexes]
    indexes = np.array(indexes)
    values = values.numpy()
    return indexes, values


def decode_outputs_polygondet(heatmap, distance, down, k=100, num_points=4, nms_radius=3):
    heatmap = heatmap.detach().cpu()
    distance = distance.detach().cpu()
    heatmap = torch.sigmoid(heatmap)
    heatmap = heatmap_nms(heatmap, nms_radius)
    batch_size = heatmap.shape[0]
    num_classes = heatmap.shape[1]
    res = []
    for i in range(batch_size):
        indexes, scores, dists = heatmap_topk_with_distance(heatmap[i], distance[i], k=k)
        coords = []
        indexes = np.array(indexes) + np.array([0.5, 0.5])
        indexes *= down
        # dists *= down

        for cls_idx in range(num_classes):
            cls_coords = np.zeros((k, num_points * 2 + 2))
            indexes_cls = indexes[cls_idx][:, ::-1]
            cls_coords[:, 0:num_points * 2] = (
                    indexes_cls.reshape((-1, 1, 2)) + dists[cls_idx].reshape((-1, num_points, 2))).reshape(
                (-1, num_points * 2))
            cls_coords[:, num_points * 2] = scores[cls_idx]
            cls_coords[:, num_points * 2 + 1] = cls_idx
            coords.append(cls_coords)
        coords = np.concatenate(coords, axis=0)
        res.append(coords)
    return np.array(res)


def decode_outputs_circledet(heatmap, distance, down, k=100, nms_radius=3):
    heatmap = heatmap.detach().cpu()
    distance = distance.detach().cpu()
    heatmap = torch.sigmoid(heatmap)
    heatmap = heatmap_nms(heatmap, nms_radius)
    batch_size = heatmap.shape[0]
    num_classes = heatmap.shape[1]
    res = []
    num_params=3
    for i in range(batch_size):
        indexes, scores, dists = heatmap_topk_with_feature(heatmap[i], distance[i], k=k)
        coords = []
        indexes = np.array(indexes) + np.array([0.5, 0.5])
        indexes *= down
        # dists *= down
        for cls_idx in range(num_classes):
            cls_coords = np.zeros((k, num_params + 2))
            indexes_cls = indexes[cls_idx][:, ::-1]
            cls_coords[:, 0:num_params] = np.c_[indexes_cls,dists[cls_idx]]
            cls_coords[:, num_params] = scores[cls_idx]
            cls_coords[:, num_params + 1] = cls_idx
            coords.append(cls_coords)
        coords = np.concatenate(coords, axis=0)
        res.append(coords)
    return np.array(res)


def decode_outputs_polygondet_polar(heatmap, distance, num_points, down, k=100):
    heatmap = heatmap.detach().cpu()
    distance = distance.detach().cpu()
    heatmap = torch.sigmoid(heatmap)
    heatmap = heatmap_nms(heatmap, 3)
    batch_size = heatmap.shape[0]
    num_classes = heatmap.shape[1]
    res = []
    for i in range(batch_size):
        indexes, scores, dists = heatmap_topk_with_distance(heatmap[i], distance[i], k=k)
        indexes = np.array(indexes) + np.array([0.5, 0.5])
        coords = []
        for cls_idx in range(num_classes):
            cls_coords = np.zeros((k, num_points * 2 + 2))
            indexes_cls = indexes[cls_idx][:, ::-1]
            indexes_cls *= down
            # distance*=down
            cls_coords[:, 0:num_points * 2] = (
                    indexes_cls.reshape((-1, 1, 2)) + polar2cartesian(dists[cls_idx].reshape((-1, num_points, 2)),
                                                                      from_degree=True)
            ).reshape((-1, num_points * 2))
            cls_coords[:, num_points * 2] = scores[cls_idx]
            cls_coords[:, num_points * 2 + 1] = cls_idx
            coords.append(cls_coords)
        coords = np.concatenate(coords, axis=0)
        res.append(coords)
    return np.array(res)


def decode_outputs_fcosdet(heatmap, distance, down, k=100):
    num_points = 2
    heatmap = heatmap.detach().cpu()
    distance = distance.detach().cpu()
    heatmap = torch.sigmoid(heatmap)
    heatmap = heatmap_nms(heatmap, 3)
    batch_size = heatmap.shape[0]
    num_classes = heatmap.shape[1]
    res = []
    for i in range(batch_size):
        indexes, scores, dists = heatmap_topk_with_distance(heatmap[i], distance[i], k=k)
        indexes = np.array(indexes) + np.array([0.5, 0.5])
        indexes *= down
        # dists *= down
        coords = []
        for cls_idx in range(num_classes):
            cls_coords = np.zeros((k, num_points * 2 + 2))
            indexes_cls = indexes[cls_idx][:, ::-1]
            cls_coords[:, 0:num_points * 2] = (
                    indexes_cls.reshape((-1, 1, 2)) + (dists[cls_idx] * np.array([-1, -1, 1, 1])).reshape(
                (-1, num_points, 2))
            ).reshape((-1, num_points * 2))
            cls_coords[:, num_points * 2] = scores[cls_idx]
            cls_coords[:, num_points * 2 + 1] = cls_idx
            coords.append(cls_coords)
        coords = np.concatenate(coords, axis=0)
        res.append(coords)
    return np.array(res)


def decode_outputs_centerdet(heatmap, distance, down, k=100):
    num_points = 2
    heatmap = heatmap.detach().cpu()
    distance = distance.detach().cpu()
    heatmap = torch.sigmoid(heatmap)
    heatmap = heatmap_nms(heatmap, 3)
    batch_size = heatmap.shape[0]
    num_classes = heatmap.shape[1]
    res = []
    for i in range(batch_size):
        indexes, scores, dists = heatmap_topk_with_distance(heatmap[i], distance[i], k=k)
        indexes = np.array(indexes) + np.array([0.5, 0.5])
        indexes *= down
        # dists *= down
        coords = []
        for cls_idx in range(num_classes):
            cls_coords = np.zeros((k, num_points * 2 + 2))
            indexes_cls = indexes[cls_idx][:, ::-1]
            cls_coords[:, 0:num_points * 2] = np.concatenate(
                (indexes_cls - dists[cls_idx] / 2, indexes_cls + dists[cls_idx] / 2), axis=1)
            cls_coords[:, num_points * 2] = scores[cls_idx]
            cls_coords[:, num_points * 2 + 1] = cls_idx
            coords.append(cls_coords)
        coords = np.concatenate(coords, axis=0)
        res.append(coords)
    return np.array(res)


def decode_outputs_polygondet_single(heatmap, num_points, down, nms_radius=3):
    '''heatmap corner'''
    k = 1
    heatmap = heatmap.detach().cpu()
    heatmap = torch.sigmoid(heatmap)
    heatmap = heatmap_nms(heatmap, nms_radius)
    batch_size = heatmap.shape[0]
    num_classes = heatmap.shape[1]
    res = []
    for i in range(batch_size):
        indexes, scores = heatmap_topk(heatmap[i], k=num_points)
        indexes = np.array(indexes) + np.array([0.5, 0.5])
        indexes *= down
        coords = []
        for cls_idx in range(num_classes):
            cls_coords = np.zeros((k, num_points * 2 + 2))
            indexes_cls = indexes[cls_idx][:, ::-1]
            cls_coords[0, 0:num_points * 2] = indexes_cls.reshape((-1))
            cls_coords[0, num_points * 2] = scores[cls_idx].min()
            cls_coords[0, num_points * 2 + 1] = cls_idx
            coords.append(cls_coords)
        coords = np.concatenate(coords, axis=0)
        res.append(coords)
    return np.array(res)
