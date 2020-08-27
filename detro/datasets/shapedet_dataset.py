import torch
import numpy as np
import os, glob, json, random
from torch.utils.data import Dataset
import cv2
from detro import utils as ut
from .det_dataset import DetDataset


class ShapeDetDataset(DetDataset):
    HEATMAP_DOWNSAMPLE = 4
    NUM_REG_PARAMS = None
    REG_MODE = 'offsets'

    __required__ = DetDataset.__required__ + ['CLASSES', 'HEATMAP_DOWNSAMPLE', 'transform','NUM_REG_PARAMS']

    def data_augmentation(self, img, objs):
        objs.sort(key=lambda obj: np.array(obj[0])[:, 0].mean())
        shapes = [ut.organize_polygon_points(shape) for shape, cls in objs]
        clses = [cls for shape, cls in objs]
        if self.transform:
            img, shapes = self.transform(img, shapes)
        objs = list(zip(shapes, clses))
        return img, objs

    def transform_points(self, points):
        mode = self.REG_MODE
        points = np.array(points)
        center = points.mean(axis=0)
        if mode == 'offsets-polar':
            points = ut.organize_polygon_points(points)
            return center, ut.cartesian2polar(points - center, to_degree=True).reshape((-1))
        if mode == 'fcos':
            points = ut.bounding_rect(points).reshape((2, 2))
            center = points.mean(axis=0)
            return center, np.abs(points - center).reshape(-1)
        if mode == 'centernet':
            p1, p2 = points = ut.bounding_rect(points).reshape((2, 2))
            w, h = abs(p1[0] - p2[0]), abs(p1[1] - p2[1])
            center = points.mean(axis=0)
            return center, np.array([w, h])
        if mode == 'circle':
            points = ut.organize_polygon_points(points)
            center = points.mean(axis=0)
            radius = 0.5 * ut.polygon_length(points) / len(points)
            return center, radius
        if mode == 'offsets':
            points = ut.organize_polygon_points(points)
            return center, (points - center).reshape((-1))
        else:
            raise

    def process_batch_annots(self, batch_annots):
        down = self.HEATMAP_DOWNSAMPLE
        batch_size = len(batch_annots)
        imw, imh = self.INPUT_SIZE
        hm_h, hm_w = int(imh / down), int(imw / down)
        batch_imgs = np.zeros((batch_size, 3, imh, imw), dtype=np.float32)
        batch_center_heatmap = np.zeros((batch_size, self.NUM_CLASSES, hm_h, hm_w), dtype=np.float32)
        batch_corner_heatmap = np.zeros((batch_size, self.NUM_CLASSES, hm_h, hm_w), dtype=np.float32)
        batch_offsets = np.zeros((batch_size, self.NUM_REG_PARAMS, hm_h, hm_w), dtype=np.float32)
        batch_offsets_mask = np.zeros((batch_size, 1, hm_h, hm_w), dtype=np.float32)
        for i, (img_path, objs) in enumerate(batch_annots):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img, objs = self.data_augmentation(img, objs)
            img = np.array(img)
            img = img / 255
            img = ut.normalize(img)
            batch_imgs[i, ...] = np.transpose(img, (2, 0, 1))
            for points, cls in objs:
                points = np.array(points)
                center, regs = self.transform_points(points)
                cx, cy = (center / down).astype(np.int)
                ut.draw_points_heatmap(batch_center_heatmap[i, cls], [[cx, cy]])
                ut.draw_points_heatmap(batch_corner_heatmap[i, cls], (points / 4).astype(np.int))
                batch_offsets[i, :, cy, cx] = regs
                batch_offsets_mask[i, 0, cy, cx] = 1
        batch_imgs = torch.from_numpy(batch_imgs).to(self.DEVICE)
        batch_center_heatmap = torch.from_numpy(batch_center_heatmap).to(self.DEVICE)
        batch_corner_heatmap = torch.from_numpy(batch_corner_heatmap).to(self.DEVICE)
        batch_offsets = torch.from_numpy(batch_offsets).to(self.DEVICE)
        batch_offsets_mask = torch.from_numpy(batch_offsets_mask).to(self.DEVICE)
        batch_labels = dict(
            center_heatmap=batch_center_heatmap,
            corner_heatmap=batch_corner_heatmap,
            offsets=batch_offsets,
            offsets_mask=batch_offsets_mask,
        )
        return batch_imgs, batch_labels


class CircleDetDataset(ShapeDetDataset):
    REG_MODE = 'circle'
    NUM_REG_PARAMS = 1
class CenterDetDataset(ShapeDetDataset):
    REG_MODE = 'centernet'
    NUM_REG_PARAMS = 2
class FCOSDetDataset(ShapeDetDataset):
    REG_MODE = 'fcos'
    NUM_REG_PARAMS = 4
class PolygonDetDataset(ShapeDetDataset):
    REG_MODE = 'offsets'



