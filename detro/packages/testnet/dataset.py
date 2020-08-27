import torch
import numpy as np
import os, glob, json, random
from torch.utils.data import Dataset
import cv2
from detro import utils as ut
from detro.dataset.det_dataset import DetDataset




class PointsDetDataset(DetDataset):

	def __init__(self, img_dir=None, label_dir=None, data_dirs=None, num_classes=None, num_points=None, classes=None,
	             batch_size=8,
	             input_size=(512, 512), params=None, transform=None, device=None):
		default_params = dict(
			heatmap_downsample=4,
			points_transform='none'
		)
		default_params.update(**(params or {}))
		params = default_params

		super().__init__(
			img_dir=img_dir, label_dir=label_dir, data_dirs=data_dirs, num_classes=num_classes, classes=classes,
			batch_size=batch_size, input_size=input_size, transform=transform, params=params, device=device,
		)
		self.num_points = num_points or self.params.get('num_points') or 4

	def data_augmentation(self, img, objs):
		objs.sort(key=lambda obj: np.array(obj[0])[:, 0].mean())
		shapes = [ut.organize_polygon_points(shape) for shape, cls in objs]
		clses = [cls for shape, cls in objs]
		if self.transform:
			img, shapes = self.transform(img, shapes)
		objs = list(zip(shapes, clses))
		return img, objs

	def transform_points(self, points):
		points_transform = self.params['points_transform'].split(',')
		points = ut.organize_polygon_points(points)
		points = np.array(points)
		center = points.mean(axis=0)
		offsets = points - center
		if 'none' in points_transform:
			return points
		if 'cartesian' in points_transform:
			offsets = offsets.reshape((-1))
		elif 'polar' in points_transform:
			offsets = ut.cartesian2polar(offsets, to_degree=True).reshape((-1))
		if 'abs' in points_transform:
			offsets = np.abs(offsets)
		return offsets

	def process_batch_annots(self, batch_annots):
		down = self.params['heatmap_downsample']
		batch_size = len(batch_annots)
		imw, imh = self.input_size
		hm_h, hm_w = int(imh / down), int(imw / down)
		batch_imgs = np.zeros((batch_size, 3, imh, imw), dtype=np.float32)
		batch_center_heatmap = np.zeros((batch_size, self.num_classes, hm_h, hm_w), dtype=np.float32)
		batch_corner_heatmap = np.zeros((batch_size, self.num_classes, hm_h, hm_w), dtype=np.float32)
		batch_offsets = np.zeros((batch_size, self.num_points * 2, hm_h, hm_w), dtype=np.float32)
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
				points=ut.organize_polygon_points(points)
				points = np.array(points)
				offsets=(points-points.mean(axis=0)).reshape((-1))
				cx,cy=center=(points.mean(axis=0)/down).astype(np.int)
				points=(points/down).astype(np.int)
				ut.draw_points_heatmap(batch_center_heatmap[i, cls], [center])
				batch_offsets[i,:,cy,cx]=offsets/down
				batch_offsets_mask[i,0,cy,cx]=1
				for pnt in points:
					ut.draw_points_heatmap(batch_corner_heatmap[i, cls], [pnt])
		batch_imgs = torch.from_numpy(batch_imgs).to(self.device)
		batch_center_heatmap = torch.from_numpy(batch_center_heatmap).to(self.device)
		batch_corner_heatmap = torch.from_numpy(batch_corner_heatmap).to(self.device)
		batch_offsets = torch.from_numpy(batch_offsets).to(self.device)
		batch_offsets_mask = torch.from_numpy(batch_offsets_mask).to(self.device)
		batch_labels = dict(
			center_heatmap=batch_center_heatmap,
			corner_heatmap=batch_corner_heatmap,
			offsets=batch_offsets,
			offsets_mask=batch_offsets_mask,
		)
		return batch_imgs, batch_labels
