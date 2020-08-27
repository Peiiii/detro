from detro import utils as ut
from .classical_detector import ClassicalDetectorBase
import numpy as np


def postprocess_result_circle(res, score_thresh=None, scaleX=None, scaleY=None, pad_x=None, pad_y=None):
    res[:, :2] = (res[:, :2] - np.array([pad_x, pad_y])) / np.array([scaleX, scaleY])
    res[:, 2] /= scaleX
    polygons = list(filter(lambda obj: obj[3] > score_thresh, res))
    polygons = np.array(polygons)
    return polygons


def postprocess_result_polygon(res, num_points, score_thresh=None, scaleX=None, scaleY=None, pad_x=None, pad_y=None):
    res[:, :num_points * 2] = (
            (res[:, :num_points * 2].reshape((-1, num_points, 2)) - np.array([pad_x, pad_y])) / np.array(
        [scaleX, scaleY])).reshape((-1, num_points * 2))
    polygons = list(filter(lambda polygon: polygon[num_points * 2] > score_thresh, res))
    polygons = np.array(polygons)
    return polygons


class CircleDetector(ClassicalDetectorBase):
    def decode_outputs(self, outputs, *args, **kwargs):
        heatmap = outputs['center_heatmap']
        offsets = outputs['offsets']
        return ut.decode_outputs_circledet(heatmap, offsets, down=self.HEATMAP_DOWNSAMPLE,
                                           k=self.MAX_OUTPUT_NUM, nms_radius=3)

    def postprocess(self, res, score_thresh=None, scaleX=None, scaleY=None, pad_x=None, pad_y=None):
        score_thresh = score_thresh if score_thresh is not None else self.SCORE_THRESH
        scaleX = self.cache['scaleX'] if scaleX is None else scaleX
        scaleY = self.cache['scaleY'] if scaleY is None else scaleY
        pad_x = self.cache['pad_x'] if pad_x is None else pad_x
        pad_y = self.cache['pad_y'] if pad_y is None else pad_y
        return postprocess_result_circle(res, score_thresh, scaleX, scaleY, pad_x, pad_y)


class PolygonDetectorBase(ClassicalDetectorBase):
    __required__ = ClassicalDetectorBase.__required__+['NUM_POINTS']
    def postprocess(self, res, score_thresh=None, scaleX=None, scaleY=None, pad_x=None, pad_y=None):
        score_thresh = score_thresh if score_thresh is not None else self.SCORE_THRESH
        scaleX = self.cache['scaleX'] if scaleX is None else scaleX
        scaleY = self.cache['scaleY'] if scaleY is None else scaleY
        pad_x = self.cache['pad_x'] if pad_x is None else pad_x
        pad_y = self.cache['pad_y'] if pad_y is None else pad_y
        return postprocess_result_polygon(res, self.NUM_POINTS, score_thresh, scaleX, scaleY, pad_x, pad_y)


class PolygonDetDetector(PolygonDetectorBase):
    def decode_outputs(self, outputs, down, k):
        num_points = self.NUM_POINTS
        heatmap = outputs['center_heatmap']
        offsets = outputs['offsets']
        return ut.decode_outputs_polygondet(heatmap, offsets, num_points=num_points, down=down, k=k)


class PolygonDetDetectorPolar(PolygonDetectorBase):

    def decode_outputs(self, outputs, down, k):
        num_points = self.NUM_POINTS
        heatmap = outputs['center_heatmap']
        offsets = outputs['offsets']
        return ut.decode_outputs_polygondet_polar(heatmap, offsets, num_points=num_points, down=down, k=k)

class FCOSDetector(PolygonDetectorBase):
    NUM_POINTS = 2
    def decode_outputs(self, outputs, down, k):
        heatmap = outputs['center_heatmap']
        offsets = outputs['offsets']
        return ut.decode_outputs_fcosdet(heatmap, offsets, down, k)


class CenterDetDetector(PolygonDetectorBase):
    NUM_POINTS = 2
    def decode_outputs(self, outputs, down, k):
        heatmap = outputs['center_heatmap']
        offsets = outputs['offsets']
        return ut.decode_outputs_centerdet(heatmap, offsets, down, k)
