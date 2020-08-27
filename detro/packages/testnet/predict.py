from detro import utils as ut
from detro.predict import ClassicalHeatmapDetectorBase, ClassicalCenterOffsetsDetector


class TestDetector(ClassicalCenterOffsetsDetector):
    @classmethod
    def make_params(cls, input_size, num_points, num_classes=None, classes=None,
                    heatmap_downsample=4, score_thresh=0.1):
        params = ClassicalCenterOffsetsDetector.make_params(input_size=input_size, num_points=num_points,
                                                            num_classes=num_classes, classes=classes,
                                                            max_output_num=1,
                                                            heatmap_downsample=heatmap_downsample,
                                                            score_thresh=score_thresh)
        return params
    def get_middle_results(self):
        outputs=self.cache['outputs']
        center_heatmap_img=self.heatmap2image(outputs['center_heatmap'][0,0])
        corner_heatmap_img=self.heatmap2image(outputs['corner_heatmap'][0,0])
        return dict(
            center_heatmap_img=center_heatmap_img,corner_heatmap_img=corner_heatmap_img
        )
    def decode_outputs(self, outputs, down, *args, **kwargs):
        num_points = self.params.num_points
        heatmap = outputs['center_heatmap']
        offsets = outputs['offsets']
        return ut.decode_outputs_polygondet(heatmap, offsets, num_points=num_points, down=down,
                                            k=self.params.max_output_num, nms_radius=3)
