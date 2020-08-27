from detro.networks.models.polygonnet import PolygonNet,PolygonDetCriterion
from detro.datasets import PolygonDetDataset
from detro.predictors import PolygonDetDetector
from detro.training import train
from detro.testing import test
from detro.configs import DetConfigBase
from wpcv.utils import data_aug, det_aug, img_aug, pil_ops
class PolygonDetConfig(DetConfigBase):
    DET_TYPE = 'polygon'
    FONT_PATH='/home/ars/sda6/work/ProjectBox/QuadNet/detro/data/msyh.ttf'
    INPUT_SIZE = (512, 512)
    BATCH_SIZE = 4
    NUM_POINTS = 4
    NUM_REG_PARAMS = NUM_POINTS*2
    HEATMAP_DOWNSAMPLE = 4
    CLASSES = ['0']
    WEIGHTS_DIR = 'weights/training'
    TEST_WEIGHTS_PATH='weights/training/model_best.pkl'
    transform = data_aug.Compose([
        det_aug.ToPILImage(),
        data_aug.RandomOrder([
            det_aug.Limitsize(600),
            # det_aug.RandomShear(20, 30),
            # det_aug.RandomRotate(30),
            det_aug.RandomTranslate([200, 200]),
            # det_aug.RandomVerticalFlip(),
            # det_aug.RandomHorizontalFlip(),
            data_aug.Zip([
                data_aug.Compose([
                    img_aug.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.4),
                    img_aug.RandomApply(pil_ops.gaussian_blur, p=0.3, radius=1),
                    img_aug.RandomApply(pil_ops.sp_noise, p=0.3),
                ]),
                data_aug.Identical()
            ]),
        ]),
        det_aug.Resize(INPUT_SIZE, keep_ratio=False, fillcolor='black'),
    ])
    criterion = staticmethod(PolygonDetCriterion)
    def get_dataset(self):
        return PolygonDetDataset(self)
    def get_model(self):
        return PolygonNet(num_classes=len(self.CLASSES),num_points=self.NUM_POINTS)
    def get_detector(self):
        return PolygonDetDetector(self,model=self.get_model(),WEIGHTS_PATH=self.TEST_WEIGHTS_PATH)
    def test(self):
        test(self)
    def train(self):
        train(self)
    def init(self):
        super().init()
        self.dataset=self.get_dataset()