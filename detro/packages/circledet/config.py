from detro.configs import DetConfigBase
from wpcv.utils import data_aug, det_aug, img_aug, pil_ops
from detro.packages.circledet import CircleDetDataset, CircleDetector, CircleNet, CircleDetCriterion, train, test
class CircleDetConfig(DetConfigBase):
    FONT_PATH='/home/ars/sda6/work/ProjectBox/QuadNet/detro/data/msyh.ttf'
    INPUT_SIZE = (512, 512)
    HEATMAP_DOWNSAMPLE = 4
    CLASSES = ['0']
    DATA_DIRS = [
        ('/home/ars/sda5/data/projects/圆检测/data/imgs_train',
         '/home/ars/sda5/data/projects/圆检测/data/train_labels', 'voc', '.xml'),
    ]
    WEIGHTS_DIR = 'weights/training'
    TEST_WEIGHTS_PATH='weights/training/model_best.pkl'
    TEST_DIR = '/home/ars/sda5/data/projects/圆检测/data/imgs_val'
    transform = data_aug.Compose([
        det_aug.ToPILImage(),
        data_aug.RandomOrder([
            det_aug.Limitsize(600),
            # det_aug.RandomShear(20, 30),
            det_aug.RandomRotate(90),
            det_aug.RandomTranslate([600, 600]),
            det_aug.RandomVerticalFlip(),
            det_aug.RandomHorizontalFlip(),
            data_aug.Zip([
                data_aug.Compose([
                    img_aug.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.4),
                    img_aug.RandomApply(pil_ops.gaussian_blur, p=0.3, radius=1),
                    img_aug.RandomApply(pil_ops.sp_noise, p=0.3),
                ]),
                data_aug.Identical()
            ]),
        ]),
        det_aug.Resize(INPUT_SIZE, keep_ratio=True, fillcolor='black'),
    ])
    criterion = staticmethod(CircleDetCriterion)
    def get_dataset(self):
        return CircleDetDataset(self)
    def get_model(self):
        return CircleNet(num_classes=len(self.CLASSES))
    def get_detector(self):
        return CircleDetector(self,model=self.get_model(),WEIGHTS_PATH=self.TEST_WEIGHTS_PATH)
    def test(self):
        test(self)
    def train(self):
        train(self)
    def init(self):
        super().init()
        self.dataset=self.get_dataset()