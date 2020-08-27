import torch
from detro.configs import ConfigBase
class PredictConfig(ConfigBase):
    INPUT_SIZE=None
    CLASSES=None
    NUM_CLASSES=None
    NUM_POINTS=None
    HEATMAP_DOWNSAMPLE=None
    MAX_OUTPUT_NUM=None
    SCORE_THRESH=None
    DEVICE=None
    WEIGHTS_PATH=None
    model=None
    transform=None

    def get_model(self):
        raise
    def init(self):
        if not self.model:
            self.model=self.get_model()
        self.model.load_state_dict(torch.load(self.WEIGHTS_PATH))
        self.model.to(self.DEVICE)
        self.model.eval()
class _HintBase:
    INPUT_SIZE = None
    CLASSES = None
    NUM_CLASSES = None
    NUM_POINTS = None
    HEATMAP_DOWNSAMPLE = None
    MAX_OUTPUT_NUM = None
    SCORE_THRESH = None
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    WEIGHTS_PATH = None
    model = None
    transform = None
    @classmethod
    def get_options(cls):
        dic = {}
        for k, v in cls.__dict__.items():
            if not (k.startswith('__') and k.endswith('__') and k not in ['init']):
                dic[k] = v
        return dic

class PredictorBase(_HintBase):
    __allowed_options__=_HintBase.get_options()
    __required__=['model','INPUT_SIZE','DEVICE']
    def __init__(self, cfg=None,**kwargs):
        self.cache={}
        if cfg:
            assert isinstance(cfg, ConfigBase)
            self.set_options(cfg)
            self.set_options(kwargs)
        self.init()
        self.check_required()
    def set_options(self, cfg):
        if not isinstance(cfg,dict):
            cfg=cfg.to_dict()
        for k, v in cfg.items():
            if k in self.__allowed_options__:
                if v is None:
                    v= getattr(self,k,None)
                setattr(self, k, v)
    def check_required(self):
        for key in self.__required__:
            if getattr(self, key, None) is None:
                raise Exception('Argument %s is required but not given.' % (key))

    def init(self):
        if self.WEIGHTS_PATH:
            self.model.load_state_dict(torch.load(self.WEIGHTS_PATH))
        if self.DEVICE:
            self.model.to(self.DEVICE)
        self.model.eval()

class Predictor(PredictorBase):
    def predict(self,img):
        pass
    def preprocess(self,img):
        pass
    def postprocess(self,res):
        pass


