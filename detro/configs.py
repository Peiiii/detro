import torch
import os
import copy

# class ConfigOld:
#     class __empty_:pass
#     @classmethod
#     def _to_dict(cls,recursive=True,soft_merge=False):
#         '''soft_merge: try to select and non-None value from [cls,cls.__base__,...]'''
#         if recursive and issubclass(cls.__base__,Config):
#             dic=cls.__base__._to_dict(recursive=recursive,soft_merge=soft_merge)
#         else:
#             dic={}
#         for k,v in cls.__dict__.items():
#             if not k.startswith('_'):
#                 if soft_merge and v is None:
#                     v_old=dic.get(k,None)
#                     if v_old is not None:
#                         v=v_old
#                 dic[k]=v
#         return dic
#     @classmethod
#     def _get(cls,key,default=__empty_):
#         if hasattr(cls,key):
#             return getattr(cls,key)
#         if issubclass(cls.__base__,Config):
#             return cls.__base__._get(key,default)
#         else:
#             if default is Config.__empty_:
#                 raise AttributeError('Config class %s has no attribute %s.'%(cls,key))
#             else:
#                 return default
# class config_metaclass(type):
#     def __new__(cls, name,bases,attrs):
#         base=bases[0] if len(bases) else None
#         if base and hasattr(base,'__attr_dict__'):
#             attr_dict=base.__attr_dict__.copy()
#         else:
#             attr_dict={}
#         for k,v in attrs.items():
#             if not k.startswith('__'):
#                 attr_dict[k]=v
#         attrs['__attr_dict__']=attr_dict
#         return type.__new__(cls,name,bases,attrs)
# class Config(metaclass=config_metaclass):
#     def __init__(self,**kwargs):
#         self.__attrs_dict__=copy.deepcopy(self.__class__.__attrs_dict__)
#         self.__attrs_dict__.update(**kwargs)
#     @classmethod
#     def __to_dict__(cls):
#         return cls.__attr_dict__.copy()
#
# class DemoBase(object):
#     class __config__(Config):
#         BATCH_SIZE=4
#         INPUT_SIZE=(512,512)
#     class __allowed_options__(Config):
#         BATCH_SIZE=None
#
#     def __init__(self,cfg=None,**kwargs):
#         self.set_attrs(self.__config__.__attrs_dict__.cppy())
#         if cfg:
#             pass
#         self.set_attrs(kwargs)
#     def set_attrs(self,attrs):
#         for k,v in attrs.items():
#             setattr(self,k,v)
#
#
# class Demo(DemoBase,object):
#     class __config__(DemoBase.__config__):
#         BATCH_SIZE = 4


# print(Demo.__config__.__to_dict__())

# print(Demo.__bases__)




class ConfigBase:
    __required__ = []
    __methods__=['init','update','train','test']
    def __init__(self, init=True, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.check_required()
        if init:
            self.init()

    def init(self):
        pass

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def check_required(cls):
        for key in cls.__required__:
            if getattr(cls, key, None) is None:
                raise Exception('Argument %s is required but not given.' % (key))

    @classmethod
    def to_dict(cls):
        if issubclass(cls.__base__,ConfigBase):
            dic=cls.__base__.to_dict()
        else:
            dic = {}
        for k, v in cls.__dict__.items():
            if not (k.startswith('__') and k.endswith('__') and k not in cls.__methods__):
                dic[k] = v
        return dic


class ConfigHintBase(ConfigBase):
    INPUT_SIZE = None
    MAX_EPOCHS = None
    CLASSES = None
    NUM_CLASSES = None
    NUM_POINTS = None
    NUM_REG_PARAMS=None
    BATCH_SIZE = None
    LEARN_RATE_INIT = None
    LEARN_RATE_END = None
    WEIGHTS_SAVE_INTERVAL = None
    WEIGHTS_DIR = None
    WEIGHTS_INIT = None
    DEVICE = None
    train_transform = None
    criterion = None
    get_model = None
    get_train_data = None
    get_val_data = None
    get_predictor = None
    TRAIN_DIR = None
    VAL_DIR = None
    TEST_DIR = None
    DATA_DIR = None
    DET_TYPE=None
    __required__ = []


class DetConfigBase(ConfigHintBase):
    BATCH_SIZE = 4
    INPUT_SIZE = (512, 512)
    MAX_EPOCHS = 200
    LEARN_RATE_INIT = 1e-4
    LEARN_RATE_END = 1e-6
    WEIGHTS_SAVE_INTERVAL = 20
    WEIGHTS_DIR = 'weights/training'
    WEIGHTS_INIT = None
    criterion = None
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    __required__ = ['get_model', 'criterion']
    __dataset__ = None
    __model__ = None

    def get_dataset(self):
        raise NotImplementedError

    def get_model(self):
        raise NotImplementedError

    def get_detector(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def init(self):
        if not os.path.exists(self.WEIGHTS_DIR):
            os.makedirs(self.WEIGHTS_DIR)

