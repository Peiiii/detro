import torch
import numpy as np
import os, glob, json, random
from torch.utils.data import Dataset
import cv2
from detro import utils as ut
from detro.configs import ConfigBase



class _HintBase:
    INPUT_SIZE = (512, 512)
    BATCH_SIZE = 4
    IMG_DIR = None
    LABEL_DIR = None
    DATA_DIRS=None
    NUM_REG_PARAMS=None
    ANNOT_FORMAT = 'voc'
    ANNOT_FILE_EXT = '.xml'
    NUM_POINTS = None
    CLASSES = None
    NUM_CLASSES = None
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = None
    CLS2IDX = None
    @classmethod
    def get_options(cls):
        dic = {}
        for k, v in cls.__dict__.items():
            if not (k.startswith('__') and k.endswith('__') and k not in ['init']):
                dic[k] = v
        return dic

class DetDatasetBase(_HintBase):
    __allowed_options__=list(_HintBase.get_options().keys())
    __required__=['BATCH_SIZE','INPUT_SIZE']
    def __init__(self,cfg=None):
        self.cfg = cfg
        if cfg:
            assert isinstance(cfg, ConfigBase)
            self.set_options(cfg)
        self.init()
        self.check_required()
    def set_options(self,cfg):
        for k, v in cfg.to_dict().items():
            if k in self.__allowed_options__:
                if v is None:
                    v_old=getattr(self,k,None)
                    v=v_old
                setattr(self, k, v)
    def check_required(self):
        for key in self.__required__:
            if getattr(self, key, None) is None:
                raise Exception('Argument %s is required but not given.' % (key))
    def init(self):
        pass


class DetDataset(DetDatasetBase):
    __required__ = ['BATCH_SIZE', 'INPUT_SIZE','CLS2IDX','DATA_DIRS','DEVICE']
    def init(self):
        if self.CLASSES:
            self.CLS2IDX = {cls: idx for idx, cls in enumerate(self.CLASSES)}
            self.NUM_CLASSES = len(self.CLASSES)
        else:
            self.CLS2IDX = None
        if self.IMG_DIR:
            self.DATA_DIRS = self.DATA_DIRS or []
            # data_dirs format: (dir,dir, labelme, .json)
            self.DATA_DIRS.append((self.IMG_DIR, self.LABEL_DIR, self.ANNOT_FORMAT, self.ANNOT_FILE_EXT))
        self.annotations = self.load_annotations_multi_dirs(self.DATA_DIRS)
        random.shuffle(self.annotations)
        self.current_index = -1
        self.num_batches = int(len(self.annotations) / self.BATCH_SIZE)
    def set_device(self, device):
        self.DEVICE = device

    def process_annotation(self, annot):
        raise NotImplementedError

    def process_batch_annots(self, batch_annots):
        raise NotImplementedError

    def create_label(self, img, shapes):
        raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self):
        self.current_index += 1
        if self.current_index == self.num_batches:
            self.current_index = -1
            random.shuffle(self.annotations)
            raise StopIteration
        batch_annots = self.annotations[self.current_index * self.BATCH_SIZE:(self.current_index + 1) * self.BATCH_SIZE]
        batch_inputs, batch_labels = self.process_batch_annots(batch_annots)
        return batch_inputs, batch_labels

    def __len__(self):
        return len(self.annotations)

    def load_annot_file_labelme(self, label_path):
        objs = []
        with open(label_path, 'r') as fp:
            annot = json.load(fp)
            shapes = annot['shapes']
            if not shapes:  # 没有标注框的就不要
                return objs
            for shape in shapes:
                points = shape['points']
                label = self.CLS2IDX[shape['label']]
                objs.append([points, label])
        return objs

    def load_annot_file_pascalvoc(self, label_path):
        annot = ut.read_pascalvoc_annotation(label_path, cls2idx=self.CLS2IDX)
        objs = [[ut.ltrb_to_quad(ltrb), cls] for ltrb, cls in annot]
        return objs

    def load_annotations_from_dir(self, img_dir, label_dir, label_ext=None, format=None):
        format = format or 'labelme'
        if not label_ext:
            if format == 'labelme':
                label_ext = '.json'
            else:
                # voc as default
                label_ext = '.xml'
        fs = []
        for s in [img_dir + '/*' + ext for ext in ['.jpg', '.png', 'jpeg', '.bmp']]:
            fs += glob.glob(s)
        annotations = []
        for i, img_path in enumerate(fs):
            label_path = label_dir + '/' + os.path.basename(img_path).rsplit('.', maxsplit=1)[0] + label_ext
            if not os.path.exists(label_path):
                continue
            if format == 'labelme':
                objs = self.load_annot_file_labelme(label_path)
            else:
                objs = self.load_annot_file_pascalvoc(label_path)
            annotations.append((img_path, objs))
        return annotations

    def load_annotations_multi_dirs(self, data_dirs):
        annotations = []
        for img_dir, label_dir, annot_format, annot_file_ext in data_dirs:
            annotations += self.load_annotations_from_dir(img_dir, label_dir, format=annot_format,
                                                          label_ext=annot_file_ext)
        return annotations
