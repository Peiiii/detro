import torch
import cv2
import numpy as np
from PIL import Image
from detro import utils as ut
from .predictor import Predictor



class ClassicalDetectorBase(Predictor):
    MAX_OUTPUT_NUM = 100
    SCORE_THRESH = 0.1
    def heatmap2image(self,heatmap,img=None, scaleX=None,scaleY=None,pad_x=None,pad_y=None):
        scaleX=self.cache['scaleX'] if scaleX is None else scaleX
        scaleY=self.cache['scaleY'] if scaleY is None else scaleY
        pad_x=self.cache['pad_x'] if pad_x is None else pad_x
        pad_y=self.cache['pad_y'] if pad_y is None else pad_y
        img = self.cache['img']
        heatmap_downsample=self.HEATMAP_DOWNSAMPLE
        heatmap = torch.nn.functional.sigmoid(heatmap)
        heatmap *= 255
        heatmap = heatmap.detach().cpu().numpy().astype(np.uint8)
        h,w=heatmap.shape
        w*=heatmap_downsample
        h*=heatmap_downsample
        heatmap=cv2.resize(heatmap,(w,h))
        heatmap=heatmap[pad_y:h-pad_y,pad_x:w-pad_x]
        new_h,new_w=int(heatmap.shape[0]/scaleY),int(heatmap.shape[1]/scaleX)
        if img is not None:
            new_h,new_w=img.shape[:2]
        heatmap=cv2.resize(heatmap,(new_w,new_h))
        im=img.copy()
        im[...,0]+=heatmap
        heatmap=cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
        return heatmap
    def preprocess(self,img):
        h, w = img.shape[:2]
        dst_w, dst_h = self.INPUT_SIZE
        scale=min(dst_w/w,dst_h/h)
        scale_x,scale_y=scale,scale
        new_h=int(scale_y*h)
        new_w=int(scale_x*w)
        pad_x=(dst_w-new_w)//2
        pad_y=(dst_h-new_h)//2
        img_padded=np.full((dst_h,dst_w,3),0)
        img_resized=cv2.resize(img,(new_w,new_h))
        img_padded[pad_y:pad_y+new_h,pad_x:pad_x+new_w]=img_resized
        img=img_padded
        img = img / 255
        img = ut.normalize(img)
        img = np.transpose(img, (2, 0, 1))
        img = np.array([img]).astype(np.float32)
        img = torch.from_numpy(img)
        img = img.to(self.DEVICE)
        return img,[scale_x,scale_y,pad_x,pad_y]
    def decode_outputs(self,outputs,down,k):
        raise NotImplementedError
    def postprocess(self,res,score_thresh):
        raise NotImplementedError
    def predict(self,img,max_output_num=None,score_thresh=None):
        if max_output_num is None:
            assert self.MAX_OUTPUT_NUM is not None
            max_output_num=self.MAX_OUTPUT_NUM
        if score_thresh is None:
            assert self.SCORE_THRESH is not None
            score_thresh=self.SCORE_THRESH
        down=self.HEATMAP_DOWNSAMPLE
        if isinstance(img,Image.Image):
            img=np.array(img).astype(np.float32)
        else:
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)
        self.cache.update(img=img)
        img,(scaleX,scaleY,pad_x,pad_y)=self.preprocess(img)
        outputs= self.model(img)
        self.cache.update(outputs=outputs,scaleX=scaleX,scaleY=scaleY,pad_x=pad_x,pad_y=pad_y)
        res=self.decode_outputs(outputs,down=down,k=max_output_num)[0]
        res=self.postprocess(res,score_thresh)
        return res