import cv2
from PIL import Image
import numpy as np

def cv2img(img):
    if isinstance(img,Image.Image):
        img=np.array(img)
        if len(img.shape)==3:img=img[:,:,::-1]
        return img
    return img
def pilimg(img):
    if isinstance(img,Image.Image):return img
    if isinstance(img,np.ndarray):
        if len(img.shape)==3:img=img[:,:,::-1]
    return Image.fromarray(np.array(img).astype(np.uint8))