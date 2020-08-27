import numpy as np
import cv2
from PIL import Image
from detro import utils as ut
def visualize(img,res):
    img=ut.pilimg(img)
    for circle in res:
        text='%.2f'%(circle[3])
        ut.draw_circle(img,center=circle[:2],radius=circle[2],text=text,fillcolor='blue')
    return img