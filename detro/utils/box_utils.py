import wpcv
import math
import numpy as np

import math
import numpy as np
from .polygon_utils import organize_quad_points,rotate_points,bounding_rect
##########################Box Operations#############################
def resize_box(box,size):
    cx,cy,w,h=ltrb_to_ccwh(box)
    nw,nh=size
    l=cx-nw//2
    r=cx+nw//2
    t=cy-nh//2
    b=cy+nh//2
    return [l,t,r,b]
def rescale_box(box,scale):
    '''rescale with  image'''
    if isinstance(scale,(tuple,list)):
        scaleX,scaleY=scale
    else:
        scaleX=scaleY=scale
    l,t,r,b=box
    l*=scaleX
    r*=scaleX
    t*=scaleY
    b*=scaleY
    return [l,t,r,b]


def rotate_boxes(boxes,angle,cx,cy,h,w):
    '''原创'''
    boxes=[list(box) for box in boxes]
    quads = [ltrb_to_quad(box) for box in boxes]
    quads=rotate_points(quads,angle,cx,cy,h,w)
    boxes=[bounding_rect(quad) for quad in quads]
    boxes=np.array(boxes)
    return boxes
def rotate_box(box, angle, cx, cy, h, w):
    return rotate_boxes([box],angle,cx,cy,h,w)
def shift_box(box,offset,limits=None):
    l,t,r,b=box
    ofx,ofy=offset
    l+=ofx
    r+=ofx
    t+=ofy
    b+=ofy
    return limit_box([l,t,r,b],limits=limits)
def translate_box(box,offset,limits=None):
    return shift_box(box,offset,limits)
def flip_box_horizontal(box,imsize):
    imw,imh=imsize
    l,t,r,b=box
    l=imw-r
    r=imw-l
    return [l,t,r,b]
def flip_box_vertical(box,imsize):
    imw,imh=imsize
    l,t,r,b=box
    t=imh-b
    b=imh-t
    return [l,t,r,b]
def pad_box(box,pad_size=5,pad_ratio=None,limits=None):
    from math import inf
    limits=limits or (-inf,-inf,inf,inf)
    l,t,r,b=box
    bh=b-t
    bw=r-l
    if pad_ratio is not None:
        if isinstance(pad_ratio,(tuple,list)):
            if len(pad_ratio)==2:
                pad_l=pad_r=int(min(bh,bw)*pad_ratio[0])
                pad_t=pad_b=int(min(bh,bw)*pad_ratio[1])
            else:
                assert len(pad_ratio)==4
                pad_l= int(min(bh, bw) * pad_ratio[0])
                pad_t= int(min(bh, bw) * pad_ratio[1])
                pad_r= int(min(bh, bw) * pad_ratio[2])
                pad_b= int(min(bh, bw) * pad_ratio[3])
        else:
            pad_l=pad_t=pad_r=pad_b=int(min(bh,bw)*pad_ratio)
    else:
        if isinstance(pad_size,(tuple,list)):
            if len(pad_size)==2:
                pad_l=pad_r=pad_size[0]
                pad_t=pad_b=pad_size[1]
            else:
                assert len(pad_size)==4
                pad_l,pad_t,pad_r,pad_b=pad_size
        else:
            pad_l=pad_t=pad_r=pad_b=pad_size
    l-=pad_l
    t-=pad_t
    r+=pad_r
    b+=pad_b
    return limit_box([l,t,r,b],limits=limits)

def limit_box(box,limits=None):
    if limits is None:return box
    if len(limits)==2:
        ml,mt=0,0
        mr,mb=limits
    else:
        assert len(limits)==4
        ml,mt,mr,mb=limits
    l,t,r,b=box
    l=max(ml,l)
    t=max(mt,t)
    r=min(mr,r)
    b=min(mb,b)
    if l>=r:
        return None
    if t>=b:return None
    return [l,t,r,b]
############################Sort Operations#############################
def min_enclosing_box(boxes):
    boxes=np.array(boxes)
    ml=np.min(boxes[:,0])
    mt=np.min(boxes[:,1])
    mr=np.max(boxes[:,2])
    mb=np.max(boxes[:,3])
    return [ml,mt,mr,mb]
def box_in_area(box,size):
    l,t,r,b=box
    mr,mb=size
    if l<0 or t<0 or r>mr or b>mb:
        return False
    else:
        return True
############################ConvertingTools#############################
def ltrb_to_quad(box):
    l,t,r,b=box
    quad=[[l,t],[r,t],[r,b],[l,b]]
    return quad

def ltrb_to_ccwh(box):
    l, t, r, b = box
    w = r - l
    h = b - t
    cx = (l + r) // 2
    cy = (t + b) // 2
    return [cx,cy,w,h]
def ccwh_to_ltrb(box):
    cx,cy,w,h=box
    l=cx-w//2
    t=cy-h//2
    r=cx+w//2
    b=cy+h//2
    return [l,t,r,b]
def oowh_to_ltrb(box):
    '''o:origin'''
    x, y, w, h = box
    return [int(i) for i in (x, y, x + w, y + h)]
def ltrb_to_oowh(box):
    l, t, r, b = box
    return [l, t, r - l, b - t]
def xywh_to_ltrb(box):
    '''same with oowh'''
    x,y,w,h=box
    return [int(i) for i in (x,y,x+w,y+h)]
def ltrb_to_xywh(box):
    l,t,r,b=box
    return [l,t,r-l,b-t]

def quad_to_four_corners(quad):
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = quad
    return [x0,y0,x1,y1,x2,y2,x3,y3]
def four_corners_to_quad(box):
    x0,y0,x1,y1,x2,y2,x3,y3 = box
    return [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]
###########################Calculations##############################
def calc_iou(box1,box2):
    l1,t1,r1,b1=box1
    l2,t2,r2,b2=box2
    w1,h1=r1-l1,b1-t1
    w2,h2=r2-l2,b2-t2
    width=min(r1,r2)-max(l1,l2)
    height=min(b1,b2)-max(t1,t2)
    width=max(width,0)
    height=max(height,0)
    area=width*height
    return area/(w1*h1+w2*h2-area)
def calc_iou2(box1,box2):
    width=min(box1[2],box2[2])-max(box1[0],box2[0])
    height=min(box1[3],box2[3])-max(box1[1],box2[1])
    width=max(width,0)
    height=max(height,0)
    area=width*height
    return area/((box1[2]-box1[0])*(box1[3]-box1[1])+(box2[2]-box2[0])*(box2[3]-box2[1])-area)
def calc_iou_batch_with_batch(boxes1,boxes2):
    box1=boxes1
    box2=boxes2
    import numpy as np
    box1=np.array(box1)
    box2=np.array(box2)
    width = np.min(np.vstack([box1[...,2], box2[...,2]]),axis=0) - np.max(np.vstack([box1[...,0], box2[...,0]]),axis=0)
    height = np.min(np.vstack([box1[...,3], box2[...,3]]),axis=0) - np.max(np.vstack([box1[...,1], box2[...,1]]),axis=0)
    width = np.clip(width,0,np.inf)
    height = np.clip(height,0,np.inf)
    area = width * height
    return area / ((box1[...,2] - box1[...,0]) * (box1[...,3] - box1[...,1]) + (box2[...,2] - box2[...,0]) * (box2[...,3] - box2[...,1]) - area)
def calc_iou_batch(boxes,box):
    '''calculate iou between batch bboxes with one bbox , 原创'''
    box1=boxes
    box2=box
    import numpy as np
    box1 = np.array(box1)
    box2 = np.array(box2)
    shape=box1.shape
    width = np.min(np.vstack([box1[..., 2], np.full(shape[:-1],box2[2])]), axis=0) - np.max(np.vstack([box1[..., 0], np.full(shape[:-1],box2[0])]),axis=0)
    height = np.min(np.vstack([box1[..., 3], np.full(shape[:-1],box2[3])]), axis=0) - np.max(np.vstack([box1[..., 1], np.full(shape[:-1],box2[1])]),axis=0)
    width = np.clip(width, 0, np.inf)
    height = np.clip(height, 0, np.inf)
    area = width * height
    return area / ((box1[...,2] - box1[...,0]) * (box1[...,3] - box1[...,1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - area)

def calc_box_area(box):
    l,t,r,b=box
    return (r-l)*(b-t)


