import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
# criterion = nn.BCEWithLogitsLoss(reduction='none')

def __focal(target, actual, alpha=1, gamma=2):
    focal = alpha * torch.pow(torch.abs(target - actual), gamma)
    return focal


def heatmap_loss(proto, coef, gt, ids):
    batch_size,num_object=ids.shape[:2]
    #print(proto.shape) # [16, 32, 128, 128]
    #print(coef.shape)  # [16, 32, 128, 128]
    #print(ids.shape) # [16, 6, 2]

    coef = F.softmax(coef, dim=1)
    ids=ids.long()
    arr=torch.zeros((coef.shape[0],ids.shape[1],coef.shape[1])).cuda()
    for batch_idx in range(coef.shape[0]):
        arr[batch_idx]=coef[batch_idx,:,ids[batch_idx,:,0],ids[batch_idx,:,1]].t()
    
    output = proto.view(batch_size, 1, 32, 128, 128) * arr.view(batch_size, num_object, 32, 1, 1)
    output=output.sum(dim=2)
    pred = torch.clamp(torch.sigmoid(output), min=1e-6, max=1-1e-6)

    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss


def patch_loss(pred, gt):
    pred = torch.clamp(torch.sigmoid(pred), min=1e-6, max=1-1e-6)

    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss


def center_loss(pred, gt):
    pred = torch.clamp(torch.sigmoid(pred), min=1e-6, max=1-1e-6)

    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss

def l1_loss(pred,label):
    loss = F.l1_loss(pred, label, size_average=False) / (len(pred) + 1e-4)
    return loss

def wh_loss(pred_wh, label_wh, mask):
    '''
    preds (B x 4 x h x w)
    gt_regr (B x 4 x h x w)
    mask (B x 1 x h x w)
    '''
    loss = F.l1_loss(pred_wh * mask, label_wh, size_average=False) / (mask.sum() + 1e-4)

    # print(loss)

    # print(pred_wh.size())
    # print(label_wh.size())
    # loss = (torch.abs(pred_wh * mask - label_wh).sum(axis=(1, 2, 3)) / (mask.sum(axis=(1, 2, 3)) + 1e-4)).sum()
    
    return loss#  / len(pred_wh)
def distance_loss(pred_wh, label_wh, mask):
    '''
    preds (B x 4 x h x w)
    gt_regr (B x 4 x h x w)
    mask (B x 1 x h x w)
    '''
    loss = F.l1_loss(pred_wh * mask, label_wh, size_average=False) / (mask.sum() + 1e-4)

    # print(loss)

    # print(pred_wh.size())
    # print(label_wh.size())
    # loss = (torch.abs(pred_wh * mask - label_wh).sum(axis=(1, 2, 3)) / (mask.sum(axis=(1, 2, 3)) + 1e-4)).sum()

    return loss#  / len(pred_wh)


# def wh_loss(pred_lurd, label_lurd, mask):
#     '''
#     preds (B x 4 x h x w)
#     gt_regr (B x 4 x h x w)
#     mask (B x 1 x h x w)
#     '''
#     pred_coor = decode(pred_lurd)
#     label_coor = decode(label_lurd)
#     giou_loss = GIOU_loss(pred_coor, label_coor, mask)
#     loss = (giou_loss.sum(dim=(1,2,3)) / (mask.sum(dim=(1, 2, 3)) + 1e-4)).mean()
#     return loss




def GIOU(boxes1, boxes2):
    """
    input_format: (xmin, ymin, xmax, ymax)
    """
    boxes1 = torch.cat([torch.min(boxes1[..., :2], boxes1[..., 2:]),
                        torch.max(boxes1[..., :2], boxes1[..., 2:])], dim=-1)
    boxes2 = torch.cat([torch.min(boxes2[..., :2], boxes2[..., 2:]),
                        torch.max(boxes2[..., :2], boxes2[..., 2:])], dim=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    # 计算出boxes1与boxes1相交部分的左上角坐标、右下角坐标
    intersection_left_up = torch.max(boxes1[..., :2], boxes2[..., :2])
    intersection_right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])

    # 因为两个boxes没有交集时，(right_down - left_up) < 0，所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
    intersection = torch.max(intersection_right_down - intersection_left_up, torch.zeros_like(intersection_right_down))
    inter_area = intersection[..., 0] * intersection[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = 1.0 * inter_area / (union_area + 1e-5)

    enclose_left_up = torch.min(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = torch.max(boxes1[..., 2:], boxes2[..., 2:])
    enclose = torch.max(enclose_right_down - enclose_left_up, torch.zeros_like(enclose_left_up))
    enclose_area = enclose[..., 0] * enclose[..., 1]
    GIOU = IOU - 1.0 * (enclose_area - union_area) / (enclose_area + 1e-5)

    return GIOU


def GIOU_loss(pred_coor, label_coor, respond_bbox):
    giou = GIOU(pred_coor, label_coor).unsqueeze(-1)
    bbox_wh = label_coor[..., 2:] - label_coor[..., :2]
    bbox_loss_scale = 2.0 - 1.0 * bbox_wh[..., 0:1] * bbox_wh[..., 1:2] / (512 ** 2)
    GIOU_loss = respond_bbox * bbox_loss_scale * (1.0 - giou)
    return GIOU_loss


def bbox_ciou(boxes1, boxes2):
    '''
    计算ciou = iou - p2/c2 - av
    pred_xywh
    label_xywh
    举例时假设pred_xywh和label_xywh的shape都是(1, 4)
    '''

    # 变成左上角坐标、右下角坐标
    boxes1_x0y0x1y1 = boxes1
    boxes2_x0y0x1y1 = boxes2
    '''
    逐个位置比较boxes1_x0y0x1y1[..., :2]和boxes1_x0y0x1y1[..., 2:]，即逐个位置比较[x0, y0]和[x1, y1]，小的留下。
    比如留下了[x0, y0]
    这一步是为了避免一开始w h 是负数，导致x0y0成了右下角坐标，x1y1成了左上角坐标。
    '''
    boxes1_x0y0x1y1 = torch.cat((torch.min(boxes1_x0y0x1y1[..., :2], boxes1_x0y0x1y1[..., 2:]),
                             torch.max(boxes1_x0y0x1y1[..., :2], boxes1_x0y0x1y1[..., 2:])), dim=-1)
    boxes2_x0y0x1y1 = torch.cat((torch.min(boxes2_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., 2:]),
                             torch.max(boxes2_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., 2:])), dim=-1)

    # 两个矩形的面积
    boxes1_area = (boxes1_x0y0x1y1[..., 2] - boxes1_x0y0x1y1[..., 0]) * (
                boxes1_x0y0x1y1[..., 3] - boxes1_x0y0x1y1[..., 1])
    boxes2_area = (boxes2_x0y0x1y1[..., 2] - boxes2_x0y0x1y1[..., 0]) * (
                boxes2_x0y0x1y1[..., 3] - boxes2_x0y0x1y1[..., 1])

    # 相交矩形的左上角坐标、右下角坐标，shape 都是 (8, 13, 13, 3, 2)
    left_up = torch.max(boxes1_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., :2])
    right_down = torch.min(boxes1_x0y0x1y1[..., 2:], boxes2_x0y0x1y1[..., 2:])

    # 相交矩形的面积inter_area。iou
    inter_section = right_down - left_up
    inter_section = torch.where(inter_section < 0.0, inter_section*0, inter_section)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / union_area

    # 包围矩形的左上角坐标、右下角坐标，shape 都是 (8, 13, 13, 3, 2)
    enclose_left_up = torch.min(boxes1_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., :2])
    enclose_right_down = torch.max(boxes1_x0y0x1y1[..., 2:], boxes2_x0y0x1y1[..., 2:])

    # 包围矩形的对角线的平方
    enclose_wh = enclose_right_down - enclose_left_up
    enclose_c2 = torch.pow(enclose_wh[..., 0], 2) + torch.pow(enclose_wh[..., 1], 2)

    # 两矩形中心点距离的平方
    p2 = torch.pow(boxes1[..., 0] - boxes2[..., 0], 2) + torch.pow(boxes1[..., 1] - boxes2[..., 1], 2)

    # 增加av。分母boxes2[..., 3]可能为0，所以加上除0保护防止nan。
    atan1 = torch.atan(boxes1[..., 2] / boxes1[..., 3])
    temp_a = torch.where(boxes2[..., 3] > 0.0, boxes2[..., 3], boxes2[..., 3] + 1.0)
    atan2 = torch.atan(boxes2[..., 2] / temp_a)
    v = 4.0 * torch.pow(atan1 - atan2, 2) / (math.pi ** 2)
    a = v / (1 - iou + v)

    ciou = iou - 1.0 * p2 / enclose_c2 - 1.0 * a * v
    return ciou


def decode(output, stride=4):
    # output: [batchsize, 4, height, width]
    # left, up, right, down ===>>> x1, y1, x2, y2
    bz=output.shape[0]
    gridsize=output.shape[-1]

    output=output.permute(0, 2, 3, 1)
    output=output.view(bz, gridsize, gridsize, 4)
    leftup,rightdown=torch.split(output, [2,2], dim=3)

    shiftx=torch.arange(0,gridsize,dtype=torch.float32)
    shifty=torch.arange(0,gridsize,dtype=torch.float32)
    shifty,shiftx=torch.meshgrid([shiftx, shifty])
    shiftx=shiftx.unsqueeze(0).repeat(bz, 1,1)
    shifty=shifty.unsqueeze(0).repeat(bz, 1,1)

    xy_grid=torch.stack([shiftx,shifty], dim=3).cuda()

    x1y1=(xy_grid+0.5)*stride - leftup
    x2y2=(xy_grid+0.5)*stride + rightdown

    coors=torch.cat((x1y1,x2y2),dim=3)
    return coors


