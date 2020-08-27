import torch
import torch.nn.functional as F
from torchvision import transforms
import os,shutil,glob
import cv2
# import wpcv
from PIL import Image
import numpy as np
def _nms(heat, kernel=3):
  hmax = F.max_pool2d(heat, kernel, stride=1, padding=(kernel - 1) // 2)
  keep = (hmax == heat).float()
  return heat * keep

def _gather_feature(feat, ind, mask=None):
  dim = feat.size(2)
  ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
  feat = feat.gather(1, ind)
  if mask is not None:
    mask = mask.unsqueeze(2).expand_as(feat)
    feat = feat[mask]
    feat = feat.view(-1, dim)
  return feat

def _topk(scores, K=4):
    batch, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, 1, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feature(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feature(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feature(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def ctdet_decode(centermap, K=4):
    detections = []
    centermap=torch.sigmoid(centermap)
    centermap = _nms(centermap)

    save_center = centermap[0, 0, ...]
    save_center = save_center.cpu().detach().numpy() * 255
    save_center = save_center.astype(np.uint8)

    centermap = centermap[:, 0, :, :]
    _, inds, clses, ys, xs = _topk(centermap, K)

    xs = xs.view(1, K, 1) + 0.5
    ys = ys.view(1, K, 1) + 0.5

    xs *= 4
    ys *= 4

    y1 = ys[:, 0, :].view(1, -1)
    x1 = xs[:, 0, :].view(1, -1)
    y2 = ys[:, 1, :].view(1, -1)
    x2 = xs[:, 1, :].view(1, -1)
    y3 = ys[:, 2, :].view(1, -1)
    x3 = xs[:, 2, :].view(1, -1)
    y4 = ys[:, 3, :].view(1, -1)
    x4 = xs[:, 3, :].view(1, -1)
    detections = torch.cat((x1, y1, x2, y2, x3, y3, x4, y4), dim=1)
    detections = detections.cpu().detach().numpy()

    if len(detections) == 0:
        detections = [[0,0,0,0,0,0,0,0,0,0]]
        return np.array(detections)
    else:
        return np.concatenate(np.array(detections), 0)
def normalize(image):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    mean = np.array([[mean]])
    std = np.array([[std]])
    image = (image.astype(np.float32) - mean) / std
    return image
def organize_quad_points(box):
    '''
    turn into clock-wise points: quad
    p0:lt,p1:rt,p2:rb,p4:lb
    '''
    p0,p3, p1,p2 = sorted(box, key=lambda p: p[0])
    p03=[p0,p3]
    p12=[p1,p2]
    p0, p3 = sorted(p03, key=lambda p: p[1])
    p1, p2 = sorted(p12, key=lambda p: p[1])
    return [p0,p1,p2,p3]
def crop_quad(img,box):
    p0, p1, p2, p3 = box
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = box
    w,h=((x1-x0+x2-x3)//2,(y3-y0+y2-y1)//2)
    w,h=int(w),int(h)
    M=cv2.getPerspectiveTransform(np.float32([p0,p1,p3,p2]),np.float32([[0,0],[w,0],[0,h],[w,h]]))
    img=cv2.warpPerspective(img,M,(w,h))
    return img
def img_preprocess2(image, target_shape,bboxes=None,  correct_box=False):
    """
    RGB转换 -> resize(resize不改变原图的高宽比) -> normalize
    并可以选择是否校正bbox
    :param image_org: 要处理的图像
    :param target_shape: 对图像处理后，期望得到的图像shape，存储格式为(h, w)
    :return: 处理之后的图像，shape为target_shape
    """
    h_target, w_target = target_shape
    h_org, w_org, _ = image.shape

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    resize_ratio = min(1.0 * w_target / w_org, 1.0 * h_target / h_org)
    resize_w = int(resize_ratio * w_org)
    resize_h = int(resize_ratio * h_org)
    image_resized = cv2.resize(image, (resize_w, resize_h))

    image_paded = np.full((h_target, w_target, 3), 128.0)
    dw = int((w_target - resize_w) / 2)
    dh = int((h_target - resize_h) / 2)
    image_paded[dh:resize_h+dh, dw:resize_w+dw,:] = image_resized
    image = image_paded / 255.0
    image = normalize(image)

    if correct_box:
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * resize_ratio + dw
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * resize_ratio + dh
        return image, bboxes
    return image,resize_ratio,dw,dh
class Rectifier:
    def __init__(self,model_path=os.path.dirname(__file__)+'/model.pth'):
        self.device=torch.device('cuda' if torch.cuda.is_available() else "cpu")
        print("device:",self.device)
        from .backbone import EfficientDetBackbone
        self.model=EfficientDetBackbone()
        self.model.to(self.device)
        state_dict=torch.load(model_path,map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.transform=transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((512,512)),
            transforms.ToTensor()
        ])
    def predict(self,img):
        # if not isinstance(img,Image.Image):
        #     img=Image.fromarray(img).convert('RGB')
        if isinstance(img,Image.Image):
            img=np.array(img).astype(np.float32)
            img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        img,ratio,dw,dh=img_preprocess2(img,target_shape=(512,512))
        # else:
        #     img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)
        # h,w=img.shape[:2]
        # rh=h/512
        # rw=w/512
        # img=cv2.resize(img,(512,512))/255
        img=torch.from_numpy(img.astype(np.float32)).transpose(1,2).transpose(0,1)
        img=img.unsqueeze(0)
        img = img.to(self.device)
        y=self.model(img)
        output = ctdet_decode(y)
        output=output.reshape((4,2))
        output[:,0]=(output[:,0]-dw)/ratio
        output[:,1]=(output[:,1]-dh)/ratio
        return output
_rectifier=None
def rectify_bxd(img):
    if isinstance(img,Image.Image):
        cvimg=np.array(img)
        cvimg=cv2.cvtColor(cvimg,cv2.COLOR_RGB2BGR)
    else:
        cvimg=img
    global _rectifier
    if not _rectifier:
        _rectifier=Rectifier()
    points=_rectifier.predict(img)
    points=organize_quad_points(points)
    img=crop_quad(cvimg,points)
    return img

def demo():
    dir="/home/ars/sda5/data/chaoyuan/datasets/detect_datasets/保险单矫正/val"
    out_dir=dir+'_out'
    import wpcv
    out=wpcv.ImageSaver(out_dir,remake_dir=True)
    out2=wpcv.ImageSaver(dir+'_out2',remake_dir=True)
    rectifier=Rectifier('/home/ars/disk/work/超远/chaoyuan-part1-dev/scripts/tests/model.pth')
    # print(rectifier.model)
    fs=glob.glob(dir+'/*.jpg')
    for i,f in enumerate(fs):
        # img=Image.open(f)
        img=cv2.imread(f)
        pilimg=Image.open(f)
        y=rectifier.predict(img)
        y=wpcv.organize_quad_points(y)
        y=np.array(y)
        # y=y.tolist()
        # print(y)
        im = wpcv.draw_polygon(pilimg,y,color='red',width=5)
        out.save(im)
        im=wpcv.crop_quad(img,y)
        out2.save(im)
        print(i,f,y)

if __name__ == '__main__':
    demo()