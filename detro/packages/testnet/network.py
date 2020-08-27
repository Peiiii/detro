from .resnet_backbone import resnet18
from torch import nn
import torch
import torch.nn.functional as F
from detro.networks.components import BiFPN, Center_layer, Offset_layer, Reg_layer, Heatmap_layer, BasicBlock
from detro.networks.losslib import center_loss, distance_loss


class FeatureFusionNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        resized = []
        size = inputs[0].size()[-2:]
        for x in inputs[1:]:
            resized.append(F.upsample(x, size))
        x = torch.cat(resized, dim=1)
        return x


class TestNetV1(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.backbone = resnet18(pretrained=True)
        self.neck = FeatureFusionNetwork()
        self.conv1 = nn.Conv2d(896, 256, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.center_layer = Heatmap_layer(in_channels=256, out_channels=num_classes)

    def forward(self, inputs):
        c1, c2, c3, c4, c5 = self.backbone(inputs)
        features = [c2, c3, c4, c5]
        features = self.neck(features)
        x = features
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        heatmap = self.center_layer(x)
        return heatmap


class TestNet(nn.Module):
    def __init__(self, num_classes=1, num_points=4):
        super().__init__()
        self.backbone = resnet18(pretrained=True)
        self.neck = FeatureFusionNetwork()
        self.conv1 = nn.Conv2d(896, 256, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        # self.corner_layer = Heatmap_layer(in_channels=256, out_channels=num_classes)
        # self.center_layer = Heatmap_layer(in_channels=256, out_channels=num_classes)
        self.hm_layer=Heatmap_layer(in_channels=256,out_channels=num_classes*2)
        # self.info_layer = Heatmap_layer(in_channels=256, out_channels=num_classes)
        self.num_classes=num_classes
        self.reg_layer = nn.Sequential(
            BasicBlock(num_classes*2,num_classes*2),
            Heatmap_layer(in_channels=num_classes * 2, out_channels=num_points * 2),
        )

    # self.reg_layer = Heatmap_layer(in_channels=num_classes * 3, out_channels=num_points*2)
    # self.reg_layer = Heatmap_layer(in_channels=256, out_channels=num_points*2)

    def forward(self, inputs):
        c1, c2, c3, c4, c5 = self.backbone(inputs)
        features = [c2, c3, c4, c5]
        features = self.neck(features)
        x = features
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        heatmap=self.hm_layer(x)
        corner_heatmap = heatmap[:,:self.num_classes]
        center_heatmap = heatmap[:,self.num_classes:]
        offsets = self.reg_layer(heatmap)
        return dict(
            center_heatmap=center_heatmap, corner_heatmap=corner_heatmap, offsets=offsets
        )


def TestCriterion(preds, labels):
    loss_center = center_loss(preds['center_heatmap'], labels['center_heatmap'])
    loss_corner = center_loss(preds['corner_heatmap'], labels['corner_heatmap'])
    loss_offsets = distance_loss(preds['offsets'], labels['offsets'], labels['offsets_mask'])
    return dict(
        loss=loss_center + loss_corner + loss_offsets,
        loss_center=loss_center,
        loss_corner=loss_corner,
        loss_offsets=loss_offsets,
    )
