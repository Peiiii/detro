# Author: Zylo117

import math

import torch
from torch import nn

from .efficientdet.model import BiFPN, Center_layer, Proto_layer, Coef_layer,Offset_layer, Heatmap_layer, EfficientNet, SeparableConvBlock


class EfficientDetBackbone(nn.Module):
    def __init__(self, num_classes=1, compound_coef=0, load_weights=False, **kwargs):
        super(EfficientDetBackbone, self).__init__()
        self.compound_coef = compound_coef

        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6]
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8]
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5]
        self.anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5.]
        self.aspect_ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])
        self.num_scales = len(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
        conv_channel_coef = {
            # the channels of P3/P4/P5.
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
        }

        num_anchors = len(self.aspect_ratios) * self.num_scales

        self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters[self.compound_coef],
                    conv_channel_coef[compound_coef],
                    True if _ == 0 else False,
                    attention=True if compound_coef < 6 else False)
              for _ in range(self.fpn_cell_repeats[compound_coef])])

        self.conv = SeparableConvBlock(self.fpn_num_filters[self.compound_coef])
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.coord_channel = self.CoordChannel
        self.num_classes = num_classes
        # self.proto_head = Proto_layer(in_channels=self.fpn_num_filters[self.compound_coef])
        # self.coef_head = Coef_layer(in_channels=self.fpn_num_filters[self.compound_coef])
        self.center = Center_layer(in_channels=self.fpn_num_filters[self.compound_coef],num_classes=num_classes)
        self.offset_layer=Offset_layer(in_channels=self.fpn_num_filters[self.compound_coef])

        self.backbone_net = EfficientNet(self.backbone_compound_coef[compound_coef], load_weights)


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


    def CoordChannel(self, feature):

        # 将x,y坐标拼接在feature map上,通道数+2,
        # 这里是将坐标信息concat到feature map上
        x_range = torch.linspace(-1, 1, feature.shape[-1], device=feature.device) # w --> x
        y_range = torch.linspace(-1, 1, feature.shape[-2], device=feature.device) # h --> y
        # 对 x_range, y_range 进行扩充 
        y, x = torch.meshgrid(y_range, x_range)
        # 将两个坐标扩成4维
        y = y.expand([feature.shape[0], 1, -1, -1])
        x = x.expand([feature.shape[0], 1, -1, -1])
        # 将坐标cancat到feature map的通道上
        coord_feature = torch.cat([x, y], 1)
        ins_feature = torch.cat([feature, coord_feature], 1)
        
        return ins_feature


    def forward(self, inputs):
        max_size = inputs.shape[-1]

        _, p3, p4, p5 = self.backbone_net(inputs)

        features = [p3, p4, p5]
        features = self.bifpn(features)

        feature = self.conv(self.upsample(features[0]))
        centermap = self.center(feature)
        offsets=self.offset_layer(feature)
        return centermap,offsets


    def init_backbone(self, path):
        state_dict = torch.load(path)
        try:
            ret = self.load_state_dict(state_dict, strict=False)
            print(ret)
        except RuntimeError as e:
            print('Ignoring ' + str(e) + '"')


if __name__ == '__main__':
    input_tensor = torch.ones((4, 3, 512, 512))
    model = EfficientDetBackbone()
    centermap = model(input_tensor)
    print(centermap.shape)
