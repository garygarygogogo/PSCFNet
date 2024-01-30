import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log
from .pvtv2 import *


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



class Relation(nn.Module):
    def __init__(self, in_channels,out_channels,c):
        super(Relation, self).__init__()
        self.scene = nn.Sequential(BasicConv2d(in_channels, out_channels, 1), nn.Conv2d(out_channels, out_channels, 1))
        self.content = BasicConv2d(c, out_channels, 1)
        self.feature = BasicConv2d(c, out_channels, 1)
        self.normalizer = nn.Sigmoid()


    def forward(self, scene_feature, features):
        content_feat = self.content(features)
        scene_feat = self.scene(scene_feature)
        relations = self.normalizer((scene_feat * content_feat).sum(dim=1, keepdim=True))
        p_feats = self.feature(features)
        refined_feats = relations * p_feats
        return refined_feats



class Saliency_Mining_Module(nn.Module):
    def __init__(self, channel1, channel2):
        super(Saliency_Mining_Module, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2

        self.up = nn.Sequential(nn.Conv2d(self.channel2, self.channel1, 7, 1, 3),
                                nn.BatchNorm2d(self.channel1), nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=2))

        self.delta_gen1 = nn.Sequential(
                        nn.Conv2d(self.channel1*2, self.channel1, kernel_size=1, bias=False),
                        nn.BatchNorm2d(self.channel1),
                        nn.Conv2d(self.channel1, 2, kernel_size=3, padding=1, bias=False)
                        )

        self.delta_gen2 = nn.Sequential(
                        nn.Conv2d(self.channel1*2, self.channel1, kernel_size=1, bias=False),
                        nn.BatchNorm2d(self.channel1),
                        nn.Conv2d(self.channel1, 2, kernel_size=3, padding=1, bias=False)
                        )
        self.input_map = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2), nn.Sigmoid())
        self.gamma = nn.Parameter(torch.ones(1))
        self.out = nn.Conv2d(self.channel1,1,3,1,1)


    def bilinear_interpolate_torch_gridsample(self, input, size, delta=0):
        out_h, out_w = size
        n, c, h, w = input.shape
        s = 1.0
        norm = torch.tensor([[[[w/s, h/s]]]]).type_as(input).to(input.device)
        w_list = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h_list = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h_list.unsqueeze(2), w_list.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + delta.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output


    def forward(self, low_stage, high_stage, in_map):
        h, w = low_stage.size(2), low_stage.size(3)
        high_stage = F.interpolate(input=high_stage, size=(h, w), mode='bilinear', align_corners=True)
        input_map = self.input_map(in_map)
        concat = torch.cat((low_stage, high_stage), 1)
        delta1 = self.delta_gen1(concat)
        delta2 = self.delta_gen2(concat)
        high_stage = self.bilinear_interpolate_torch_gridsample(high_stage, (h, w), delta1)
        low_stage = self.bilinear_interpolate_torch_gridsample(low_stage, (h, w), delta2)

        high_stage += low_stage
        f_feature = self.gamma * high_stage * input_map + high_stage
        out = self.out(f_feature)

        return f_feature, out


class SOD_pvt(nn.Module):
    def __init__(self,channel=32, n_class=1):
        super(SOD_pvt, self).__init__()
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrain_models/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        self.conv_4 = BasicConv2d(512,channel,1)
        self.conv_3 = BasicConv2d(320,channel,1)
        self.conv_2 = BasicConv2d(128,channel,1)
        self.conv_1= BasicConv2d(64,channel,1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.rea_3 = Relation(channel,channel,channel)
        self.rea_2 = Relation(channel,channel,channel)
        self.rea_1 = Relation(channel,channel,channel)


        self.out_4 = nn.Conv2d(channel,n_class,3,1,1)
        self.prob = nn.Sigmoid()

        self.SMM_3 = Saliency_Mining_Module(channel,channel)
        self.SMM_2 = Saliency_Mining_Module(channel,channel)
        self.SMM_1 = Saliency_Mining_Module(channel,channel)


        


    def forward(self, x):
        layer1,layer2,layer3,layer4 = self.backbone(x)

        layer1 = self.conv_1(layer1)
        layer2 = self.conv_2(layer2)
        layer3 = self.conv_3(layer3)
        layer4 = self.conv_4(layer4)

        seg_4 = self.out_4(layer4)
        layer4_prob = self.prob(seg_4) * layer4

        semantic = self.gap(layer4_prob)
        layer3 = self.rea_3(semantic, layer3) + layer3
        layer2 = self.rea_2(semantic, layer2) + layer2
        layer1 = self.rea_1(semantic, layer1) + layer1

        fusion, seg_3 = self.SMM_3(layer3,layer4, seg_4)
        fusion, seg_2 = self.SMM_2(layer2,fusion, seg_3)
        fusion, seg_1 = self.SMM_1(layer1,fusion, seg_2)


        return F.upsample(seg_4, x.size()[2:], mode='bilinear', align_corners=True),F.upsample(seg_3, x.size()[2:], mode='bilinear', align_corners=True),\
        F.upsample(seg_2, x.size()[2:], mode='bilinear', align_corners=True),F.upsample(seg_1, x.size()[2:], mode='bilinear', align_corners=True)



