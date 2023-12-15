# from "Multitask Semantic Boundary Awareness Network for Remote Sensing Image Segmentation"

import torch
import torch.nn as nn

import torch.nn.functional as F

class BAM(nn.Module):

    def __init__(self, low_channels, high_channels, mid_channels=16 ):
        super(BAM, self).__init__()

        self.low_conv = nn.Sequential(
            nn.Conv2d(low_channels, mid_channels, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
            )
        
        self.high_conv1 = nn.Sequential(
            nn.Conv2d(high_channels, mid_channels, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
        )
        self.high_conv2 = nn.Conv2d(mid_channels, 1, 1)

        self.alpha = nn.Sigmoid()

        self.conv = nn.Sequential(
            nn.Conv2d(mid_channels*2, low_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(low_channels),
            nn.ReLU()
        )

    def forward(self, low_feat, high_feat):

        _,_,H,W = low_feat.size()

        s_i = self.low_conv(low_feat)

        s_j = self.high_conv1(high_feat)
        s_j = F.interpolate(s_j, (H,W), mode = "bilinear", align_corners=True)

        s_j_alpha = self.high_conv2(s_j)

        s_i = s_i * (1-self.alpha(s_j_alpha))

        x_out = self.conv(torch.cat((s_i, s_j), dim = 1))

        return x_out








