
# modified Boundary-Aware Bilateral Fusion Network for Cloud Detection(2023)

import torch 
import torch.nn as nn 
import torch.nn.functional as F

from blocks import Bottleneck

## low feature * gate + low fe

class BFM(nn.Module):
    
    """
    Semantic Enhancement Module

    """

    def __init__(self, low_channels, high_channels):
        super(BFM, self).__init__()

        self.low_conv1 = Bottleneck(low_channels, low_channels//4)
        self.low_conv2 = nn.Conv2d(low_channels, high_channels//2, 1)
        
        self.high_conv = nn.Conv2d(high_channels, high_channels//2, 1 )

        self.fuse_gate = nn.Sequential(
            nn.BatchNorm2d(high_channels),
            nn.ReLU(),
            nn.Conv2d(high_channels, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()

        )


   
    def forward(self, low_feat, high_feat):

        _, _, H, W = low_feat.size()

        x_low = self.low_conv1(low_feat)
        x_low = self.low_conv2(x_low)


        x_high = self.high_conv(high_feat)
        x_high = F.interpolate(x_high, (H,W), mode="bilinear", align_corners=True)

        x_fuse = torch.cat((x_low, x_high), dim=1)

        gate_out = self.fuse_gate(x_fuse)

        x_low = low_feat + low_feat * gate_out

        return x_low