# modified strip pooling 


import torch
import torch.nn as nn


# -------------------------------------------------------------
# NewSPBlock: 将strippooling 和平均池化和自适应最大化池化结合起来
# -------------------------------------------------------------


class NEWSPBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None):
        super(NEWSPBlock, self).__init__()
        midplanes = outplanes

        self.conv1 = nn.Conv2d(inplanes, midplanes, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn1 = norm_layer(midplanes)
        self.conv2 = nn.Conv2d(inplanes, midplanes, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn2 = norm_layer(midplanes)
        self.conv3 = nn.Conv2d(midplanes, outplanes, kernel_size=1, bias=True)

        self.pool1_1 = nn.AdaptiveAvgPool2d((None, 1))
        self.pool1_2 = nn.AdaptiveMaxPool2d((None, 1))

        self.pool2_1 = nn.AdaptiveAvgPool2d((1, None))
        self.pool2_2 = nn.AdaptiveMaxPool2d((1, None))

        self.conv_pool = nn.Conv2d(2, 1, kernel_size=(1,1), bias = False)

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        _, _, h, w = x.size()

        x1_1 = self.pool1_1(x)
        print(x1_1.size())
        x1_2 = self.pool1_2(x)
        print(x1_2.size())

        x1 = torch.cat((x1_1, x1_2), dim = 3)
        print(x1.size())
        x1 = x1.permute(0, 3, 2, 1).contiguous()
        print(x1.size())
        x1 = self.conv_pool(x1)
        print(x1.size())
        x1 = x1.permute(0, 3, 2, 1).contiguous()
        print(x1.size())

        x1 = self.conv1(x1)
        print(x1.size())
        x1 = self.bn1(x1)
        print(x1.size())
        x1 = x1.expand(-1, -1, h, w)
        print(x1.size())
        #x1 = F.interpolate(x1, (h, w))

        x2_1 = self.pool2_1(x)
        print(x2_1.size())
        x2_2 = self.pool2_2(x)
        print(x2_2.size())

        x2 = torch.cat((x2_1, x2_2), dim = 2)
        print(x2.size())
        x2 = x2.permute(0, 2, 1, 3).contiguous()
        print(x2.size())
        x2 = self.conv_pool(x2)
        print(x2.size())
        x2 = x2.permute(0, 2, 1, 3).contiguous()
        print(x2.size())

        x2 = self.conv2(x2)
        print(x2.size())
        x2 = self.bn2(x2)
        print(x2.size())
        x2 = x2.expand(-1, -1, h, w)
        print(x2.size())
        #x2 = F.interpolate(x2, (h, w))

        x = self.relu(x1 + x2)
        print(x.size())
        x = self.conv3(x).sigmoid()
        print(x.size())
        return x


if __name__ == "__main__":

    in_put = torch.randn(4,64,128,128)

    model  = NEWSPBlock(64,64,norm_layer=nn.BatchNorm2d)

    out = model(in_put)

