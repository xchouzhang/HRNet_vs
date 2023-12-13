# Strip Pooling: Rethinking spatial pooling for scene parsing (CVPR 2020)
import torch
import torch.nn as nn

class SPBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None):
        super(SPBlock, self).__init__()
        midplanes = outplanes
        self.conv1 = nn.Conv2d(inplanes, midplanes, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn1 = norm_layer(midplanes)
        self.conv2 = nn.Conv2d(inplanes, midplanes, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn2 = norm_layer(midplanes)
        self.conv3 = nn.Conv2d(midplanes, outplanes, kernel_size=1, bias=True)
        self.pool1 = nn.AdaptiveAvgPool2d((None, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((1, None))
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.pool1(x)
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = x1.expand(-1, -1, h, w)
        #x1 = F.interpolate(x1, (h, w))

        x2 = self.pool2(x)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = x2.expand(-1, -1, h, w)
        #x2 = F.interpolate(x2, (h, w))

        x = self.relu(x1 + x2)
        x = self.conv3(x).sigmoid()
        return x


if __name__ == "__main__":

    in_put = torch.randn(4,64,128,128)

    model = SPBlock(64,64,norm_layer=nn.BatchNorm2d)

    out = model(in_put)

    print(out.size())