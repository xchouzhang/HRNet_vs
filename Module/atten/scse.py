import torch 
import torch.nn as nn


class CSE(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CSE, self).__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
   
    def forward(self, x):
        return x * self.cSE(x)


class SSE(nn.Module):
    def __init__(self, in_channels):
        super(SSE, self).__init__()
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.sSE(x)



class SCSE(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SCSE, self).__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)
    



if __name__ == '__main__':

        in_put=torch.randn(4,512,64,64)
        scse = SCSE(512)

        output=scse(in_put)
        print(output.shape)