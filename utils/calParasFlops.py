# -*- coding: utf-8 -*-
# @Company: Aerospace Information  Research Institute, Chinese Academy of Sciences
# @Time :  
# @Author : Xiaoping,Zhang
# @File : calParasFlops.py


import torch
from Module.hrnet import HighResolutionNet as HR1
from Module.seg_hrnet import HighResolutionNet as HR2
from Module.seg_hrnet import get_seg_model
from thop import profile,clever_format
from torchsummary import summary

from torchstat import stat

# 第一种
def calParasFlops(model,input):
    flops,params =  profile(model,inputs = (input,))
    flops, params = clever_format([flops, params], "%.3f")
    print("The FLOPs:{}".format(flops) , "The parameters:{}".format(params))
    return flops,params

# 第二种,后面的input_size必须是三维的
# from torchstat import stat
# stat(model, input_size=(4, 512, 512))

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = torch.randn(1, 4, 512, 512).to(device)

    # #hr1
    # net1 = HR1(1).to(device)
    # flops, paras = calParasFlops(net1, input)
    # summary(net1, (4, 512, 512))

    # hr2
    from config.default import _C

    cfg = _C
    cfg.defrost()
    cfg.merge_from_file('../seg_hrnet.yaml')
    cfg.freeze()
    print(cfg)

    net = get_seg_model(cfg)
    flops, paras = calParasFlops(net, input)

    stat(net,(4, 512, 512))
    # summary(net, (4, 512, 512))



