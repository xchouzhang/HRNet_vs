import os
import sys

sys.path.append(r"..")

import numpy as np
import torch
from torchvision.models import resnet18
import time

from config import update_config
from config import default_config

from Module.seg_hrnet import HighResolutionNet





if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = torch.randn(1, 4, 512, 512).to(device)

    # cfg_file = "seg_hrnet.yaml"
    # cfg_file = "seg_se_hrnet.yaml"
    # cfg_file = "seg_cbam_hrnet.yaml"
    # cfg_file = "seg_scse_hrnet.yaml"
    # cfg_file = "seg_coord_hrnet.yaml"
    # cfg_file = "seg_trip_hrnet.yaml"
    # cfg_file = "seg_sp_hrnet.yaml"
    # cfg_file = "seg_se_gc_hrnet.yaml"
    cfg_file = "seg_newsp_hrnet.yaml"

    cfg = update_config(default_config, r"../config/" + cfg_file)
    model = HighResolutionNet(cfg).to(device)

    model.eval()


    # Warn-up
    for _ in range(5):
        start = time.time()
        outputs = model(input)
        torch.cuda.synchronize()
        end = time.time()
        print('Time:{}ms'.format((end-start)*1000))

    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
        outputs = model(input)
    print(prof.key_averages().table())

