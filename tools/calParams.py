'''
Author: your name
Date: 2024-02-27 11:23:11
LastEditors: your name
LastEditTime: 2024-04-11 14:05:41
Description: 
FilePath: \HRNet_vs\tools\calParams.py
'''
import sys
sys.path.append("..")

import torch

from config import update_config
from config import default_config

from utils.calParasFlops import calParasFlops, calFlopsMacsParas 
from Module.seg_hrnet import HighResolutionNet




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input = torch.randn(1, 4, 512, 512).to(device)

cfgfileList = [
    # "seg_hrnet.yaml",
    # "seg_se_hrnet.yaml"
    # "seg_cbam_hrnet.yaml",
    # "seg_scse_hrnet.yaml",
    # "seg_coord_hrnet.yaml",
    # "seg_trip_hrnet.yaml",
    # "seg_sp_hrnet.yaml",
    # "seg_se_gc_hrnet.yaml",
    # "seg_newsp_hrnet.yaml"
    "seg_hrnet_32.yaml",

]

for cfg_file in cfgfileList:
    cfg = update_config(default_config, r"../config/" + cfg_file)
    model = HighResolutionNet(cfg).to(device)
    # macs1, params1 = calParasFlops(model, input)
    flops, macs2, params2 = calFlopsMacsParas(model, input)
