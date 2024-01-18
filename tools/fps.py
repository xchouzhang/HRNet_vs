import sys
sys.path.append("..")

import time
import torch

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
    # cfg_file = "seg_newsp_hrnet.yaml"

    cfg = update_config(default_config, r"../config/" + cfg_file)
    model = HighResolutionNet(cfg).to(device)

    model.eval()

    # 预热
    for _ in range(10):
        with torch.no_grad():
            model(input)


    # 测量推理时间
    start_time = time.time()

    for _ in range(100):
        with torch.no_grad():
            model(input)

    end_time = time.time()

    # 计算FPS

    fps = 100 / (end_time - start_time)
    print("FPS: {:.3f}".format(fps))

   
