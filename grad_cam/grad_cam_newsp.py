import sys
sys.path.append(r"..")

import cv2
import torch
import torchvision
import torch.functional as F
import numpy as np

from osgeo import gdal
from PIL import Image

from config import default_config, update_config
from Module.seg_hrnet import HighResolutionNet

from dataset.datasets import read_tiff, preprocess

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from collections import OrderedDict

# 获取所有层的名称
def get_all_layers(model, prefix=''):
    all_layers = OrderedDict()
    for name, layer in model.named_children():
        full_name = prefix + '.' + name if prefix else name
        if isinstance(layer, torch.nn.Module):
            all_layers[full_name] = layer
            all_layers.update(get_all_layers(layer, full_name))
        else:
            all_layers[full_name] = layer
    return all_layers



class SegmentationModelOutputWrapper(torch.nn.Module):
    def __init__(self, model): 
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model
        
    def forward(self, x):
        seg_out = self.model(x)
        return seg_out


# --------------------------------------------------------------------------------------------
# 实现语义分割模型某个层的可视化
# --------------------------------------------------------------------------------------------

# path-需要修改
img_path = r"./img/768.tif"
# best_model_path = r"../checkpoints_seg_hrnet_BS_4_EPOCHS_100_time_2023-12-13_09_37_58/CP_epoch40.pth"
# best_model_path = r"../checkpoints_seg_cbam_hrnet_BS_4_EPOCHS_100_time_2023-12-12_17_58_28/CP_epoch40.pth"
best_model_path = r"../checkpointsseg_se_hrnet_BS_4_EPOCHS_100_time_2023-12-12_17_21_01/best_model.pth"
# best_model_path = r"../checkpoints_seg_newsp_hrnet_BS_4_EPOCHS_100_time_2023-12-14_11_24_27/best_model.pth"
# cfg_file_name = "seg_hrnet.yaml"
# cfg_file_name = "seg_cbam_hrnet.yaml"
cfg_file_name = "seg_se_hrnet.yaml"
# cfg_file_name = "seg_newsp_hrnet.yaml"



# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# input
img = read_tiff(img_path) 
img = np.transpose(img,(1,2,0)) 
img_tensor = torchvision.transforms.ToTensor()(img)
img_tensor = torch.unsqueeze(img_tensor,dim = 0).to(device=device, dtype=torch.float32)


# select models and load pth
cfg = update_config(default_config, r"../config/" + cfg_file_name)
model = HighResolutionNet(cfg).to(device)
model.load_state_dict(torch.load(best_model_path, map_location=device)['model_state'])
# model.load_state_dict(torch.load(best_model_path, map_location=device))

model.eval()


# # 获取模型所有层最底层的名称
# all_layers = get_all_layers(model)
# for name in all_layers:
#     print(name)


# redefine model and inference
model = SegmentationModelOutputWrapper(model)
output = model(img_tensor)

# define classes of datasets
sem_classes = [
    '__background__', 'greenhouse'
]

sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
greenhouse_category = sem_class_to_idx["greenhouse"]

greenhouse_mask = torch.where(torch.sigmoid(output).squeeze() > 0.5, torch.tensor(1), torch.tensor(0)).detach().cpu().numpy()
greenhouse_mask_uint8 = 255 * np.uint8(greenhouse_mask == greenhouse_category)
greenhouse_mask_float = np.float32(greenhouse_mask == greenhouse_category)


class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        
    def __call__(self, model_output):
        return (model_output[0, :, : ] * self.mask).sum()


# 遥感影像前三波段的可视化,不知道对不对
img_ = img[:,:,:3]
img_1 = np.float32(img_)/255


# select visualized layers

# target_layers =[model.model.layer1[3]]
target_layers =[model.model.stage3[0].branches[0][3]]
targets = [SemanticSegmentationTarget(greenhouse_category, greenhouse_mask_float)]

# cam 实例化
with GradCAM(model=model, target_layers=target_layers) as cam:
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0, :]

    # cam_image = show_cam_on_image(img_, grayscale_cam, use_rgb=True, colormap=cv2.COLORMAP_VIRIDIS, image_weight=0.5)
    cam_image = show_cam_on_image(img_1, grayscale_cam, use_rgb=True,image_weight=0)
    


# save png
cam_image = Image.fromarray(cam_image)
cam_image.show()

cam_image.save(r"./img/768/768_se_stage3-0-branch0-3.jpg")
# cam_image.save(r"./img/768/768_newsp-layer1-3.jpg")



img_save = Image.fromarray(img_)
img_save.save(r"./img/768/768.jpg")



# 如果输出是多波段的，则可以去官网查看语义分割教程

















