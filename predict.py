# -*- coding: utf-8 -*-
# @Company: Aerospace Information  Research Institute, Chinese Academy of Sciences
# @Time :  
# @Author : Xiaoping,Zhang
# @File : predict.py



import os
import glob
import numpy as np
import torch
import cv2
import shutil

import torchvision
from dataset.datasets import read_tiff
from config import default_config
from config import update_config


from evaluate.evaluate import evaluate, flip_inference


def predict(cfg_file, best_model, test_imgs_path, pre_masks_path, save_edge = True):

    # device
    device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')

    # model
    if save_edge == True:
        
        from Module.seg_hrnet_withedge import HighResolutionNet
        net = HighResolutionNet(update_config(default_config, cfg_file), 4 ,1).to(device)
    
    else:
        from Module.seg_hrnet import HighResolutionNet
        net = HighResolutionNet(update_config(default_config, cfg_file), 4, 1).to(device)

    # .pth load
    net.load_state_dict(torch.load('./pth/best_model.pth', map_location=device)['model_state'])


    # make save_mask_path
    if not os.path.exists(pre_masks_path):
        os.makedirs(pre_masks_path)
        "新建pre mask 文件夹"
    else:
        shutil.rmtree(pre_masks_path)
        os.makedirs(pre_masks_path)

    

    



    # each img path
    test_img_path = os.path.join(test_imgs_path, os.listdir(test_imgs_path))  
    

    



if __name__ == "__main__":

    device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')




    

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    # 加载网络，图片4通道，分类为1
    cfg = update_config(default_config, 'seg_hrnet.yaml')
    net = MyNet(cfg, 4,1)

    # 将网络拷贝到deivce中
    net.to(device)
    # 加载模型参数
    # net.load_state_dict(torch.load('./pth/CP_epoch20.pth', map_location=device))
    net.load_state_dict(torch.load('./pth/best_model.pth', map_location=device)['model_state'])
    # 测试模式
    net.eval()
    # 读取所有图片路径
    testMaskPath = r'/home/zhangxiaoping/experiments/0_My/mynet_edge/Test Set/Input images'
    testMaskName = os.listdir(testMaskPath)

    # 遍历所有图片
    print('predicting...................................')
    for testmaskname in testMaskName:
        testmaskpath = os.path.join(testMaskPath,testmaskname)
        # 保存结果地址
        save_res_path =  os.path.join(testMaskPath.replace('Input images','predict mask'),testmaskname.split('.')[0]+'_res.tif')
        edge_res_path = os.path.join(testMaskPath.replace('Input images','predict edge'),testmaskname.split('.')[0]+'_edgeres.tif')
        # 读取图片
        img = read_tiff(testmaskpath) #   [4,256,256]
        img = np.transpose(img,(1,2,0)) #[256,256,4]
        # 转为tensor
        img_tensor = torchvision.transforms.ToTensor()(img)
        img_tensor = torch.unsqueeze(img_tensor,dim = 0)
        # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)

        # 预测

        # pred = net(img_tensor)                                     #[1,2,256,256]
        # # 提取结果
        # pred2num = torch.argmax(pred,dim=1)
        # pred2num = torch.squeeze(pred2num)
        # res = np.array(pred2num.data.cpu())
        # res[res== 0] = 0
        # res[res == 1] = 255
        seg_pred, edge_pred = flip_inference(net,img_tensor,flip=True)

        seg_prednum = torch.sigmoid(seg_pred).squeeze()

        seg_prednum[seg_prednum > 0.5] = 255
        seg_prednum[seg_prednum <= 0.5] = 0
        seg_res = np.array(seg_prednum.cpu().data, dtype=np.uint8)

        edge_prednum = torch.sigmoid(edge_pred).squeeze()

        edge_prednum[edge_prednum > 0.5] = 255
        edge_prednum[edge_prednum <= 0.5] = 0
        edge_res = np.array(edge_prednum.cpu().data, dtype=np.uint8)

        # 保存图片
        cv2.imwrite(save_res_path, seg_res)
        cv2.imwrite(edge_res_path, edge_res)

    print("results are out")