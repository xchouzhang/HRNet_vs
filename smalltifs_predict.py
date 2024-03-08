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


def predict(cfg_file, best_model_path, test_imgs_path, pred_msks_path, save_edge = False, flip = False):

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
    net.load_state_dict(torch.load(best_model_path, map_location=device))


    # make pred_mask_path
    if not os.path.exists(pred_msks_path):
        os.makedirs(pred_msks_path)
        print("新建pred mask文件夹")
    else:
        shutil.rmtree(pred_msks_path)
        print("删除已有pred mask文件夹")
        os.makedirs(pred_msks_path)
        print("重新新建pred mask文件夹")


    # make pred_edges_path
    if save_edge == True:
        pred_edges_path = pred_msks_path.replace('msk_pred','edge_pred')

        if not os.path.exists(pred_edges_path):
            os.makedirs(pred_edges_path)
            print("新建edge mask文件夹")
        else:
            shutil.rmtree(pred_edges_path)
            print("删除已有edge mask文件夹")
            os.makedirs(pred_edges_path)
            print("重新新建edge mask文件夹")

        

    # each img path
    test_imgs_path_name = [tuple([os.path.join(test_imgs_path, each), each]) for each in os.listdir(test_imgs_path)]
    
    print("---------------------------------------------------------")
    print('predicting....')
    print("---------------------------------------------------------")


    # predict

    for each_test_img_path, each_test_img_name in test_imgs_path_name:
        
        # read img
        img = read_tiff(each_test_img_path) 
        img = np.transpose(img,(1,2,0)) 

        # to tensor
        img_tensor = torchvision.transforms.ToTensor()(img)
        img_tensor = torch.unsqueeze(img_tensor,dim = 0).to(device=device, dtype=torch.float32)


        # each save path 
        each_pred_msk_name = each_test_img_name
        each_pred_msk_path = os.path.join(pred_msks_path, each_pred_msk_name.split(".")[0] + "_hrnetv2_msk.jpg")

        if save_edge:
            each_pred_edge_name = each_test_img_name
            each_pred_edge_path = os.path.join(pred_edges_path, each_pred_edge_name.split(".")[0] + "_hrnetv2_edge.jpg")

        # save 
        
        if not save_edge:

            msk_pred = flip_inference(net, img_tensor, flip=flip, with_edge=save_edge)
            msk_prednum = torch.sigmoid(msk_pred).squeeze()
            msk_prednum[msk_prednum > 0.5] = 255
            msk_prednum[msk_prednum <= 0.5] = 0

            msk_pred = np.array(msk_prednum.cpu().data, dtype=np.uint8)

            # 保存图片
            cv2.imwrite(each_pred_msk_path, msk_pred)
    

        else: 
            msk_pred, edge_pred = flip_inference(net,img_tensor,flip=True, with_edge=save_edge)
            msk_prednum = torch.sigmoid(msk_pred).squeeze()
            msk_prednum[msk_prednum > 0.5] = 255
            msk_prednum[msk_prednum <= 0.5] = 0

            msk_pred = np.array(msk_prednum.cpu().data, dtype=np.uint8)

            edge_prednum = torch.sigmoid(edge_pred).squeeze()
            edge_prednum[edge_prednum > 0.5] = 255
            edge_prednum[edge_prednum <= 0.5] = 0

            edge_pred = np.array(edge_prednum.cpu().data, dtype=np.uint8)

            # 保存图片
            cv2.imwrite(each_pred_msk_path, msk_pred)
            cv2.imwrite(each_pred_edge_path, edge_pred)

    print("Predicting has been finished!")
                



if __name__ == "__main__":

    predict("./config/seg_hrnet.yaml", 
            "./checkpoints_seg_hrnet_BS_4_EPOCHS_100_time_2023-12-13_09_37_58/CP_epoch40.pth",
            "./test_image/input images", 
            "./test_image/msk_pred", 
            save_edge = False, 
            flip = False)
    
    
