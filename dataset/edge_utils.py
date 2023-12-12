# -*- coding: utf-8 -*-
# @Company: Aerospace Information  Research Institute, Chinese Academy of Sciences
# @Time :  
# @Author : Xiaoping,Zhang
# @File :edge_utils.py

import cv2
import os
import torch
import glob
import torchvision
import numpy as np
import torch.nn as  nn
from PIL import Image
from scipy.ndimage.morphology import distance_transform_edt

def mask_to_onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector

    """
    _mask = [mask == (i+1) for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)


def onehot_to_mask(mask):
    """
    Converts a mask (K,H,W) to (H,W)
    """
    _mask = np.argmax(mask, axis=0)
    _mask[_mask != 0] += 1
    return _mask


def onehot_to_multiclass_edges(mask, radius, num_classes):
    """
    Converts a segmentation mask (K,H,W) to an edgemap (K,H,W)

    """
    if radius < 0:
        return mask

    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

    channels = []
    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :]) + distance_transform_edt(1.0 - mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        dist = (dist > 0).astype(np.uint8)
        channels.append(dist)

    return np.array(channels)


def onehot_to_binary_edges(mask, radius, num_classes):
    """
    Converts a segmentation mask (K,H,W) to a binary edgemap (H,W)

    """

    if radius < 0:
        return mask

    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

    edgemap = np.zeros(mask.shape[1:])

    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :]) + distance_transform_edt(1.0 - mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        edgemap += dist
    edgemap = np.expand_dims(edgemap, axis=0)
    edgemap = (edgemap > 0).astype(np.uint8)
    return edgemap




class  SpatialVariance(nn.Module):
    def __init__(self, pool_size=3):
        super(SpatialVariance,self).__init__()
        self.pool_size = pool_size

    def forward(self,x):
        return torch.abs(x - nn.MaxPool2d(3,1,padding=1)(x))


def grade(img):
    x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    # 转回到unit8
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    mi=np.min(dst)
    ma=np.max(dst)
    return (dst-mi)/(ma-mi)


def canny(img):
    edge = cv2.Canny(img, 5, 5, apertureSize=3)
    edge = cv2.dilate(edge, kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(2,2)))

    return edge



if __name__ == '__main__':

# # 第一种取边界的方法，one_hot,from GSCNN
#     label = cv2.imread(r"C:\Users\ZXP\Desktop\5.tif",cv2.IMREAD_GRAYSCALE)
#     label = np.array(label)
#
#     label[label == 255] = 1
#     label[label ==0] = 0
#
#     edge_map = mask_to_onehot(label,2)
#     edge_map = onehot_to_binary_edges(edge_map,1,2)
#
#     edge_map = np.squeeze(edge_map)
#     edge_map[edge_map == 1] =255
#     edge_map[edge_map == 0] =0
#
#     edge_out = cv2.imwrite(r"C:\Users\ZXP\Desktop\5_onehot2edge_out.tif",edge_map)
#
# # 第二种取边界的方法,Spatial-Vrariance form Joint-...

#     mask = cv2.imread(r"C:\Users\ZXP\Desktop\5.tif",cv2.IMREAD_GRAYSCALE)
#     # print(mask.shape)
#     mask =  torchvision.transforms.ToTensor()(mask)
#     print(mask.size())
#     edge_tensor  = SpatialVariance()(mask)
#     print(edge_tensor.size())
#     edge = np.array(edge_tensor).astype(np.uint8)
#     # print(edge.shape)
#     edge = np.squeeze(edge)
#     edge[edge == 1] =255
#     edge[edge == 0] = 0
#
#     edge_map = cv2.imwrite(r"C:\Users\ZXP\Desktop\5_spatialvar_out.tif",edge)
# #
# 第三种采用canny或者sober算子进行边缘的采集
# sobel算子,具体原理和参数也不懂

    def grade(img):
        x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        # 转回到unit8
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        mi=np.min(dst)
        ma=np.max(dst)
        return (dst-mi)/(ma-mi)

    img = cv2.imread(r'C:\Users\ZXP\Desktop\5.tif',cv2.IMREAD_GRAYSCALE)
    # out shape [H,W]
    out = grade(img)
    print(out.shape,np.unique(out),out.dtype)
    out[out > 0 ] = 255
    cv2.imwrite(r'C:\Users\ZXP\Desktop\5_sobel_out.tif',out)


# # copy from https://github1s.com/ycszen/TorchSeg/blob/HEAD/model/dfn/cityscapes.dfn.R101_v1c/dataloader.py
# # 原理不是很清楚
# 第四种采用canny算子产生边缘
    def canny(img):
        edge = cv2.Canny(img, 5, 5, apertureSize=3)
        edge = cv2.dilate(edge, kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(2,2)))

        return edge

    mask = cv2.imread(r'C:\Users\ZXP\Desktop\5.tif', cv2.IMREAD_GRAYSCALE)
    out = canny(mask)
    print(out.dtype)
    print(np.unique(out))

    cv2.imwrite(r'C:\Users\ZXP\Desktop\5_canny_out.tif', out)

#
# # 用来生成所有训练集和我验证集的edges
# # generate edges for training datasets
#     mask_map_paths = glob.glob(r'E:\experiments\1.samples\img_label_version1\data_512\Training Set\Mask maps'+'/*.tif')
#     mask_map_names = os.listdir(r'E:\experiments\1.samples\img_label_version1\data_512\Training Set\Mask maps')
#
#     assert len(mask_map_paths) == len(mask_map_names)
#     mask_map_num = len(mask_map_names)
#
#     for i in range(mask_map_num):
#         mask = cv2.imread(mask_map_paths[i], cv2.IMREAD_GRAYSCALE)
#         # print(mask.shape)
#         mask =  torchvision.transforms.ToTensor()(mask)
#         # print(mask.size())
#         edge_tensor  = SpatialVariance()(mask)
#         # print(edge_tensor.size())
#         edge = np.array(edge_tensor).astype(np.uint8)
#         # print(edge.shape)
#         edge = np.squeeze(edge)
#         edge[edge == 1] =255
#         edge[edge == 0] = 0
#
#         edge_map = cv2.imwrite(r'E:\experiments\1.samples\img_label_version1\data_512\Training Set\Edges\{}'.
#                                format(mask_map_names[i]),edge)
#
#
#
#     mask_map_paths = glob.glob(r'E:\experiments\1.samples\img_label_version1\data_512\Validation Set\Mask maps' + '/*.tif')
#     mask_map_names = os.listdir(r'E:\experiments\1.samples\img_label_version1\data_512\Validation Set\Mask maps')
#
#     assert len(mask_map_paths) == len(mask_map_names)
#     mask_map_num = len(mask_map_names)
#
#     for i in range(mask_map_num):
#         mask = cv2.imread(mask_map_paths[i], cv2.IMREAD_GRAYSCALE)
#         # print(mask.shape)
#         mask = torchvision.transforms.ToTensor()(mask)
#         # print(mask.size())
#         edge_tensor = SpatialVariance()(mask)
#         # print(edge_tensor.size())
#         edge = np.array(edge_tensor).astype(np.uint8)
#         # print(edge.shape)
#         edge = np.squeeze(edge)
#         edge[edge == 1] = 255
#         edge[edge == 0] = 0
#
#         edge_map = cv2.imwrite(r'E:\experiments\1.samples\img_label_version1\data_512\Validation Set\Edges\{}'.
#                                format(mask_map_names[i]), edge)














