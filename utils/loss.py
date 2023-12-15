# -*- coding: utf-8 -*-
# @Company: Aerospace Information  Research Institute, Chinese Academy of Sciences
# @Time :  
# @Author : Xiaoping,Zhang
# @File : loss.py

import torch
import torch.nn as nn
import cv2
import numpy as np
import torchvision.transforms as tf


#  这是根据源代码中的代码修改而成,和下面使用了torch的API计算的结果一样

# class Maskloss(nn.Module):
#     def __init__(self, w=2, name= 'mask_loss'):
#         super(Maskloss, self).__init__()
#         self.w = w
#
#     def forward(self,mask_pred,mask_gt):
#
#         mask_type = torch.float32
#         mask_gt =mask_gt.float()
#
#         wp = torch.ones_like(mask_gt, dtype=mask_type)*self.w
#         wn = torch.ones_like(mask_gt, dtype=mask_type)
#         weights = torch.where(mask_gt > 0.5, wp, wn)
#
#         loss = mask_gt * torch.log(torch.clamp(mask_pred, 1e-7, 1)) + (1 - mask_gt) * torch.log(
#             torch.clamp(1 - mask_pred, 1e-7, 1))
#         loss = weights * loss
#         return torch.mean(-1.0 * loss)



class Segloss(nn.Module):
    def __init__(self, w = 1,name = 'mask_loss'):
        super(Segloss, self).__init__()

        self.w = w
        self.name = name
        # self.sigmoid = nn.Sigmoid()

    def forward(self, mask_pred , mask_gt):

        # mask_pred = self.sigmoid(mask_pred)
        mask_type = torch.float32
        mask_gt = mask_gt.float()

        wp = torch.ones_like(mask_gt, dtype = mask_type)* self.w
        wn = torch.ones_like(mask_gt, dtype = mask_type)

        weights = torch.where(mask_gt > 0.5, wp, wn)
        # print(weights.size())
        # loss = mask_gt * torch.log(torch.clamp(mask_pred,1e-7,1)) + (1-mask_gt) * torch.log(torch.clamp(1-mask_pred,1e-7,1))
        # loss = torch.mul(weights, loss)
        # loss =torch.mean(-1.0*loss)

        loss = nn.BCEWithLogitsLoss(weight = weights)(mask_pred ,mask_gt)

        return loss




# 边缘损失函数
class BoundaryLoss(nn.Module):
    def __init__(self, name = 'boundary loss'):
        super(BoundaryLoss, self).__init__()
        self.name = name
        self.sigmoid = nn.Sigmoid()

    def forward(self,mask_pred, mask_gt):

        mask_pred = self.sigmoid(mask_pred)

        mask_gt = mask_gt.float()
        beta = 1.0 - torch.mean(mask_gt)
        loss = beta * mask_gt * torch.log(torch.clamp(mask_pred,1e-7, 1)) + (1-beta) * (1-mask_gt) * torch.log(torch.clamp(1-mask_pred,1e-7,1))
        return torch.mean(-1.0*loss)



# 先暂时定义联合损失函数

class JointLoss(nn.Module):
    def __init__(self, w = 1, alpha = 2,name = 'Joint loss'):
        super(JointLoss, self).__init__()

        self.w = w
        self.alpha = alpha
        self.name = name

    def forward(self, mask_pred, mask_gt, edge_pred, edge_gt):
        loss = Segloss(self.w)(mask_pred, mask_gt) + BoundaryLoss()(edge_pred, edge_gt)*self.alpha
        return loss




if __name__ == '__main__':


    torch.manual_seed(3)
    input = torch.randn(1,1,512,512)

    # print(mask_pred.size())
    mask_gt =cv2.imread(r'260.tif', cv2.IMREAD_GRAYSCALE)
    mask_gt = tf.ToTensor()(mask_gt)
    mask_gt = torch.unsqueeze(mask_gt,dim= 0)
    mask_gt = mask_gt.float()

    loss = Segloss(w=1)
    loss1 = nn.BCEWithLogitsLoss()

    l = loss(input,mask_gt)
    l1 = loss1(input,mask_gt)
    print(l, l1)









