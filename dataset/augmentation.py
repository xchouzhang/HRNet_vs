# -*- coding: utf-8 -*-
# @Company: Aerospace Information  Research Institute, Chinese Academy of Sciences
# @Time :  
# @Author : Xiaoping,Zhang
# @File : augmentation.py


import os
import cv2
import numpy as np
import random
import torch
import random
import torch.utils

from osgeo import gdal
from torchvision import transforms


"""version 1"""
class Augmentation():
    def __init__(self,img):
        self.array = img
    # 翻转
    def flip(self,action):
        # 水平翻转

        if action == 'hor':
            img_flip = np.flip(self.array,2)
        # 垂直翻转
        elif action == 'vet':
            img_flip  = np.flip(self.array,1)
        # 水平加垂直翻转
        elif action == 'all':
            img_flip  = np.flip(self.array,(1,2))
        else:
            img_flip  = self.array
        self.array = img_flip            # 图像，标签矩阵的格式都是[bands,height,width]

    # 旋转,angle 为角度
    def rotation(self,angle):
        c,h,w = self.array.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle,1)
        for i in range(c):
            self.array [i]= cv2.warpAffine(self.array[i],M,(w,h))

        # img_rotation = self.img
        # return img_rotation



""" version 2 """
def read_tiff(fl):
    dataset = gdal.Open(fl)
    if not dataset:
        print(fl + 'filename can not open')
    else:
        im_width = dataset.RasterXSize
        im_height = dataset.RasterYSize
        im_bands = dataset.RasterCount
        im_proj = dataset.GetProjection()
        im_geotrans = dataset.GetGeoTransform()
        im_data = dataset.ReadAsArray(0, 0, im_width, im_height)

        del dataset
        return im_data                                                              # [bands,height,width],比如说[4,512,512]

def read_label(fl):
    label = cv2.imread(fl,cv2.IMREAD_GRAYSCALE)                                      # [h,w]=>[512,512]
    return label


def write_tif(im_data, path):
    # 判断栅格数据的数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, im_width, im_height, im_bands, datatype)

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

    del dataset



# copy from(https://github1s.com/feevos/resuneta/blob/master/src/semseg_aug_cv2.py)
class ParamsRange(dict):
    def __init__(self):

        self['flip_mode'] = ['x', 'y', 'xy', 'None']

        self['center_range'] = (256,256)
        self['rot_range'] = [0,90,180,270]

        # random_scale
        # self['scale_range']=[0.5,0.75,1.0,1.25,1.5]
        self['scale_range']=[1.0]

class SegAugmentation(object):
    def __init__(self, params_range = ParamsRange(), prob=1.0, one_hot=False):
        self.one_hot = one_hot
        self.range = params_range
        self.prob = prob
        assert self.prob <= 1, "prob must be in range [0,1], you gave prob::{}".format(prob)


    def _flip(self, _img, _mask, mode):
        """

        :param _img: the size of inputs,(c,h,w)
        :param _mask: the size of mask,(h,w)
        :param mode: the mode of reflection,'x','y','xy','None'
        :return: the output of transformation
        """
        # 沿着x轴
        if mode == 'x':
            img_z = _img[:, ::-1, :]
            if self.one_hot:
                mask_z = _mask[:, ::-1, :]  # 1hot representation
            else:
                mask_z = _mask[::-1, :]  # standard (int's representation)
        # 沿着y轴
        elif mode == 'y':
            img_z = _img[:, :, ::-1]
            if self.one_hot:
                mask_z = _mask[:, :, ::-1]  # 1hot representation
            else:
                mask_z = _mask[:, ::-1]  # standard (int's representation)
        # 对角翻转
        elif mode == 'xy':
            img_z = _img[:, ::-1, ::-1]
            if self.one_hot:
                mask_z = _mask[:, ::-1, ::-1]  # 1hot representation
            else:
                mask_z = _mask[::-1, ::-1]  # standard (int's representation)
        else:
            img_z = _img
            mask_z = _mask

        return img_z, mask_z

    def _rotation_scale(self, _img, _mask, _center, _angle, _scale):
        """

         :param _img: (c,h,w)
         :param _mask: (h,w)
         :param _center: (w方向,h方向)
         :param _angle:(角度)
         :param _scale:(尺度)
         :return:
         """
        # imgT:(h,w,c)
        imgT = _img.transpose([1, 2, 0])
        if (self.one_hot):
            maskT = _mask.transpose([1, 2, 0])
        else:
            maskT = _mask

        rows, cols = imgT.shape[:-1]

        # Produces affine rotation matrix, with center, for angle, and optional zoom in/out scale
        tRotMat = cv2.getRotationMatrix2D(_center, _angle, _scale)

        img_trans = cv2.warpAffine(imgT, tRotMat, (cols, rows), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT)  # """,flags=cv2.INTER_CUBIC,"""
        mask_trans = cv2.warpAffine(maskT, tRotMat, (cols, rows), flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_REFLECT)

        # img_trans:(h,w,c)
        img_trans = img_trans.transpose([2, 0, 1])
        if (self.one_hot):
            mask_trans = mask_trans.transpose([2, 0, 1])

        return img_trans, mask_trans

    def rand_flip(self, _img, _mask):

        mode = random.choice(self.range['flip_mode'])

        return  self._flip(_img, _mask, mode)

    def rand_rotation_scale(self, _img, _mask):

        center = self.range['center_range']
        angle = random.choice(self.range['rot_range'])
        scale = random.choice(self.range['scale_range'])

        return self._rotation_scale(_img, _mask, center, angle, scale)

    def __call__(self, _img, _mask):
        rand = np.random.rand()
        if (rand <= self.prob):
            img ,mask = self.rand_flip(_img , _mask)
            img ,mask = self.rand_rotation_scale(img , mask)
            return img, mask
        else:
            return _img, _mask



if __name__ == '__main__':

    img = read_tiff(r"C:\Users\ZXP\Desktop\ms_experiment\img.tif")
    mask = read_label(r"C:\Users\ZXP\Desktop\ms_experiment\mask.tif")
    print(img.shape)
    print(mask.shape)

    img_out,mask_out = SegAugmentation()(img,mask)
    # img_out,mask_out = shift_rot_zoom(img,mask,(256,256),0,0.75)
    print(img_out.shape)
    print(mask_out.shape)
    print(np.unique(mask_out))

    write_tif(img_out,r"C:\Users\ZXP\Desktop\ms_experiment\img_out.tif")
    cv2.imwrite(r"C:\Users\ZXP\Desktop\ms_experiment\mask_out.tif", mask_out)




















