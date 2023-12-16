# -*- coding: utf-8 -*-
# @Company: Aerospace Information  Research Institute, Chinese Academy of Sciences
# @Time :
# @Author : Xiaoping,Zhang
# @File : datasets1.py

# 写这个脚本的原因是将训练集和验证集的图像预处理分开，训练集是要进行数据加强的，而验证集是不需要的


import os
from osgeo import gdal
import cv2
import numpy as np
np.seterr(divide='ignore',invalid='ignore')
import torch
import random
import torch.utils
import dataset.edge_utils as edge_utils

from torchvision import transforms
from dataset.augmentation import SegAugmentation





# 打开TIF图像
# 返回的维度为[c,h,w]
def read_tiff(fl):
    dataset = gdal.Open(fl)
    if not dataset:
        print(fl + 'filename can not open')
    else:
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        # 坐标再也没有其它作用了额，就可以省略了
        # proj = dataset.GetProjection()
        # geotrans = dataset.GetGeoTransform()
        img = dataset.ReadAsArray(0, 0, width, height)
        del dataset
        return img                                                           # [bands,height,width],比如说[4,512,512]

# 打开掩模图像
def read_label(fl):
    label = cv2.imread(fl,cv2.IMREAD_GRAYSCALE)
    # label = label[np.newaxis, :, :]                                          # [h,w]=>[1,512,512]
    return label


#
# def preprocess(img_fl,label_fl,train_transform,label_transform,edge_mode, aug):
#     # 打开训练样本影像和标签
#     img = read_tiff(img_fl)
#     label = read_label(label_fl)
#     # 数据增强实现
#     if aug:
#         img,label = SegAugmentation()(img,label)
#     label = label[np.newaxis, :, :]
#     assert img.shape == (4,512,512)
#     assert label.shape ==(1,512,512)
#     if edge_mode =
#     # 将img转化为[h,w,4],转化为自然图像读出来的顺序,便于后面转化为Tensor
#     # 将label转化为[h,w,1]
#     img = np.transpose(img, (1, 2, 0)).copy()
#     label = np.transpose(label, (1, 2, 0)).copy()
#     # assert img.shape ==(512,512,4)
#     # assert label.shape ==(512,512,1)
#     # 将array格式转为为torch格式
#     img = train_transform(img)
#     label = label_transform(label)
#     assert  img.size() == (4,512,512)
#     assert label.size() == (1, 512, 512)
#     # 降维,label由三维降到二维
#     label = torch.squeeze(label,0)
#     return (img,label)


def preprocess(img_fl,label_fl,train_transform,label_transform, aug, edge_mode):
    """
    edge_mode:['canny','sobel','spv'] spv refers to SpatialVariance
    """
    # 打开训练样本影像和标签
    img = read_tiff(img_fl)
    label = read_label(label_fl)
    # 数据增强实现
    if aug:
        img,label = SegAugmentation()(img,label)
    else:
        pass


    if edge_mode in ['canny','sobel']:
        if edge_mode == 'sobel':
            edge = edge_utils.grade(label)
            edge[edge > 0] = 255
            edge = edge.astype('uint8')
        else:
            edge = edge_utils.canny(label)

        # 增加维度
        label = label[np.newaxis, :, :]
        edge = edge[np.newaxis, :, :]
        assert img.shape == (4,512,512)
        assert label.shape == (1,512,512)
        assert edge.shape == (1,512,512)

        # 将img转化为[h,w,4],转化为自然图像读出来的顺序,便于后面转化为Tensor
        # 将label转化为[h,w,1]
        img = np.transpose(img, (1, 2, 0)).copy()
        label = np.transpose(label, (1, 2, 0)).copy()
        edge = np.transpose(edge, (1, 2, 0)).copy()
        assert img.shape == (512,512,4)
        assert label.shape == (512,512,1)
        assert edge.shape == (512,512,1)

        # 将array格式转为为tensor格式
        img = train_transform(img)
        label = label_transform(label)
        edge = label_transform(edge)
        assert img.size() == (4, 512, 512)
        assert label.size() == (1, 512, 512)
        assert edge.size() == (1,512,512)

        # 降维,label由三维降到二维
        # label = label.squeeze()
        # edge = edge.squeeze()

    elif edge_mode == 'spv':

        label = label[np.newaxis, :, :]

        # 将img转化为[h,w,4]，将label转化为[h,w,1]
        img = np.transpose(img, (1, 2, 0)).copy()
        label = np.transpose(label, (1, 2, 0)).copy()

        # 将array格式转为为torch格式
        img = train_transform(img)
        label = label_transform(label)

        assert img.size() == (4, 512, 512)
        assert label.size() == (1, 512, 512)
        # edge
        edge = edge_utils.SpatialVariance()(label)
        assert edge.size() == (1,512,512)

        # 降维,label由三维降到二维
        # label = label.squeeze()
        # edge = edge.squeeze()
    else:
        raise NotImplementedError("No such edgemode")

    return (img, label, edge)



class Mydatasets(torch.utils.data.Dataset):
    def __init__(self, img_pth, label_pth, train_transform, label_transform, with_edge=False, edge_mode='canny',aug=True):
        names = os.listdir(img_pth)
        names = [name for name in names if name.endswith('.tif')]
        num = len(names)
        img_sets = []
        for name in names:
            img_path = os.path.join(img_pth, name)
            label_path = os.path.join(label_pth, name)
            img_sets.append((img_path, label_path))

        assert num == len(img_sets)

        self.number = num
        self.img_sets = img_sets
        self.train_transform = train_transform
        self.label_transform = label_transform
        self.aug = aug
        self.with_edge = with_edge
        self.edge_mode = edge_mode

    # 返回edge_label和seg_label,维度都是3维
    def __getitem__(self, index):
        img_path, label_path = self.img_sets[index]
        # 通过索引得到经过图像加强以及Tensor转化的张量

        img, mask, edgemap = preprocess(img_path, label_path, self.train_transform,
                                        self.label_transform, self.aug,  self.edge_mode)

        if self.with_edge:
            return img, mask, edgemap
        else:
            return img, mask

    def __len__(self):
        return self.number


if __name__ == '__main__':
    # 对于训练数据和标签图像的预处理
    train_transform = transforms.ToTensor()
    label_transform = transforms.ToTensor()

    train_ds = Mydatasets(r'E:\experiments\1.samples\img_label_version1/data_512/Training Set/Input images',
                          r'E:\experiments\1.samples\img_label_version1/data_512/Training Set/Mask maps',
                          train_transform,
                          label_transform,
                          with_edge=True,
                          edge_mode='sobel',
                          aug = True)
    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4)

    valid_ds = Mydatasets(r'E:\experiments\1.samples\img_label_version1/data_512/Validation Set/Input images',
                          r'E:\experiments\1.samples\img_label_version1/data_512/Validation Set/Mask maps',
                          train_transform,
                          label_transform,
                          with_edge=True,
                          edge_mode='sobel',
                          aug = False)
    valid_dataloader = torch.utils.data.DataLoader(valid_ds, batch_size=4, shuffle=True)


    print('train_img test.........................................................')
    print('train_img size:',train_ds[0][0].size(),'train_label size:',train_ds[0][1].size(),
            'train_edge size:',train_ds[0][2].size())
    test1 = next(iter(train_dataloader))
    print('train_img_batch size:',test1[0].size(), 'train_label_batch size:',test1[1].size(),
          'train_edge_batch size:',test1[2].size())
    print('train_img type:',test1[0].dtype, 'train_label type:',test1[1].dtype, 'train_edge type:',test1[2].dtype)
    # print(test1[0])
    print(torch.unique(test1[2]))

    print('valid_img test.........................................................')
    print('valid_img size:',valid_ds[0][0].size(), 'valid_label size:',valid_ds[0][1].size(),
    'valid_edge size:',valid_ds[0][2].size())
    test2 = next(iter(valid_dataloader))
    # print(test2)
    print('valid_img_batch size:',test2[0].size(),'valid_label_batch size:', test2[1].size(),
          'valid_edge_batch size:', test2[2].size())
    print('valid_img type:',test2[0].dtype, 'valid_label type:',test2[1].dtype,
          'valid_edge type:',test2[2].dtype)
    print(torch.unique(test2[2]))

#
#
# 测试一下看添加了edge之后的datasets是否正确
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


    img,label,edge = train_ds[10]
    print(type(edge))

    img = np.array(img, dtype=np.float32)
    label = np.array(label.squeeze(), dtype=np.uint8)
    edge = np.array(edge.squeeze(), dtype=np.uint8)
    print(np.unique(label))
    print(np.unique(edge))

    label[label == 1] = 255
    label[label == 0] = 0

    edge[edge == 1] = 255
    edge[edge == 0] = 0




    write_tif(img, r'C:\Users\ZXP\Desktop\2\2.tif')
    cv2.imwrite(r'C:\Users\ZXP\Desktop\2\2_label.tif',label)
    cv2.imwrite(r'C:\Users\ZXP\Desktop\2\2_edge.tif',edge)

# 验证loss能否正确运行
#     from model.dfn import DFN
#     from utils.loss import Jointloss
#
#     img, label, edge = test1
#     print(img.size())
#     print(label.size())
#     print(edge.size())
#
#     pred = DFN(2,1)(img)
#     criterion = Jointloss()
#     loss =criterion(pred,label,edge)
















