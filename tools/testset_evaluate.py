import os
import sys

sys.path.append(r"..")

import datetime
import argparse

import torch
import torchvision

from dataset.datasets import Mydatasets
from torch.utils.data import DataLoader
from utils.utils import get_logger

from evaluate.evaluate import evaluate

from config import update_config
from config import default_config

from Module.seg_hrnet import HighResolutionNet


def testset_evaluate(test_img_path, test_mask_path, device, flip, with_edge):

    # Setup datasets and dataloader
    train_transform = torchvision.transforms.ToTensor()
    label_transform = torchvision.transforms.ToTensor()

    test_datasets = Mydatasets(test_img_path, 
                               test_mask_path, 
                               train_transform, 
                               label_transform, 
                               aug=False)

    test_dataloader = DataLoader(test_datasets,
                                 batch_size = 1,
                                 shuffle = False,
                                 num_workers=4,
                                )
    logger.info("Train Datasets and Validation Datasets has been created!")

    
    #  Model
    model.eval()

    if with_edge == False:
        running_metrics_test = evaluate(test_dataloader, model, device, flip=flip, with_edge=with_edge)
        score, class_iou = running_metrics_test.get_scores()

        for k,v in score.items():
            logger.info("{}: {}".format(k, v))

        for k,v in class_iou.items():
            logger.info("{}: {}".format(k, v))


def  get_args():

    # 创建一个参数对象
    parser = argparse.ArgumentParser(description='Train the model on images and masks')

    # 定义数据集的路径
    parser.add_argument('-testimgpth', '--testset-image-pth', type=str,
                        default=r'/data/sgData/Test Set/img',
                        help='the path of the testset img', dest='testimgpth')
    parser.add_argument('-testmaskpth', '--testset-mask-pth', type=str,
                        default=r'/data/sgData/Test Set/msk',
                        help='the path of the the testset mask', dest='testmaskpth')

    # 加载模型
    # checkpoints_seg_hrnet_BS_4_EPOCHS_100_time_2023-12-13_09_37_58
    # checkpointsseg_se_hrnet_BS_4_EPOCHS_100_time_2023-12-12_17_21_01
    # checkpoints_seg_cbam_hrnet_BS_4_EPOCHS_100_time_2023-12-12_17_58_28
    # checkpoints_seg_scse_hrnet_BS_4_EPOCHS_100_time_2023-12-13_03_45_42
    # checkpoints_seg_coord_hrnet_BS_4_EPOCHS_100_time_2023-12-13_04_47_04
    # checkpoints_seg_trip_hrnet_BS_4_EPOCHS_100_time_2023-12-13_07_32_17
    # checkpoints_seg_sp_hrnet_BS_4_EPOCHS_100_time_2023-12-13_08_47_41
    # checkpoints_seg_se_gc_hrnet_BS_4_EPOCHS_100_time_2023-12-14_07_25_12
    # checkpoints_seg_newsp_hrnet_BS_4_EPOCHS_100_time_2023-12-14_11_24_27


    parser.add_argument('-f', '--load', dest='load', type=str, 
                        default=r'../checkpoints_seg_se_gc_hrnet_BS_4_EPOCHS_100_time_2023-12-14_07_25_12/best_model.pth',
                        help='Load model from a .pth file')
    # log
    parser.add_argument('-log', '--log', type=str, default=r'./evaluate_log',
                        help='Create a trainging log file', dest='log')
 
    args = parser.parse_args()

    return args




if __name__ == '__main__':

    # 调用所有的参数
    args = get_args()

    # Setup logger
    if not os.path.exists(args.log):
        os.makedirs(args.log)
    logger = get_logger(args.log)
    logger.info('Start Evaluating TestSet!')


    # Setup Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    # faster convolutions, but more memory
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    logger.info("Using device:{}".format(device))


    # 加载预训练模型
    # cfg_file = "seg_hrnet.yaml"
    # cfg_file = "seg_se_hrnet.yaml"
    # cfg_file = "seg_cbam_hrnet.yaml"
    # cfg_file = "seg_scse_hrnet.yaml"
    # cfg_file = "seg_coord_hrnet.yaml"
    # cfg_file = "seg_trip_hrnet.yaml"
    # cfg_file = "seg_sp_hrnet.yaml"
    cfg_file = "seg_se_gc_hrnet.yaml"
    # cfg_file = "seg_newsp_hrnet.yaml"

    cfg = update_config(default_config, r"../config/" + cfg_file)
    model = HighResolutionNet(cfg).to(device)


    logger.info(f'Network:\n'
                 f'\t{model.in_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n')
    
    if args.load:
        model.load_state_dict(torch.load(args.load, map_location=device)['model_state'])
        logger.info(f' Pretrained model loaded from file:{args.load}')


    # 增加一个计算时间的txt功能的实现
    start_time = datetime.datetime.now()

    testset_evaluate(args.testimgpth, args.testmaskpth, device,flip=False, with_edge=False)

    end_time = datetime.datetime.now()
    logger.info("The evaluation is over!")
    log_time = "测试总时间: " + str((end_time - start_time).seconds / 60) + "m"
    logger.info(log_time)








