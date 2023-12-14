# -*- coding: utf-8 -*-
# @Company: Aerospace Information  Research Institute, Chinese Academy of Sciences
# @Time :
# @Author : Xiaoping,Zhang
# @File : trainv4.py


import os,sys
import argparse
import torchvision
import datetime

import time
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from utils.utils import get_logger
from utils.metric import runningScore,averageMeter


from torch.utils.tensorboard import SummaryWriter


from dataset.datasets import Mydatasets
from torch.utils.data import DataLoader

from config import default_config
from config import update_config
from Module.seg_hrnet import HighResolutionNet




os.environ["CUDA_VISIBLE_DEVICES"] = '0'


# def setdir(fl_pth):
#     # 构建存放log的文件夹
#     # 如果log文件夹存在的话，则删除该文件夹，并重新构建新文件夹
#     if not os.path.exists(fl_pth):
#         os.mkdir(fl_pth)
#     else:
#         shutil.rmtree(fl_pth)
#         os.mkdir(fl_pth)

#
# def accuracy(logits, target):
#     # logits: (n, c, h, w), target: (n, h, w)   n代表了batch_size,c代表了波段数
#     n, c, h, w = logits.size()
#     if c == 2:
#         output = torch.argmax(logits, dim=1)
#     else:
#     # 需要判断一下输出维度为1时，代码是否正确
#         assert c == 1
#         probs = torch.sigmoid(logits)
#         output = (probs>0.5).float()
#         output = torch.squeeze(output)
#         target = torch.squeeze(target)
#
#     right_number = torch.sum(output == target).float()
#     right_number = right_number.cpu()
#     return float(right_number/(n*h*w))

# Validation

def validation(model, criterion, val_dataloader, device):
    running_metrics_val = runningScore(2)
    val_loss_meter = averageMeter()

    model.eval()
    with torch.no_grad():
        val_loader =  tqdm(val_dataloader ,desc='Validation')
        for i_val, (valid_img, valid_mask) in enumerate(val_loader):

            valid_img = valid_img.to(device, dtype=torch.float32)
            if model.n_classes > 1:
                valid_mask = valid_mask.to(device, dtype=torch.long)
            else:
                valid_mask = valid_mask.to(device, dtype=torch.float32)

            valid_pred = model(valid_img)
            val_loss = criterion(valid_pred, valid_mask)

            if model.n_classes > 1:
                pred = valid_pred.data.max(1)[1].cpu().numpy()
                gt = valid_mask.data.cpu().numpy()
            else:
                pred = (torch.sigmoid(valid_pred) > 0.5).float()
                pred = pred.squeeze().cpu().numpy()
                gt = valid_mask.squeeze().cpu().numpy()

            running_metrics_val.update(gt, pred)
            val_loss_meter.update(val_loss.item())

            val_loader.set_postfix({"Validation Loss":'{:.4f}'.format(val_loss_meter.avg),
                                   "Mean_IoU":'{:.4f}'.format(running_metrics_val.get_scores()[0]['Mean_IoU'])})


    return running_metrics_val, val_loss_meter


def train(args,
          img_path, mask_path, val_img_path, val_mask_path,
          model, epochs, batch_size, lr, dir_checkpoint, logger,
          resume='',save_cp=True):

    # Setup seeds
    # torch.manual_seed(cfg.get("seed", 1337))
    # torch.cuda.manual_seed(cfg.get("seed", 1337))
    # np.random.seed(cfg.get("seed", 1337))
    # random.seed(cfg.get("seed", 1337))

    # Setup Writer,visualize the precession
    writer = SummaryWriter(comment=f'_{args.cfg_file.split(".")[0]}_StepLR_{lr}_BS_{batch_size}_EPOCHS_{epochs}')

    # Setup Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # faster convolutions, but more memory
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    logger.info("Using device:{}".format(device))

    # Setup DataLoader
    train_transform = torchvision.transforms.ToTensor()
    label_transform = torchvision.transforms.ToTensor()

    # Setup datasets
    train_datasets = Mydatasets(img_path, mask_path, train_transform, label_transform,aug=True)
    val_datasets = Mydatasets(val_img_path, val_mask_path, train_transform, label_transform, aug=False)

    # Loader
    train_dataloader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=4,
                                  pin_memory=True,
                                  drop_last=True
                                  )
    val_dataloader = DataLoader(val_datasets,batch_size = batch_size,shuffle = False,
                                pin_memory=True,
                                drop_last=True
                                )
    logger.info("Train Datasets and Validation Datasets has been created!")


    # Set checkpoints
    if save_cp:
        dir_checkpoint += f'_{args.cfg_file.split(".")[0]}_BS_{batch_size}_EPOCHS_{epochs}_time_' \
                          f'{str(datetime.datetime.now()).split(".")[0].replace(" ", "_").replace(":", "_")}'
        os.mkdir(dir_checkpoint)
        logger.info('Created checkpoint directory!')

    # Setup Model
    model = model.to(device)

    # Setup optimizer, lr_scheduler and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # optimizer = optim.Adam(model.parameters(),lr=lr, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)

    # loss Function
    if model.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    elif model.n_classes == 1:
        # 将来可以将Dice_Loss+BCELoss合成一步
        # 目前是sigmoid函数和BCEloss合成为了一步
        criterion = nn.BCEWithLogitsLoss()
    else:
        pass

    logger.info(f'''Starting Training:
           Epochs:          {epochs}
           Batch size:      {batch_size}
           Learning rate:   {lr}
           Training size:   {len(train_datasets)}
           Validation size: {len(val_datasets)}
           Checkpoints:     {save_cp}
           Device:          {device.type}
           Optimizer:       {optimizer}
           LossFunction:    {criterion}
       ''')

    # starting params
    best_iou = -100.0
    start_epoch = 0
    globe_step = 0

    # optionally resume from a checkpoint
    if bool(resume) is not False:
        if os.path.isfile(resume):
            logger.info("=>Loading model and optimizer from checkpoint '{}'".format(resume))

            checkpoint = torch.load(resume)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_epoch = checkpoint["epoch"]
            logger.info(
                "Loaded checkpoint '{}' (iter"
                " {})".format(
                    resume, checkpoint["epoch"]
                )
            )
        else:
            logger.info("No checkpoint found at '{}'".format(resume))
    else:
        logger.info("There is no resume")

    for epoch in range(start_epoch,epochs):
        print("Epoch {}/{}".format(epoch, epochs-1))
        print("-" * 100)

        # logging accuracy and loss
        batch_time = averageMeter()

        # running training and valition
        # iterate over data

        model.train()
        for i_iter,(train_img,train_mask) in enumerate(train_dataloader):

            # get the start time
            start_t = time.time()

            # get the inputs and wrap in Variable
            train_img = train_img.to(device,dtype = torch.float32)
            if model.n_classes > 1:
                train_mask = train_mask.to(device, dtype=torch.long)  # [4,512,512]
            else:
                train_mask = train_mask.to(device, dtype=torch.float32)

            assert train_img.shape[1] == model.in_channels, \
            f'modelwork has been defined with {model.in_channels} input channels, ' \
            f'but loaded images have {train_img.shape[1]} channels. Please check that ' \
            'the images are loaded incorrectly.'

            # forward
            train_pred = model(train_img)
            # loss
            loss = criterion(train_pred, train_mask)
            # zero the parameter gradients
            optimizer.zero_grad()
            # backward
            loss.backward()
            optimizer.step()

            batch_time.update(time.time()-start_t)

            # 先把loss写出来
            if (globe_step + 1) % 4 == 0:
                train_msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.4f}' .format(
                      epoch, epochs-1, i_iter + 1, len(train_datasets)//batch_size,
                      batch_time.avg, [x['lr'] for x in optimizer.param_groups], loss.item())

                logger.info(train_msg)
                writer.add_scalar("loss/train_loss", loss.item(), globe_step + 1)

            # Validatiuon
            if (globe_step + 1) % 420 == 0:
                running_metrics_val, val_loss_meter = validation(model, criterion, val_dataloader, device)

                writer.add_scalar("loss/val_loss", val_loss_meter.avg, globe_step + 1)

                val_msg = "Validation Loss: \t{:.4f}".format(val_loss_meter.avg)
                logger.info(val_msg)

                score, class_iou = running_metrics_val.get_scores()
                for k, v in score.items():
                    logger.info("{}: {}".format(k, v))
                    # writer.add_scalar("val_metrics/{}".format(k), v, globe_step + 1)
                    writer.add_scalar("val_metrics/{}".format(k), v, epoch)

                for k, v in class_iou.items():
                    logger.info("{}: {}".format(k, v))
                    # writer.add_scalar("val_metrics/cls_{}".format(k), v, globe_step + 1)
                    writer.add_scalar("val_metrics/cls_{}".format(k), v, epoch)

                if score["Mean_IoU"] >= best_iou:
                    best_iou = score["Mean_IoU"]
                    state = {
                        "epoch": epoch + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_iou": best_iou,
                    }
                    torch.save(state, 'best_model.pth')

                    logger.info("Saved best model!")

            globe_step +=1


        # Update learning rate
        scheduler.step()
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch+1)

        if (epoch+1) % 40 == 0:
            torch.save(model.state_dict(), os.path.join(dir_checkpoint, f'CP_epoch{epoch + 1}.pth'))
            logger.info(f'Checkpoint {epoch + 1} saved !')

        torch.cuda.empty_cache()

    writer.close()





def  get_args():

    # 创建一个参数对象
    parser = argparse.ArgumentParser(description='Train the model on images and masks')

    # 定义数据集的路径
    parser.add_argument('-img_path', '--train-img-path', type=str,
                        default=r'/data/sgData/Train Set/img',
                        help='the path of the train img', dest='img_path')
    parser.add_argument('-mask_path', '--train-mask-pth', type=str,
                        default=r'/data/sgData/Train Set/msk',
                        help='the path of the mask maps', dest='mask_path')

    parser.add_argument('-valimgpth', '--validation-image-pth', type=str,
                        default=r'/data/sgData/Val Set/img',
                        help='the path of the validation img', dest='valimgpath')
    parser.add_argument('-valmaskpth', '--validation-mask-pth', type=str,
                        default=r'/data/sgData/Val Set/msk',
                        help='the path of the the validation mask', dest='valmaskpth')

    # 训练超参数
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=100,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=4 ,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='The initial learning rate', dest='lr')
    parser.add_argument('-d','--device',default='cuda',help='gpu or cpu',dest='device')

    # 模型参数
    # 读取配置文件
    parser.add_argument('-cfg_file', '--cfg-file', type=str, default=r'seg_se_hrnet.yaml',
                        help='the name of config file', dest='cfg_file')

    # log and checkpoint
    parser.add_argument('-log', '--log', type=str, default=r'./log',
                        help='Create a trainging log file', dest='log')
    parser.add_argument('-resume', '--resume', type = str, default='', dest='resume' )
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-dir', '--dir-checkpoints',type=str, default=r'./checkpoints',
                        help='the path of the mode weight file',dest = 'checkpoints')

    parser.add_argument('-losstype', '--loss-type',type=str, default=r'bceloss',
                        help='the type of loss ',dest = 'losstype')

    args = parser.parse_args()

    return args




if __name__ == '__main__':

    # 调用所有的参数
    args = get_args()

    # Setup logger
    if not os.path.exists(args.log):
        os.makedirs(args.log)

    logger = get_logger(args.log)
    logger.info('Start Beginning!')

    try:
        if  args.losstype =='crossentropy':
            n_classes = 2
        elif  args.losstype == 'bceloss':
            n_classes = 1
    except:
        logger.warning('No such Losstype!')



    cfg = update_config(default_config, r"./config/" + args.cfg_file)
    model = HighResolutionNet(cfg)

    logger.info(f'Network:\n'
                 f'\t{model.in_channels} input channels\n'
                 f'\t{args.losstype} loss function\n')

    # 加载预训练模型
    if args.load:
        model.load_state_dict(
            torch.load(args.load)
        )
        logger.info(f' Pretrained model loaded from file:{args.load}')


    # 增加一个计算时间的txt功能的实现
    start_time = datetime.datetime.now()
    try:
        train(args,
              img_path=args.img_path,
              mask_path=args.mask_path,
              val_img_path=args.valimgpath,
              val_mask_path=args.valmaskpth,
              model=model,
              epochs=args.epochs,
              batch_size=args.batchsize,
              lr=args.lr,
              dir_checkpoint=args.checkpoints,
              logger=logger
              )
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'INTERRUPTED.pth')
        logger.info(' KeyboardInterrupt:Saved interrupt!!!')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

    end_time = datetime.datetime.now()
    logger.info("The training is over!")
    log_time = "训练总时间: " + str((end_time - start_time).seconds / 60) + "m"
    logger.info(log_time)
    with open('Training Time.txt', 'w') as f:
        f.write(log_time)














