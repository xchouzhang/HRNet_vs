# -*- coding: utf-8 -*-
# @Company: Aerospace Information  Research Institute, Chinese Academy of Sciences
# @Time :  
# @Author : Xiaoping,Zhang
# @File : .py



import torch
from tqdm import tqdm
from utils.metric import runningScore


def flip_inference(model, image, flip=False):
    batch, _, ori_height, ori_width = image.size()
    assert batch == 1, "only supporting batchsize 1."
    # image = image.numpy()[0].transpose((1, 2, 0)).copy()     #(512,512,4)
    #
    # final_pred = torch.zeros([1, self.num_classes,
    #                           ori_height, ori_width]).cuda()

    seg_pred, edge_pred = model(image)

    if flip:
        # 水平翻转
        hflip_img = image.cpu().numpy()[:, :, :, ::-1]      # size:[1,4,512,512]
        seg_hflip_output, edge_hflip_output = model(torch.from_numpy(hflip_img.copy()).cuda())

        seg_hflip_pred = seg_hflip_output.data.cpu().numpy().copy()
        edge_hflip_pred = edge_hflip_output.data.cpu().numpy().copy()

        seg_hflip_pred = torch.from_numpy(seg_hflip_pred[:, :, :, ::-1].copy()).cuda()
        edge_hflip_pred = torch.from_numpy(edge_hflip_pred[:, :, :, ::-1].copy()).cuda()

        seg_pred += seg_hflip_pred
        edge_pred += edge_hflip_pred


        # 垂直翻转
        vflip_img = image.cpu().numpy()[:, :, ::-1, :]
        seg_vflip_output, edge_vflip_output = model(torch.from_numpy(vflip_img.copy()).cuda())


        seg_vflip_pred = seg_vflip_output.data.cpu().numpy().copy()
        edge_vflip_pred = edge_vflip_output.data.cpu().numpy().copy()

        seg_vflip_pred = torch.from_numpy(seg_vflip_pred[:, :, ::-1, :].copy()).cuda()
        edge_vflip_pred = torch.from_numpy(edge_vflip_pred[:, :, ::-1, :].copy()).cuda()

        seg_pred += seg_vflip_pred
        edge_pred += edge_vflip_pred


        seg_pred = seg_pred / 3
        edge_pred = edge_pred / 3

    return seg_pred, edge_pred



def evaluate(test_loader, model, device):

    test_running_metrics = runningScore(2)
    test_edge_running_metrics = runningScore(2)

    model.eval()
    with torch.no_grad():
        test_loader = tqdm(test_loader, desc='Test Set')
        for index, (test_image, test_mask, test_edge) in enumerate(test_loader):
            test_image = test_image.to(device)
            test_mask = test_mask.to(device)
            test_edge = test_edge.to(device)

            test_mask_pred, test_edge_pred = flip_inference(model, test_image, flip=True)

            mask_pred = (torch.sigmoid(test_mask_pred) > 0.5).float()
            mask_pred = mask_pred.squeeze().cpu().numpy()
            mask_gt = test_mask.squeeze().cpu().numpy()

            edge_pred = (torch.sigmoid(test_edge_pred) > 0.5).float()
            edge_pred = edge_pred.squeeze().cpu().numpy()
            edge_gt = test_edge.squeeze().cpu().numpy()



            test_running_metrics.update(mask_gt, mask_pred)
            test_edge_running_metrics.update(edge_gt, edge_pred)

            test_loader.set_postfix({"Mask_MIoU": '{:.4f}'.format(test_running_metrics.get_scores()[0]['Mean_IoU']),
                                    "Edge_MIoU": '{:.4f}'.format(test_edge_running_metrics.get_scores()[0]['Mean_IoU'])})


    return test_running_metrics, test_edge_running_metrics






























