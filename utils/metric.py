# -*- coding: utf-8 -*-
# @Company: Aerospace Information  Research Institute, Chinese Academy of Sciences
# @Time :  
# @Author : Xiaoping,Zhang
# @File : .py


# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np


"""
混淆矩阵
P\L     P    N
P      TP    FP
N      FN    TN
"""

class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask].astype(int), minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        # overall accuracy
        acc = np.diag(hist).sum() / hist.sum()
        # class accuracy
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        # mean_acc
        mean_acc = np.nanmean(acc_cls)
        # class Recall
        recall_cls = np.diag(hist)/hist.sum(axis = 0)
        # F1:
        f1_score = 2*acc_cls*recall_cls/(acc_cls+recall_cls)
        # iou
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        # mean_iou
        mean_iu = np.nanmean(iu)
        # FWIOU
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        # every class iou
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Overall_Acc": float("{:.4f}".format(acc*100)),
                "Precsion": float("{:.3f}".format(acc_cls[1]*100)),
                "Recall":float("{:.3f}".format(recall_cls[1]*100)),
                "F1_Score":float("{:.3f}".format(f1_score[1]*100)),
                "P_IoU": float("{:.3f}".format(iu[1]*100)),
                "Mean_Precsion":float("{:.3f}".format(mean_acc*100)),
                "Mean_IoU": float("{:.3f}".format(mean_iu*100)),
                "FWIoU":float("{:.3f}".format(fwavacc*100))
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




if __name__ == '__main__':
    metric = runningScore(2)
    print(metric.confusion_matrix)