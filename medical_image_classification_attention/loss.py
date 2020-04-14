import torch
import torch.nn as nn
from torchvision import transforms, utils
import torch.nn.functional as F

# https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/


class FocalLoss(nn.Module):
    def __init__(self, gamma, if_custom_mean=False, if_logit=False):
        super(FocalLoss, self).__init__()
        self.gamma=gamma
        self.division_factor = 1.0
        if if_logit:
            self.forward = self.forward_logit
        else:
            if if_custom_mean:
                self.forward = self.forward_custom_mean
            else:
                self.forward = self.forward_normal

    def set_division_factor(self, x):
        self.division_factor = float(x)

    def forward_logit(self, logit, label):
        logit = logit.view(logit.size(0), -1)  # N,C,H,W => N, H*W (C=1)
        label = label.view(label.size(0), -1)  # N,H,W => N, H*W

        BCE = F.binary_cross_entropy_with_logits(logit, label, reduction='none')
        loss = ((1 - torch.exp(-BCE)) ** self.gamma) * BCE
        return loss.mean(dim=1, keepdim=False)

    def forward_normal(self, pred, label):
        '''
        :param logit: NCHW or NCDHW, C=1
        :param label: NHW or NDHW
        :return:
        '''
        pred = pred.view(pred.size(0), -1)  # N,C,H,W => N, H*W (C=1)
        label = label.view(label.size(0), -1)  # N,H,W => N, H*W

        BCE = F.binary_cross_entropy(pred, label, reduction='none')
        loss= ((1 - torch.exp(-BCE)) ** self.gamma) * BCE

        return loss.mean(dim=1, keepdim=False)
        # return loss.sum(dim=1, keepdim=False)

    def forward_custom_mean(self, pred, label):
        '''
        :param logit: NHW or NDHW, C=1
        :param label: NHW or NDHW
        :return:
        '''

        pred = pred.view(pred.size(0), -1)  # N,C,H,W => N, H*W (C=1)
        label = label.view(label.size(0), -1)  # N,H,W => N, H*W

        BCE = F.binary_cross_entropy(pred, label, reduction='none')
        loss= ((1 - torch.exp(-BCE)) ** self.gamma) * BCE

        # return loss.mean(dim=1, keepdim=False)
        return (loss.sum(dim=1, keepdim=False)/self.division_factor)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, label):
        '''

        :param pred: NHW or NDHW
        :param label: NHW or NDHW
        :return:
        '''
        pred = pred.view(pred.size(0), -1)  # NHW or NDHW => N * ~
        label = label.view(label.size(0), -1)  # NHW or NDHW => N * ~

        numerator = 2.0 * torch.sum( torch.mul(pred, label), dim=1, keepdim=False)
        denominator = torch.sum(pred, dim=1, keepdim=False) + torch.sum(label, dim=1, keepdim=False)

        return -torch.div(numerator, denominator)

