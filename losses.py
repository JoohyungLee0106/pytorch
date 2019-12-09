import torch
import torch.nn.functional as F

def focal_loss(logits, target, alpha=1.0, gamma=4.0):
    '''

    :param logits: 1st dimension shall be the channel
    :param target: 1st dimension shall be the channel
    :return:
    '''

    loss_BCE = F.binary_cross_entropy_with_logits(logits, target,
                                                          reduction='none')
    loss_focal = torch.mean(((1 - torch.exp(-loss_BCE)) ** gamma) * loss_BCE * alpha, dim=list(range(1, len(target.size()))))

    return loss_focal

def dice_loss(logits, target):
    '''

    :param logits: 1st dimension shall be the channel
    :param target: 1st dimension shall be the channel
    :return:
    '''
    list_without_zero = list(range(1, len(target.size())))
    preds = torch.sigmoid(logits)
    numerator = 2*torch.sum(preds*target, dim=list_without_zero)+0.0000001
    denominator = torch.sum(preds+target, dim=list_without_zero)+0.0000001

    return 1-numerator/denominator

def dice_focal_loss(logits, target, alpha=1.0, gamma=4.0):
    '''

    :param logits: 1st dimension shall be the channel
    :param target: 1st dimension shall be the channel
    :return:
    '''

    list_without_zero = list(range(1, len(target.size())))
    preds = torch.sigmoid(logits)
    numerator = 2 * torch.sum(preds * target, dim=list_without_zero) + 0.0000001
    denominator = torch.sum(preds + target, dim=list_without_zero) + 0.0000001
    loss_dice = 1-numerator/denominator


    loss_BCE = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    loss_focal = torch.mean(((1 - torch.exp(-loss_BCE)) ** gamma) * loss_BCE * alpha, dim=list_without_zero)

    return loss_dice + loss_focal