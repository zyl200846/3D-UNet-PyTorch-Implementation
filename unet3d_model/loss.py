# _*_ coding: utf-8 _*_
# Author: Jielong
# @Time: 27/08/2019 15:29
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(DiceLoss, self).__init__()
        # smooth factor
        self.epsilon = epsilon

    def forward(self, targets, logits):
        batch_size = targets.size(0)
        # log_prob = torch.sigmoid(logits)
        logits = logits.view(batch_size, -1).type(torch.FloatTensor)
        targets = targets.view(batch_size, -1).type(torch.FloatTensor)
        intersection = (logits * targets).sum(-1)
        dice_score = 2. * intersection / ((logits + targets).sum(-1) + self.epsilon)
        # dice_score = 1 - dice_score.sum() / batch_size
        return torch.mean(1. - dice_score)


class AsymmetricLoss(nn.Module):
    def __init__(self, beta):
        super(AsymmetricLoss, self).__init__()
        # hyper-parameter for balancing precision and recall
        self.beta = beta

    def forward(self, targets, logits):
        pass


if __name__ == "__main__":
    import numpy as np
    yt = np.random.random(size=(2, 1, 3, 3, 3))
    # print(yt)
    yt = torch.from_numpy(yt)
    yp = np.zeros(shape=(2, 1, 3, 3, 3))
    yp = yp + 1
    yp = torch.from_numpy(yp)
    # print(yp)
    dl = DiceLoss()
    print(dl(yp, yt).item())
