# _*_ coding: utf-8 _*_
# Author: Jielong
# @Time: 27/08/2019 15:29
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, targets, logits):
        batch_size = targets.size(0)
        logits_prob = torch.sigmoid(logits)
        intersection = logits_prob.view(batch_size, -1) * targets.view(batch_size, -1)
        dice_score = 2. * (intersection.sum(1) + self.epsilon) / (logits_prob.view(batch_size, -1).sum(1) +
                                                                  targets.view(batch_size, -1).sum(1) +
                                                                  self.epsilon)
        dice_score = 1 - dice_score.sum() / batch_size
        return dice_score


if __name__ == "__main__":
    y_true = torch.randn(2, 3, 4, 3, 3)
    y_pred = torch.randn(2, 3, 4, 3, 3)
    dl = DiceLoss()
    print(dl(y_true, y_pred).item())
