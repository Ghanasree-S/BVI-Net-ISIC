"""Combined BCE + Dice loss, weighted equally (lambda1 = lambda2 = 1), per Eq. (9)."""
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.flatten(1)
        target = target.flatten(1)
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, lambda1=1.0, lambda2=1.0):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def forward(self, pred, target):
        return self.lambda1 * self.bce(pred, target) + self.lambda2 * self.dice(pred, target)
