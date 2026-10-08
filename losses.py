"""Combined BCE + Dice loss, weighted equally (lambda1 = lambda2 = 1), per Eq. (9)."""
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, per_sample=True):
        """Builds the Dice loss.

        Hyperparameters:
            smooth (float, default=1e-6): small constant added to numerator
                and denominator to avoid division-by-zero when a mask is
                entirely empty (no lesion pixels).
            per_sample (bool, default=True): True = Dice per sample and
                channel (skin). False = Dice per channel pooled over the
                whole batch (liver/brain): with per-sample Dice, every slice
                whose tumor/ET channel is empty is only satisfied by an
                exactly-zero prediction, which collapsed the small classes to
                "predict nothing" (tumor sensitivity 0.0 on LiTS).
        """
        super().__init__()
        self.smooth = smooth
        self.per_sample = per_sample

    def forward(self, pred, target):
        """Computes 1 - Dice coefficient between predicted and target masks.

        Parameters:
            pred (torch.Tensor): predicted probabilities, shape (B, 1, H, W),
                values in [0, 1] (post-sigmoid).
            target (torch.Tensor): ground-truth binary mask, same shape.

        Returns:
            torch.Tensor (scalar): mean Dice loss over the batch (0 = perfect
            overlap, 1 = no overlap).
        """
        # Dice is computed per sample AND per output channel, then averaged, so
        # small classes (liver TUMOR, brain ET) weigh as much as large ones
        # (liver, WT) instead of being swamped by them. For the single-channel
        # skin model this is identical to a plain per-sample Dice.
        if self.per_sample:
            pred = pred.flatten(2)      # (B, C, H*W)
            target = target.flatten(2)
        else:
            pred = pred.transpose(0, 1).flatten(1).unsqueeze(0)  # (1, C, B*H*W)
            target = target.transpose(0, 1).flatten(1).unsqueeze(0)
        intersection = (pred * target).sum(dim=2)
        union = pred.sum(dim=2) + target.sum(dim=2)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, lambda1=1.0, lambda2=1.0, per_sample_dice=True):
        """Builds the combined loss (Eq. 9 of the paper: Loss = lambda1*BCE + lambda2*Dice).

        Hyperparameters:
            lambda1 (float, default=1.0): weight on the BCE (pixel-wise
                classification) term. The paper sets lambda1 = lambda2 = 1.
            lambda2 (float, default=1.0): weight on the Dice (region overlap)
                term. Increasing this relative to lambda1 pushes the model to
                prioritize overall mask shape over per-pixel confidence.
        """
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss(per_sample=per_sample_dice)
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def forward(self, pred, target):
        """Computes the weighted sum of BCE and Dice losses.

        Parameters:
            pred (torch.Tensor): predicted probabilities, shape (B, 1, H, W).
            target (torch.Tensor): ground-truth binary mask, same shape.

        Returns:
            torch.Tensor (scalar): combined training loss.
        """
        return self.lambda1 * self.bce(pred, target) + self.lambda2 * self.dice(pred, target)
