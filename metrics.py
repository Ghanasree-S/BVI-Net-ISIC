"""Evaluation metrics matching Table V of the paper: Dice, mIoU, Accuracy,
Specificity, Sensitivity, and ASSD (Average Symmetric Surface Distance).
"""
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, binary_erosion


def _binarize(pred, thresh=0.5):
    """Thresholds a soft prediction into a hard 0/1 mask.

    Parameters:
        pred (torch.Tensor): predicted probabilities, any shape, values in [0, 1].
        thresh (float, default=0.5): decision boundary -- pixels with
            probability above this are classified as "lesion". This is a
            hyperparameter of evaluation, not training; raising it trades
            sensitivity for specificity.

    Returns:
        torch.Tensor: same shape as pred, values in {0.0, 1.0}.
    """
    return (pred > thresh).float()


def dice_score(pred, target, eps=1e-6):
    """Computes the Dice coefficient (2x overlap / total area) between a
    predicted and ground-truth mask.

    Parameters:
        pred (torch.Tensor): predicted probabilities (binarized internally).
        target (torch.Tensor): ground-truth binary mask, same shape as pred.
        eps (float, default=1e-6): smoothing constant to avoid division by
            zero on empty masks.

    Returns:
        float: Dice score in [0, 1], higher is better.
    """
    pred, target = _binarize(pred), target
    inter = (pred * target).sum()
    return ((2 * inter + eps) / (pred.sum() + target.sum() + eps)).item()


def miou(pred, target, eps=1e-6):
    """Computes mean Intersection-over-Union between predicted and ground-truth masks.

    Parameters:
        pred (torch.Tensor): predicted probabilities (binarized internally).
        target (torch.Tensor): ground-truth binary mask, same shape as pred.
        eps (float, default=1e-6): smoothing constant to avoid division by
            zero on empty masks.

    Returns:
        float: IoU in [0, 1], higher is better.
    """
    pred, target = _binarize(pred), target
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return ((inter + eps) / (union + eps)).item()


def confusion_counts(pred, target):
    """Computes the four confusion-matrix counts for a binarized prediction.

    Parameters:
        pred (torch.Tensor): predicted probabilities (binarized internally).
        target (torch.Tensor): ground-truth binary mask, same shape as pred.

    Returns:
        tuple[int, int, int, int]: (true_positive, true_negative,
        false_positive, false_negative) pixel counts.
    """
    pred, target = _binarize(pred), target
    tp = ((pred == 1) & (target == 1)).sum().item()
    tn = ((pred == 0) & (target == 0)).sum().item()
    fp = ((pred == 1) & (target == 0)).sum().item()
    fn = ((pred == 0) & (target == 1)).sum().item()
    return tp, tn, fp, fn


def accuracy(pred, target):
    """Computes pixel-wise classification accuracy.

    Parameters:
        pred (torch.Tensor): predicted probabilities.
        target (torch.Tensor): ground-truth binary mask.

    Returns:
        float: (TP + TN) / total pixels, in [0, 1].
    """
    tp, tn, fp, fn = confusion_counts(pred, target)
    return (tp + tn) / max(tp + tn + fp + fn, 1)


def specificity(pred, target):
    """Computes specificity (true negative rate): how well background pixels
    are correctly identified as non-lesion.

    Parameters:
        pred (torch.Tensor): predicted probabilities.
        target (torch.Tensor): ground-truth binary mask.

    Returns:
        float: TN / (TN + FP), in [0, 1].
    """
    tp, tn, fp, fn = confusion_counts(pred, target)
    return tn / max(tn + fp, 1)


def sensitivity(pred, target):
    """Computes sensitivity (true positive rate / recall): how well lesion
    pixels are correctly identified.

    Parameters:
        pred (torch.Tensor): predicted probabilities.
        target (torch.Tensor): ground-truth binary mask.

    Returns:
        float: TP / (TP + FN), in [0, 1].
    """
    tp, tn, fp, fn = confusion_counts(pred, target)
    return tp / max(tp + fn, 1)


def assd(pred, target):
    """Computes the Average Symmetric Surface Distance between predicted and
    ground-truth mask boundaries, on a single 2D mask pair.

    Parameters:
        pred (torch.Tensor): predicted probabilities for one sample,
            squeezable to a 2D mask (binarized internally).
        target (torch.Tensor): ground-truth binary mask for the same sample.

    Returns:
        float: mean boundary-to-boundary distance in pixels (lower is
        better); NaN if either mask is empty.
    """
    p = _binarize(pred).squeeze().cpu().numpy().astype(bool)
    t = target.squeeze().cpu().numpy().astype(bool)
    if p.sum() == 0 or t.sum() == 0:
        return float("nan")

    def surface(mask):
        eroded = binary_erosion(mask)
        return mask ^ eroded

    p_surf, t_surf = surface(p), surface(t)
    dt_t = distance_transform_edt(~t_surf)
    dt_p = distance_transform_edt(~p_surf)
    d1 = dt_t[p_surf]
    d2 = dt_p[t_surf]
    if len(d1) == 0 or len(d2) == 0:
        return float("nan")
    return float(np.concatenate([d1, d2]).mean())


def all_metrics(pred, target):
    """Convenience wrapper computing every metric for one prediction/target pair.

    Parameters:
        pred (torch.Tensor): predicted probabilities for one sample.
        target (torch.Tensor): ground-truth binary mask for the same sample.

    Returns:
        dict[str, float]: keys "dice", "miou", "accuracy", "specificity",
        "sensitivity", "assd".
    """
    return {
        "dice": dice_score(pred, target),
        "miou": miou(pred, target),
        "accuracy": accuracy(pred, target),
        "specificity": specificity(pred, target),
        "sensitivity": sensitivity(pred, target),
        "assd": assd(pred, target),
    }
