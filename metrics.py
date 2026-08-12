"""Evaluation metrics matching Table V of the paper: Dice, mIoU, Accuracy,
Specificity, Sensitivity, and ASSD (Average Symmetric Surface Distance).
"""
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, binary_erosion


def _binarize(pred, thresh=0.5):
    return (pred > thresh).float()


def dice_score(pred, target, eps=1e-6):
    pred, target = _binarize(pred), target
    inter = (pred * target).sum()
    return ((2 * inter + eps) / (pred.sum() + target.sum() + eps)).item()


def miou(pred, target, eps=1e-6):
    pred, target = _binarize(pred), target
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return ((inter + eps) / (union + eps)).item()


def confusion_counts(pred, target):
    pred, target = _binarize(pred), target
    tp = ((pred == 1) & (target == 1)).sum().item()
    tn = ((pred == 0) & (target == 0)).sum().item()
    fp = ((pred == 1) & (target == 0)).sum().item()
    fn = ((pred == 0) & (target == 1)).sum().item()
    return tp, tn, fp, fn


def accuracy(pred, target):
    tp, tn, fp, fn = confusion_counts(pred, target)
    return (tp + tn) / max(tp + tn + fp + fn, 1)


def specificity(pred, target):
    tp, tn, fp, fn = confusion_counts(pred, target)
    return tn / max(tn + fp, 1)


def sensitivity(pred, target):
    tp, tn, fp, fn = confusion_counts(pred, target)
    return tp / max(tp + fn, 1)


def assd(pred, target):
    """Average symmetric surface distance, computed on a single 2D mask pair."""
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
    return {
        "dice": dice_score(pred, target),
        "miou": miou(pred, target),
        "accuracy": accuracy(pred, target),
        "specificity": specificity(pred, target),
        "sensitivity": sensitivity(pred, target),
        "assd": assd(pred, target),
    }
