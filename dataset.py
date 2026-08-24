"""PyTorch Dataset for the prepared ISIC2018-a split."""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class ISICDataset(Dataset):
    def __init__(self, root, split="train", augment=None):
        """Indexes the image/mask pairs for one split of the prepared dataset.

        Parameters:
            root (str): path to the prepared dataset folder (output of
                data/prepare_isic_a.py), containing train/val/test
                subfolders each with images/ and masks/.
            split (str, default="train"): which subfolder to load --
                "train", "val", or "test".
            augment (callable or None, default=None): an albumentations
                Compose object (see transforms.py) applied only when set --
                pass None for val/test splits to keep evaluation unbiased.
        """
        self.img_dir = Path(root) / split / "images"
        self.mask_dir = Path(root) / split / "masks"
        self.files = sorted(p.name for p in self.img_dir.glob("*.png"))
        self.augment = augment

    def __len__(self):
        """Returns the number of image/mask pairs in this split."""
        return len(self.files)

    def __getitem__(self, idx):
        """Loads, normalizes, optionally augments, and tensor-izes one
        image/mask pair.

        Parameters:
            idx (int): index into the file list.

        Returns:
            tuple:
                img (torch.Tensor): shape (3, H, W), float32, values in [0, 1].
                mask (torch.Tensor): shape (1, H, W), float32, binary (0 or 1).
        """
        name = self.files[idx]
        img = cv2.imread(str(self.img_dir / name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mask = cv2.imread(str(self.mask_dir / name), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        if self.augment is not None:
            augmented = self.augment(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]

        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        return img, mask


class LiTSDataset(Dataset):
    """PyTorch Dataset for the prepared LiTS17 split (output of
    data/prepare_lits.py). Two-label, multi-label output: Liver and Tumor
    masks stacked as separate channels (tumor is a subset of liver, so
    both are supplied for BVINet(num_classes=2) with per-channel sigmoid).
    """

    def __init__(self, root, split="train", augment=None):
        """Indexes the image/liver-mask/tumor-mask triples for one split.

        Parameters:
            root (str): path to the prepared dataset folder (output of
                data/prepare_lits.py), containing train/val/test
                subfolders each with images/, masks_liver/, masks_tumor/.
            split (str, default="train"): "train", "val", or "test".
            augment (callable or None, default=None): an albumentations
                Compose applied to (image, mask) jointly -- pass a mask
                with an extra channel dim so both liver and tumor masks
                get the identical spatial transform. None for val/test.
        """
        self.img_dir = Path(root) / split / "images"
        self.liver_dir = Path(root) / split / "masks_liver"
        self.tumor_dir = Path(root) / split / "masks_tumor"
        self.files = sorted(p.name for p in self.img_dir.glob("*.png"))
        self.augment = augment

    def __len__(self):
        """Returns the number of slices in this split."""
        return len(self.files)

    def __getitem__(self, idx):
        """Loads, normalizes, optionally augments, and tensor-izes one
        CT slice + its (liver, tumor) mask pair.

        Parameters:
            idx (int): index into the file list.

        Returns:
            tuple:
                img (torch.Tensor): shape (1, H, W), float32 grayscale CT
                    slice (already HU-windowed + normalized by
                    prepare_lits.py), values in [0, 1].
                mask (torch.Tensor): shape (2, H, W), float32, channel 0 =
                    liver, channel 1 = tumor, each binary (0 or 1).
        """
        name = self.files[idx]
        img = cv2.imread(str(self.img_dir / name), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        liver = (cv2.imread(str(self.liver_dir / name), cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
        tumor = (cv2.imread(str(self.tumor_dir / name), cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
        mask = np.stack([liver, tumor], axis=-1)  # (H, W, 2)

        if self.augment is not None:
            augmented = self.augment(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]

        img = torch.from_numpy(img).unsqueeze(0).float()
        mask = torch.from_numpy(mask.transpose(2, 0, 1)).float()
        return img, mask


class BraTSDataset(Dataset):
    """PyTorch Dataset for the prepared BraTS19 split (output of
    data/prepare_brats.py). 4-channel input (T1, T1ce, T2, FLAIR stacked
    as image channels instead of RGB), three-label multi-label output:
    Whole Tumor / Tumor Core / Enhancing Tumor (nested/overlapping regions,
    for BVINet(in_channels=4, num_classes=3) with per-channel sigmoid).
    """

    def __init__(self, root, split="train", augment=None):
        """Indexes the image/WT/TC/ET quadruples for one split.

        Parameters:
            root (str): path to the prepared dataset folder (output of
                data/prepare_brats.py), containing train/val/test
                subfolders each with images/ (.npy, 4-channel),
                masks_wt/, masks_tc/, masks_et/ (.png).
            split (str, default="train"): "train", "val", or "test".
            augment (callable or None, default=None): an albumentations
                Compose applied to (image, mask) jointly. None for
                val/test splits to keep evaluation unbiased.
        """
        self.img_dir = Path(root) / split / "images"
        self.wt_dir = Path(root) / split / "masks_wt"
        self.tc_dir = Path(root) / split / "masks_tc"
        self.et_dir = Path(root) / split / "masks_et"
        self.files = sorted(p.name for p in self.img_dir.glob("*.npy"))
        self.augment = augment

    def __len__(self):
        """Returns the number of slices in this split."""
        return len(self.files)

    def __getitem__(self, idx):
        """Loads, normalizes, optionally augments, and tensor-izes one
        4-modality MRI slice + its (WT, TC, ET) mask triple.

        Parameters:
            idx (int): index into the file list.

        Returns:
            tuple:
                img (torch.Tensor): shape (4, H, W), float32, values in
                    [0, 1] (already z-score normalized + rescaled by
                    prepare_brats.py) -- channel order T1, T1ce, T2, FLAIR.
                mask (torch.Tensor): shape (3, H, W), float32, channel
                    order WT, TC, ET, each binary (0 or 1).
        """
        stem = Path(self.files[idx]).stem
        img = np.load(self.img_dir / self.files[idx]).astype(np.float32) / 255.0  # (H, W, 4)
        mask_name = f"{stem}.png"
        wt = (cv2.imread(str(self.wt_dir / mask_name), cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
        tc = (cv2.imread(str(self.tc_dir / mask_name), cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
        et = (cv2.imread(str(self.et_dir / mask_name), cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
        mask = np.stack([wt, tc, et], axis=-1)  # (H, W, 3)

        if self.augment is not None:
            augmented = self.augment(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]

        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask.transpose(2, 0, 1)).float()
        return img, mask


# Registry of the three organ datasets from the base paper: which Dataset
# class to load, the model's expected in_channels/num_classes, and the
# per-channel names used when reporting Dice/mIoU broken down by class
# (Table V/VI/VII of the paper). Shared by train.py and evaluate.py so the
# two never drift out of sync.
DATASETS = {
    "isic": dict(cls=ISICDataset, in_channels=3, num_classes=1, class_names=["lesion"]),
    "lits": dict(cls=LiTSDataset, in_channels=1, num_classes=2, class_names=["liver", "tumor"]),
    "brats": dict(cls=BraTSDataset, in_channels=4, num_classes=3, class_names=["WT", "TC", "ET"]),
}
