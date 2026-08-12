"""PyTorch Dataset for the prepared ISIC2018-a split."""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class ISICDataset(Dataset):
    def __init__(self, root, split="train", augment=None):
        self.img_dir = Path(root) / split / "images"
        self.mask_dir = Path(root) / split / "masks"
        self.files = sorted(p.name for p in self.img_dir.glob("*.png"))
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
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
