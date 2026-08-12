"""Evaluates a trained checkpoint on the test split and reports the same metrics
as Table V of the paper. Optionally saves qualitative image/GT/prediction triptychs.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ISICDataset
from metrics import all_metrics
from models import BVINet


def save_triptych(img, mask, pred, out_path):
    img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    mask_np = (mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
    pred_np = ((pred.squeeze().cpu().numpy() > 0.5) * 255).astype(np.uint8)

    mask_rgb = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR)
    pred_rgb = cv2.cvtColor(pred_np, cv2.COLOR_GRAY2BGR)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    combined = np.concatenate([img_bgr, mask_rgb, pred_rgb], axis=1)
    cv2.imwrite(str(out_path), combined)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--data_dir", default="data/isic2018a")
    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BVINet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    test_ds = ISICDataset(args.data_dir, split="test")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    out_dir = Path(args.out_dir)
    if args.visualize:
        out_dir.mkdir(exist_ok=True)

    accum = {k: [] for k in ("dice", "miou", "accuracy", "specificity", "sensitivity", "assd")}
    with torch.no_grad():
        for i, (img, mask) in enumerate(tqdm(test_loader, desc="evaluating")):
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            m = all_metrics(pred, mask)
            for k, v in m.items():
                if not np.isnan(v):
                    accum[k].append(v)

            if args.visualize and i < 12:
                save_triptych(img[0], mask[0], pred[0], out_dir / f"sample_{i:03d}.png")

    print("\n=== Test set results (ISIC2018-a) ===")
    for k, vals in accum.items():
        print(f"{k:>12}: {np.mean(vals):.4f}")


if __name__ == "__main__":
    main()
