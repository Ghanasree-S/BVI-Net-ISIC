"""Evaluates a trained checkpoint on the test split and reports the same metrics
as Table V of the paper. Optionally saves qualitative image/GT/prediction triptychs.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DATASETS
from metrics import per_class_metrics
from models import BVINet


def save_triptych(img, mask, pred, out_path, class_idx=0):
    """Saves a side-by-side [input | ground truth | prediction] image for
    qualitative inspection.

    Parameters:
        img (torch.Tensor): input image, shape (in_channels, H, W), values
            in [0, 1] -- only the first 3 channels are used for display
            (BraTS's 4-channel MRI input isn't directly viewable as RGB;
            this shows T1/T1ce/T2 as a pseudo-color composite).
        mask (torch.Tensor): ground-truth mask, shape (num_classes, H, W).
        pred (torch.Tensor): predicted probabilities, shape (num_classes, H, W).
        out_path (str or Path): file path to write the combined PNG to.
        class_idx (int, default=0): which output channel to visualize, for
            multi-label organs (e.g. 0="liver", 1="tumor").
    """
    display_img = img[:3] if img.shape[0] >= 3 else img.expand(3, -1, -1)
    img_np = (display_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    mask_np = (mask[class_idx].cpu().numpy() * 255).astype(np.uint8)
    pred_np = ((pred[class_idx].cpu().numpy() > 0.5) * 255).astype(np.uint8)  # 0.5 = decision threshold

    mask_rgb = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR)
    pred_rgb = cv2.cvtColor(pred_np, cv2.COLOR_GRAY2BGR)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    combined = np.concatenate([img_bgr, mask_rgb, pred_rgb], axis=1)
    cv2.imwrite(str(out_path), combined)


def main():
    """Parses command-line arguments, loads a trained checkpoint, and runs
    it over the test split to report Table-V-style metrics.

    Command-line hyperparameters/parameters:
        --dataset (str, default="isic"): which organ dataset to evaluate --
            "isic", "lits", or "brats" -- selects the Dataset class and the
            model's in_channels/num_classes via dataset.DATASETS, same
            registry train.py uses (must match what --checkpoint was
            actually trained with).
        --checkpoint (str, default="checkpoints/best.pt"): path to the
            trained model weights (output of train.py).
        --data_dir (str, default="data/isic2018a"): path to the prepared
            dataset.
        --visualize (flag): if set, saves the first 12 test-set predictions
            as image/GT/prediction triptychs to --out_dir.
        --out_dir (str, default="outputs"): where triptych images are saved.
        --channels (list[int] of 5, default=None): encoder channel widths --
            MUST exactly match what the checkpoint was trained with, since
            it determines the number of weight tensors and their shapes.
        --gcn_nodes (int, default=32): GCN-Attention node count -- MUST
            match the checkpoint's training configuration for the same reason.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), default="isic")
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--data_dir", default="data/isic2018a")
    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--channels", type=int, nargs=5, default=None,
                     help="Must match the widths used when the checkpoint was trained")
    ap.add_argument("--gcn_nodes", type=int, default=32)
    args = ap.parse_args()

    organ = DATASETS[args.dataset]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BVINet(
        in_channels=organ["in_channels"],
        num_classes=organ["num_classes"],
        channels=args.channels,
        gcn_nodes=args.gcn_nodes,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    test_ds = organ["cls"](args.data_dir, split="test")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = organ["class_names"]
    accum = {name: {k: [] for k in ("dice", "miou", "accuracy", "specificity", "sensitivity", "assd")}
             for name in class_names}
    with torch.no_grad():
        for i, (img, mask) in enumerate(tqdm(test_loader, desc="evaluating")):
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            per_class = per_class_metrics(pred, mask, class_names)
            for name, m in per_class.items():
                for k, v in m.items():
                    if not np.isnan(v):
                        accum[name][k].append(v)

            if args.visualize and i < 12:
                save_triptych(img[0], mask[0], pred[0], out_dir / f"sample_{i:03d}.png")

    print(f"\n=== Test set results ({args.dataset}) ===")
    results = {}
    for name in class_names:
        print(f" -- {name} --")
        results[name] = {}
        for k, vals in accum[name].items():
            results[name][k] = float(np.mean(vals))
            print(f"{k:>12}: {results[name][k]:.4f}")

    # Test-set metrics for the backend to report alongside live predictions
    # (a new upload has no ground truth, so its own Dice can't be computed).
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"dataset": args.dataset, "num_test": len(test_ds), "per_class": results}, f, indent=2)
    print(f"Saved {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
