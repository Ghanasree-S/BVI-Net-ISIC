"""Sweeps a few encoder channel-width configs to see how close we can get to
the paper's claimed 0.026M parameters, and what it costs in Dice.

Uses a reduced epoch budget per config (not the full 50) since this is a
comparative sweep, not a final training run -- once you pick a config, retrain
it with the full protocol (train.py --epochs 50) for your final reported number.
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import ISICDataset
from losses import BCEDiceLoss
from metrics import dice_score
from models import BVINet
from transforms import get_train_augment

# Each entry: (name, channels list, gcn_nodes). Smaller widths as we go down.
CONFIGS = [
    ("current (8-16-32-64-128, N=32)", [8, 16, 32, 64, 128], 32),
    ("narrow-1 (4-8-16-32-64, N=16)", [4, 8, 16, 32, 64], 16),
    ("narrow-2 (2-4-8-16-32, N=8)", [2, 4, 8, 16, 32], 8),
]


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss, total_dice, n = 0.0, 0.0, 0
    torch.set_grad_enabled(train)
    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        pred = model(img)
        loss = criterion(pred, mask)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * img.size(0)
        total_dice += dice_score(pred.detach(), mask) * img.size(0)
        n += img.size(0)
    return total_loss / n, total_dice / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/isic2018a")
    ap.add_argument("--epochs", type=int, default=20,
                     help="Reduced epoch budget per config for a fast comparative sweep")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--out_json", default="width_search_results.json")
    args = ap.parse_args()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        print(f"GPU memory free: {free / 1e9:.2f} GB / {total / 1e9:.2f} GB total")
        if free / total < 0.5:
            print("WARNING: less than half the GPU memory is free. If this is the start "
                  "of a fresh script run, a previous process likely still holds memory -- "
                  "restart the Kaggle/Colab kernel before continuing.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = ISICDataset(args.data_dir, split="train", augment=get_train_augment())
    val_ds = ISICDataset(args.data_dir, split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    results = []
    for name, channels, gcn_nodes in CONFIGS:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # release previous config's activations before starting next
        print(f"\n=== {name} ===")
        model = BVINet(channels=channels, gcn_nodes=gcn_nodes).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {n_params:,} ({n_params / 1e6:.4f}M)")

        criterion = BCEDiceLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        best_dice = 0.0
        for epoch in range(1, args.epochs + 1):
            _, train_dice = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            _, val_dice = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
            best_dice = max(best_dice, val_dice)
            print(f"  epoch {epoch:02d} | train_dice={train_dice:.4f} val_dice={val_dice:.4f}")

        results.append({
            "config": name,
            "channels": channels,
            "gcn_nodes": gcn_nodes,
            "params": n_params,
            "params_M": round(n_params / 1e6, 4),
            "best_val_dice": round(best_dice, 4),
        })

        del model, optimizer, criterion  # free this config's GPU memory before the next one
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n=== Width search summary ===")
    print(f"{'Config':38s} {'Params (M)':>12s} {'Best Val Dice':>15s}")
    for r in results:
        print(f"{r['config']:38s} {r['params_M']:12.4f} {r['best_val_dice']:15.4f}")

    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out_json}")


if __name__ == "__main__":
    main()
