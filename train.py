"""Training loop matching the paper's settings (Sec. IV-A):
AdamW, initial lr 0.001, batch size 64, 50 epochs, early stop on 5-epoch
val-Dice plateau, combined BCE+Dice loss.
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ISICDataset
from losses import BCEDiceLoss
from metrics import dice_score
from models import BVINet


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss, total_dice, n = 0.0, 0.0, 0
    torch.set_grad_enabled(train)
    for img, mask in tqdm(loader, leave=False):
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
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--out_dir", default="checkpoints")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = ISICDataset(args.data_dir, split="train")
    val_ds = ISICDataset(args.data_dir, split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = BVINet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params / 1e6:.4f}M)")

    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    best_dice, patience_left = 0.0, args.patience
    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_dice = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} train_dice={train_dice:.4f} "
              f"| val_loss={val_loss:.4f} val_dice={val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            patience_left = args.patience
            torch.save(model.state_dict(), out_dir / "best.pt")
            print(f"  -> new best (val_dice={best_dice:.4f}), checkpoint saved")
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"Early stopping at epoch {epoch} (no val_dice improvement in {args.patience} epochs)")
                break

    print(f"Training done. Best val_dice = {best_dice:.4f}")


if __name__ == "__main__":
    main()
