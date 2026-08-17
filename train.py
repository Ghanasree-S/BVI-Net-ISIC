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
from transforms import get_train_augment


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    """Runs one full pass over a DataLoader, either training or evaluating.

    Parameters:
        model (nn.Module): the BVINet instance being trained/evaluated.
        loader (DataLoader): yields (image, mask) batches.
        criterion (nn.Module): loss function (BCEDiceLoss).
        optimizer (torch.optim.Optimizer): only stepped when train=True.
        device (torch.device): "cuda" or "cpu".
        train (bool, default=True): if True, runs backprop + optimizer step
            and puts the model in train() mode (enables dropout/BN update);
            if False, runs in eval() mode with gradients disabled.

    Returns:
        tuple[float, float]: (mean_loss, mean_dice) over every sample in the loader.
    """
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
    """Parses hyperparameters from the command line, builds the model/data
    pipeline, and runs the full training loop with early stopping.

    Command-line hyperparameters:
        --data_dir (str, default="data/isic2018a"): path to the prepared
            dataset (output of data/prepare_isic_a.py).
        --epochs (int, default=50): maximum training epochs -- matches the
            paper's protocol; early stopping may end training sooner.
        --batch_size (int, default=64): samples per gradient step. Paper
            uses 64; reduced to 8 in practice when using real Mamba, since
            its selective-scan at 256x256 is memory-heavy (see README).
        --lr (float, default=0.001): initial AdamW learning rate, matches
            the paper's stated value.
        --patience (int, default=15): epochs to wait for a val_dice
            improvement before early-stopping. The paper's own patience (5)
            was found to stop too early in practice.
        --out_dir (str, default="checkpoints"): where the best checkpoint
            (best.pt) is saved.
        --channels (list[int] of 5, default=None): overrides the encoder's
            5-stage channel widths (see models/bvi_net.py BVINet.__init__)
            -- the main lever for shrinking total parameter count.
        --gcn_nodes (int, default=32): overrides the GCN-Attention graph
            node count N used at every skip connection (see
            models/gcn_attention.py) -- second lever on parameter count.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/isic2018a")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--patience", type=int, default=15)  # 5 stopped training too early in testing
    ap.add_argument("--out_dir", default="checkpoints")
    ap.add_argument("--channels", type=int, nargs=5, default=None,
                     help="Override encoder channel widths, e.g. --channels 4 8 16 32 64 "
                          "to shrink the model toward the paper's claimed 0.026M params")
    ap.add_argument("--gcn_nodes", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = ISICDataset(args.data_dir, split="train", augment=get_train_augment())
    val_ds = ISICDataset(args.data_dir, split="val")  # no augmentation for honest evaluation
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = BVINet(channels=args.channels, gcn_nodes=args.gcn_nodes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params / 1e6:.4f}M)")

    criterion = BCEDiceLoss()  # lambda1 = lambda2 = 1.0 (Eq. 9 of the paper)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )  # halves lr if val_dice plateaus for 3 epochs, squeezes out extra gains before early stop

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    best_dice, patience_left = 0.0, args.patience
    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_dice = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_dice)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} train_dice={train_dice:.4f} "
              f"| val_loss={val_loss:.4f} val_dice={val_dice:.4f} | lr={current_lr:.6f}")

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
