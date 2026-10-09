"""Training loop matching the paper's settings (Sec. IV-A):
AdamW, initial lr 0.001, batch size 64, 50 epochs, early stop on 5-epoch
val-Dice plateau, combined BCE+Dice loss.
"""
import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from dataset import DATASETS
from losses import BCEDiceLoss
from metrics import per_class_metrics
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
    total_loss, n = 0.0, 0
    inter, denom = 0.0, 0.0  # per-channel totals over the whole pass
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
        n += img.size(0)
        # Dice accumulated over the whole loader (per channel, then averaged)
        # rather than averaged per batch: a batch with no tumor scores 1.0 by
        # predicting nothing, which made the per-batch average noisy and
        # rewarded a model that never predicts the small class.
        binary = (pred.detach() > 0.5).float()
        inter = inter + (binary * mask).sum(dim=(0, 2, 3))
        denom = denom + binary.sum(dim=(0, 2, 3)) + mask.sum(dim=(0, 2, 3))
    dice = ((2 * inter + 1e-6) / (denom + 1e-6)).mean().item()
    return total_loss / n, dice


def main():
    """Parses hyperparameters from the command line, builds the model/data
    pipeline, and runs the full training loop with early stopping.

    Command-line hyperparameters:
        --dataset (str, default="isic"): which organ dataset to train on --
            "isic" (skin, RGB, 1-class), "lits" (liver, grayscale CT,
            2-class Liver+Tumor), or "brats" (brain, 4-channel MRI,
            3-class WT/TC/ET). Selects the Dataset class and the model's
            in_channels/num_classes automatically via the DATASETS registry
            -- no need to pass --channels/--in_channels separately.
        --data_dir (str, default="data/isic2018a"): path to the prepared
            dataset (output of data/prepare_isic_a.py, prepare_lits.py, or
            prepare_brats.py, matching --dataset).
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
    ap.add_argument("--dataset", choices=list(DATASETS), default="isic")
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
    ap.add_argument("--ablate", nargs="*", default=[], choices=["gabor", "global", "gcn"],
                    help="Ablation: remove these components (Gabor bank / FA-VSSM pathway / GCN skips)")
    ap.add_argument("--init_checkpoint", default=None,
                    help="Start from these weights (fine-tuning) instead of random init")
    ap.add_argument("--oversample_class", type=int, default=None,
                    help="Output channel whose positive slices get sampled more often (1 = LiTS tumor)")
    ap.add_argument("--oversample_factor", type=float, default=3.0,
                    help="Sampling weight of slices containing --oversample_class (others = 1)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    organ = DATASETS[args.dataset]
    DatasetClass = organ["cls"]
    train_ds = DatasetClass(args.data_dir, split="train",
                            augment=get_train_augment(organ["in_channels"]))
    val_ds = DatasetClass(args.data_dir, split="val")  # no augmentation for honest evaluation
    if args.oversample_class is None:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    else:
        # Rare-class oversampling: liver tumor appears in a minority of slices, so
        # weight those slices up (same epoch length, sampled with replacement).
        plain = DatasetClass(args.data_dir, split="train")
        has_pos = [bool(plain[i][1][args.oversample_class].any()) for i in range(len(plain))]
        weights = [args.oversample_factor if p else 1.0 for p in has_pos]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=2)
        print(f"Oversampling class {args.oversample_class}: {sum(has_pos)}/{len(has_pos)} positive slices "
              f"x{args.oversample_factor}")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = BVINet(
        in_channels=organ["in_channels"],
        num_classes=organ["num_classes"],
        channels=args.channels,
        gcn_nodes=args.gcn_nodes,
        ablate=args.ablate,
    ).to(device)
    if args.init_checkpoint:
        model.load_state_dict(torch.load(args.init_checkpoint, map_location=device))
        print(f"Initialised from {args.init_checkpoint}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params / 1e6:.4f}M)")

    # lambda1 = lambda2 = 1.0 (Eq. 9 of the paper); batch-pooled Dice for multi-class organs
    criterion = BCEDiceLoss(per_sample_dice=organ["num_classes"] == 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )  # halves lr if val_dice plateaus for 3 epochs, squeezes out extra gains before early stop

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    best_dice, patience_left = 0.0, args.patience
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_dice = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_dice = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_dice)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} train_dice={train_dice:.4f} "
              f"| val_loss={val_loss:.4f} val_dice={val_dice:.4f} | lr={current_lr:.6f} | {time.time() - epoch_start:.0f}s")

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

    if organ["num_classes"] > 1:
        # Per-class breakdown (paper Table VI/VII: Liver+Tumor or WT/TC/ET reported
        # separately, not pooled) using the best checkpoint just saved.
        model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
        model.eval()
        totals = {name: {} for name in organ["class_names"]}
        counts = {name: 0 for name in organ["class_names"]}
        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.to(device), mask.to(device)
                pred = model(img)
                per_class = per_class_metrics(pred, mask, organ["class_names"])
                for name, m in per_class.items():
                    for k, v in m.items():
                        totals[name][k] = totals[name].get(k, 0.0) + v * img.size(0)
                    counts[name] += img.size(0)
        print("Per-class validation metrics (best checkpoint):")
        for name in organ["class_names"]:
            avg = {k: v / counts[name] for k, v in totals[name].items()}
            print(f"  {name}: " + " ".join(f"{k}={v:.4f}" for k, v in avg.items()))


if __name__ == "__main__":
    main()
