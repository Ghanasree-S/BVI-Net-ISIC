"""Builds the ISIC2018 train/val/test split, resized to 256x256.

Two modes:
  1. --val_raw_dir given: uses ISIC2018's OFFICIAL validation set (100 images,
     public masks) as the val split -- matches the paper's val count exactly.
     The train pool (2594 official training images) has a small held-out
     chunk carved off as our own test set, since the paper's real test split
     (1000 images) has never had its ground truth masks publicly released --
     that specific split is unreachable by anyone without official ISIC
     challenge submission access.
  2. --val_raw_dir omitted (default): falls back to a self-split of the
     training pool only, as before.
"""
import argparse
import random
from pathlib import Path

import cv2
from tqdm import tqdm

IMG_SUFFIX = ".jpg"
MASK_SUFFIX = "_segmentation.png"
SIZE = 256


def find_dirs(raw_dir, img_pattern, mask_pattern, required=True):
    raw = Path(raw_dir)
    try:
        img_dir = next(raw.rglob(img_pattern))
        mask_dir = next(raw.rglob(mask_pattern))
    except StopIteration:
        if not required:
            return None, None
        print(f"Could not find {img_pattern} / {mask_pattern} under {raw}")
        print("Directory tree found:")
        for p in sorted(raw.rglob("*"))[:50]:
            print(" ", p)
        raise
    img_dir = img_dir if img_dir.is_dir() else img_dir.parent
    mask_dir = mask_dir if mask_dir.is_dir() else mask_dir.parent
    return img_dir, mask_dir


def save_pair(img_path, mask_dir, out_dir, split):
    stem = img_path.stem
    mask_path = mask_dir / f"{stem}{MASK_SUFFIX}"
    if not mask_path.exists():
        return False

    img = cv2.imread(str(img_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(str(out_dir / split / "images" / f"{stem}.png"), img)
    cv2.imwrite(str(out_dir / split / "masks" / f"{stem}.png"), mask)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default=str(Path(__file__).parent / "isic2018_raw"))
    ap.add_argument("--val_raw_dir", default=None,
                     help="Path to official ISIC2018 validation set (see "
                          "download_isic_official_val.sh). If given, val split "
                          "matches the paper's 100-image official validation set.")
    ap.add_argument("--out_dir", default=str(Path(__file__).parent / "isic2018a"))
    ap.add_argument("--fraction", type=float, default=1.0,
                     help="1.0 = full ISIC2018 training pool; 0.25 = ISIC2018-a subset")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_fraction", type=float, default=0.1,
                     help="Fraction of the training pool held out as our own test "
                          "set (only used when --val_raw_dir is given, since the "
                          "paper's real 1000-image test set has no public masks)")
    ap.add_argument("--splits", type=float, nargs=3, default=(0.8, 0.1, 0.1),
                     help="train/val/test fractions (only used in self-split fallback mode)")
    args = ap.parse_args()

    img_dir, mask_dir = find_dirs(args.raw_dir, "*Training_Input*", "*Training_GroundTruth*")
    images = sorted(p for p in img_dir.glob(f"*{IMG_SUFFIX}"))
    random.Random(args.seed).shuffle(images)
    subset = images[: int(len(images) * args.fraction)]
    print(f"ISIC2018 training pool total: {len(images)} | using: {len(subset)}")

    out = Path(args.out_dir)
    for split in ("train", "val", "test"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "masks").mkdir(parents=True, exist_ok=True)

    if args.val_raw_dir:
        val_img_dir, val_mask_dir = find_dirs(
            args.val_raw_dir, "*Validation_Input*", "*Validation_GroundTruth*"
        )
        val_images = sorted(p for p in val_img_dir.glob(f"*{IMG_SUFFIX}"))
        print(f"Official validation set found: {len(val_images)} images (matches paper's split)")

        n_test = int(len(subset) * args.test_fraction)
        test_images = subset[:n_test]
        train_images = subset[n_test:]
        print(f"Train: {len(train_images)} (official pool minus self-carved test) | "
              f"Val: {len(val_images)} (official) | "
              f"Test: {len(test_images)} (self-carved -- paper's real test masks are not public)")

        split_map = (
            [("train", p, mask_dir) for p in train_images]
            + [("val", p, val_mask_dir) for p in val_images]
            + [("test", p, mask_dir) for p in test_images]
        )
    else:
        n = len(subset)
        n_train = int(n * args.splits[0])
        n_val = int(n * args.splits[1])
        print("No --val_raw_dir given -- falling back to self-split of training pool only.")
        split_map = (
            [("train", p, mask_dir) for p in subset[:n_train]]
            + [("val", p, mask_dir) for p in subset[n_train:n_train + n_val]]
            + [("test", p, mask_dir) for p in subset[n_train + n_val:]]
        )

    for split, img_path, this_mask_dir in tqdm(split_map, desc="resizing + splitting"):
        save_pair(img_path, this_mask_dir, out, split)

    print(f"Done. Dataset saved to: {out}")


if __name__ == "__main__":
    main()
