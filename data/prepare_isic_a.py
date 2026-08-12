"""Builds ISIC2018-a: a random 1/4 subset of ISIC2018 Task 1, resized to 256x256,
split train/val/test, matching the paper's ablation-subset protocol (Sec. IV-A).
"""
import argparse
import random
from pathlib import Path

import cv2
from tqdm import tqdm

IMG_SUFFIX = ".jpg"
MASK_SUFFIX = "_segmentation.png"
SIZE = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default=str(Path(__file__).parent / "isic2018_raw"))
    ap.add_argument("--out_dir", default=str(Path(__file__).parent / "isic2018a"))
    ap.add_argument("--fraction", type=float, default=1.0,
                     help="1.0 = full ISIC2018 (matches paper's Table V); 0.25 = ISIC2018-a subset")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits", type=float, nargs=3, default=(0.8, 0.1, 0.1),
                     help="train/val/test fractions")
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    img_dir = next(raw.glob("*Training_Input*"))
    mask_dir = next(raw.glob("*Training_GroundTruth*"))

    images = sorted(p for p in img_dir.glob(f"*{IMG_SUFFIX}"))
    random.Random(args.seed).shuffle(images)
    subset = images[: int(len(images) * args.fraction)]
    print(f"ISIC2018 total: {len(images)} | ISIC2018-a subset: {len(subset)}")

    n = len(subset)
    n_train = int(n * args.splits[0])
    n_val = int(n * args.splits[1])
    split_map = (
        [("train", p) for p in subset[:n_train]]
        + [("val", p) for p in subset[n_train:n_train + n_val]]
        + [("test", p) for p in subset[n_train + n_val:]]
    )

    out = Path(args.out_dir)
    for split in ("train", "val", "test"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "masks").mkdir(parents=True, exist_ok=True)

    for split, img_path in tqdm(split_map, desc="resizing + splitting"):
        stem = img_path.stem
        mask_path = mask_dir / f"{stem}{MASK_SUFFIX}"
        if not mask_path.exists():
            continue

        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(str(out / split / "images" / f"{stem}.png"), img)
        cv2.imwrite(str(out / split / "masks" / f"{stem}.png"), mask)

    print(f"Done. ISIC2018-a saved to: {out}")


if __name__ == "__main__":
    main()
