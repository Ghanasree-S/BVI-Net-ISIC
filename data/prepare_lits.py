"""Builds the LiTS17 train/val/test split, resized to 448x448.

Each raw sample is a pair of 3D CT volumes (`volume-N.nii`/`.nii.gz` +
`segmentation-N.nii`/`.nii.gz`) covering the whole abdomen. This script:
  1. loads each volume with nibabel,
  2. clips to a liver HU window and normalizes to [0,1] (raw CT values are
     Hounsfield Units, typically -1000..3000 -- without windowing the model
     sees mostly noise),
  3. slices the 3D volume into individual 2D axial slices,
  4. discards slices with no liver/tumor label at all (paper keeps only
     19,163 liver + 7,190 tumor labeled slices, not the full ~58K),
  5. splits at the VOLUME level (7:1:2) so no slice from the same patient
     leaks across train/val/test,
  6. writes two label channels per slice: liver mask and tumor mask.

LiTS17 segmentation labels: 0 = background, 1 = liver, 2 = tumor (tumor is
a subset of liver, so "liver" mask below includes label 2 pixels too).
"""
import argparse
import random
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
from tqdm import tqdm

SIZE = 448  # paper's stated input size for LiTS (different from ISIC's 256x256)
LIVER_WINDOW = (-100, 400)  # HU clip range commonly used for liver CT (min, max)


def hu_to_uint8(slice_hu, window=LIVER_WINDOW):
    """Clips a raw Hounsfield-Unit CT slice to a liver window and rescales
    to an 8-bit grayscale image.

    Parameters:
        slice_hu (np.ndarray): 2D array of raw HU values for one axial slice.
        window (tuple[float, float], default=LIVER_WINDOW): (min_hu, max_hu)
            clip range -- values outside are clamped before rescaling.
            [-100, 400] HU is a standard liver window; narrower than a
            full-body window so liver/tumor contrast survives normalization.

    Returns:
        np.ndarray: same shape, dtype uint8, values in [0, 255].
    """
    lo, hi = window
    clipped = np.clip(slice_hu, lo, hi)
    normalized = (clipped - lo) / (hi - lo)  # -> [0, 1]
    return (normalized * 255).astype(np.uint8)


def find_volume_pairs(raw_dir):
    """Pairs up every `volume-N` file with its matching `segmentation-N`
    file under the raw LiTS17 directory, regardless of `.nii`/`.nii.gz`
    extension or nesting depth.

    Parameters:
        raw_dir (str or Path): root folder of the raw LiTS17 download
            (e.g. the Kaggle-mounted `lits17-liver-tumor-segmentation` dir).

    Returns:
        list[tuple[int, Path, Path]]: (volume_id, volume_path, seg_path),
        sorted by volume_id.
    """
    raw = Path(raw_dir)
    volumes = {}
    segmentations = {}
    for p in raw.rglob("volume-*.nii*"):
        vol_id = int(p.stem.split("-")[1].split(".")[0])
        volumes[vol_id] = p
    for p in raw.rglob("segmentation-*.nii*"):
        vol_id = int(p.stem.split("-")[1].split(".")[0])
        segmentations[vol_id] = p

    pairs = [(vid, volumes[vid], segmentations[vid]) for vid in sorted(volumes) if vid in segmentations]
    if not pairs:
        print(f"Could not find volume-N/segmentation-N pairs under {raw}")
        print("Directory tree found:")
        for p in sorted(raw.rglob("*"))[:50]:
            print(" ", p)
        raise FileNotFoundError("No matching LiTS17 volume/segmentation pairs")
    return pairs


def save_volume_slices(vol_id, vol_path, seg_path, out_dir, split, min_label_fraction=0.0):
    """Loads one CT volume + its segmentation, slices both into 2D axial
    slices, discards unlabeled slices, and writes the labeled ones out.

    Parameters:
        vol_id (int): patient/volume identifier, used as a filename prefix
            so slices from the same volume never collide across splits.
        vol_path (Path): path to the raw `volume-N.nii(.gz)` CT scan.
        seg_path (Path): path to the matching `segmentation-N.nii(.gz)`
            label volume (0=background, 1=liver, 2=tumor).
        out_dir (Path): root of the prepared dataset (contains
            train/val/test subfolders).
        split (str): which split folder to write into ("train", "val", or
            "test") -- determined by the volume-level split, so every slice
            from this volume goes to the same split.
        min_label_fraction (float, default=0.0): slices are kept only if
            at least this fraction of pixels are labeled liver or tumor;
            0.0 keeps every slice with a nonzero label pixel (paper's
            "labeled slices only" filter), raise it to be stricter.

    Returns:
        int: number of slices written for this volume.
    """
    vol = nib.load(str(vol_path)).get_fdata()
    seg = nib.load(str(seg_path)).get_fdata()
    n_slices = vol.shape[2]
    written = 0

    for z in range(n_slices):
        seg_slice = seg[:, :, z]
        liver_mask = (seg_slice >= 1).astype(np.uint8)  # label 1 or 2
        tumor_mask = (seg_slice == 2).astype(np.uint8)  # label 2 only

        label_fraction = liver_mask.mean()
        if label_fraction <= min_label_fraction:
            continue  # discard unlabeled/empty slices to match paper's filtered slice count

        img_slice = hu_to_uint8(vol[:, :, z])
        img_slice = cv2.resize(img_slice, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        liver_mask = cv2.resize(liver_mask * 255, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        tumor_mask = cv2.resize(tumor_mask * 255, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)

        name = f"vol{vol_id:03d}_slice{z:04d}.png"
        cv2.imwrite(str(out_dir / split / "images" / name), img_slice)
        cv2.imwrite(str(out_dir / split / "masks_liver" / name), liver_mask)
        cv2.imwrite(str(out_dir / split / "masks_tumor" / name), tumor_mask)
        written += 1

    return written


def main():
    """Parses hyperparameters and builds the resized LiTS17 train/val/test
    dataset folders used by dataset.py's LiTSDataset.

    Command-line hyperparameters/parameters:
        --raw_dir (str): path to the raw LiTS17 download (containing
            volume-N.nii(.gz) / segmentation-N.nii(.gz) pairs), e.g. the
            Kaggle-mounted dataset folder.
        --out_dir (str, default="data/lits17a"): where the prepared,
            resized dataset is written.
        --seed (int, default=42): random seed for shuffling volumes before
            the split, so the split is reproducible across runs.
        --splits (tuple[float, float, float], default=(0.7, 0.1, 0.2)):
            train/val/test fractions, applied at the VOLUME level (not
            slice level) to avoid leaking slices from the same patient
            across splits -- matches the paper's stated 7:1:2 protocol.
        --min_label_fraction (float, default=0.0): forwarded to
            save_volume_slices -- minimum fraction of liver-labeled pixels
            required to keep a slice.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default=str(Path(__file__).parent / "lits17_raw"))
    ap.add_argument("--out_dir", default=str(Path(__file__).parent / "lits17a"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits", type=float, nargs=3, default=(0.7, 0.1, 0.2),
                     help="train/val/test fractions, applied at the volume level")
    ap.add_argument("--min_label_fraction", type=float, default=0.0)
    args = ap.parse_args()

    pairs = find_volume_pairs(args.raw_dir)
    random.Random(args.seed).shuffle(pairs)
    print(f"LiTS17 volumes found: {len(pairs)}")

    n = len(pairs)
    n_train = int(n * args.splits[0])
    n_val = int(n * args.splits[1])
    split_assignment = (
        [("train", p) for p in pairs[:n_train]]
        + [("val", p) for p in pairs[n_train:n_train + n_val]]
        + [("test", p) for p in pairs[n_train + n_val:]]
    )
    print(f"Volume split -> train: {n_train} | val: {n_val} | test: {n - n_train - n_val}")

    out = Path(args.out_dir)
    for split in ("train", "val", "test"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "masks_liver").mkdir(parents=True, exist_ok=True)
        (out / split / "masks_tumor").mkdir(parents=True, exist_ok=True)

    total_written = 0
    for split, (vol_id, vol_path, seg_path) in tqdm(split_assignment, desc="slicing volumes"):
        total_written += save_volume_slices(
            vol_id, vol_path, seg_path, out, split, args.min_label_fraction
        )

    print(f"Done. {total_written} labeled slices written to: {out}")


if __name__ == "__main__":
    main()
