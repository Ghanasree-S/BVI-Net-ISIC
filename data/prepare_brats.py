"""Builds the BraTS19 (LGG-only) train/val/test split, resized to 240x240.

Each raw case is a folder containing 4 co-registered MRI modalities
(T1, T1ce, T2, FLAIR) plus a segmentation volume. This script:
  1. loads all 4 modalities + the segmentation with nibabel,
  2. per-modality z-score normalizes each volume (brain-extracted MRI has
     no fixed intensity scale like CT HU, so min-max/window clipping
     doesn't apply -- z-score per volume is the standard BraTS preprocessing),
  3. stacks the 4 modalities as input channels (T1, T1ce, T2, FLAIR) instead
     of RGB, so the model's `in_channels` becomes 4,
  4. slices each 3D volume into 2D axial slices, discarding slices with no
     tumor label at all,
  5. splits at the CASE level (7:1:2) so slices from one patient never leak
     across train/val/test,
  6. builds 3 nested/overlapping label channels per slice: Whole Tumor (WT,
     labels 1+2+4), Tumor Core (TC, labels 1+4), Enhancing Tumor (ET, label
     4 only) -- BraTS raw labels are 0=background, 1=necrotic core,
     2=edema, 4=enhancing tumor (3 is intentionally unused upstream).
  7. restricts to LGG cases only, per the paper's stated scope (76 patients,
     avoiding full HGG+LGG complexity).
"""
import argparse
import random
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
from tqdm import tqdm

SIZE = 240  # paper's stated input size for BraTS
MODALITY_SUFFIXES = ["t1", "t1ce", "t2", "flair"]  # fixed channel order -> in_channels=4
SEG_SUFFIX = "seg"


def zscore_normalize(volume):
    """Normalizes one MRI modality volume to zero mean / unit variance over
    its nonzero (brain-tissue) voxels, then rescales to [0, 1] for image
    encoding.

    Parameters:
        volume (np.ndarray): raw 3D MRI intensities for one modality.

    Returns:
        np.ndarray: same shape, float values in [0, 1].
    """
    mask = volume > 0
    if mask.sum() == 0:
        return np.zeros_like(volume, dtype=np.float32)
    mean, std = volume[mask].mean(), volume[mask].std()
    normalized = (volume - mean) / (std + 1e-6)
    normalized = np.clip(normalized, -5, 5)  # clip extreme outlier voxels before rescale
    return ((normalized + 5) / 10).astype(np.float32)  # -> [0, 1]


def find_case_dirs(raw_dir, lgg_only=True):
    """Locates every per-patient case folder under the raw BraTS19 dataset,
    optionally restricted to the LGG (low-grade glioma) cohort.

    Parameters:
        raw_dir (str or Path): root of the raw BraTS19 download (official
            CBICA/Synapse export, or a Kaggle mirror with the same layout).
        lgg_only (bool, default=True): if True, only returns case folders
            under a path containing "LGG" -- matches the paper's stated use
            of the 76 LGG cases only, citing full HGG+LGG as unnecessary
            complexity for this scope.

    Returns:
        list[Path]: one directory per case, each expected to contain
        `<case>_t1.nii.gz`, `_t1ce.nii.gz`, `_t2.nii.gz`, `_flair.nii.gz`,
        `_seg.nii.gz`.
    """
    raw = Path(raw_dir)
    seg_files = sorted(raw.rglob(f"*_{SEG_SUFFIX}.nii*"))
    if lgg_only:
        seg_files = [p for p in seg_files if "LGG" in str(p).upper()]
    case_dirs = sorted({p.parent for p in seg_files})
    if not case_dirs:
        print(f"Could not find BraTS case folders (lgg_only={lgg_only}) under {raw}")
        print("Directory tree found:")
        for p in sorted(raw.rglob("*"))[:50]:
            print(" ", p)
        raise FileNotFoundError("No matching BraTS19 case folders")
    return case_dirs


def load_case(case_dir):
    """Loads and z-score normalizes all 4 modalities plus the raw
    segmentation volume for one BraTS case.

    Parameters:
        case_dir (Path): per-patient folder containing the 4 modality
            files and the segmentation file (see find_case_dirs).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            stacked (H, W, D, 4) float32 array of normalized modalities,
            seg (H, W, D) raw integer label volume.
    """
    case_id = case_dir.name
    modality_vols = []
    for suffix in MODALITY_SUFFIXES:
        matches = list(case_dir.glob(f"*_{suffix}.nii*"))
        if not matches:
            raise FileNotFoundError(f"Missing modality '{suffix}' in {case_dir}")
        vol = nib.load(str(matches[0])).get_fdata()
        modality_vols.append(zscore_normalize(vol))
    stacked = np.stack(modality_vols, axis=-1)  # (H, W, D, 4)

    seg_matches = list(case_dir.glob(f"*_{SEG_SUFFIX}.nii*"))
    seg = nib.load(str(seg_matches[0])).get_fdata()
    return stacked, seg


def save_case_slices(case_id, stacked, seg, out_dir, split, slice_stride=1):
    """Slices one case's stacked 4-modality volume + segmentation into 2D
    axial slices, discards tumor-free slices, and writes labeled ones out.

    Parameters:
        case_id (str): patient case identifier, used as a filename prefix.
        stacked (np.ndarray): (H, W, D, 4) normalized modality volume, see
            load_case.
        seg (np.ndarray): (H, W, D) raw BraTS label volume (0/1/2/4).
        out_dir (Path): root of the prepared dataset (contains
            train/val/test subfolders).
        split (str): which split folder to write into -- determined at the
            case level so all slices from one patient share a split.
        slice_stride (int, default=1): keep only every Nth axial slice
            (z % slice_stride == 0) before the tumor-free filter -- adjacent
            slices are nearly identical, so this cuts training time ~Nx at
            little cost in information.

    Returns:
        int: number of slices written for this case.
    """
    n_slices = stacked.shape[2]
    written = 0

    for z in range(0, n_slices, slice_stride):
        seg_slice = seg[:, :, z]
        wt = (seg_slice > 0).astype(np.uint8)                       # 1 + 2 + 4
        tc = np.isin(seg_slice, [1, 4]).astype(np.uint8)            # 1 + 4
        et = (seg_slice == 4).astype(np.uint8)                      # 4 only

        if wt.sum() == 0:
            continue  # discard slices with no tumor at all

        img_slice = (stacked[:, :, z, :] * 255).astype(np.uint8)    # (H, W, 4)
        img_slice = cv2.resize(img_slice, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        wt_r = cv2.resize(wt * 255, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        tc_r = cv2.resize(tc * 255, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        et_r = cv2.resize(et * 255, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)

        name = f"{case_id}_slice{z:04d}.npy"
        np.save(out_dir / split / "images" / name, img_slice)  # 4-channel, .npy since PNG caps at 4 channels via RGBA (order mismatch)
        mask_name = f"{case_id}_slice{z:04d}.png"
        cv2.imwrite(str(out_dir / split / "masks_wt" / mask_name), wt_r)
        cv2.imwrite(str(out_dir / split / "masks_tc" / mask_name), tc_r)
        cv2.imwrite(str(out_dir / split / "masks_et" / mask_name), et_r)
        written += 1

    return written


def main():
    """Parses hyperparameters and builds the resized BraTS19 (LGG-only)
    train/val/test dataset folders used by dataset.py's BraTSDataset.

    Command-line hyperparameters/parameters:
        --raw_dir (str): path to the raw BraTS19 download (official
            CBICA/Synapse export or a licensed Kaggle mirror with the same
            per-case folder layout).
        --out_dir (str, default="data/brats19a"): where the prepared,
            resized dataset is written.
        --seed (int, default=42): random seed for shuffling cases before
            the split.
        --splits (tuple[float, float, float], default=(0.7, 0.1, 0.2)):
            train/val/test fractions, applied at the CASE level to avoid
            leaking slices from the same patient across splits -- matches
            the paper's stated 7:1:2 protocol.
        --lgg_only (bool, default=True): restrict to the 76 LGG cases, per
            the paper's stated scope; pass --no-lgg_only to include HGG too.
        --slice_stride (int, default=2): keep every Nth axial slice, see
            save_case_slices. Use 1 for every tumor-bearing slice.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default=str(Path(__file__).parent / "brats19_raw"))
    ap.add_argument("--out_dir", default=str(Path(__file__).parent / "brats19a"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits", type=float, nargs=3, default=(0.7, 0.1, 0.2),
                     help="train/val/test fractions, applied at the case level")
    ap.add_argument("--lgg_only", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--slice_stride", type=int, default=2)
    args = ap.parse_args()

    case_dirs = find_case_dirs(args.raw_dir, lgg_only=args.lgg_only)
    random.Random(args.seed).shuffle(case_dirs)
    print(f"BraTS19 cases found ({'LGG only' if args.lgg_only else 'HGG+LGG'}): {len(case_dirs)}")

    n = len(case_dirs)
    n_train = int(n * args.splits[0])
    n_val = int(n * args.splits[1])
    split_assignment = (
        [("train", d) for d in case_dirs[:n_train]]
        + [("val", d) for d in case_dirs[n_train:n_train + n_val]]
        + [("test", d) for d in case_dirs[n_train + n_val:]]
    )
    print(f"Case split -> train: {n_train} | val: {n_val} | test: {n - n_train - n_val}")

    out = Path(args.out_dir)
    for split in ("train", "val", "test"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "masks_wt").mkdir(parents=True, exist_ok=True)
        (out / split / "masks_tc").mkdir(parents=True, exist_ok=True)
        (out / split / "masks_et").mkdir(parents=True, exist_ok=True)

    total_written = 0
    for split, case_dir in tqdm(split_assignment, desc="slicing cases"):
        stacked, seg = load_case(case_dir)
        total_written += save_case_slices(case_dir.name, stacked, seg, out, split, args.slice_stride)

    print(f"Done. {total_written} labeled slices written to: {out}")


if __name__ == "__main__":
    main()
