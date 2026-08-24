# TODO — Final Review: Extending to Liver (LiTS17) and Brain (BraTS19)

Status: ISIC2018 (skin) segmentation is complete (Dice 0.8631, 27,312 params,
within 5% of the paper's claimed 0.026M). This file tracks what's left for
the two remaining tasks from the base paper, plus the frontend.

---

## 1. Liver — LiTS17

- [ ] **Source the dataset.** Kaggle mirror found: `javariatahir123/lits17-liver-tumor-segmentation`.
      131 CT volumes, NIfTI format (`.nii`/`.nii.gz`), ~58,638 total slices
      across all volumes. Add via Kaggle "+ Add Input" like the ISIC dataset.
      **Blocked on manual download — needs a Kaggle session, not doable from here.**
- [x] **3D → 2D slicing script** (`data/prepare_lits.py`): loads `.nii`/`.nii.gz`
      volumes via `nibabel`, slices into 2D axial images. Written, untested
      against real data (no dataset downloaded yet).
- [x] **CT windowing / HU normalization:** liver window [-100, 400] HU clip
      → normalize to [0,1], implemented in `hu_to_uint8()`.
- [x] **Filter to labeled slices only:** `save_volume_slices()` discards
      slices with no liver/tumor label (`--min_label_fraction`).
- [x] **Two-label masks:** writes separate `masks_liver/` + `masks_tumor/`
      folders; `LiTSDataset` in `dataset.py` stacks them as a (2, H, W)
      multi-label target for `BVINet(num_classes=2)`.
- [x] **Split:** volume-level 7:1:2 split in `prepare_lits.py` (shuffles
      whole volumes before splitting, not slices — no leakage).
- [x] **Resize to 448×448** — `SIZE = 448` in `prepare_lits.py`.
- [x] **Retrain BVI-Net**: `train.py --dataset lits` now auto-selects
      `LiTSDataset`, `in_channels=1`, `num_classes=2`. **Actually running
      this still needs the dataset downloaded and a GPU training session.**
- [x] **Report per-class Dice/mIoU**: `metrics.per_class_metrics()` added;
      `train.py` prints a per-class (Liver/Tumor) breakdown on the best
      checkpoint after training.

## 2. Brain — BraTS19

- [ ] **Source the dataset.** Official access requires CBICA/Synapse
      registration (data use agreement) — start this early, approval can
      take time. **This is a manual human approval step, not automatable.**
- [x] **Multi-modal 3D → 2D slicing** (`data/prepare_brats.py`): stacks
      T1/T1ce/T2/FLAIR as 4 input channels, z-score normalizes each
      modality. Written, untested against real data (no dataset yet).
- [x] **Three nested labels:** WT/TC/ET masks written separately;
      `BraTSDataset` stacks them as (3, H, W) for `BVINet(num_classes=3)`.
- [x] **Use only LGG cases**: `find_case_dirs(..., lgg_only=True)` is the
      default filter. Case-level 7:1:2 split.
- [x] **Resize to 240×240** — `SIZE = 240` in `prepare_brats.py`.
- [x] **Retrain and evaluate**: `train.py --dataset brats` auto-selects
      `BraTSDataset`, `in_channels=4`, `num_classes=3`, and prints the
      WT/TC/ET per-class breakdown. **Still needs the approved dataset and
      a GPU training session to actually run.**

## 3. Frontend

- [x] Dashboard prompt written for AI Studio (multi-organ tab selector:
      Skin / Liver / Brain, skin fully wired, others show "coming soon"
      until their models are ready).
- [x] Backend API (`backend/app.py`, FastAPI) with `/predict` and
      `/api/predict`: accepts `organ_type` + base64 image, loads the
      matching `BVINet` checkpoint per organ (`ORGAN_CONFIG`), runs real
      inference, returns mask + metrics as JSON. Response shape matches
      `frontend/server.ts`'s mock payload so the frontend fetch call
      doesn't need to change. Skin will work once `checkpoints/bvi_net_skin_best.pt`
      is copied in; liver/brain 404 until their checkpoints exist (by design).
- [ ] Wire the AI Studio frontend's mock `/predict` call (in `frontend/server.ts`)
      to this real API — point it at `http://localhost:8000/predict` (run via
      `uvicorn backend.app:app`) instead of `generateSyntheticMaskPng`.
- [ ] Add overlay visualization (translucent mask on top of original image)
      in the backend response or frontend rendering.

## 4. Nice-to-haves (if time remains)

- [ ] Inference speed (FPS) benchmark vs. paper's claimed 292 FPS.
- [ ] Ablation table extending the width-search results (Dice at 0.66M vs
      37K vs 27.3K params) into the final report/PPT.
- [ ] Update `main.tex` with Liver/Brain results once available (new
      slides: "Liver Results", "Brain Results", updated Conclusion).
