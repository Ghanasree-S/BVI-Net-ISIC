# TODO — Final Review: Extending to Liver (LiTS17) and Brain (BraTS19)

Status: ISIC2018 (skin) segmentation is complete (Dice 0.8631, 27,312 params,
within 5% of the paper's claimed 0.026M). This file tracks what's left for
the two remaining tasks from the base paper, plus the frontend.

---

## 1. Liver — LiTS17

- [ ] **Source the dataset.** Kaggle mirror found: `javariatahir123/lits17-liver-tumor-segmentation`.
      131 CT volumes, NIfTI format (`.nii`/`.nii.gz`), ~58,638 total slices
      across all volumes. Add via Kaggle "+ Add Input" like the ISIC dataset.
- [ ] **3D → 2D slicing script** (`data/prepare_lits.py`): each CT volume is a
      3D array (512×512×N slices) — must be sliced into individual 2D images.
      Use `nibabel` to load `.nii` files.
- [ ] **CT windowing / HU normalization:** raw CT values are in Hounsfield
      Units (typically -1000 to +1000s); apply a liver-window clip (e.g.
      [-100, 400] HU is common for liver) before normalizing to [0,1] —
      without this the images will look like noise to the model.
- [ ] **Filter to labeled slices only:** paper uses 19,163 liver-labeled +
      7,190 tumor-labeled slices out of the full volume set (discards
      unlabeled/empty slices to balance classes).
- [ ] **Two-label masks:** each slice has a "Liver" mask and a "Tumor" mask
      (tumor is a subset of liver). Model output should be `num_classes=2`
      (BVINet already supports this via `num_classes` param) with sigmoid
      per-channel (multi-label, not multi-class, since tumor overlaps liver).
- [ ] **Split:** paper uses 7:1:2 (train:val:test) of the 131 volumes,
      volume-level split (not slice-level) so no data leakage between splits.
- [ ] **Resize to 448×448** (paper's stated input size for LiTS — different
      from ISIC's 256×256).
- [ ] **Retrain BVI-Net** with `--num_classes 2`, same hyperparameters
      (BCE+Dice loss, AdamW lr=0.001, batch size reduced for real-Mamba
      memory as before).
- [ ] **Report per-class Dice/mIoU** (separately for Liver and Tumor,
      matching paper's Table VI structure) — will need a small metrics.py
      extension to break down per-channel instead of only global.

## 2. Brain — BraTS19

- [ ] **Source the dataset.** Official access requires CBICA/Synapse
      registration (data use agreement) — start this early, approval can
      take time. Check for a Kaggle mirror as a faster alternative if one
      with proper licensing exists.
- [ ] **Multi-modal 3D → 2D slicing** (`data/prepare_brats.py`): each case has
      4 MRI modalities (T1, T1c, T2, FLAIR) that must be stacked as input
      channels (so model input becomes 4-channel instead of RGB 3-channel —
      requires changing BVINet's `in_channels` param, already supported).
- [ ] **Three nested labels:** Whole Tumor (WT), Tumor Core (TC), Enhancing
      Tumor (ET) — again `num_classes=3` with sigmoid (nested/overlapping
      regions, not mutually exclusive classes).
- [ ] **Use only LGG cases** (76 patients) as the paper does, citing BraTS19's
      full HGG+LGG complexity as unnecessary for this scope. Split 7:1:2.
- [ ] **Resize to 240×240** (paper's stated size for BraTS).
- [ ] **Retrain and evaluate**, same protocol as above, report per-class
      Dice/mIoU/etc. matching paper's Table VII.

## 3. Frontend

- [x] Dashboard prompt written for AI Studio (multi-organ tab selector:
      Skin / Liver / Brain, skin fully wired, others show "coming soon"
      until their models are ready).
- [ ] Once Liver/Brain checkpoints exist: build a small backend API
      (FastAPI) with one `/predict` endpoint that:
      - accepts `organ_type` + image
      - loads the matching `BVINet` checkpoint (different `channels`,
        `num_classes`, `in_channels` per organ)
      - runs inference, returns mask + metrics as JSON
- [ ] Wire the AI Studio frontend's mock `/predict` call to this real API.
- [ ] Add overlay visualization (translucent mask on top of original image)
      in the backend response or frontend rendering.

## 4. Nice-to-haves (if time remains)

- [ ] Inference speed (FPS) benchmark vs. paper's claimed 292 FPS.
- [ ] Ablation table extending the width-search results (Dice at 0.66M vs
      37K vs 27.3K params) into the final report/PPT.
- [ ] Update `main.tex` with Liver/Brain results once available (new
      slides: "Liver Results", "Brain Results", updated Conclusion).
