# Status — BVI-Net multi-organ replication

All three tasks from the base paper are trained, evaluated and served in the dashboard.
Numbers below are held-out test sets; see README §1 for the full table.

## Done

- [x] **Skin (ISIC2018)** — Dice 0.863, 27,312 params (Kaggle v1).
- [x] **Liver (LiTS17)** — global Dice liver 0.928 / tumor 0.566, 27,277 params (Kaggle v7,
      `andrewmvd` mirror, stride 5, 3834 slices).
- [x] **Brain (BraTS19 LGG)** — global Dice WT 0.862 / TC 0.711 / ET 0.744, 27,342 params (Kaggle v6).
- [x] Small-class collapse fixed: batch-pooled Dice loss for multi-class organs, dataset-level
      validation Dice for early stopping, `global_dice` in `evaluate.py` / metrics JSONs.
- [x] `evaluate.py` writes `outputs/<organ>/metrics.json`; notebook packages checkpoints + metrics.
- [x] CPU inference of real-Mamba checkpoints via `models/mamba_ref.py` (matches GPU masks
      pixel-for-pixel on saved test samples).
- [x] FastAPI backend: all organs, test-set metrics, per-image confidence, real params/GFLOPs,
      `.npy` input + FLAIR preview for brain.
- [x] Frontend wired to the backend (no simulated masks/metrics), overlay view, liver/brain enabled,
      real LiTS17 test slices as samples, errors surfaced in the UI.
- [x] README rewritten; slides (`ppt/main.tex`) updated with liver/brain results, the loss fix,
      qualitative test images and the dashboard.

## Remaining / nice-to-have

- [ ] Real ISIC + BraTS demo samples: run `notebooks/kaggle-export-samples.ipynb` on Kaggle (CPU,
      no Mamba build) and copy `demo_samples.zip` contents into `frontend/public/samples/`.
- [ ] Liver tumor is the weakest class (0.566 global Dice): oversample tumor slices, use every
      labelled slice, longer training.
- [ ] GPU FPS benchmark vs. the paper's 292 FPS (CPU latency is ~4–16 s per image).
- [ ] Ablations (Gabor sharing / FA-VSSM / GCN skips) and BraTS HGG cases.
