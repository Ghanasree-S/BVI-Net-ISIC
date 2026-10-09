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
- [x] Real held-out demo samples for every organ (4 ISIC, 2 LiTS, 3 BraTS from different patients),
      exported by `notebooks/kaggle-export-samples.ipynb`; drawn placeholder samples removed.

- [x] Liver fine-tune with tumor oversampling (`--init_checkpoint`, `--oversample_class`): small gain, adopted.
- [x] GPU FPS benchmark (`benchmark.py`): skin 47, brain 46, liver 30 FPS on a Tesla T4.

- [x] Batched 4-direction scan: 1.5–1.8× faster on GPU (skin 87, brain 84, liver 44 FPS on a T4).
- [x] Ablations on ISIC2018: w/o GCN −1.75, w/o FA-VSSM −0.61, w/o Gabor +0.67 Dice.

## Remaining / nice-to-have

- [ ] Liver tumor is still the weakest class (0.567 global Dice). Fine-tuning with 3× tumor-slice
      oversampling (v8) gave only +0.001 global / +0.046 per-image Dice; next: every labelled slice,
      tumor-only crops around the liver, or a dedicated tumor stage.
- [ ] Speed: 44–87 FPS on a T4 vs. the paper's 292 FPS — next: fp16 / CUDA graphs / torch.compile.
- [ ] BraTS HGG cases; repeat ablations over several seeds (differences are near run-to-run noise).
