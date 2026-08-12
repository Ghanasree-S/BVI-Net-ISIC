# BVI-Net Replication — Skin Lesion Segmentation (ISIC2018-a)

Replication of **"Ultra-Lightweight Network for Medical Image Segmentation Inspired by
Bio-Visual Interaction"** (Cai, Fan, Zhu, Fang — *IEEE Transactions on Circuits and
Systems for Video Technology*, Vol. 35, No. 4, April 2025, DOI: 10.1109/TCSVT.2024.3507383).

Scope: only the **skin lesion segmentation** task on the **ISIC2018-a** subset (the
paper's own 1/4-sized ablation subset of ISIC2018), building only the proposed **BVI-Net**
model — no baseline/SOTA comparison models, no ablation variants, no liver/brain tasks.

---

## 1. What the model does

Given a dermoscopy (skin) image, BVI-Net outputs a binary mask marking which pixels
belong to the lesion. It has three pieces, inspired by the human visual system:

| Component | File | What it simulates | What it does |
|---|---|---|---|
| Local Pathway | `models/gabor_conv.py` | Ventral stream (detail vision) | Gabor-function-initialized conv kernels tuned to V1/V2/V4 orientation bias — extracts fine lesion edges |
| Global Pathway (FA-VSSM) | `models/fa_vssm.py` | Dorsal stream (fast, big-picture vision) | State-space model scanning the image 4 ways (line/column/Z-curve/Hilbert) + pooling-based attention — quickly finds roughly where the lesion is |
| GCN Attention skip connection | `models/gcn_attention.py` | Multi-level visual integration | Projects fused features into a graph, runs graph convolution, adds refined detail back during upsampling |
| Full model | `models/bvi_net.py` | — | Encoder-decoder combining all three, guided by Global→Local multiplication |

**Known deviation (documented, not hidden):** the paper's Global Pathway uses the
`mamba_ssm` selective-scan library, which frequently fails to build on Colab (CUDA
compile step). `fa_vssm.py` tries `mamba_ssm` first and automatically falls back to a
depthwise-conv + gating approximation of the same idea if it's unavailable. Report
whichever path you ended up training with.

---

## 2. Project structure

```
BVI-Net-ISIC/
├── README.md                  <- this file
├── requirements.txt
├── data/
│   ├── download_isic.sh       <- fetches ISIC2018 Task 1 images + masks
│   └── prepare_isic_a.py      <- builds the ISIC2018-a 1/4 subset, resizes to 256x256
├── models/
│   ├── gabor_conv.py          <- Local Pathway
│   ├── fa_vssm.py             <- Global Pathway
│   ├── gcn_attention.py       <- skip connection
│   └── bvi_net.py             <- assembles the full model
├── train.py                   <- training loop (BCE+Dice loss, AdamW, early stop)
├── evaluate.py                <- Dice / mIoU / Acc / Spe / Sen / ASSD on test set
└── notebooks/
    └── run_on_colab.ipynb     <- Colab-ready notebook wrapping the above
```

---

## 3. Step-by-step plan

### Step 1 — Environment setup
```bash
pip install -r requirements.txt
```
On Colab, run the first cell of `notebooks/run_on_colab.ipynb` — it attempts
`pip install mamba-ssm` and reports whether it succeeded (fallback is automatic either way).

### Step 2 — Get the dataset
```bash
bash data/download_isic.sh          # downloads ISIC2018 Task 1 training images + ground truth
python data/prepare_isic_a.py       # randomly samples 1/4, resizes to 256x256, train/val/test split
```

### Step 3 — Sanity-check the model builds and runs
```bash
python -c "from models.bvi_net import BVINet; import torch; m = BVINet(); print(m(torch.randn(1,3,256,256)).shape)"
```
Expect output shape `(1, 1, 256, 256)`.

### Step 4 — Train
```bash
python train.py --data_dir data/isic2018a --epochs 50 --batch_size 64 --lr 0.001
```
Matches the paper's settings: combined BCE+Dice loss (equal weight), AdamW, initial
lr 0.001, batch size 64, early stop if validation Dice doesn't improve for 5 epochs.

### Step 5 — Evaluate
```bash
python evaluate.py --checkpoint checkpoints/best.pt --data_dir data/isic2018a
```
Reports Dice, mIoU, Accuracy, Specificity, Sensitivity, ASSD — the same metrics
Table V in the paper uses, so your numbers are directly comparable to their reported
BVI-Net row (not to the other SOTA rows, since we're not building those).

### Step 6 — Visualize a few predictions
`evaluate.py --visualize` saves a handful of image/ground-truth/prediction triptychs
to `outputs/` — useful for your report's qualitative section (paper's Fig. 5 equivalent).

### Step 7 — Write up
Report: your Dice/mIoU/etc. vs. the paper's reported ISIC numbers, training curve,
a few qualitative examples, and — if you hit the Mamba fallback — a short note on
that deviation and why it doesn't change the core replication claim.

---

## 4. Feasibility summary

- Dataset: ISIC2018-a, ~650 images, free public download, no access approval needed
- Model: 0.026M parameters — trains in well under an hour on a Colab T4 for 50 epochs
- Main risk: `mamba_ssm` install — mitigated by automatic fallback in `fa_vssm.py`
- Estimated time: dataset+setup (half a day), model assembly/debugging (1-1.5 days),
  training+evaluation (half a day), write-up (half a day) — fits a 3-5 day budget
