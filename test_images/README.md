# Test images for the dashboard

Real scans from the **held-out test splits** (never seen during training).
Open http://localhost:3000, pick the organ, click "Browse Computer" and choose a file.

| Folder | Upload this | Notes |
|---|---|---|
| `skin/`  | `skin_test_1..4.png` | ISIC2018 dermoscopy, 256×256 RGB (small → very large lesions) |
| `liver/` | `liver_ct_test_1..2.png` | LiTS17 axial CT, liver window −100…400 HU, 448×448 grayscale |
| `brain/` | `brain_test_1..3.npy` | BraTS19 LGG, 4-channel MRI (T1, T1ce, T2, FLAIR), 240×240 — upload the **.npy** |

`brain_test_*_preview_FLAIR.png` are only previews for you to look at; the brain model needs the `.npy`.
