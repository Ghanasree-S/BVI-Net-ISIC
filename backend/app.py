"""FastAPI inference backend for BVI-Net.

One `/predict` endpoint that accepts `organ_type` + an image, loads the
matching BVINet checkpoint (different in_channels/num_classes/channels
per organ), runs real inference, and returns a mask + metrics as JSON --
the "real API" the frontend's mock server.ts `/predict` route (see
frontend/server.ts) is meant to be swapped for once Liver/Brain
checkpoints exist (see TODO.md section 3).

Response shape intentionally matches server.ts's mock payload
(`mask_base64`, `metrics`, `model_info`) so the frontend fetch call
doesn't need to change when pointed at this backend instead.

Run with: uvicorn backend.app:app --host 0.0.0.0 --port 8000
"""
import base64
import io
import json
import os
import time
from pathlib import Path

# Checkpoints are trained on Kaggle with the real CUDA mamba_ssm, which this
# CPU backend doesn't have -- use the parameter-compatible pure-PyTorch port
# (models/mamba_ref.py) so their state_dicts load. Must be set before
# importing models.
os.environ.setdefault("BVI_MAMBA_REF", "1")

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import BVINet
from models.fa_vssm import HAS_MAMBA

CHECKPOINT_DIR = Path(os.environ.get("BVI_CHECKPOINT_DIR", Path(__file__).parent.parent / "checkpoints"))

# Per-organ model config: must match what each organ was actually trained
# with (see train.py's DATASETS registry) -- wrong in_channels/num_classes
# here will fail to load the checkpoint's state_dict.
ORGAN_CONFIG = {
    "skin": dict(
        checkpoint="bvi_net_skin_best.pt",
        in_channels=3, num_classes=1, input_size=256,
        channels=[4, 8, 8, 16, 16], gcn_nodes=8,
        class_names=["lesion"], dataset="ISIC2018 (Skin Lesion)",
    ),
    "liver": dict(
        checkpoint="bvi_net_liver_best.pt",
        in_channels=1, num_classes=2, input_size=448,
        channels=[4, 8, 8, 16, 16], gcn_nodes=8,
        class_names=["liver", "tumor"], dataset="LiTS17 (Liver Tumor CT)",
    ),
    "brain": dict(
        checkpoint="bvi_net_brain_best.pt",
        in_channels=4, num_classes=3, input_size=240,
        channels=[4, 8, 8, 16, 16], gcn_nodes=8,
        class_names=["WT", "TC", "ET"], dataset="BraTS19 (Brain Glioma MRI)",
    ),
}

app = FastAPI(title="BVI-Net Inference Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model_cache = {}
_flops_cache = {}


class PredictRequest(BaseModel):
    organ_type: str = "skin"
    image: str  # base64-encoded image (data URL or raw base64)


def load_model(organ_type):
    """Loads (and caches) the BVINet checkpoint for one organ.

    Parameters:
        organ_type (str): one of "skin", "liver", "brain" -- key into
            ORGAN_CONFIG.

    Returns:
        tuple[nn.Module, dict]: the loaded model in eval() mode, and its
        ORGAN_CONFIG entry.

    Raises:
        HTTPException(404): if organ_type is unknown, or its checkpoint
        file doesn't exist yet (liver/brain until their training in
        TODO.md sections 1-2 is complete).
    """
    if organ_type not in ORGAN_CONFIG:
        raise HTTPException(404, f"Unknown organ_type '{organ_type}'")
    if organ_type in _model_cache:
        return _model_cache[organ_type], ORGAN_CONFIG[organ_type]

    cfg = ORGAN_CONFIG[organ_type]
    ckpt_path = CHECKPOINT_DIR / cfg["checkpoint"]
    if not ckpt_path.exists():
        raise HTTPException(
            404,
            f"No checkpoint for '{organ_type}' yet at {ckpt_path} "
            "-- train it first (see TODO.md).",
        )

    model = BVINet(
        in_channels=cfg["in_channels"],
        num_classes=cfg["num_classes"],
        channels=cfg["channels"],
        gcn_nodes=cfg["gcn_nodes"],
    )
    if not HAS_MAMBA:
        raise HTTPException(500, "Mamba backend unavailable -- cannot load a real-Mamba checkpoint")
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    _model_cache[organ_type] = model
    _flops_cache[organ_type] = count_gflops(model, cfg)
    return model, cfg


def count_gflops(model, cfg):
    """GFLOPs of one forward pass at the organ's input size, counted by
    torch's FlopCounterMode (convolutions + matmuls, which dominate; the
    elementwise selective-scan ops aren't counted)."""
    from torch.utils.flop_counter import FlopCounterMode
    x = torch.zeros(1, cfg["in_channels"], cfg["input_size"], cfg["input_size"])
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        model(x)
    return round(counter.get_total_flops() / 1e9, 3)


def load_test_metrics(organ_type):
    """Reads the checkpoint's test-set metrics written by evaluate.py
    (outputs/<organ>/metrics.json, copied next to the checkpoint as
    bvi_net_<organ>_metrics.json).

    A newly uploaded image has no ground-truth mask, so Dice/IoU/etc. can't
    be measured for it -- the dashboard shows the model's held-out test-set
    scores instead, flagged with metrics_source="test_set".

    Returns:
        dict or None: per-class metrics of the first class, or None if the
        metrics file hasn't been copied in yet.
    """
    path = CHECKPOINT_DIR / f"bvi_net_{organ_type}_metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    first_class = ORGAN_CONFIG[organ_type]["class_names"][0]
    per_class = data["per_class"]
    return per_class.get(first_class) or next(iter(per_class.values()))


def decode_npy(raw, size, in_channels):
    """Decodes a .npy slice (BraTS: (H, W, 4) uint8, as written by
    data/prepare_brats.py) into a (1, C, size, size) tensor in [0, 1]."""
    arr = np.load(io.BytesIO(raw), allow_pickle=False)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.shape[-1] != in_channels:
        raise HTTPException(400, f"Expected {in_channels} channels, got array of shape {arr.shape}")
    arr = cv2.resize(arr.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.max() > 1.0:
        arr = arr / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1).copy()).unsqueeze(0)


def decode_image(data_url, size, in_channels):
    """Decodes a base64 image string into a model-ready input tensor.

    Parameters:
        data_url (str): base64-encoded image, optionally prefixed with a
            `data:image/...;base64,` header (browser <input type=file>
            output).
        size (int): target square resolution, matching the organ's
            trained input size (256/448/240).
        in_channels (int): 3 for RGB (skin), 1 for grayscale CT (liver).
            Brain's 4-channel MRI input isn't representable by a single
            uploaded photo/PNG, so brain requests must supply a
            pre-stacked array -- not handled by this simple decode path.

    Returns:
        torch.Tensor: shape (1, in_channels, size, size), float32, [0, 1].
    """
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    if raw[:6] == bytes([0x93]) + b"NUMPY":  # .npy magic
        return decode_npy(raw, size, in_channels)
    if in_channels not in (1, 3):
        raise HTTPException(
            400,
            f"This model needs a {in_channels}-channel input (T1/T1ce/T2/FLAIR) -- "
            "upload a .npy slice from data/prepare_brats.py, not a regular image.",
        )
    arr = np.frombuffer(raw, dtype=np.uint8)
    flag = cv2.IMREAD_COLOR if in_channels == 3 else cv2.IMREAD_GRAYSCALE
    img = cv2.imdecode(arr, flag)
    if img is None:
        raise HTTPException(400, "Could not decode image")
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    if in_channels == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)
    else:
        img = img[None, :, :]
    tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
    return tensor


def encode_mask_png(mask_uint8):
    """Encodes a single-channel uint8 mask array as a base64 PNG data URL.

    Parameters:
        mask_uint8 (np.ndarray): 2D array, values in [0, 255].

    Returns:
        str: `data:image/png;base64,...` data URL.
    """
    ok, buf = cv2.imencode(".png", mask_uint8)
    if not ok:
        raise HTTPException(500, "Failed to encode mask")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


@app.get("/api/health")
def health():
    """Reports which organ checkpoints are currently available to serve."""
    availability = {
        organ: (CHECKPOINT_DIR / cfg["checkpoint"]).exists()
        for organ, cfg in ORGAN_CONFIG.items()
    }
    return {
        "status": "online",
        "service": "BVI-Net Inference Engine",
        "checkpoints_available": availability,
    }


@app.post("/predict")
@app.post("/api/predict")
def predict(req: PredictRequest):
    """Runs BVINet inference for one organ + image.

    Parameters (JSON body):
        organ_type (str): "skin", "liver", or "brain".
        image (str): base64-encoded input image.

    Returns:
        dict: mask_base64 (first predicted channel, PNG data URL),
        per-class metrics-shaped placeholders are omitted here since
        ground truth isn't available at inference time (only Dice/mIoU
        computed during training/eval have a target to compare against),
        model_info (checkpoint/architecture details).
    """
    organ_type = req.organ_type.lower().strip()
    model, cfg = load_model(organ_type)

    start = time.time()
    img_tensor = decode_image(req.image, cfg["input_size"], cfg["in_channels"])
    with torch.no_grad():
        pred = model(img_tensor)  # (1, num_classes, H, W), sigmoid probabilities
    inference_ms = (time.time() - start) * 1000

    # Confidence: mean probability the model assigns to its own decision,
    # over the predicted region (or over the whole image if nothing is found).
    p = pred[0, 0]
    fg = p > 0.5
    confidence = float(p[fg].mean()) if fg.any() else float((1 - p).mean())
    n_params = sum(t.numel() for t in model.parameters())
    test = load_test_metrics(organ_type) or {}

    channel_masks = {}
    for c, name in enumerate(cfg["class_names"]):
        mask_uint8 = (pred[0, c] > 0.5).float().mul(255).byte().numpy()
        channel_masks[name] = encode_mask_png(mask_uint8)

    return {
        "success": True,
        "organ_type": organ_type,
        "status": "production_ready",
        "mask_base64": channel_masks[cfg["class_names"][0]],
        "masks_by_class": channel_masks,
        "metrics": {
            "dice": test.get("dice", 0.0),
            "iou": test.get("miou", 0.0),
            "sensitivity": test.get("sensitivity", 0.0),
            "specificity": test.get("specificity", 0.0),
            "confidence": round(confidence, 4),
            "inference_time_ms": round(inference_ms, 2),
            "model_size": f"{n_params:,} parameters ({n_params * 4 / 1e6:.2f} MB)",
            "parameter_count": n_params,
            "flops_gflops": _flops_cache.get(organ_type, 0.0),
            "metrics_source": "test_set" if test else "unavailable",
            "foreground_fraction": round(float(fg.float().mean()), 4),
        },
        "model_info": {
            "name": "BVI-Net",
            "version": "v1.0",
            "architecture": "Gabor Local + Mamba Global + GCN Attention",
            "dataset": cfg["dataset"],
            "checkpoint": cfg["checkpoint"],
            "input_resolution": f"{cfg['input_size']}x{cfg['input_size']}x{cfg['in_channels']}",
            "class_names": cfg["class_names"],
        },
    }
