"""Inference-speed benchmark (FPS) of BVI-Net for each organ's input size.

Times batch-1 forward passes after a warm-up, the setting the paper's FPS
figure refers to. Weights don't affect speed, so trained checkpoints are
optional (loaded when present so the run also sanity-checks them).

Usage: python benchmark.py [--iters 200] [--out outputs/benchmark.json]
"""
import argparse
import json
import time
from pathlib import Path

import torch

from models import BVINet

ORGANS = {  # name: (in_channels, num_classes, input_size)
    "skin": (3, 1, 256),
    "liver": (1, 2, 448),
    "brain": (4, 3, 240),
}


def bench(model, x, iters, device):
    """Returns mean milliseconds per batch-1 forward pass."""
    with torch.no_grad():
        for _ in range(20):  # warm-up: cuDNN autotune, Mamba kernel JIT
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(iters):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--checkpoint_dir", default="checkpoints")
    ap.add_argument("--out", default="outputs/benchmark.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    results = {"device": name}
    for organ, (ic, nc, size) in ORGANS.items():
        model = BVINet(in_channels=ic, num_classes=nc, channels=[4, 8, 8, 16, 16], gcn_nodes=8).to(device).eval()
        ckpt = Path(args.checkpoint_dir) / f"bvi_net_{organ}_best.pt"
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location=device))
        ms = bench(model, torch.rand(1, ic, size, size, device=device), args.iters, device)
        results[organ] = {"input": f"{size}x{size}x{ic}", "ms": round(ms, 3), "fps": round(1000 / ms, 1),
                          "params": sum(p.numel() for p in model.parameters())}
        print(f"{organ:>5} {size}x{size}x{ic}: {ms:.2f} ms/img  ->  {1000 / ms:.1f} FPS on {name}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
