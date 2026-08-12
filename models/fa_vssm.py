"""Global Pathway: Fast-Attention Visual State Space Module (FA-VSSM).

Tries to use the real Mamba selective-scan (mamba_ssm) for the state-space channel.
If that package is not installed/buildable (common on Colab), falls back to a
depthwise-conv + gated-linear-unit approximation of a selective SSM. This is a
documented deviation for feasibility, not a silent substitution -- see README.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except Exception:
    HAS_MAMBA = False


def _scan_order(h, w, mode):
    """Return a permutation of flattened pixel indices for the given scan mode."""
    idx = torch.arange(h * w).view(h, w)
    if mode == "line":
        return idx.flatten()
    if mode == "column":
        return idx.t().flatten()
    if mode == "zcurve":
        # simple bit-interleaving Z-order (only exact for power-of-two dims)
        ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        z = torch.zeros_like(ys)
        for b in range(max(h, w).bit_length()):
            z |= ((xs >> b) & 1) << (2 * b)
            z |= ((ys >> b) & 1) << (2 * b + 1)
        order = torch.argsort(z.flatten())
        return order
    if mode == "hilbert":
        # fallback: hilbert curve approximation via recursive index (small grids only)
        return _hilbert_order(h, w)
    raise ValueError(mode)


def _hilbert_order(h, w):
    n = max(h, w)
    n = 1 << (n - 1).bit_length()

    def d2xy(n, d):
        x = y = 0
        t = d
        s = 1
        while s < n:
            rx = 1 & (t // 2)
            ry = 1 & (t ^ rx)
            if ry == 0:
                if rx == 1:
                    x, y = s - 1 - x, s - 1 - y
                x, y = y, x
            x += s * rx
            y += s * ry
            t //= 4
            s *= 2
        return x, y

    pts = [d2xy(n, d) for d in range(n * n)]
    order = [y * w + x for x, y in pts if x < w and y < h]
    return torch.tensor(order, dtype=torch.long)


class SimpleSSMBranch(nn.Module):
    """Lightweight selective-scan approximation used when mamba_ssm is unavailable:
    depthwise causal conv (captures local sequence context) gated by a linear unit,
    applied along the scanned pixel order.
    """

    def __init__(self, dim, kernel_size=7):
        super().__init__()
        self.dw = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size - 1, groups=dim)
        self.gate = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, seq):  # seq: (B, L, C)
        x = seq.transpose(1, 2)  # (B, C, L)
        x = self.dw(x)[..., : seq.size(1)]
        x = x.transpose(1, 2)  # (B, L, C)
        g = torch.sigmoid(self.gate(seq))
        return self.out(x * g)


class StateSpaceChannel(nn.Module):
    """Scans the feature map 4 ways (line/column/Z-curve/Hilbert), runs an SSM along
    each order, un-scans and averages the results (Sec. III-B of the paper).
    """

    MODES = ("line", "column", "zcurve", "hilbert")

    def __init__(self, dim):
        super().__init__()
        if HAS_MAMBA:
            self.ssm = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        else:
            self.ssm = SimpleSSMBranch(dim)
        self._order_cache = {}

    def _get_order(self, h, w, mode, device):
        key = (h, w, mode)
        if key not in self._order_cache:
            self._order_cache[key] = _scan_order(h, w, mode).to(device)
        return self._order_cache[key]

    def forward(self, x):  # x: (B, C, H, W)
        b, c, h, w = x.shape
        flat = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        outs = []
        for mode in self.MODES:
            order = self._get_order(h, w, mode, x.device)
            scanned = flat[:, order, :]
            processed = self.ssm(scanned)
            unscanned = torch.zeros_like(processed)
            unscanned[:, order, :] = processed
            outs.append(unscanned)
        merged = torch.stack(outs, dim=0).mean(0)  # average the 4 scan directions
        return merged.transpose(1, 2).view(b, c, h, w)


class FastAttentionChannel(nn.Module):
    """Eq. (5): adaptive channel attention from avg+max pooled features."""

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return attn


class FAVSSM(nn.Module):
    """Full Global Pathway block: fast-attention channel gates the state-space channel."""

    def __init__(self, dim):
        super().__init__()
        self.fast_attn = FastAttentionChannel(dim)
        self.ssm_channel = StateSpaceChannel(dim)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        attn = self.fast_attn(x)
        ssm_out = self.ssm_channel(x)
        return self.norm(x + attn * ssm_out)
