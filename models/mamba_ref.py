"""Pure-PyTorch drop-in for mamba_ssm.Mamba, for CPU inference.

A checkpoint trained with the real (CUDA) mamba_ssm on Kaggle stores Mamba
weights (in_proj, conv1d, x_proj, dt_proj, A_log, D, out_proj). fa_vssm's
SimpleSSMBranch fallback has different weights, so it can't load that
checkpoint. This module re-implements mamba_ssm's Mamba forward pass
(mamba_simple.py slow path + selective_scan_ref) with the SAME parameter
names and shapes, so a GPU-trained state_dict loads and runs on a machine
without mamba_ssm (e.g. the Windows backend).

Enabled by setting the env var BVI_MAMBA_REF=1 before importing models
(backend/app.py does this). Training still uses the real mamba_ssm.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _logcumsumexp_scan(log_decay, inp):
    """Computes h_t = sum_{s<=t} exp(cum_t - cum_s) * inp_s along dim 2, i.e.
    the linear recurrence h_t = exp(log_decay_t) * h_{t-1} + inp_t with h_0 = 0,
    in one vectorized pass instead of a Python loop over t.

    Done in log space (positive and negative parts of `inp` separately) so the
    decay products never under/overflow, even over 200K-step sequences.

    Parameters:
        log_decay (torch.Tensor): (B, D, L, N), log of the per-step decay
            (delta * A, always <= 0).
        inp (torch.Tensor): (B, D, L, N), per-step input (delta * B * u).

    Returns:
        torch.Tensor: (B, D, L, N) hidden states h_t.
    """
    cum = torch.cumsum(log_decay, dim=2)
    tiny = torch.finfo(inp.dtype).tiny
    pos = torch.logcumsumexp(torch.log(inp.clamp(min=0) + tiny) - cum, dim=2)
    neg = torch.logcumsumexp(torch.log((-inp).clamp(min=0) + tiny) - cum, dim=2)
    return torch.exp(cum + pos) - torch.exp(cum + neg)


def selective_scan(u, delta, A, B, C, D, z, delta_bias):
    """Same math as mamba_ssm.ops.selective_scan_interface.selective_scan_ref
    (delta_softplus=True, real A, input-dependent B/C of shape (B, N, L)).

    Returns:
        torch.Tensor: (B, D, L), same dtype as u.
    """
    dtype_in = u.dtype
    u, delta, B, C = u.double(), delta.double(), B.double(), C.double()
    delta = F.softplus(delta + delta_bias[..., None].double())
    A = A.double()
    log_decay = torch.einsum("bdl,dn->bdln", delta, A)
    deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
    h = _logcumsumexp_scan(log_decay, deltaB_u)
    y = torch.einsum("bdln,bnl->bdl", h, C)
    out = y + u * D.double()[:, None]
    out = out * F.silu(z.double())
    return out.to(dtype_in)


class Mamba(nn.Module):
    """Parameter-compatible stand-in for mamba_ssm.Mamba (see module docstring).

    Hyperparameters match mamba_ssm.Mamba's defaults: d_state=16, d_conv=4,
    expand=2, dt_rank=ceil(d_model/16). Initialization is not reproduced --
    this class is only meant for loading trained weights.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto",
                 conv_bias=True, bias=False, **_ignored):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                groups=self.d_inner, padding=d_conv - 1, bias=conv_bias)
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, hidden_states):
        """hidden_states: (B, L, d_model) -> (B, L, d_model)."""
        seqlen = hidden_states.shape[1]
        xz = self.in_proj(hidden_states).transpose(1, 2)  # (B, 2*d_inner, L)
        x, z = xz.chunk(2, dim=1)
        x = self.act(self.conv1d(x)[..., :seqlen])

        x_dbl = self.x_proj(x.transpose(1, 2))  # (B, L, dt_rank + 2*d_state)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = (dt @ self.dt_proj.weight.t()).transpose(1, 2)  # (B, d_inner, L), bias added in scan
        B = B.transpose(1, 2)  # (B, d_state, L)
        C = C.transpose(1, 2)

        A = -torch.exp(self.A_log.float())
        y = selective_scan(x, dt, A, B, C, self.D.float(), z, self.dt_proj.bias.float())
        return self.out_proj(y.transpose(1, 2))
