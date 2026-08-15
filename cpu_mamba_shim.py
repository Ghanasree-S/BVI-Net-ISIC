"""Import this before anything that builds a BVINet, when running on a
machine without an NVIDIA GPU. Patches mamba_ssm's Mamba module to use its
pure-PyTorch reference ops (selective_scan_ref) instead of the CUDA-only
kernels, so the real mamba_ssm module runs on CPU instead of falling back
to fa_vssm's conv approximation. Much slower than the CUDA path -- only
meant for local sanity checks, not real training runs.
"""
import mamba_ssm.modules.mamba_simple as _ms
from mamba_ssm.ops.selective_scan_interface import selective_scan_ref as _selective_scan_ref

_ms.causal_conv1d_fn = None
_ms.causal_conv1d_update = None
_ms.selective_scan_fn = _selective_scan_ref
