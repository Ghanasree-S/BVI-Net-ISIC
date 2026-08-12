"""Gabor-initialized convolution for the Local Pathway (simulates V1/V2/V4 orientation-selective receptive fields)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def gabor_kernel(kernel_size, theta, sigma=2.0, lam=4.0, psi=0.0, gamma=0.5):
    half = kernel_size // 2
    y, x = torch.meshgrid(
        torch.arange(-half, half + 1, dtype=torch.float32),
        torch.arange(-half, half + 1, dtype=torch.float32),
        indexing="ij",
    )
    x_theta = x * math.cos(theta) + y * math.sin(theta)
    y_theta = -x * math.sin(theta) + y * math.cos(theta)
    gb = torch.exp(-(x_theta ** 2 + gamma ** 2 * y_theta ** 2) / (2 * sigma ** 2))
    gb = gb * torch.cos(2 * math.pi * x_theta / lam + psi)
    return gb


class GaborConv2d(nn.Module):
    """Convolution layer whose kernel is initialized by a Gabor function fused with
    the Cardinal/Radial orientation-bias model described in the paper (Eq. 1-4).
    Weights across the four canonical orientations {0,45,90,135} are shared to keep
    the layer ultra-lightweight, then fine-tuned adaptively during training.
    """

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2,
                 orientations=(0, 45, 90, 135), stage="v1v2"):
        super().__init__()
        assert out_channels % len(orientations) == 0, \
            "out_channels must be divisible by number of orientations for weight sharing"
        self.stride = stride
        self.padding = padding
        self.orientations = orientations
        self.groups = len(orientations)
        self.out_per_group = out_channels // self.groups

        # one shared base kernel per orientation, applied to every input channel group
        base = torch.stack([
            gabor_kernel(kernel_size, math.radians(o)) for o in orientations
        ])  # (num_orientations, k, k)
        weight = base.unsqueeze(1).unsqueeze(1)  # (num_orientations, 1, 1, k, k)
        weight = weight.repeat(1, self.out_per_group, in_channels, 1, 1)
        weight = weight.reshape(out_channels, in_channels, kernel_size, kernel_size)

        # V1/V2 uses combined cardinal+radial bias (all 4 orientations equally likely,
        # modulation 18%); V4 uses cardinal-only bias (0/90 weighted higher, modulation 26%).
        mod = 0.18 if stage == "v1v2" else 0.26
        bias_scale = torch.ones(out_channels)
        if stage == "v4":
            for i, o in enumerate(orientations):
                start, end = i * self.out_per_group, (i + 1) * self.out_per_group
                bias_scale[start:end] = (1 + mod) if o in (0, 90) else (1 - mod)
        else:
            for i, o in enumerate(orientations):
                start, end = i * self.out_per_group, (i + 1) * self.out_per_group
                bias_scale[start:end] = 1 + mod if o in (0, 45, 90, 135) else 1
        weight = weight * bias_scale.view(-1, 1, 1, 1)

        self.weight = nn.Parameter(weight)  # adapts during training (not frozen)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)


class LocalPathwayBlock(nn.Module):
    """One block of the Local Pathway: Gabor-init conv + BN + activation."""

    def __init__(self, in_channels, out_channels, stage="v1v2"):
        super().__init__()
        self.conv = GaborConv2d(in_channels, out_channels, stage=stage)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
