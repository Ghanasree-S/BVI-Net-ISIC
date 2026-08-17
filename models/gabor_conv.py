"""Gabor-initialized convolution for the Local Pathway (simulates V1/V2/V4 orientation-selective receptive fields).

True weight sharing: a single small kernel per orientation (4 x k x k params
total, independent of channel width) is applied depthwise to every input
channel, then a lightweight 1x1 pointwise conv mixes/projects to the output
width. This matches the paper's stated design ("weights of different
orientation kernels in the same layer are shared... achieve ultra-lightweight
requirement") -- our earlier version only used Gabor *initialization* while
still storing a full dense conv weight, which didn't actually save parameters.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def gabor_kernel(kernel_size, theta, sigma=2.0, lam=4.0, psi=0.0, gamma=0.5):
    """Builds a single 2D Gabor filter kernel used to initialise the Local
    Pathway's orientation-selective receptive fields (simulates V1 neurons).

    Parameters (hyperparameters of the filter shape):
        kernel_size (int): spatial size of the square kernel (k x k). Fixed at 5
            for all Local Pathway layers -- controls the receptive field size.
        theta (float): filter orientation in radians. One of {0, 45, 90, 135}
            degrees is used per orientation channel -- this is what makes the
            kernel "orientation-selective".
        sigma (float, default=2.0): standard deviation of the Gaussian envelope.
            Larger sigma = wider, smoother filter; controls how localized the
            edge response is.
        lam (float, default=4.0): wavelength of the sinusoidal carrier. Smaller
            lam = the filter responds to finer/higher-frequency edges.
        psi (float, default=0.0): phase offset of the sinusoid. 0 gives a
            symmetric (edge-detecting) filter.
        gamma (float, default=0.5): spatial aspect ratio. Values <1 elongate
            the filter along the preferred orientation (more edge-like,
            less blob-like).

    Returns:
        torch.Tensor of shape (kernel_size, kernel_size): the Gabor kernel.
    """
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
    """Depthwise Gabor filtering (shared per orientation, ~4*k*k params total)
    followed by a pointwise 1x1 projection to out_channels. The Cardinal/Radial
    orientation-bias model (Eq. 1-4 of the paper) is applied as a fixed scale
    on the pointwise projection's per-orientation-group output, so V1/V2 vs V4
    stages still get different orientation emphasis without adding parameters.
    """

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2,
                 orientations=(0, 45, 90, 135), stage="v1v2"):
        """Builds the shared Gabor kernels and the pointwise projection layer.

        Hyperparameters:
            in_channels (int): number of input feature channels. Comes from
                the previous encoder stage's output width.
            out_channels (int): number of output feature channels for this
                stage. Set by the model's encoder channel-width schedule
                (e.g. 4-8-8-16-16 in the final lightweight config).
            kernel_size (int, default=5): spatial size of the shared Gabor
                kernel -- larger values see more context per pixel but add a
                little compute (parameter count is unaffected since the
                kernel is shared, not per-channel).
            stride (int, default=1): convolution stride. Kept at 1 so spatial
                resolution is preserved within a stage (downsampling happens
                separately via MaxPool2d in EncoderStage).
            padding (int, default=2): zero-padding so output spatial size
                matches input size (same-padding for kernel_size=5).
            orientations (tuple, default=(0,45,90,135)): the four canonical
                orientations (degrees) simulating V1 orientation columns.
                This count also fixes how many "groups" the depthwise
                filtering produces per input channel.
            stage (str, default="v1v2"): either "v1v2" or "v4". Selects which
                Cardinal/Radial orientation-bias modulation to apply (Eq. 3-4
                of the paper) -- V1/V2 layers weight all orientations equally
                (+18%), V4 layers favor horizontal/vertical orientations more
                strongly (+26%), matching biological visual cortex data.
        """
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.orientations = orientations
        self.num_orient = len(orientations)
        self.in_channels = in_channels

        # shared depthwise kernel: one per orientation, reused across every input
        # channel -- this is the actual weight-sharing the paper describes.
        base = torch.stack([
            gabor_kernel(kernel_size, math.radians(o)) for o in orientations
        ])  # (num_orient, k, k)
        self.depthwise_weight = nn.Parameter(base.unsqueeze(1))  # (num_orient, 1, k, k), adapts during training

        # pointwise projection mixes the num_orient*in_channels filtered maps to out_channels
        self.pointwise = nn.Conv2d(in_channels * self.num_orient, out_channels, kernel_size=1)

        # Cardinal/Radial bias (Eq. 3-4): V1/V2 stages favor all 4 orientations
        # equally (18% modulation), V4 favors 0/90 more (26% modulation).
        mod = 0.18 if stage == "v1v2" else 0.26
        bias = torch.ones(self.num_orient)
        for i, o in enumerate(orientations):
            if stage == "v4":
                bias[i] = (1 + mod) if o in (0, 90) else (1 - mod)
            else:
                bias[i] = 1 + mod
        self.register_buffer("orientation_bias", bias)

    def forward(self, x):
        """Applies the shared Gabor depthwise filter then the pointwise projection.

        Parameters:
            x (torch.Tensor): input feature map, shape (B, in_channels, H, W).

        Returns:
            torch.Tensor of shape (B, out_channels, H, W): orientation-filtered,
            channel-mixed features.
        """
        b, c, h, w = x.shape
        # apply each of the 4 shared kernels to every input channel (grouped depthwise trick)
        weight = self.depthwise_weight.repeat(c, 1, 1, 1)  # (c*num_orient, 1, k, k), view of small param
        weight = weight * self.orientation_bias.repeat(c).view(-1, 1, 1, 1)
        x_rep = x.repeat_interleave(self.num_orient, dim=1)  # (B, c*num_orient, H, W)
        filtered = F.conv2d(x_rep, weight, stride=self.stride, padding=self.padding, groups=c * self.num_orient)
        return self.pointwise(filtered)


class LocalPathwayBlock(nn.Module):
    """One block of the Local Pathway: Gabor depthwise conv + pointwise + BN + activation."""

    def __init__(self, in_channels, out_channels, stage="v1v2"):
        """Wraps GaborConv2d with normalization and activation.

        Hyperparameters:
            in_channels (int): input feature channel count for this stage.
            out_channels (int): output feature channel count for this stage.
            stage (str, default="v1v2"): "v1v2" or "v4" -- forwarded to
                GaborConv2d to select the orientation-bias modulation
                (see GaborConv2d.__init__ for details).
        """
        super().__init__()
        self.conv = GaborConv2d(in_channels, out_channels, stage=stage)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        """Runs Gabor conv -> BatchNorm -> GELU activation.

        Parameters:
            x (torch.Tensor): input feature map, shape (B, in_channels, H, W).

        Returns:
            torch.Tensor of shape (B, out_channels, H, W): activated local
            (edge/detail) features for this encoder stage.
        """
        return self.act(self.bn(self.conv(x)))


"""
                    LOCAL PATHWAY
                          │
                          ↓
                  Input feature map
                          │
                          ↓
              ┌─────────────────────┐
              │   4 Gabor Filters   │
              │                     │
              │  0°  45°  90°  135°│
              └─────────────────────┘
                          │
                          ↓
             Orientation-specific
                    features
                          │
                          ↓
                 1 × 1 Conv
                          │
                          ↓
                Channel mixing
                          │
                          ↓
                    BN + GELU
                          │
                          ↓
                  Local features
"""
