"""BVI-Net: full encoder-decoder assembling Local Pathway, Global Pathway (FA-VSSM),
and GCN-Attention skip connections (Fig. 1 of the paper).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gabor_conv import LocalPathwayBlock
from .fa_vssm import FAVSSM
from .gcn_attention import GCNAttention

# Local Pathway: first 2 stages simulate V1/V2 (combined cardinal+radial bias),
# last 3 stages simulate V4 (cardinal bias only) -- per Sec III-A.
STAGE_ROLES = ["v1v2", "v1v2", "v4", "v4", "v4"]
CHANNELS = [8, 16, 32, 64, 128]  # kept tiny to stay near the paper's 0.026M params


class EncoderStage(nn.Module):
    """One resolution level: parallel Local + Global pathway, Global guides Local."""

    def __init__(self, in_ch, out_ch, stage_role):
        super().__init__()
        self.local = LocalPathwayBlock(in_ch, out_ch, stage=stage_role)
        self.global_in = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.global_path = FAVSSM(out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        local_feat = self.local(x)                       # fine edge features
        global_feat = self.global_path(self.global_in(x))  # fast global context

        # Global Pathway guides Local Pathway to locate the lesion (multiplicative gating)
        gate = torch.sigmoid(global_feat)
        fused = local_feat * gate + local_feat             # residual-gated fusion

        skip = fused  # kept at this resolution for the skip connection
        down = self.pool(fused)
        return down, skip


class DecoderStage(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, num_nodes=32):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, skip_ch, kernel_size=2, stride=2)
        self.gcn_skip = GCNAttention(skip_ch, num_nodes=num_nodes)
        self.fuse = nn.Sequential(
            nn.Conv2d(skip_ch * 2, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip = self.gcn_skip(skip)  # multi-level feature fusion via graph convolution
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class BVINet(nn.Module):
    """Ultra-lightweight dual-pathway segmentation network. The paper reports 0.026M
    params; this replication keeps the same architectural philosophy (dual pathway +
    GCN skip) at a comparably tiny width, though the paper does not publish every
    exact layer dimension so widths here are a reasonable reconstruction, not a
    guaranteed byte-for-byte match.
    """

    def __init__(self, in_channels=3, num_classes=1, channels=None, gcn_nodes=32):
        super().__init__()
        channels = channels if channels is not None else CHANNELS  # override to shrink/grow the model
        chans = [in_channels] + channels
        self.encoders = nn.ModuleList([
            EncoderStage(chans[i], chans[i + 1], STAGE_ROLES[i]) for i in range(len(channels))
        ])

        # Decode from the bottleneck back up through all 5 skip levels (reverse order).
        rev = list(reversed(channels))
        decoder_out = rev[1:] + [rev[-1]]  # last stage keeps the smallest width for the head
        self.decoders = nn.ModuleList([
            DecoderStage(in_ch=rev[i], skip_ch=rev[i], out_ch=decoder_out[i], num_nodes=gcn_nodes)
            for i in range(len(rev))
        ])
        self.head = nn.Conv2d(decoder_out[-1], num_classes, kernel_size=1)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)

        skips = list(reversed(skips))  # deepest skip first, matches decoder order
        for dec, skip in zip(self.decoders, skips):
            x = dec(x, skip)

        out = self.head(x)  # already back at input resolution after 5 upsamples
        return torch.sigmoid(out)
