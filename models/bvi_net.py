"""BVI-Net: full encoder-decoder assembling Local Pathway, Global Pathway (FA-VSSM),
and GCN-Attention skip connections (Fig. 1 of the paper).
"""
"""
BVI-Net
│
├── Encoder
│   ├── Local Pathway → Gabor
│   └── Global Pathway → FA-VSSM
│
├── Skip connections
│   └── GCN Attention
│
└── Decoder
    └── Upsampling + feature fusion
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gabor_conv import LocalPathwayBlock
from .fa_vssm import FAVSSM
from .gcn_attention import GCNAttention

"""
                 Input
                   │
           ┌───────┴───────┐
           ↓               ↓
      Local Pathway    Global Pathway
        (Gabor)          (FA-VSSM)
           ↓               ↓
     local features    global features
           └───────┬───────┘
                   ↓
              interaction
                   ↓
                 Pool
"""

# Local Pathway: first 2 stages simulate V1/V2 (combined cardinal+radial bias),
# last 3 stages simulate V4 (cardinal bias only) -- per Sec III-A.
STAGE_ROLES = ["v1v2", "v1v2", "v4", "v4", "v4"]
CHANNELS = [8, 16, 32, 64, 128]  # default encoder widths (overridden by --channels for the lightweight config)


class EncoderStage(nn.Module):
    """One resolution level: parallel Local + Global pathway, Global guides Local."""

    def __init__(self, in_ch, out_ch, stage_role):
        """Builds the Local Pathway block, the Global Pathway block, and pooling.

        Hyperparameters:
            in_ch (int): input channel width for this stage (output width of
                the previous stage, or 3 for the first stage's raw RGB input).
            out_ch (int): output channel width for this stage. Together the
                sequence of in_ch/out_ch across all 5 stages defines the
                model's channel-width schedule (e.g. 4-8-8-16-16), the
                single biggest lever on total parameter count.
            stage_role (str): "v1v2" or "v4" -- forwarded to LocalPathwayBlock
                to select which orientation-bias modulation the Gabor filters
                use at this depth (see gabor_conv.py).
        """
        super().__init__()
        self.local = LocalPathwayBlock(in_ch, out_ch, stage=stage_role)
        self.global_in = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.global_path = FAVSSM(out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        """Runs the Local and Global pathways in parallel, fuses them via
        gating, and downsamples for the next stage.

        Parameters:
            x (torch.Tensor): input feature map, shape (B, in_ch, H, W).

        Returns:
            tuple:
                down (torch.Tensor): pooled features, shape
                    (B, out_ch, H/2, W/2), passed to the next encoder stage.
                skip (torch.Tensor): pre-pooling fused features, shape
                    (B, out_ch, H, W), stored for the matching decoder stage's
                    skip connection.
        """
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
        """Builds the upsampling transpose-conv, the GCN-attention skip
        module, and the post-concatenation fusion conv.

        Hyperparameters:
            in_ch (int): channel width of the incoming (deeper) decoder
                feature map, before upsampling.
            skip_ch (int): channel width of the matching encoder skip
                connection at this resolution (equals in_ch by construction
                in BVINet, since each decoder stage's transpose-conv maps
                back to the same width as its skip).
            out_ch (int): channel width to project to after fusing upsampled
                + skip features -- becomes in_ch for the next (shallower)
                decoder stage.
            num_nodes (int, default=32): graph node count forwarded to
                GCNAttention (see gcn_attention.py) -- reduced to 8 in the
                final lightweight configuration.
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, skip_ch, kernel_size=2, stride=2)
        self.gcn_skip = GCNAttention(skip_ch, num_nodes=num_nodes)
        self.fuse = nn.Sequential(
            nn.Conv2d(skip_ch * 2, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x, skip):
        """Upsamples the deeper feature map, refines the skip connection via
        GCN attention, concatenates, and fuses with a conv block.

        Parameters:
            x (torch.Tensor): deeper decoder feature map, shape (B, in_ch, h, w).
            skip (torch.Tensor): matching encoder skip feature map, shape
                (B, skip_ch, H, W) where H,W is double h,w (one resolution
                level shallower).

        Returns:
            torch.Tensor of shape (B, out_ch, H, W): fused decoder features
            for this resolution level.
        """
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
        """Assembles the full 5-stage encoder-decoder.

        Hyperparameters (the ones actually swept in width_search.py / passed
        via --channels and --gcn_nodes on the command line):
            in_channels (int, default=3): input image channels (RGB).
            num_classes (int, default=1): output mask channels (1 = binary
                lesion/background segmentation).
            channels (list[int] or None, default=None): the 5 encoder-stage
                channel widths, e.g. [4, 8, 8, 16, 16] for the final
                lightweight config (27,312 total params) vs. the original
                [8, 16, 32, 64, 128] (777,584 params). This is the single
                largest lever on model size -- every width must be divisible
                by 4 (one per Gabor orientation). Falls back to module-level
                CHANNELS if not given.
            gcn_nodes (int, default=32): graph node count N used by every
                GCNAttention skip connection (see gcn_attention.py). Reduced
                to 8 in the final lightweight config -- second-largest lever
                on parameter count after `channels`.
        """
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
        """Runs the full encoder-decoder forward pass.

        Parameters:
            x (torch.Tensor): input image batch, shape (B, in_channels, H, W)
                -- H=W=256 throughout this project.

        Returns:
            torch.Tensor of shape (B, num_classes, H, W): predicted
            segmentation mask with values in (0, 1) (sigmoid-activated).
        """
        skips = []
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)

        skips = list(reversed(skips))  # deepest skip first, matches decoder order
        for dec, skip in zip(self.decoders, skips):
            x = dec(x, skip)

        out = self.head(x)  # already back at input resolution after 5 upsamples
        return torch.sigmoid(out)


"""
                          INPUT IMAGE
                              │
                              ↓
                    ┌──────────────────┐
                    │   ENCODER × 5    │
                    └──────────────────┘
                              │
             ┌────────────────┴────────────────┐
             ↓                                 ↓
      LOCAL PATHWAY                     GLOBAL PATHWAY
        Gabor Conv                         FA-VSSM
             │                                 │
             ↓                                 ↓
      Local / Edge                    Global / Context
       Features                           Features
             │                                 │
             └──────────────┬──────────────────┘
                            ↓
                    GLOBAL → LOCAL
                       GATING
                            │
                            ↓
                     FUSED FEATURES
                            │
                    ┌───────┴───────┐
                    ↓               ↓
               SKIP FEATURE       MaxPool
                    │               │
                    │               ↓
                    │          NEXT ENCODER
                    │               │
                    └───────────────┘
                            .
                            .
                     5 ENCODER STAGES
                            │
                            ↓
                       BOTTLENECK
                            │
                            ↓
                    ┌───────────────┐
                    │    DECODER    │
                    └───────────────┘
                            │
                    Upsampling
                            │
                            ↓
                  GCN ATTENTION
                    on skip feature
                            │
                            ↓
                  Feature Concatenation
                            │
                            ↓
                     Conv + BN + GELU
                            │
                            ↓
                     Repeat × 5
                            │
                            ↓
                   SEGMENTATION HEAD
                            │
                            ↓
                     LESION MASK
"""
