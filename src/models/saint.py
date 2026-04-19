"""
SAINT: Spatially Aware Interpolation NeTwork for Medical Slice Synthesis.

Re-implementation based on Peng et al., "SAINT: Spatially Aware Interpolation
NeTwork for Medical Slice Synthesis" (CVPR 2020).

Pipeline for a volume (H, W, D) with only z-axis downsampled (D_lr = D / R):
    1. Coronal branch (2D EDSR) does SR on each coronal slice
       (shape (D_lr, W) -> (D_hr, W)) for every y in [0, H).
    2. Sagittal branch does SR on each sagittal slice
       (shape (D_lr, H) -> (D_hr, H)) for every x in [0, W).
    3. Both produce full HR volume estimates of shape (H, W, D_hr).
    4. Fusion module fuses the two axial slice estimates at a target z into
       the final (H, W) prediction.

Volume reshape / fusion orchestration lives in `trainer_saint.py`.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """EDSR residual block: Conv -> ReLU -> Conv with residual scaling."""

    def __init__(self, channels: int, res_scale: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.conv1(x), inplace=True)
        y = self.conv2(y)
        return x + y * self.res_scale


class ZAxisUpsampler(nn.Module):
    """Upsample a 2D feature map only along the last spatial axis (z)."""

    def __init__(self, scale: int, channels: int):
        super().__init__()
        # Transposed conv targeting last axis only (kernel 1 along H).
        self.up = nn.ConvTranspose2d(
            channels, channels,
            kernel_size=(1, 2 * scale),
            stride=(1, scale),
            padding=(0, scale // 2),
        )
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


class EDSRZAxisSR(nn.Module):
    """EDSR-like 2D network that super-resolves only along the last axis.

    Input : (B, 1, L, Z_lr) -- L is either H (coronal) or W (sagittal).
    Output: (B, 1, L, Z_hr) with Z_hr = Z_lr * scale.
    """

    def __init__(
        self,
        scale: int = 2,
        num_feats: int = 64,
        num_blocks: int = 16,
        res_scale: float = 0.1,
    ):
        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(1, num_feats, kernel_size=3, padding=1)
        self.body = nn.Sequential(
            *[ResBlock(num_feats, res_scale) for _ in range(num_blocks)],
            nn.Conv2d(num_feats, num_feats, kernel_size=3, padding=1),
        )
        self.upsampler = ZAxisUpsampler(scale, num_feats)
        self.tail = nn.Conv2d(num_feats, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_z = x.shape[-1]
        h = self.head(x)
        b = self.body(h)
        h = h + b
        up = self.upsampler(h)
        out = self.tail(up)
        target_z = orig_z * self.scale
        if out.shape[-1] != target_z:
            out = F.interpolate(
                out, size=(out.shape[-2], target_z),
                mode="bilinear", align_corners=False,
            )
        return out


class FusionModule(nn.Module):
    """Fuse two HR axial-slice estimates into a single prediction."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, cor_est: torch.Tensor, sag_est: torch.Tensor) -> torch.Tensor:
        """cor_est, sag_est: (B, 1, H, W) -> (B, 1, H, W)."""
        x = torch.cat([cor_est, sag_est], dim=1)
        return self.net(x)


class SAINT(nn.Module):
    """Full SAINT model: two SR branches + fusion, trained per sparse ratio.

    The model exposes `coronal_net`, `sagittal_net`, and `fusion`. A plain
    forward call does fusion of two precomputed 2D axial estimates. Volume-
    level inference is done in `trainer_saint.py`.
    """

    def __init__(
        self,
        scale: int = 2,
        num_feats: int = 64,
        num_blocks: int = 16,
        res_scale: float = 0.1,
        fusion_hidden: int = 16,
    ):
        super().__init__()
        self.scale = scale
        self.coronal_net = EDSRZAxisSR(
            scale=scale, num_feats=num_feats,
            num_blocks=num_blocks, res_scale=res_scale,
        )
        self.sagittal_net = EDSRZAxisSR(
            scale=scale, num_feats=num_feats,
            num_blocks=num_blocks, res_scale=res_scale,
        )
        self.fusion = FusionModule(hidden=fusion_hidden)

    def forward(
        self, cor_axial: torch.Tensor, sag_axial: torch.Tensor
    ) -> torch.Tensor:
        """Fuse two axial-slice estimates.

        cor_axial, sag_axial: (B, 1, H, W) HR axial slices derived from the
            coronal and sagittal branches respectively.
        Returns fused prediction of the same shape.
        """
        return self.fusion(cor_axial, sag_axial)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_saint_from_config(cfg: dict) -> SAINT:
    """Build SAINT from a nested dict."""
    s = cfg.get("saint", {})
    return SAINT(
        scale=int(s.get("scale", 2)),
        num_feats=int(s.get("num_feats", 64)),
        num_blocks=int(s.get("num_blocks", 16)),
        res_scale=float(s.get("res_scale", 0.1)),
        fusion_hidden=int(s.get("fusion_hidden", 16)),
    )
