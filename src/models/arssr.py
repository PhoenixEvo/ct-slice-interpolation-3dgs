"""
ArSSR: Arbitrary-Scale Super-Resolution via Implicit Neural Representation.

Re-implementation based on Wu et al., "An Arbitrary Scale Super-Resolution
Approach for 3-Dimensional Magnetic Resonance Image Using Implicit Neural
Representation" (IEEE JBHI 2022), adapted for CT slice interpolation on
CT-ORG: only z-axis is downsampled, xy resolution is preserved.

Reference: https://github.com/iwuqing/ArSSR
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock3D(nn.Module):
    """Single RDB block (3D variant) used in RDN encoder."""

    def __init__(self, channels: int, growth: int, num_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                nn.Conv3d(channels + i * growth, growth, kernel_size=3, padding=1)
            )
        self.lff = nn.Conv3d(
            channels + num_layers * growth, channels, kernel_size=1, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [x]
        for conv in self.layers:
            y = F.relu(conv(torch.cat(feats, dim=1)), inplace=True)
            feats.append(y)
        out = self.lff(torch.cat(feats, dim=1))
        return out + x


class RDNEncoder3D(nn.Module):
    """Lightweight 3D RDN encoder that produces a feature volume.

    Output has same spatial shape as input (no spatial downsampling), so we
    can interpolate features at arbitrary continuous coordinates.
    """

    def __init__(
        self,
        in_channels: int = 1,
        feat_channels: int = 64,
        num_blocks: int = 4,
        growth: int = 32,
        num_layers: int = 4,
    ):
        super().__init__()
        self.feat_channels = feat_channels
        self.sfe1 = nn.Conv3d(in_channels, feat_channels, kernel_size=3, padding=1)
        self.sfe2 = nn.Conv3d(feat_channels, feat_channels, kernel_size=3, padding=1)

        self.blocks = nn.ModuleList(
            [ResidualDenseBlock3D(feat_channels, growth, num_layers)
             for _ in range(num_blocks)]
        )

        self.gff_1x1 = nn.Conv3d(feat_channels * num_blocks, feat_channels, kernel_size=1)
        self.gff_3x3 = nn.Conv3d(feat_channels, feat_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode LR volume (B, 1, D, H, W) -> feature (B, C, D, H, W)."""
        f_neg1 = self.sfe1(x)
        f_0 = self.sfe2(f_neg1)

        block_feats = []
        out = f_0
        for block in self.blocks:
            out = block(out)
            block_feats.append(out)

        gff = self.gff_3x3(self.gff_1x1(torch.cat(block_feats, dim=1)))
        return gff + f_neg1


class INRDecoder(nn.Module):
    """Coordinate-based MLP decoder.

    Given a feature vector sampled at a query coordinate plus a relative
    coordinate offset, outputs a scalar intensity in [0, 1] (sigmoid).
    """

    def __init__(self, feat_channels: int = 64, hidden: int = 256, num_layers: int = 5):
        super().__init__()
        # Input: feature (C) + relative coord (3)
        layers: List[nn.Module] = []
        in_dim = feat_channels + 3
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, feat: torch.Tensor, rel_coord: torch.Tensor) -> torch.Tensor:
        """feat: (N, C); rel_coord: (N, 3) in [-1, 1]; returns (N, 1)."""
        x = torch.cat([feat, rel_coord], dim=-1)
        return self.mlp(x)


class ArSSR(nn.Module):
    """Arbitrary-scale SR via INR.

    Volume convention (matches this repo): tensors shaped (B, 1, D, H, W)
    where D is z (through-plane), H is y, W is x. We only super-resolve
    along the D axis; H and W are preserved because CT-ORG in-plane
    resolution is already high.
    """

    def __init__(
        self,
        feat_channels: int = 64,
        num_blocks: int = 4,
        growth: int = 32,
        rdn_layers: int = 4,
        decoder_hidden: int = 256,
        decoder_layers: int = 5,
    ):
        super().__init__()
        self.encoder = RDNEncoder3D(
            in_channels=1,
            feat_channels=feat_channels,
            num_blocks=num_blocks,
            growth=growth,
            num_layers=rdn_layers,
        )
        self.decoder = INRDecoder(
            feat_channels=feat_channels,
            hidden=decoder_hidden,
            num_layers=decoder_layers,
        )
        self.feat_channels = feat_channels

    def encode(self, lr_volume: torch.Tensor) -> torch.Tensor:
        """lr_volume: (B, 1, D_lr, H, W) -> (B, C, D_lr, H, W)."""
        return self.encoder(lr_volume)

    def query(
        self,
        feat_vol: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """Query intensity at continuous coords.

        Args:
            feat_vol: (B, C, D_lr, H, W) feature volume from encoder.
            coords: (B, N, 3) normalized coords in [-1, 1], order (z, y, x)
                    matching the feature volume's (D, H, W) axes.

        Returns:
            (B, N, 1) predicted intensities in [0, 1].
        """
        B, C, D, H, W = feat_vol.shape
        N = coords.shape[1]

        # grid_sample expects (B, C, D, H, W) volumes and
        # grid of shape (B, D_out, H_out, W_out, 3) with coords (x, y, z).
        # We use (1, 1, N, 3) and feed (x, y, z). coords arg above is (z, y, x),
        # so flip to (x, y, z) for grid_sample.
        grid = coords.flip(-1).unsqueeze(1).unsqueeze(1)  # (B, 1, 1, N, 3)
        sampled = F.grid_sample(
            feat_vol, grid,
            mode="bilinear", padding_mode="border", align_corners=True,
        )  # (B, C, 1, 1, N)
        feat_at_coord = sampled.squeeze(2).squeeze(2).transpose(1, 2)  # (B, N, C)

        # Relative coord: here we pass the continuous coord itself (already
        # normalized). This gives the decoder a positional cue in addition to
        # bilinearly-interpolated features, which is sufficient for z-axis SR.
        rel = coords  # (B, N, 3)

        feat_flat = feat_at_coord.reshape(B * N, C)
        rel_flat = rel.reshape(B * N, 3)
        out = self.decoder(feat_flat, rel_flat)  # (B*N, 1)
        out = torch.sigmoid(out)
        return out.reshape(B, N, 1)

    def forward(
        self,
        lr_volume: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        feat = self.encode(lr_volume)
        return self.query(feat, coords)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
