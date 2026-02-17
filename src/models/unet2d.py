"""
U-Net 2D baseline model for slice interpolation.
Input: 2 adjacent observed slices -> Output: interpolated middle slice.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DoubleConv(nn.Module):
    """Double convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet2D(nn.Module):
    """U-Net 2D for slice interpolation.

    Takes 2 input slices (concatenated as channels) and outputs 1 slice.
    Architecture: 4-level encoder-decoder with skip connections.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        features: List[int] = None,
    ):
        """Initialize U-Net.

        Args:
            in_channels: Number of input channels (2 for adjacent slices).
            out_channels: Number of output channels (1 for single slice).
            features: List of feature dimensions for each level.
        """
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        self.upconvs = nn.ModuleList()

        # Encoder path
        prev_channels = in_channels
        for feat in features:
            self.encoders.append(DoubleConv(prev_channels, feat))
            prev_channels = feat

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder path
        reversed_features = list(reversed(features))
        prev_channels = features[-1] * 2
        for feat in reversed_features:
            self.upconvs.append(
                nn.ConvTranspose2d(prev_channels, feat, 2, stride=2)
            )
            self.decoders.append(DoubleConv(feat * 2, feat))
            prev_channels = feat

        # Final output
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, 2, H, W) - two adjacent slices.

        Returns:
            Output tensor (B, 1, H, W) - interpolated slice.
        """
        # Pad to ensure divisible by 2^num_levels
        orig_h, orig_w = x.shape[2], x.shape[3]
        pad_h = (16 - orig_h % 16) % 16
        pad_w = (16 - orig_w % 16) % 16
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        # Encoder
        skip_connections = []
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skip_connections = list(reversed(skip_connections))
        for i, (upconv, decoder) in enumerate(
            zip(self.upconvs, self.decoders)
        ):
            x = upconv(x)
            skip = skip_connections[i]

            # Handle size mismatch
            if x.shape != skip.shape:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=False
                )

            x = torch.cat([skip, x], dim=1)
            x = decoder(x)

        x = self.final_conv(x)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :orig_h, :orig_w]

        return x

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
