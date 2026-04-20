"""
Tri-plane Implicit Neural Representation for per-volume CT reconstruction.

Represents a 3D CT volume as three 2D learnable feature grids on axis-aligned
planes (xy, xz, yz) + a small MLP decoder. For a 3D query (x, y, z), features
are bilinearly sampled from each plane and summed, then decoded to intensity.

Design rationale:
- Self-supervised per-volume training (similar to 3DGS setup), no external
  training data required.
- Explicitly exploits x-y correlation via the xy feature grid, which cubic
  along z cannot capture.
- Much smaller than full 3D voxel grid and more expressive than pure MLP
  (NeRF-style) thanks to the feature planes.

Reference: EG3D (Chan et al. 2022), K-Planes (Fridovich-Keil et al. 2023).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TriPlaneINR(nn.Module):
    """Tri-plane feature grids + MLP decoder for CT volume representation.

    Parameters
    ----------
    volume_shape : (H, W, D)
        Physical volume extent in voxel units. Query coordinates are
        expected in this range and are normalized to [-1, 1] internally.
    feat_dim : int
        Number of feature channels on each plane.
    plane_res : tuple of (res_xy, res_xz, res_yz)
        Spatial resolution of each feature plane. Can be lower than the
        actual volume resolution; bilinear sampling handles interpolation.
    mlp_hidden : int
        Width of MLP hidden layers.
    mlp_layers : int
        Number of MLP hidden layers (excluding input/output).
    num_freq : int
        Number of positional-encoding frequencies per coordinate
        (0 = disabled).
    use_residual_base : bool
        If True, the output is interpreted as a residual to be added to an
        external base (e.g. cubic). Intensity head is zero-initialized so
        initial residual is approximately zero.
    """

    def __init__(
        self,
        volume_shape: Tuple[int, int, int],
        feat_dim: int = 32,
        plane_res: Tuple[int, int, int] = (128, 128, 128),
        mlp_hidden: int = 64,
        mlp_layers: int = 3,
        num_freq: int = 4,
        use_residual_base: bool = True,
    ) -> None:
        super().__init__()
        self.volume_shape = volume_shape
        self.feat_dim = feat_dim
        self.num_freq = num_freq
        self.use_residual_base = use_residual_base

        H, W, D = volume_shape
        res_xy, res_xz, res_yz = plane_res

        # Each plane: (1, feat_dim, res_a, res_b)
        # xy plane uses (H, W) axes, xz uses (H, D), yz uses (W, D).
        init_scale = 0.1
        self.plane_xy = nn.Parameter(
            torch.randn(1, feat_dim, res_xy, res_xy) * init_scale
        )
        self.plane_xz = nn.Parameter(
            torch.randn(1, feat_dim, res_xz, res_xz) * init_scale
        )
        self.plane_yz = nn.Parameter(
            torch.randn(1, feat_dim, res_yz, res_yz) * init_scale
        )

        # Decoder MLP
        in_dim = feat_dim
        if num_freq > 0:
            # PE adds 2 * 3 * num_freq extra features (sin/cos per freq per axis)
            in_dim += 2 * 3 * num_freq

        layers = []
        prev = in_dim
        for _ in range(mlp_layers):
            layers.append(nn.Linear(prev, mlp_hidden))
            layers.append(nn.GELU())
            prev = mlp_hidden
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)

        if use_residual_base:
            # Zero-init last layer so initial output is ~0 (pure base)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def _positional_encoding(self, coords_norm: torch.Tensor) -> torch.Tensor:
        """Apply NeRF-style sin/cos positional encoding.

        Args:
            coords_norm: (N, 3) in [-1, 1].

        Returns:
            Encoded features (N, 2*3*num_freq).
        """
        if self.num_freq <= 0:
            return coords_norm.new_zeros(coords_norm.shape[0], 0)
        freqs = (2.0 ** torch.arange(self.num_freq, device=coords_norm.device)) * math.pi
        # (N, 3, num_freq)
        scaled = coords_norm.unsqueeze(-1) * freqs.view(1, 1, -1)
        pe = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        return pe.reshape(coords_norm.shape[0], -1)

    def _normalize_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """Map voxel coords (x in [0,H), y in [0,W), z in [0,D)) to [-1, 1]."""
        H, W, D = self.volume_shape
        sizes = coords.new_tensor([H, W, D], dtype=coords.dtype)
        # Normalize to [-1, 1] (grid_sample convention)
        return 2.0 * (coords / (sizes - 1).clamp(min=1)) - 1.0

    @staticmethod
    def _sample_plane(
        plane: torch.Tensor, coords_2d: torch.Tensor
    ) -> torch.Tensor:
        """Bilinear sample a 2D feature plane at given normalized coords.

        Args:
            plane: (1, C, H_p, W_p).
            coords_2d: (N, 2) in [-1, 1] with order (x, y).

        Returns:
            (N, C) features.
        """
        # grid_sample expects grid of shape (N, H_out, W_out, 2) with
        # (x, y) in last dim. Fold N into H_out dim.
        N = coords_2d.shape[0]
        grid = coords_2d.view(1, N, 1, 2)
        sampled = F.grid_sample(
            plane, grid, mode="bilinear",
            padding_mode="border", align_corners=True,
        )
        return sampled.squeeze(-1).squeeze(0).transpose(0, 1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Query the tri-plane representation at arbitrary 3D coordinates.

        Args:
            coords: (N, 3) in voxel units (x in [0,H), y in [0,W), z in [0,D)).

        Returns:
            Predicted intensity residual (N,) -- scalar per point.
        """
        coords_norm = self._normalize_coords(coords)
        xn, yn, zn = coords_norm[:, 0], coords_norm[:, 1], coords_norm[:, 2]

        # Build 2D coords for each plane. Convention: grid_sample's x=W, y=H
        # axis. We keep (first_axis, second_axis) order consistent with how
        # plane_* tensors were allocated.
        coords_xy = torch.stack([yn, xn], dim=-1)  # plane_xy: (H,W), grid x=W
        coords_xz = torch.stack([zn, xn], dim=-1)  # plane_xz: (H,D), grid x=D
        coords_yz = torch.stack([zn, yn], dim=-1)  # plane_yz: (W,D), grid x=D

        feat_xy = self._sample_plane(self.plane_xy, coords_xy)
        feat_xz = self._sample_plane(self.plane_xz, coords_xz)
        feat_yz = self._sample_plane(self.plane_yz, coords_yz)

        feat = feat_xy + feat_xz + feat_yz  # (N, C)

        pe = self._positional_encoding(coords_norm)
        if pe.shape[-1] > 0:
            feat = torch.cat([feat, pe], dim=-1)

        h = self.mlp(feat)
        out = self.head(h).squeeze(-1)
        return out

    @torch.no_grad()
    def render_slice(
        self,
        z_value: float,
        image_height: Optional[int] = None,
        image_width: Optional[int] = None,
        base_slice: Optional[torch.Tensor] = None,
        chunk: int = 65536,
    ) -> torch.Tensor:
        """Render a full 2D slice at given z.

        Args:
            z_value: z-coordinate in voxel units.
            image_height, image_width: output resolution. Defaults to
                volume shape.
            base_slice: optional (1, H, W) tensor to add to the INR
                residual output (used in residual mode).
            chunk: number of points per decoding chunk (memory control).

        Returns:
            Slice tensor (1, H, W) on the same device as the model.
        """
        device = self.plane_xy.device
        H, W, _ = self.volume_shape
        H_out = image_height or H
        W_out = image_width or W

        ys = torch.arange(H_out, device=device, dtype=torch.float32)
        xs = torch.arange(W_out, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        zz = torch.full_like(xx, float(z_value))
        coords = torch.stack([yy, xx, zz], dim=-1).reshape(-1, 3)

        outputs = []
        for start in range(0, coords.shape[0], chunk):
            end = min(start + chunk, coords.shape[0])
            outputs.append(self.forward(coords[start:end]))
        slice_flat = torch.cat(outputs, dim=0)
        slice_img = slice_flat.view(1, H_out, W_out)

        if base_slice is not None:
            slice_img = slice_img + base_slice
        return slice_img
