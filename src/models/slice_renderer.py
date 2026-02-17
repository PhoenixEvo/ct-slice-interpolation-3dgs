"""
Differentiable Slice Renderer for 3D Gaussian Splatting on CT volumes.

Renders axis-aligned 2D slices from a set of 3D Gaussians.
Optimized with tile-based culling and z-threshold filtering.
All operations are pure PyTorch for automatic differentiation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SliceRenderer(nn.Module):
    """Differentiable renderer that produces 2D slices from 3D Gaussians.

    For a target slice at z-position z_t, each Gaussian contributes:
        w_z = exp(-0.5 * ((z_t - mu_z) / sigma_z)^2)
        G_2d(x,y) = exp(-0.5 * [((x-mu_x)/sigma_x)^2 + ((y-mu_y)/sigma_y)^2])
        contribution = intensity * opacity * w_z * G_2d(x,y)

    Uses alpha compositing or weighted summation to combine contributions.
    """

    def __init__(
        self,
        image_height: int,
        image_width: int,
        tile_size: int = 16,
        z_threshold: float = 3.0,
        render_mode: str = "alpha",
    ):
        """Initialize slice renderer.

        Args:
            image_height: Height of the output slice in pixels.
            image_width: Width of the output slice in pixels.
            tile_size: Tile size for tiled rendering optimization.
            z_threshold: Only include Gaussians within z_threshold * sigma_z.
            render_mode: 'alpha' for alpha compositing, 'weighted' for weighted sum.
        """
        super().__init__()
        self.H = image_height
        self.W = image_width
        self.tile_size = tile_size
        self.z_threshold = z_threshold
        self.render_mode = render_mode

        # Pre-compute pixel coordinate grids
        y_coords = torch.arange(image_height, dtype=torch.float32)
        x_coords = torch.arange(image_width, dtype=torch.float32)
        # grid_y: (H, W), grid_x: (H, W)
        self.register_buffer(
            "grid_y", y_coords.unsqueeze(1).expand(image_height, image_width)
        )
        self.register_buffer(
            "grid_x", x_coords.unsqueeze(0).expand(image_height, image_width)
        )

    def forward(
        self,
        positions: torch.Tensor,
        scales: torch.Tensor,
        opacity: torch.Tensor,
        intensity: torch.Tensor,
        z_target: float,
    ) -> torch.Tensor:
        """Render a 2D slice at the given z-position.

        Args:
            positions: Gaussian centers (N, 3) - (x, y, z).
            scales: Gaussian scales (N, 3) - (sx, sy, sz), positive.
            opacity: Gaussian opacities (N,) in [0, 1].
            intensity: Gaussian intensities (N,).
            z_target: Z-position of the target slice.

        Returns:
            Rendered slice (1, H, W).
        """
        # Step 1: Z-filtering - only keep Gaussians near the target slice
        z_dist = (z_target - positions[:, 2]) / (scales[:, 2] + 1e-8)
        z_mask = torch.abs(z_dist) < self.z_threshold
        
        if z_mask.sum() == 0:
            # No Gaussians contribute - return zeros
            return torch.zeros(
                1, self.H, self.W,
                device=positions.device, dtype=positions.dtype
            )

        # Filter to active Gaussians
        pos_active = positions[z_mask]       # (K, 3)
        scale_active = scales[z_mask]        # (K, 3)
        opacity_active = opacity[z_mask]     # (K,)
        intensity_active = intensity[z_mask] # (K,)

        # Step 2: Compute z-weights
        z_diff = z_target - pos_active[:, 2]  # (K,)
        z_weight = torch.exp(
            -0.5 * (z_diff / (scale_active[:, 2] + 1e-8)) ** 2
        )  # (K,)

        # Effective per-Gaussian weight
        eff_weight = opacity_active * z_weight  # (K,)

        # Step 3: Render using tile-based approach for memory efficiency
        if self.H * self.W * pos_active.shape[0] > 50_000_000:
            # Use tiled rendering for large volumes
            return self._render_tiled(
                pos_active, scale_active, eff_weight, intensity_active
            )
        else:
            # Direct rendering for small cases
            return self._render_direct(
                pos_active, scale_active, eff_weight, intensity_active
            )

    def _render_direct(
        self,
        positions: torch.Tensor,
        scales: torch.Tensor,
        weights: torch.Tensor,
        intensities: torch.Tensor,
    ) -> torch.Tensor:
        """Direct (non-tiled) rendering for small number of Gaussians.

        Args:
            positions: Active Gaussian positions (K, 3).
            scales: Active Gaussian scales (K, 3).
            weights: Effective weights (K,) = opacity * z_weight.
            intensities: Gaussian intensities (K,).

        Returns:
            Rendered slice (1, H, W).
        """
        K = positions.shape[0]

        # Gaussian centers in x, y
        mu_x = positions[:, 0]  # (K,)
        mu_y = positions[:, 1]  # (K,)
        sigma_x = scales[:, 0]  # (K,)
        sigma_y = scales[:, 1]  # (K,)

        # Compute 2D Gaussian for each pixel
        # grid_y: (H, W), grid_x: (H, W)
        # Expand for broadcasting: (H, W, 1) vs (1, 1, K)
        dx = self.grid_y.unsqueeze(-1) - mu_x.view(1, 1, K)  # (H, W, K)
        dy = self.grid_x.unsqueeze(-1) - mu_y.view(1, 1, K)  # (H, W, K)

        # Normalized squared distances
        dx_norm = dx / (sigma_x.view(1, 1, K) + 1e-8)
        dy_norm = dy / (sigma_y.view(1, 1, K) + 1e-8)

        # 2D Gaussian values
        gauss_2d = torch.exp(-0.5 * (dx_norm ** 2 + dy_norm ** 2))  # (H, W, K)

        if self.render_mode == "alpha":
            return self._alpha_composite(
                gauss_2d, weights, intensities
            )
        else:
            return self._weighted_sum(
                gauss_2d, weights, intensities
            )

    def _render_tiled(
        self,
        positions: torch.Tensor,
        scales: torch.Tensor,
        weights: torch.Tensor,
        intensities: torch.Tensor,
    ) -> torch.Tensor:
        """Tile-based rendering for memory efficiency with many Gaussians.

        Divides the image into tiles and only processes Gaussians that
        contribute to each tile.

        Args:
            positions: Active Gaussian positions (K, 3).
            scales: Active Gaussian scales (K, 3).
            weights: Effective weights (K,).
            intensities: Gaussian intensities (K,).

        Returns:
            Rendered slice (1, H, W).
        """
        T = self.tile_size
        num_tiles_h = math.ceil(self.H / T)
        num_tiles_w = math.ceil(self.W / T)

        output = torch.zeros(
            self.H, self.W,
            device=positions.device, dtype=positions.dtype
        )

        mu_x = positions[:, 0]
        mu_y = positions[:, 1]
        sigma_x = scales[:, 0]
        sigma_y = scales[:, 1]

        for ti in range(num_tiles_h):
            for tj in range(num_tiles_w):
                # Tile boundaries
                r_start = ti * T
                r_end = min((ti + 1) * T, self.H)
                c_start = tj * T
                c_end = min((tj + 1) * T, self.W)

                # Filter Gaussians that could affect this tile
                # A Gaussian affects the tile if its center is within
                # 3*sigma of the tile boundaries
                tile_mask = (
                    (mu_x + 3 * sigma_x >= r_start)
                    & (mu_x - 3 * sigma_x < r_end)
                    & (mu_y + 3 * sigma_y >= c_start)
                    & (mu_y - 3 * sigma_y < c_end)
                )

                if tile_mask.sum() == 0:
                    continue

                # Get tile-local Gaussians
                t_pos = positions[tile_mask]
                t_scales = scales[tile_mask]
                t_weights = weights[tile_mask]
                t_intensities = intensities[tile_mask]
                K_t = t_pos.shape[0]

                # Tile pixel coordinates
                tile_grid_y = self.grid_y[r_start:r_end, c_start:c_end]  # (th, tw)
                tile_grid_x = self.grid_x[r_start:r_end, c_start:c_end]

                th = r_end - r_start
                tw = c_end - c_start

                # Compute 2D Gaussian
                dx = tile_grid_y.unsqueeze(-1) - t_pos[:, 0].view(1, 1, K_t)
                dy = tile_grid_x.unsqueeze(-1) - t_pos[:, 1].view(1, 1, K_t)

                dx_norm = dx / (t_scales[:, 0].view(1, 1, K_t) + 1e-8)
                dy_norm = dy / (t_scales[:, 1].view(1, 1, K_t) + 1e-8)

                gauss_2d = torch.exp(-0.5 * (dx_norm ** 2 + dy_norm ** 2))

                if self.render_mode == "alpha":
                    tile_result = self._alpha_composite(
                        gauss_2d, t_weights, t_intensities
                    )
                    output[r_start:r_end, c_start:c_end] = tile_result.squeeze(0)
                else:
                    tile_result = self._weighted_sum(
                        gauss_2d, t_weights, t_intensities
                    )
                    output[r_start:r_end, c_start:c_end] = tile_result.squeeze(0)

        return output.unsqueeze(0)

    @staticmethod
    def _alpha_composite(
        gauss_2d: torch.Tensor,
        weights: torch.Tensor,
        intensities: torch.Tensor,
    ) -> torch.Tensor:
        """Alpha compositing of Gaussian contributions.

        Approximates front-to-back alpha compositing by sorting by weight
        and accumulating.

        Args:
            gauss_2d: 2D Gaussian values (H, W, K) or (th, tw, K).
            weights: Per-Gaussian weights (K,).
            intensities: Per-Gaussian intensities (K,).

        Returns:
            Composited image (1, H, W) or (1, th, tw).
        """
        # Alpha for each Gaussian at each pixel
        alpha = gauss_2d * weights.view(1, 1, -1)  # (H, W, K)
        alpha = torch.clamp(alpha, 0.0, 0.99)

        # Sort by weight (descending) for consistent compositing
        sort_idx = torch.argsort(weights, descending=True)
        alpha = alpha[:, :, sort_idx]
        sorted_intensities = intensities[sort_idx]

        # Front-to-back compositing
        # T_i = prod(1 - alpha_j, j < i)
        one_minus_alpha = 1.0 - alpha  # (H, W, K)
        # Cumulative product of (1-alpha) for transmittance
        # Shift by 1: transmittance for first Gaussian is 1
        transmittance = torch.cumprod(one_minus_alpha, dim=-1)
        # Shift: T_0 = 1, T_1 = (1-a_0), T_2 = (1-a_0)(1-a_1), ...
        transmittance = torch.cat(
            [
                torch.ones_like(transmittance[:, :, :1]),
                transmittance[:, :, :-1],
            ],
            dim=-1,
        )

        # Weighted sum: C = sum(T_i * alpha_i * c_i)
        contribution = (
            transmittance * alpha * sorted_intensities.view(1, 1, -1)
        )
        rendered = contribution.sum(dim=-1)  # (H, W)

        return rendered.unsqueeze(0)

    @staticmethod
    def _weighted_sum(
        gauss_2d: torch.Tensor,
        weights: torch.Tensor,
        intensities: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted sum of Gaussian contributions (normalized).

        Args:
            gauss_2d: 2D Gaussian values (H, W, K).
            weights: Per-Gaussian weights (K,).
            intensities: Per-Gaussian intensities (K,).

        Returns:
            Rendered image (1, H, W).
        """
        # Weight each Gaussian
        weighted = gauss_2d * weights.view(1, 1, -1)  # (H, W, K)

        # Intensity-weighted sum
        numerator = (weighted * intensities.view(1, 1, -1)).sum(dim=-1)
        denominator = weighted.sum(dim=-1) + 1e-8

        rendered = numerator / denominator  # (H, W)
        return rendered.unsqueeze(0)

    def render_volume_slices(
        self,
        positions: torch.Tensor,
        scales: torch.Tensor,
        opacity: torch.Tensor,
        intensity: torch.Tensor,
        z_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Render multiple slices at given z-positions.

        Args:
            positions: Gaussian centers (N, 3).
            scales: Gaussian scales (N, 3).
            opacity: Gaussian opacities (N,).
            intensity: Gaussian intensities (N,).
            z_indices: Z-positions to render at (M,).

        Returns:
            Rendered slices (M, 1, H, W).
        """
        slices = []
        for z in z_indices:
            z_val = z.item() if isinstance(z, torch.Tensor) else z
            rendered = self.forward(positions, scales, opacity, intensity, z_val)
            slices.append(rendered)

        return torch.stack(slices)
