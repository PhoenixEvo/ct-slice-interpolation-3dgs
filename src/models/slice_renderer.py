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
        tile_size: int = 64,
        z_threshold: float = 3.0,
        render_mode: str = "weighted",
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
        rotation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Render a 2D slice at the given z-position.

        Args:
            positions: Gaussian centers (N, 3) - (x, y, z).
            scales: Gaussian scales (N, 3) - (sx, sy, sz), positive.
            opacity: Gaussian opacities (N,) in [0, 1].
            intensity: Gaussian intensities (N,).
            z_target: Z-position of the target slice.
            rotation: Optional unit quaternions (N, 4) encoding per-Gaussian
                rotation. If None or ~identity, falls back to axis-aligned
                fast separable rendering.

        Returns:
            Rendered slice (1, H, W).
        """
        use_rotation = (
            rotation is not None
            and self._has_nontrivial_rotation(rotation)
        )

        if use_rotation:
            return self._render_rotated(
                positions, scales, opacity, intensity, rotation, z_target
            )

        # --- Axis-aligned fast path (original behaviour) ---
        z_dist = (z_target - positions[:, 2]) / (scales[:, 2] + 1e-8)
        z_mask = torch.abs(z_dist) < self.z_threshold

        if z_mask.sum() == 0:
            # Maintain gradient flow so backward() does not crash.
            grad_anchor = (positions.sum() + scales.sum()
                           + opacity.sum() + intensity.sum()) * 0.0
            return grad_anchor.reshape(1, 1, 1).expand(1, self.H, self.W)

        pos_active = positions[z_mask]
        scale_active = scales[z_mask]
        opacity_active = opacity[z_mask]
        intensity_active = intensity[z_mask]

        z_diff = z_target - pos_active[:, 2]
        z_weight = torch.exp(
            -0.5 * (z_diff / (scale_active[:, 2] + 1e-8)) ** 2
        )
        eff_weight = opacity_active * z_weight

        if self.render_mode == "weighted":
            return self._render_separable(
                pos_active, scale_active, eff_weight, intensity_active
            )

        K = pos_active.shape[0]
        if self.H * self.W * K > 50_000_000:
            return self._render_tiled(
                pos_active, scale_active, eff_weight, intensity_active
            )
        else:
            return self._render_direct(
                pos_active, scale_active, eff_weight, intensity_active
            )

    @staticmethod
    def _has_nontrivial_rotation(q: torch.Tensor, tol: float = 1e-4) -> bool:
        """Quickly check if quaternions deviate from identity (w~1, xyz~0).

        Returns True if any Gaussian's rotation is non-identity beyond tol.
        We intentionally use a scalar check here to keep the fast path
        active when the model was initialized without rotation training.
        """
        if q is None:
            return False
        # Maximum absolute deviation from identity
        with torch.no_grad():
            max_xyz = q[:, 1:].abs().max().item()
            max_wdev = (q[:, 0].abs() - 1).abs().max().item()
        return max_xyz > tol or max_wdev > tol

    def _render_rotated(
        self,
        positions: torch.Tensor,
        scales: torch.Tensor,
        opacity: torch.Tensor,
        intensity: torch.Tensor,
        rotation: torch.Tensor,
        z_target: float,
    ) -> torch.Tensor:
        """Render a slice through fully-oriented 3D Gaussians.

        For each Gaussian with center mu, scale s and rotation R, the
        covariance is Sigma = R * diag(s^2) * R^T. The conditional 2D
        distribution of (x, y) given z = z_t is itself a 2D Gaussian with
            mu_cond = mu_xy + Sigma[:2, 2] * (z_t - mu_z) / Sigma[2, 2]
            Sigma_cond = Sigma[:2, :2] - Sigma[:2, 2] Sigma[2, :2] / Sigma[2, 2]
        and a marginal amplitude
            w_z = exp(-0.5 * (z_t - mu_z)^2 / Sigma[2, 2]).

        We then evaluate each pixel's 2D Mahalanobis distance with the
        per-Gaussian Sigma_cond^{-1} and weighted-sum the contributions
        (same compositing as the axis-aligned path). Uses tile-based
        processing when memory footprint would exceed 50M floats.
        """
        # Build 3D covariance matrices (K, 3, 3)
        from .gaussian_volume import quaternion_to_rotation_matrix
        R = quaternion_to_rotation_matrix(rotation)  # (N, 3, 3)
        s2 = scales ** 2  # (N, 3)
        # Sigma = R @ diag(s2) @ R^T
        cov = torch.einsum("nij,nj,nkj->nik", R, s2, R)  # (N, 3, 3)

        cov_zz = cov[:, 2, 2].clamp_min(1e-6)
        sigma_z_eff = cov_zz.sqrt()
        z_dist = (z_target - positions[:, 2]) / sigma_z_eff

        # Mask by effective z distance
        z_mask = torch.abs(z_dist) < self.z_threshold
        if z_mask.sum() == 0:
            grad_anchor = (positions.sum() + scales.sum() + opacity.sum()
                           + intensity.sum() + rotation.sum()) * 0.0
            return grad_anchor.reshape(1, 1, 1).expand(1, self.H, self.W)

        pos_a = positions[z_mask]
        cov_a = cov[z_mask]
        opacity_a = opacity[z_mask]
        intensity_a = intensity[z_mask]

        cov_xy = cov_a[:, :2, :2]               # (K, 2, 2)
        cov_xz = cov_a[:, :2, 2]                # (K, 2)
        cov_zz_a = cov_a[:, 2, 2].clamp_min(1e-6)  # (K,)
        z_diff = z_target - pos_a[:, 2]         # (K,)

        # Conditional 2D mean and covariance (Schur complement)
        mu_cond = pos_a[:, :2] + cov_xz * (z_diff / cov_zz_a).unsqueeze(-1)
        # outer product (K, 2, 2)
        outer = cov_xz.unsqueeze(-1) * cov_xz.unsqueeze(-2)
        Sigma_cond = cov_xy - outer / cov_zz_a.view(-1, 1, 1)
        # Ensure positive-definite via small jitter
        eye2 = torch.eye(2, device=Sigma_cond.device, dtype=Sigma_cond.dtype)
        Sigma_cond = Sigma_cond + 1e-6 * eye2

        # Per-Gaussian inverse (K, 2, 2) and marginal z weight
        Sigma_inv = torch.linalg.inv(Sigma_cond)
        z_weight = torch.exp(-0.5 * z_diff * z_diff / cov_zz_a)  # (K,)
        eff_weight = opacity_a * z_weight

        return self._render_mahalanobis(
            mu_cond, Sigma_inv, eff_weight, intensity_a
        )

    def _render_mahalanobis(
        self,
        mu_cond: torch.Tensor,   # (K, 2)
        Sigma_inv: torch.Tensor, # (K, 2, 2)
        weights: torch.Tensor,   # (K,)
        intensities: torch.Tensor,  # (K,)
    ) -> torch.Tensor:
        """Evaluate sum of 2D Gaussians with arbitrary covariance.

        Uses tile-based processing when K is large to limit peak memory.
        Falls back to a single (H, W, K) evaluation otherwise.
        """
        K = mu_cond.shape[0]
        total_mem = self.H * self.W * K
        if total_mem <= 20_000_000:
            return self._mahalanobis_direct(
                mu_cond, Sigma_inv, weights, intensities,
                r_start=0, r_end=self.H, c_start=0, c_end=self.W,
            )
        return self._mahalanobis_tiled(
            mu_cond, Sigma_inv, weights, intensities
        )

    def _mahalanobis_direct(
        self,
        mu_cond: torch.Tensor,
        Sigma_inv: torch.Tensor,
        weights: torch.Tensor,
        intensities: torch.Tensor,
        r_start: int, r_end: int, c_start: int, c_end: int,
    ) -> torch.Tensor:
        """Direct evaluation of 2D Mahalanobis Gaussians on a (sub-)grid."""
        K = mu_cond.shape[0]
        grid_y = self.grid_y[r_start:r_end, c_start:c_end]  # (h, w)
        grid_x = self.grid_x[r_start:r_end, c_start:c_end]

        dx = grid_y.unsqueeze(-1) - mu_cond[:, 0].view(1, 1, K)  # (h, w, K)
        dy = grid_x.unsqueeze(-1) - mu_cond[:, 1].view(1, 1, K)

        a = Sigma_inv[:, 0, 0].view(1, 1, K)
        b = Sigma_inv[:, 0, 1].view(1, 1, K)
        c = Sigma_inv[:, 1, 1].view(1, 1, K)
        # 2D Mahalanobis: a*dx^2 + 2*b*dx*dy + c*dy^2
        mahal = a * dx * dx + 2.0 * b * dx * dy + c * dy * dy
        gauss_2d = torch.exp(-0.5 * mahal)

        if self.render_mode == "alpha":
            rendered = self._alpha_composite(gauss_2d, weights, intensities)
        else:
            rendered = self._weighted_sum(gauss_2d, weights, intensities)
        # rendered is (1, h, w); embed back into full slice size only when
        # rendering full image
        if r_start == 0 and r_end == self.H and c_start == 0 and c_end == self.W:
            return rendered
        return rendered  # tiled path handles stitching

    def _mahalanobis_tiled(
        self,
        mu_cond: torch.Tensor,
        Sigma_inv: torch.Tensor,
        weights: torch.Tensor,
        intensities: torch.Tensor,
    ) -> torch.Tensor:
        T = self.tile_size
        num_tiles_h = math.ceil(self.H / T)
        num_tiles_w = math.ceil(self.W / T)
        output = torch.zeros(
            self.H, self.W,
            device=mu_cond.device, dtype=mu_cond.dtype,
        )

        # Precompute 3-sigma bounding radius per Gaussian for tile culling.
        # For 2D Gaussian, max radius ~ 3 / sqrt(lambda_min(Sigma_inv)).
        with torch.no_grad():
            # Eigenvalues of Sigma_inv are reciprocals of eigenvalues of Sigma
            a = Sigma_inv[:, 0, 0]
            b = Sigma_inv[:, 0, 1]
            c = Sigma_inv[:, 1, 1]
            tr = a + c
            det = a * c - b * b
            disc = (tr * tr * 0.25 - det).clamp_min(0.0)
            sqrt_disc = disc.sqrt()
            lam_min = (tr * 0.5 - sqrt_disc).clamp_min(1e-6)
            radius = (3.0 / lam_min.sqrt()).clamp_max(float(max(self.H, self.W)))
            mu_x = mu_cond[:, 0]
            mu_y = mu_cond[:, 1]

        for ti in range(num_tiles_h):
            for tj in range(num_tiles_w):
                r_start = ti * T
                r_end = min((ti + 1) * T, self.H)
                c_start = tj * T
                c_end = min((tj + 1) * T, self.W)

                tile_mask = (
                    (mu_x + radius >= r_start)
                    & (mu_x - radius < r_end)
                    & (mu_y + radius >= c_start)
                    & (mu_y - radius < c_end)
                )
                if tile_mask.sum() == 0:
                    continue

                tile_rendered = self._mahalanobis_direct(
                    mu_cond[tile_mask],
                    Sigma_inv[tile_mask],
                    weights[tile_mask],
                    intensities[tile_mask],
                    r_start, r_end, c_start, c_end,
                )
                output[r_start:r_end, c_start:c_end] = tile_rendered.squeeze(0)

        return output.unsqueeze(0)

    def _render_separable(
        self,
        positions: torch.Tensor,
        scales: torch.Tensor,
        weights: torch.Tensor,
        intensities: torch.Tensor,
    ) -> torch.Tensor:
        """Fast separable rendering exploiting axis-aligned Gaussians.

        Since G(x,y) = G_x(x) * G_y(y), the weighted sum can be computed
        via two matrix multiplications instead of materializing the full
        (H, W, K) tensor. Reduces memory from O(H*W*K) to O(H*K + W*K)
        and leverages cuBLAS for ~50-100x speedup over tiled rendering.
        """
        K = positions.shape[0]
        mu_x = positions[:, 0]
        mu_y = positions[:, 1]
        sigma_x = scales[:, 0] + 1e-8
        sigma_y = scales[:, 1] + 1e-8

        y_coords = self.grid_y[:, 0]  # (H,)
        x_coords = self.grid_x[0, :]  # (W,)

        gauss_row = torch.exp(
            -0.5 * ((y_coords.unsqueeze(1) - mu_x.unsqueeze(0)) / sigma_x.unsqueeze(0)) ** 2
        )  # (H, K)
        gauss_col = torch.exp(
            -0.5 * ((x_coords.unsqueeze(1) - mu_y.unsqueeze(0)) / sigma_y.unsqueeze(0)) ** 2
        )  # (W, K)

        wi = weights * intensities  # (K,)
        numerator = (gauss_row * wi.unsqueeze(0)) @ gauss_col.T    # (H, W)
        denominator = (gauss_row * weights.unsqueeze(0)) @ gauss_col.T  # (H, W)

        rendered = numerator / (denominator + 1e-8)
        return rendered.unsqueeze(0)  # (1, H, W)

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
