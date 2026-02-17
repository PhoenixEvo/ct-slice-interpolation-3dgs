"""
3D Gaussian Volume representation for CT slice interpolation.
Each Gaussian has: position (x,y,z), scale (sx,sy,sz), opacity, intensity.
Axis-aligned (no rotation) for simplicity and efficiency.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple


class GaussianVolume(nn.Module):
    """3D Gaussian Volume for representing CT data.

    Each Gaussian is parameterized by:
        - position: (x, y, z) center in volume coordinates
        - log_scale: log(sx, sy, sz) for numerical stability
        - raw_opacity: pre-sigmoid opacity value
        - intensity: scalar grayscale intensity value
    """

    def __init__(
        self,
        num_gaussians: int,
        volume_shape: Tuple[int, int, int],
        device: str = "cuda",
    ):
        """Initialize Gaussian Volume.

        Args:
            num_gaussians: Initial number of Gaussians.
            volume_shape: (H, W, D) shape of the CT volume.
            device: Computation device.
        """
        super().__init__()
        self.volume_shape = volume_shape
        self.device = device
        H, W, D = volume_shape

        # Learnable parameters
        self.positions = nn.Parameter(
            torch.zeros(num_gaussians, 3, device=device)
        )
        self.log_scales = nn.Parameter(
            torch.zeros(num_gaussians, 3, device=device)
        )
        self.raw_opacity = nn.Parameter(
            torch.zeros(num_gaussians, device=device)
        )
        self.intensity = nn.Parameter(
            torch.zeros(num_gaussians, device=device)
        )

        # Track gradient accumulation for densification
        self.register_buffer(
            "grad_accum", torch.zeros(num_gaussians, device=device)
        )
        self.register_buffer(
            "grad_count", torch.zeros(num_gaussians, device=device)
        )

    @property
    def num_gaussians(self) -> int:
        """Current number of Gaussians."""
        return self.positions.shape[0]

    @property
    def scales(self) -> torch.Tensor:
        """Get scales from log_scales (ensure positive)."""
        return torch.exp(self.log_scales)

    @property
    def opacity(self) -> torch.Tensor:
        """Get opacity from raw_opacity via sigmoid."""
        return torch.sigmoid(self.raw_opacity)

    def get_params(self) -> Dict[str, torch.Tensor]:
        """Get all Gaussian parameters as a dictionary.

        Returns:
            Dictionary with positions, scales, opacity, intensity.
        """
        return {
            "positions": self.positions,
            "scales": self.scales,
            "opacity": self.opacity,
            "intensity": self.intensity,
        }

    @classmethod
    def from_volume_grid(
        cls,
        volume: np.ndarray,
        observed_indices: np.ndarray,
        subsample_xy: int = 4,
        init_scale_xy: float = 2.0,
        init_scale_z: float = 1.0,
        init_opacity: float = 0.8,
        device: str = "cuda",
    ) -> "GaussianVolume":
        """Initialize Gaussians from a grid on observed slices.

        Places Gaussians on a subsampled grid at each observed z-position,
        with intensity initialized from the CT volume.

        Args:
            volume: Preprocessed CT volume (H, W, D), values in [0, 1].
            observed_indices: Indices of observed slices.
            subsample_xy: Subsampling factor in x and y.
            init_scale_xy: Initial scale in x and y directions.
            init_scale_z: Initial scale in z direction.
            init_opacity: Initial opacity (pre-sigmoid value).
            device: Computation device.

        Returns:
            Initialized GaussianVolume instance.
        """
        H, W, D = volume.shape

        # Create subsampled grid positions on observed slices
        x_coords = np.arange(0, H, subsample_xy).astype(np.float32)
        y_coords = np.arange(0, W, subsample_xy).astype(np.float32)
        z_coords = observed_indices.astype(np.float32)

        # Create meshgrid
        xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")
        positions = np.stack(
            [xx.ravel(), yy.ravel(), zz.ravel()], axis=-1
        )  # (N, 3)

        num_gaussians = positions.shape[0]

        # Initialize intensities from volume
        x_idx = np.clip(xx.ravel().astype(int), 0, H - 1)
        y_idx = np.clip(yy.ravel().astype(int), 0, W - 1)
        z_idx = np.clip(zz.ravel().astype(int), 0, D - 1)
        intensities = volume[x_idx, y_idx, z_idx]

        # Create instance
        model = cls(num_gaussians, (H, W, D), device)

        # Set parameters
        with torch.no_grad():
            model.positions.data = torch.from_numpy(positions).float().to(device)
            model.log_scales.data = torch.log(
                torch.tensor(
                    [init_scale_xy, init_scale_xy, init_scale_z],
                    device=device,
                ).expand(num_gaussians, 3)
            ).clone()
            # Inverse sigmoid for opacity initialization
            opacity_val = np.clip(init_opacity, 1e-4, 1 - 1e-4)
            raw_op = np.log(opacity_val / (1 - opacity_val))
            model.raw_opacity.data.fill_(raw_op)
            model.intensity.data = (
                torch.from_numpy(intensities).float().to(device)
            )

        # Reset gradient accumulators
        model.grad_accum = torch.zeros(num_gaussians, device=device)
        model.grad_count = torch.zeros(num_gaussians, device=device)

        return model

    @classmethod
    def from_volume_adaptive(
        cls,
        volume: np.ndarray,
        observed_indices: np.ndarray,
        subsample_xy: int = 4,
        edge_boost: float = 2.0,
        init_scale_xy: float = 2.0,
        init_scale_z: float = 1.0,
        init_opacity: float = 0.8,
        device: str = "cuda",
    ) -> "GaussianVolume":
        """Initialize Gaussians with higher density near edges.

        Uses Sobel edge detection to place more Gaussians near organ
        boundaries for better detail preservation.

        Args:
            volume: Preprocessed CT volume (H, W, D).
            observed_indices: Indices of observed slices.
            subsample_xy: Base subsampling factor.
            edge_boost: Factor to increase density near edges.
            init_scale_xy: Initial scale in x, y.
            init_scale_z: Initial scale in z.
            init_opacity: Initial opacity.
            device: Computation device.

        Returns:
            Initialized GaussianVolume with adaptive density.
        """
        from scipy import ndimage

        H, W, D = volume.shape
        all_positions = []
        all_intensities = []

        for z_idx in observed_indices:
            slice_2d = volume[:, :, z_idx]

            # Detect edges using Sobel
            edge_x = ndimage.sobel(slice_2d, axis=0)
            edge_y = ndimage.sobel(slice_2d, axis=1)
            edge_mag = np.sqrt(edge_x ** 2 + edge_y ** 2)
            edge_threshold = np.percentile(edge_mag, 75)

            # Coarse grid
            x_coarse = np.arange(0, H, subsample_xy).astype(np.float32)
            y_coarse = np.arange(0, W, subsample_xy).astype(np.float32)
            xx_c, yy_c = np.meshgrid(x_coarse, y_coarse, indexing="ij")

            for xi, yi in zip(xx_c.ravel(), yy_c.ravel()):
                xi_int = int(np.clip(xi, 0, H - 1))
                yi_int = int(np.clip(yi, 0, W - 1))

                all_positions.append([xi, yi, float(z_idx)])
                all_intensities.append(slice_2d[xi_int, yi_int])

                # Add denser points near edges
                if edge_mag[xi_int, yi_int] > edge_threshold:
                    fine_step = max(1, subsample_xy // int(edge_boost))
                    for dx in range(-fine_step, fine_step + 1, fine_step):
                        for dy in range(-fine_step, fine_step + 1, fine_step):
                            if dx == 0 and dy == 0:
                                continue
                            nx = xi + dx
                            ny = yi + dy
                            if 0 <= nx < H and 0 <= ny < W:
                                nxi = int(nx)
                                nyi = int(ny)
                                all_positions.append(
                                    [float(nx), float(ny), float(z_idx)]
                                )
                                all_intensities.append(slice_2d[nxi, nyi])

        positions = np.array(all_positions, dtype=np.float32)
        intensities = np.array(all_intensities, dtype=np.float32)
        num_gaussians = positions.shape[0]

        # Create instance
        model = cls(num_gaussians, (H, W, D), device)

        with torch.no_grad():
            model.positions.data = torch.from_numpy(positions).float().to(device)
            model.log_scales.data = torch.log(
                torch.tensor(
                    [init_scale_xy, init_scale_xy, init_scale_z],
                    device=device,
                ).expand(num_gaussians, 3)
            ).clone()
            opacity_val = np.clip(init_opacity, 1e-4, 1 - 1e-4)
            raw_op = np.log(opacity_val / (1 - opacity_val))
            model.raw_opacity.data.fill_(raw_op)
            model.intensity.data = (
                torch.from_numpy(intensities).float().to(device)
            )

        model.grad_accum = torch.zeros(num_gaussians, device=device)
        model.grad_count = torch.zeros(num_gaussians, device=device)

        return model

    def densify_and_prune(
        self,
        grad_threshold: float = 0.0005,
        opacity_threshold: float = 0.01,
        max_gaussians: int = 500000,
    ) -> Dict[str, int]:
        """Adaptive densification and pruning of Gaussians.

        Clone Gaussians with large gradients and prune those with
        low opacity, following the original 3DGS strategy.

        Args:
            grad_threshold: Gradient threshold for densification.
            opacity_threshold: Opacity threshold for pruning.
            max_gaussians: Maximum allowed number of Gaussians.

        Returns:
            Dictionary with densification/pruning statistics.
        """
        stats = {"cloned": 0, "pruned": 0, "before": self.num_gaussians}

        # Compute average gradient magnitude
        avg_grad = self.grad_accum / (self.grad_count + 1e-8)

        # --- Densification: clone high-gradient Gaussians ---
        if self.num_gaussians < max_gaussians:
            clone_mask = avg_grad > grad_threshold
            num_clone = clone_mask.sum().item()

            if num_clone > 0:
                # Limit cloning to not exceed max
                budget = max_gaussians - self.num_gaussians
                if num_clone > budget:
                    topk = torch.topk(avg_grad, budget)
                    clone_mask = torch.zeros_like(clone_mask)
                    clone_mask[topk.indices] = True
                    num_clone = budget

                # Clone parameters
                new_positions = self.positions.data[clone_mask].clone()
                new_log_scales = self.log_scales.data[clone_mask].clone()
                new_raw_opacity = self.raw_opacity.data[clone_mask].clone()
                new_intensity = self.intensity.data[clone_mask].clone()

                # Add small perturbation to positions
                new_positions += torch.randn_like(new_positions) * 0.5

                # Concatenate
                self.positions = nn.Parameter(
                    torch.cat([self.positions.data, new_positions])
                )
                self.log_scales = nn.Parameter(
                    torch.cat([self.log_scales.data, new_log_scales])
                )
                self.raw_opacity = nn.Parameter(
                    torch.cat([self.raw_opacity.data, new_raw_opacity])
                )
                self.intensity = nn.Parameter(
                    torch.cat([self.intensity.data, new_intensity])
                )

                stats["cloned"] = num_clone

        # --- Pruning: remove low-opacity Gaussians ---
        with torch.no_grad():
            opacity = torch.sigmoid(self.raw_opacity.data)
            keep_mask = opacity > opacity_threshold

            # Also prune Gaussians outside the volume bounds
            H, W, D = self.volume_shape
            in_bounds = (
                (self.positions.data[:, 0] >= -10)
                & (self.positions.data[:, 0] < H + 10)
                & (self.positions.data[:, 1] >= -10)
                & (self.positions.data[:, 1] < W + 10)
                & (self.positions.data[:, 2] >= -5)
                & (self.positions.data[:, 2] < D + 5)
            )
            keep_mask = keep_mask & in_bounds

            num_pruned = (~keep_mask).sum().item()
            if num_pruned > 0:
                self.positions = nn.Parameter(self.positions.data[keep_mask])
                self.log_scales = nn.Parameter(self.log_scales.data[keep_mask])
                self.raw_opacity = nn.Parameter(
                    self.raw_opacity.data[keep_mask]
                )
                self.intensity = nn.Parameter(self.intensity.data[keep_mask])

                stats["pruned"] = num_pruned

        # Reset gradient accumulators
        n = self.num_gaussians
        self.grad_accum = torch.zeros(n, device=self.positions.device)
        self.grad_count = torch.zeros(n, device=self.positions.device)

        stats["after"] = self.num_gaussians
        return stats

    def accumulate_gradients(self) -> None:
        """Accumulate position gradients for densification decisions."""
        if self.positions.grad is not None:
            grad_mag = self.positions.grad.norm(dim=-1)
            n = min(grad_mag.shape[0], self.grad_accum.shape[0])
            self.grad_accum[:n] += grad_mag[:n].detach()
            self.grad_count[:n] += 1

    def reset_opacity(self, new_value: float = 0.5) -> None:
        """Reset all opacities to a given value.

        Useful periodically to allow pruning of unnecessary Gaussians.

        Args:
            new_value: New opacity value (pre-sigmoid).
        """
        opacity_val = np.clip(new_value, 1e-4, 1 - 1e-4)
        raw = np.log(opacity_val / (1 - opacity_val))
        with torch.no_grad():
            self.raw_opacity.data.fill_(raw)
