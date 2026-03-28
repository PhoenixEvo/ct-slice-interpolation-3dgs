"""
Reconstruction loss functions for CT slice interpolation.
Includes L1, L2, SSIM, FFT high-frequency, and combined losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class L1Loss(nn.Module):
    """L1 (Mean Absolute Error) reconstruction loss."""

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute L1 loss, optionally spatially weighted.

        Args:
            prediction: Predicted slice (B, 1, H, W) or (1, H, W).
            target: Ground truth slice, same shape as prediction.
            weight_map: Optional spatial weight map, same spatial dims.

        Returns:
            Scalar loss value.
        """
        if weight_map is not None:
            return (torch.abs(prediction - target) * weight_map).mean()
        return F.l1_loss(prediction, target)


class L2Loss(nn.Module):
    """L2 (Mean Squared Error) reconstruction loss."""

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute L2 loss."""
        return F.mse_loss(prediction, target)


class SSIMLoss(nn.Module):
    """Structural Similarity (SSIM) loss.

    SSIM measures structural similarity between two images.
    Loss = 1 - SSIM (to minimize).
    """

    def __init__(self, window_size: int = 7, data_range: float = 1.0):
        """Initialize SSIM loss.

        Args:
            window_size: Size of the Gaussian window.
            data_range: Range of pixel values.
        """
        super().__init__()
        self.window_size = window_size
        self.data_range = data_range
        self.C1 = (0.01 * data_range) ** 2
        self.C2 = (0.03 * data_range) ** 2

        # Create Gaussian window
        self.register_buffer(
            "window", self._create_window(window_size)
        )

    @staticmethod
    def _create_window(size: int) -> torch.Tensor:
        """Create a 2D Gaussian window for SSIM computation."""
        sigma = 1.5
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        g = g / g.sum()
        window_2d = g.unsqueeze(1) * g.unsqueeze(0)
        return window_2d.unsqueeze(0).unsqueeze(0)

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute SSIM loss = 1 - SSIM."""
        # Ensure 4D
        if prediction.dim() == 3:
            prediction = prediction.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0)

        C = prediction.shape[1]
        window = self.window.expand(C, -1, -1, -1).to(prediction.device)
        pad = self.window_size // 2

        # Compute means
        mu_pred = F.conv2d(prediction, window, padding=pad, groups=C)
        mu_target = F.conv2d(target, window, padding=pad, groups=C)

        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_cross = mu_pred * mu_target

        # Compute variances
        sigma_pred_sq = (
            F.conv2d(prediction ** 2, window, padding=pad, groups=C) - mu_pred_sq
        )
        sigma_target_sq = (
            F.conv2d(target ** 2, window, padding=pad, groups=C) - mu_target_sq
        )
        sigma_cross = (
            F.conv2d(prediction * target, window, padding=pad, groups=C) - mu_cross
        )

        # SSIM formula
        numerator = (2 * mu_cross + self.C1) * (2 * sigma_cross + self.C2)
        denominator = (mu_pred_sq + mu_target_sq + self.C1) * (
            sigma_pred_sq + sigma_target_sq + self.C2
        )

        ssim_map = numerator / (denominator + 1e-8)
        ssim_val = ssim_map.mean()

        return 1.0 - ssim_val


class FFTHighFreqLoss(nn.Module):
    """FFT-based high-frequency loss for CT slice interpolation.

    Penalizes differences in high-frequency components between prediction
    and target in the Fourier domain. This forces the model to capture
    sharp edges (bone boundaries, organ walls) rather than just optimizing
    smooth regions.

    Reference: I3Net (2024), SFCLI-Net (2025) frequency domain learning.
    """

    def __init__(self, cutoff_ratio: float = 0.3):
        """Initialize FFT high-frequency loss.

        Args:
            cutoff_ratio: Fraction of max frequency below which to zero out.
                          0.3 means keep frequencies above 30% of max.
        """
        super().__init__()
        self.cutoff_ratio = cutoff_ratio
        self._mask_cache = {}

    def _get_high_freq_mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Create or retrieve cached high-frequency mask."""
        key = (H, W, str(device))
        if key not in self._mask_cache:
            cy, cx = H // 2, W // 2
            y = torch.arange(H, device=device, dtype=torch.float32) - cy
            x = torch.arange(W, device=device, dtype=torch.float32) - cx
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            radius = torch.sqrt(xx ** 2 + yy ** 2)
            max_radius = max(cy, cx)
            mask = (radius > self.cutoff_ratio * max_radius).float()
            self._mask_cache[key] = mask
        return self._mask_cache[key]

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute FFT high-frequency loss.

        Args:
            prediction: Predicted slice (..., H, W).
            target: Ground truth slice (..., H, W).

        Returns:
            Scalar high-frequency loss.
        """
        if prediction.dim() == 3:
            prediction = prediction.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0)

        H, W = prediction.shape[-2], prediction.shape[-1]

        # Compute 2D FFT and shift zero-frequency to center
        pred_fft = torch.fft.fftshift(torch.fft.fft2(prediction))
        target_fft = torch.fft.fftshift(torch.fft.fft2(target))

        # Apply high-frequency mask
        mask = self._get_high_freq_mask(H, W, prediction.device)

        # L1 difference in frequency domain (magnitude)
        diff = torch.abs(pred_fft - target_fft) * mask
        # Normalize by number of high-freq components to keep loss scale stable
        num_hf = mask.sum().clamp(min=1.0)
        return diff.sum() / (num_hf * prediction.shape[0] * prediction.shape[1])


class CombinedReconstructionLoss(nn.Module):
    """Combined L1 + SSIM reconstruction loss with optional spatial weighting."""

    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 0.1,
        data_range: float = 1.0,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.l1_loss = L1Loss()
        self.ssim_loss = SSIMLoss(data_range=data_range)

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute combined L1 + SSIM loss.

        Args:
            prediction: Predicted slice.
            target: Ground truth slice.
            weight_map: Optional spatial weight map for L1 term.
        """
        loss = self.l1_weight * self.l1_loss(prediction, target, weight_map)
        if self.ssim_weight > 0:
            loss = loss + self.ssim_weight * self.ssim_loss(prediction, target)
        return loss


class MultiScaleReconstructionLoss(nn.Module):
    """Multi-scale L1 + SSIM loss computed at multiple resolutions.

    Computes reconstruction loss at the original resolution and at
    successively downsampled versions (1/2, 1/4). This encourages
    the model to capture both coarse global structure and fine local
    details, acting as an implicit coarse-to-fine objective.
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 0.1,
        num_scales: int = 3,
        scale_weights: Optional[list] = None,
        data_range: float = 1.0,
    ):
        super().__init__()
        self.num_scales = num_scales
        self.scale_weights = scale_weights or [1.0, 0.5, 0.25][:num_scales]
        self.losses = nn.ModuleList([
            CombinedReconstructionLoss(l1_weight, ssim_weight, data_range)
            for _ in range(num_scales)
        ])

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute multi-scale reconstruction loss.

        Args:
            prediction: Predicted slice.
            target: Ground truth slice.
            weight_map: Optional spatial weight map (downsampled at each scale).
        """
        if prediction.dim() == 3:
            prediction = prediction.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0)
        if weight_map is not None and weight_map.dim() == 3:
            weight_map = weight_map.unsqueeze(0)

        total = self.scale_weights[0] * self.losses[0](prediction, target, weight_map)

        pred_down = prediction
        tgt_down = target
        wm_down = weight_map
        for s in range(1, self.num_scales):
            pred_down = F.avg_pool2d(pred_down, 2)
            tgt_down = F.avg_pool2d(tgt_down, 2)
            if wm_down is not None:
                wm_down = F.avg_pool2d(wm_down, 2)
            total = total + self.scale_weights[s] * self.losses[s](pred_down, tgt_down, wm_down)

        weight_sum = sum(self.scale_weights[:self.num_scales])
        return total / weight_sum
