"""
Reconstruction loss functions for CT slice interpolation.
Includes L1, L2, SSIM, and combined losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class L1Loss(nn.Module):
    """L1 (Mean Absolute Error) reconstruction loss."""

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute L1 loss.

        Args:
            prediction: Predicted slice (B, 1, H, W) or (1, H, W).
            target: Ground truth slice, same shape as prediction.

        Returns:
            Scalar loss value.
        """
        return F.l1_loss(prediction, target)


class L2Loss(nn.Module):
    """L2 (Mean Squared Error) reconstruction loss."""

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute L2 loss.

        Args:
            prediction: Predicted slice.
            target: Ground truth slice.

        Returns:
            Scalar loss value.
        """
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
        """Create a 2D Gaussian window for SSIM computation.

        Args:
            size: Window size.

        Returns:
            Gaussian window tensor (1, 1, size, size).
        """
        sigma = 1.5
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        g = g / g.sum()
        window_2d = g.unsqueeze(1) * g.unsqueeze(0)
        return window_2d.unsqueeze(0).unsqueeze(0)

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute SSIM loss = 1 - SSIM.

        Args:
            prediction: Predicted image (B, C, H, W).
            target: Ground truth image (B, C, H, W).

        Returns:
            Scalar SSIM loss.
        """
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


class CombinedReconstructionLoss(nn.Module):
    """Combined L1 + SSIM reconstruction loss."""

    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 0.1,
        data_range: float = 1.0,
    ):
        """Initialize combined loss.

        Args:
            l1_weight: Weight for L1 loss.
            ssim_weight: Weight for SSIM loss.
            data_range: Data range for SSIM.
        """
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.l1_loss = L1Loss()
        self.ssim_loss = SSIMLoss(data_range=data_range)

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute combined loss.

        Args:
            prediction: Predicted image.
            target: Ground truth image.

        Returns:
            Scalar combined loss.
        """
        loss = self.l1_weight * self.l1_loss(prediction, target)
        if self.ssim_weight > 0:
            loss = loss + self.ssim_weight * self.ssim_loss(prediction, target)
        return loss
