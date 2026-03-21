"""
Regularization losses for 3DGS CT slice interpolation.
Includes smoothness along z-axis, edge preservation, and total variation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .reconstruction import CombinedReconstructionLoss, MultiScaleReconstructionLoss


class SmoothnessLoss(nn.Module):
    """Smoothness regularization along the z-axis.

    Encourages intensity continuity between adjacent rendered slices.
    L_smooth = mean(|I(z) - I(z+1)|)
    """

    def forward(
        self,
        slice_current: torch.Tensor,
        slice_adjacent: torch.Tensor,
    ) -> torch.Tensor:
        """Compute smoothness loss between two adjacent slices.

        Args:
            slice_current: Current rendered slice (1, H, W) or (B, 1, H, W).
            slice_adjacent: Adjacent rendered slice, same shape.

        Returns:
            Scalar smoothness loss.
        """
        return F.l1_loss(slice_current, slice_adjacent)


class EdgePreservationLoss(nn.Module):
    """Edge preservation loss using Sobel gradient matching.

    Ensures that the predicted slice preserves edge structures
    present in the ground truth, important for organ boundaries.

    L_edge = mean(|grad(pred) - grad(gt)|)
    """

    def __init__(self):
        super().__init__()
        # Sobel kernels
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _compute_gradient(self, image: torch.Tensor) -> torch.Tensor:
        """Compute gradient magnitude using Sobel operator.

        Args:
            image: Input image (B, 1, H, W) or (1, H, W).

        Returns:
            Gradient magnitude, same shape as input.
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        grad_x = F.conv2d(image, self.sobel_x, padding=1)
        grad_y = F.conv2d(image, self.sobel_y, padding=1)
        gradient = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

        return gradient

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute edge preservation loss.

        Args:
            prediction: Predicted slice (1, H, W) or (B, 1, H, W).
            target: Ground truth slice, same shape.

        Returns:
            Scalar edge loss.
        """
        pred_grad = self._compute_gradient(prediction)
        target_grad = self._compute_gradient(target)
        return F.l1_loss(pred_grad, target_grad)


class TotalVariationLoss(nn.Module):
    """Anisotropic Total Variation loss for spatial denoising.

    Penalizes high-frequency noise in rendered slices while preserving
    sharp edges. Uses anisotropic TV which sums absolute differences
    along each axis independently (less edge-blurring than isotropic TV).

    L_tv = mean(|I[i+1,j] - I[i,j]| + |I[i,j+1] - I[i,j]|)
    """

    def forward(self, prediction: torch.Tensor) -> torch.Tensor:
        if prediction.dim() == 3:
            prediction = prediction.unsqueeze(0)
        diff_h = torch.abs(prediction[:, :, 1:, :] - prediction[:, :, :-1, :])
        diff_w = torch.abs(prediction[:, :, :, 1:] - prediction[:, :, :, :-1])
        return diff_h.mean() + diff_w.mean()


class TotalLoss(nn.Module):
    """Total loss combining multi-scale reconstruction and regularization.

    L_total = L_rec_ms + lambda_smooth * L_smooth + lambda_edge * L_edge
              + lambda_tv * L_tv

    Supports regularization annealing via set_progress() for coarse-to-fine
    training: smoothness starts high and decays, allowing the model to first
    capture global anatomy then refine local detail.
    """

    def __init__(
        self,
        lambda_smooth: float = 0.01,
        lambda_edge: float = 0.005,
        lambda_tv: float = 0.001,
        l1_weight: float = 1.0,
        ssim_weight: float = 0.1,
        multiscale: bool = True,
        data_range: float = 1.0,
    ):
        super().__init__()
        self.lambda_smooth_init = lambda_smooth
        self.lambda_edge = lambda_edge
        self.lambda_tv = lambda_tv
        self._lambda_smooth = lambda_smooth

        if multiscale:
            self.reconstruction_loss = MultiScaleReconstructionLoss(
                l1_weight=l1_weight, ssim_weight=ssim_weight,
                num_scales=3, data_range=data_range,
            )
        else:
            self.reconstruction_loss = CombinedReconstructionLoss(
                l1_weight=l1_weight, ssim_weight=ssim_weight,
                data_range=data_range,
            )
        self.smoothness_loss = SmoothnessLoss()
        self.edge_loss = EdgePreservationLoss()
        self.tv_loss = TotalVariationLoss()

    def set_progress(self, progress: float) -> None:
        """Update regularization weights based on training progress.

        Smoothness decays from 2x initial to 0.5x initial over training,
        implementing coarse-to-fine refinement.

        Args:
            progress: Training progress in [0, 1].
        """
        # Smooth decay: 2.0 * init -> 0.5 * init over training
        scale = 2.0 - 1.5 * min(progress, 1.0)
        self._lambda_smooth = self.lambda_smooth_init * scale

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        adjacent_pred: Optional[torch.Tensor] = None,
        adjacent_target: Optional[torch.Tensor] = None,
    ) -> dict:
        l_rec = self.reconstruction_loss(prediction, target)
        l_edge = self.edge_loss(prediction, target)
        l_tv = self.tv_loss(prediction)

        l_smooth = torch.tensor(0.0, device=prediction.device)
        if adjacent_pred is not None:
            if adjacent_target is not None:
                l_smooth = self.smoothness_loss(adjacent_pred, adjacent_target)
            else:
                l_smooth = self.smoothness_loss(prediction, adjacent_pred)

        total = (l_rec
                 + self._lambda_smooth * l_smooth
                 + self.lambda_edge * l_edge
                 + self.lambda_tv * l_tv)

        return {
            "total": total,
            "reconstruction": l_rec,
            "smoothness": l_smooth,
            "edge": l_edge,
            "tv": l_tv,
        }
