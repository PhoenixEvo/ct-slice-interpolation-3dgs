"""
Regularization losses for 3DGS CT slice interpolation.
Includes smoothness along z-axis and edge preservation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .reconstruction import CombinedReconstructionLoss


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


class TotalLoss(nn.Module):
    """Total loss combining reconstruction and regularization.

    L_total = L_rec + lambda_smooth * L_smooth + lambda_edge * L_edge
    """

    def __init__(
        self,
        lambda_smooth: float = 0.01,
        lambda_edge: float = 0.005,
        l1_weight: float = 1.0,
        ssim_weight: float = 0.1,
        data_range: float = 1.0,
    ):
        """Initialize total loss.

        Args:
            lambda_smooth: Weight for smoothness regularization.
            lambda_edge: Weight for edge preservation loss.
            l1_weight: Weight for L1 reconstruction loss.
            ssim_weight: Weight for SSIM loss component.
            data_range: Data range for SSIM computation.
        """
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_edge = lambda_edge

        self.reconstruction_loss = CombinedReconstructionLoss(
            l1_weight=l1_weight, ssim_weight=ssim_weight, data_range=data_range
        )
        self.smoothness_loss = SmoothnessLoss()
        self.edge_loss = EdgePreservationLoss()

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        adjacent_pred: Optional[torch.Tensor] = None,
        adjacent_target: Optional[torch.Tensor] = None,
    ) -> dict:
        """Compute total loss with all components.

        Args:
            prediction: Predicted slice.
            target: Ground truth slice.
            adjacent_pred: Optional adjacent predicted slice for smoothness.
            adjacent_target: Optional adjacent target slice.

        Returns:
            Dictionary with total loss and individual components.
        """
        # Reconstruction loss (always computed)
        l_rec = self.reconstruction_loss(prediction, target)

        # Edge preservation loss
        l_edge = self.edge_loss(prediction, target)

        # Smoothness loss (if adjacent slice available)
        l_smooth = torch.tensor(0.0, device=prediction.device)
        if adjacent_pred is not None:
            if adjacent_target is not None:
                l_smooth = self.smoothness_loss(adjacent_pred, adjacent_target)
            else:
                l_smooth = self.smoothness_loss(prediction, adjacent_pred)

        # Total loss
        total = l_rec + self.lambda_smooth * l_smooth + self.lambda_edge * l_edge

        return {
            "total": total,
            "reconstruction": l_rec,
            "smoothness": l_smooth,
            "edge": l_edge,
        }
