"""
Regularization losses for 3DGS CT slice interpolation.
Includes smoothness along z-axis, edge preservation, total variation,
and FFT high-frequency loss integration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .reconstruction import (
    CombinedReconstructionLoss,
    MultiScaleReconstructionLoss,
    FFTHighFreqLoss,
)


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
        """Compute smoothness loss between two adjacent slices."""
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
        """Compute gradient magnitude using Sobel operator."""
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
        """Compute edge preservation loss."""
        pred_grad = self._compute_gradient(prediction)
        target_grad = self._compute_gradient(target)
        return F.l1_loss(pred_grad, target_grad)

    def compute_weight_map(
        self, target: torch.Tensor, weight_max: float = 3.0,
    ) -> torch.Tensor:
        """Compute HU gradient-based spatial weight map.

        Returns a weight map where regions with strong gradients (organ
        boundaries, bone interfaces) receive higher weight, guiding the
        model to focus reconstruction accuracy on clinically important
        structures.

        Args:
            target: Ground truth slice (1, H, W) or (B, 1, H, W).
            weight_max: Maximum weight boost at strongest gradients.

        Returns:
            Weight map with values in [1.0, weight_max], same shape as target.
        """
        grad_mag = self._compute_gradient(target)
        # Normalize gradient to [0, 1]
        gmax = grad_mag.max()
        if gmax > 1e-8:
            grad_norm = grad_mag / gmax
        else:
            grad_norm = grad_mag
        # Map to [1.0, weight_max]
        weight = 1.0 + grad_norm * (weight_max - 1.0)
        return weight


class TotalVariationLoss(nn.Module):
    """Anisotropic Total Variation loss for spatial denoising.

    Penalizes high-frequency noise in rendered slices while preserving
    sharp edges. Uses anisotropic TV which sums absolute differences
    along each axis independently (less edge-blurring than isotropic TV).
    """

    def forward(self, prediction: torch.Tensor) -> torch.Tensor:
        if prediction.dim() == 3:
            prediction = prediction.unsqueeze(0)
        diff_h = torch.abs(prediction[:, :, 1:, :] - prediction[:, :, :-1, :])
        diff_w = torch.abs(prediction[:, :, :, 1:] - prediction[:, :, :, :-1])
        return diff_h.mean() + diff_w.mean()


class CrossViewConsistencyLoss(nn.Module):
    """Sagittal/Coronal consistency loss on a local rendered mini-volume.

    H3b rationale: the 3DGS renderer operates on axial (xy) slices, so
    without extra supervision nothing enforces that the reconstructed
    volume is coherent when viewed sagittally (xz) or coronally (yz).
    This loss takes a short stack of rendered neighboring slices and
    asks them to match the same stack cut from the cubic (or any) base,
    in three aspects:

    1. Direct L1 match on the stack values (anchors the residual).
    2. Through-plane z-gradient match: encourages the local volume's
       z-variations to mirror the base's z-variations -> sagittal and
       coronal edges behave consistently.
    3. Optional in-plane gradient match on the sagittal and coronal
       cross-sections (Sobel along z inside those slices).

    Expected weight: very small (~0.001) to act as a regularizer without
    dominating the reconstruction loss.
    """

    def __init__(
        self,
        use_gradient: bool = True,
        lambda_zgrad: float = 1.0,
        lambda_crosssec: float = 0.5,
    ) -> None:
        super().__init__()
        self.use_gradient = use_gradient
        self.lambda_zgrad = lambda_zgrad
        self.lambda_crosssec = lambda_crosssec

    def forward(
        self,
        pred_stack: torch.Tensor,
        base_stack: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-view consistency loss.

        Args:
            pred_stack: (D_local, H, W) or (1, D_local, H, W) stack of
                rendered slices at consecutive z positions.
            base_stack: same shape as pred_stack; e.g. cubic base at the
                same z positions.

        Returns:
            Scalar loss tensor.
        """
        # Flatten batch dim if present
        if pred_stack.dim() == 4 and pred_stack.shape[0] == 1:
            pred_stack = pred_stack.squeeze(0)
            base_stack = base_stack.squeeze(0)
        if pred_stack.dim() != 3:
            return pred_stack.sum() * 0.0

        # Direct L1 on the local mini-volume (very small weight via caller)
        l_direct = F.l1_loss(pred_stack, base_stack)

        if not self.use_gradient or pred_stack.shape[0] < 2:
            return l_direct

        # Through-plane (z) gradient match: (D-1, H, W)
        dz_pred = pred_stack[1:] - pred_stack[:-1]
        dz_base = base_stack[1:] - base_stack[:-1]
        l_zgrad = F.l1_loss(dz_pred, dz_base)

        # Sagittal cross-sections at a few random y columns: (D, H)
        # Coronal cross-sections at a few random x rows: (D, W)
        # We use the in-plane gradient along the spatial axis of those views
        # (i.e. Sobel H in sagittal, Sobel W in coronal).
        D_loc, H, W = pred_stack.shape
        n_samples = min(4, max(1, H // 32))
        y_idx = torch.linspace(0, W - 1, n_samples, device=pred_stack.device).long()
        x_idx = torch.linspace(0, H - 1, n_samples, device=pred_stack.device).long()

        sag_pred = pred_stack[:, :, y_idx]  # (D, H, n)
        sag_base = base_stack[:, :, y_idx]
        cor_pred = pred_stack[:, x_idx, :]  # (D, n, W)
        cor_base = base_stack[:, x_idx, :]

        # Spatial gradient in each cross-section (first-order diff)
        def _grad_h(t: torch.Tensor) -> torch.Tensor:
            return t[:, 1:, ...] - t[:, :-1, ...]

        def _grad_w(t: torch.Tensor) -> torch.Tensor:
            return t[..., 1:] - t[..., :-1]

        l_sag = F.l1_loss(_grad_h(sag_pred), _grad_h(sag_base))
        l_cor = F.l1_loss(_grad_w(cor_pred), _grad_w(cor_base))

        return (
            l_direct
            + self.lambda_zgrad * l_zgrad
            + self.lambda_crosssec * (l_sag + l_cor)
        )


class TotalLoss(nn.Module):
    """Total loss combining multi-scale reconstruction, FFT, and regularization.

    L_total = L_rec_ms + lambda_smooth * L_smooth + lambda_edge * L_edge
              + lambda_tv * L_tv + lambda_fft * L_fft

    Supports:
    - Regularization annealing via set_progress() for coarse-to-fine training
    - FFT high-frequency loss for sharp edge preservation
    - HU gradient-based spatial weighting for clinical structure focus
    """

    def __init__(
        self,
        lambda_smooth: float = 0.01,
        lambda_edge: float = 0.005,
        lambda_tv: float = 0.001,
        lambda_fft: float = 0.01,
        fft_cutoff: float = 0.3,
        lambda_residual: float = 0.0,
        lambda_crossview: float = 0.0,
        l1_weight: float = 1.0,
        ssim_weight: float = 0.1,
        hu_gradient_weight: bool = False,
        hu_weight_max: float = 3.0,
        multiscale: bool = True,
        data_range: float = 1.0,
    ):
        super().__init__()
        self.lambda_smooth_init = lambda_smooth
        self.lambda_edge = lambda_edge
        self.lambda_tv = lambda_tv
        self.lambda_fft = lambda_fft
        self.lambda_residual = lambda_residual
        self.lambda_crossview = lambda_crossview
        self._lambda_smooth = lambda_smooth
        self.hu_gradient_weight = hu_gradient_weight
        self.hu_weight_max = hu_weight_max

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
        self.crossview_loss = CrossViewConsistencyLoss()

        # FFT high-frequency loss (new)
        if lambda_fft > 0:
            self.fft_loss = FFTHighFreqLoss(cutoff_ratio=fft_cutoff)
        else:
            self.fft_loss = None

    def set_progress(self, progress: float) -> None:
        """Update regularization weights based on training progress.

        Smoothness decays from 2x initial to 0.5x initial over training,
        implementing coarse-to-fine refinement. FFT loss ramps up from
        0.5x to 1.5x to increasingly enforce high-frequency fidelity.

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
        residual_output: Optional[torch.Tensor] = None,
        crossview_pred_stack: Optional[torch.Tensor] = None,
        crossview_base_stack: Optional[torch.Tensor] = None,
    ) -> dict:
        """Compute total loss with all components.

        Args:
            prediction: Final predicted slice (1, H, W). In residual mode,
                        this is (cubic_base + 3dgs_output).
            target: Ground truth slice (1, H, W).
            adjacent_pred: Optional adjacent rendered slice.
            adjacent_target: Optional adjacent ground truth slice.
            residual_output: Optional raw 3DGS output (before adding base).
                            Used for residual magnitude penalty.

        Returns:
            Dictionary with total and component losses.
        """
        # Compute HU gradient weight map if enabled
        weight_map = None
        if self.hu_gradient_weight:
            weight_map = self.edge_loss.compute_weight_map(
                target, self.hu_weight_max
            )

        # Reconstruction loss (with optional spatial weighting)
        l_rec = self.reconstruction_loss(prediction, target, weight_map)

        # Edge preservation
        l_edge = self.edge_loss(prediction, target)

        # Total variation
        l_tv = self.tv_loss(prediction)

        # Smoothness
        if adjacent_pred is not None:
            if adjacent_target is not None:
                l_smooth = self.smoothness_loss(adjacent_pred, adjacent_target)
            else:
                l_smooth = self.smoothness_loss(prediction, adjacent_pred)
        else:
            l_smooth = prediction.sum() * 0.0

        # FFT high-frequency loss
        if self.fft_loss is not None and self.lambda_fft > 0:
            l_fft = self.fft_loss(prediction, target)
        else:
            l_fft = prediction.sum() * 0.0

        # Residual magnitude penalty (L2): push raw 3DGS output towards zero
        # This prevents artifacts from noisy residuals at target positions
        if residual_output is not None and self.lambda_residual > 0:
            l_residual = (residual_output ** 2).mean()
        else:
            l_residual = prediction.sum() * 0.0

        # Cross-view consistency loss (H3b)
        if (
            self.lambda_crossview > 0
            and crossview_pred_stack is not None
            and crossview_base_stack is not None
        ):
            l_crossview = self.crossview_loss(
                crossview_pred_stack, crossview_base_stack,
            )
        else:
            l_crossview = prediction.sum() * 0.0

        total = (l_rec
                 + self._lambda_smooth * l_smooth
                 + self.lambda_edge * l_edge
                 + self.lambda_tv * l_tv
                 + self.lambda_fft * l_fft
                 + self.lambda_residual * l_residual
                 + self.lambda_crossview * l_crossview)

        return {
            "total": total,
            "reconstruction": l_rec,
            "smoothness": l_smooth,
            "edge": l_edge,
            "tv": l_tv,
            "fft": l_fft,
            "residual": l_residual,
            "crossview": l_crossview,
        }
