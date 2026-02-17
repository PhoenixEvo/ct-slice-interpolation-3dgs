"""
Classical interpolation baselines for CT slice interpolation.
Provides nearest, linear, and cubic interpolation along the z-axis.
"""

import numpy as np
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from typing import Dict, List, Optional


class ClassicalInterpolator:
    """Classical interpolation methods along the z-axis."""

    METHODS = ["nearest", "linear", "cubic"]

    def __init__(self, method: str = "linear"):
        """Initialize interpolator.

        Args:
            method: Interpolation method - 'nearest', 'linear', or 'cubic'.
        """
        if method not in self.METHODS:
            raise ValueError(
                f"Unknown method '{method}'. Choose from {self.METHODS}"
            )
        self.method = method

    def interpolate_volume(
        self,
        observed_slices: np.ndarray,
        observed_indices: np.ndarray,
        target_indices: np.ndarray,
    ) -> np.ndarray:
        """Interpolate target slices from observed slices.

        Args:
            observed_slices: Observed slices array (H, W, N_obs).
            observed_indices: Z-indices of observed slices.
            target_indices: Z-indices of target slices to interpolate.

        Returns:
            Interpolated slices array (H, W, N_target).
        """
        H, W, N_obs = observed_slices.shape

        # Build coordinate grids
        x = np.arange(H)
        y = np.arange(W)
        z_obs = observed_indices.astype(np.float64)

        # Create interpolator
        interpolator = RegularGridInterpolator(
            (x, y, z_obs),
            observed_slices,
            method=self._scipy_method(),
            bounds_error=False,
            fill_value=None,  # Extrapolate via nearest
        )

        # Generate target coordinates
        results = np.zeros((H, W, len(target_indices)), dtype=np.float32)
        for i, z_target in enumerate(target_indices):
            # Create meshgrid for this target slice
            xx, yy = np.meshgrid(x, y, indexing="ij")
            zz = np.full_like(xx, z_target, dtype=np.float64)
            points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)

            interpolated = interpolator(points).reshape(H, W)
            results[:, :, i] = interpolated.astype(np.float32)

        return results

    def interpolate_single_slice(
        self,
        slice_before: np.ndarray,
        slice_after: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Interpolate a single slice between two adjacent slices.

        Args:
            slice_before: Previous slice (H, W).
            slice_after: Next slice (H, W).
            alpha: Interpolation position in [0, 1]. 0.5 = midpoint.

        Returns:
            Interpolated slice (H, W).
        """
        if self.method == "nearest":
            return slice_before if alpha < 0.5 else slice_after

        elif self.method == "linear":
            return (
                (1 - alpha) * slice_before + alpha * slice_after
            ).astype(np.float32)

        elif self.method == "cubic":
            # For single pair, cubic reduces to linear
            # Full cubic requires 4 slices; use volume-based method instead
            return (
                (1 - alpha) * slice_before + alpha * slice_after
            ).astype(np.float32)

        raise ValueError(f"Unknown method: {self.method}")

    def _scipy_method(self) -> str:
        """Map method name to scipy method string."""
        mapping = {
            "nearest": "nearest",
            "linear": "linear",
            "cubic": "cubic",
        }
        return mapping[self.method]

    @staticmethod
    def interpolate_all_methods(
        observed_slices: np.ndarray,
        observed_indices: np.ndarray,
        target_indices: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Run all interpolation methods and return results.

        Args:
            observed_slices: Observed slices array (H, W, N_obs).
            observed_indices: Z-indices of observed slices.
            target_indices: Z-indices of target slices.

        Returns:
            Dictionary mapping method name to interpolated slices.
        """
        results = {}
        for method in ClassicalInterpolator.METHODS:
            interp = ClassicalInterpolator(method=method)
            results[method] = interp.interpolate_volume(
                observed_slices, observed_indices, target_indices
            )
        return results
