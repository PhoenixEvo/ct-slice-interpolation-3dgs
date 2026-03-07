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

    @staticmethod
    def interpolate_target_slice(
        volume: np.ndarray,
        observed_indices: np.ndarray,
        target_z: int,
        method: str,
    ) -> np.ndarray:
        """Interpolate a single target slice directly from the volume.

        Memory-efficient: only reads needed slices from the volume without
        creating large intermediate arrays. Peak usage = 4 slices max.

        Args:
            volume: Full CT volume (H, W, D).
            observed_indices: Sorted array of observed z-indices.
            target_z: Z-index of the slice to interpolate.
            method: 'nearest', 'linear', or 'cubic'.

        Returns:
            Interpolated slice (H, W) as float32.
        """
        obs = np.asarray(observed_indices)

        if method == "nearest":
            nearest_z = obs[np.argmin(np.abs(obs - target_z))]
            return volume[:, :, nearest_z].copy()

        # Find bracketing observed slices
        left_mask = obs <= target_z
        right_mask = obs >= target_z

        if not left_mask.any() or not right_mask.any():
            nearest_z = obs[np.argmin(np.abs(obs - target_z))]
            return volume[:, :, nearest_z].copy()

        z1 = obs[left_mask][-1]
        z2 = obs[right_mask][0]

        if z1 == z2:
            return volume[:, :, z1].copy()

        alpha = (target_z - z1) / (z2 - z1)

        if method == "linear":
            result = np.empty_like(volume[:, :, 0])
            np.multiply(volume[:, :, z1], 1 - alpha, out=result)
            result += alpha * volume[:, :, z2]
            return result

        # Cubic: Catmull-Rom with 4 control points
        z0_cands = obs[obs < z1]
        z3_cands = obs[obs > z2]
        if len(z0_cands) == 0 or len(z3_cands) == 0:
            result = np.empty_like(volume[:, :, 0])
            np.multiply(volume[:, :, z1], 1 - alpha, out=result)
            result += alpha * volume[:, :, z2]
            return result

        z0 = z0_cands[-1]
        z3 = z3_cands[0]
        t = alpha
        t2 = t * t
        t3 = t2 * t
        w0 = -0.5 * t3 + t2 - 0.5 * t
        w1 = 1.5 * t3 - 2.5 * t2 + 1.0
        w2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
        w3 = 0.5 * t3 - 0.5 * t2

        result = (
            w0 * volume[:, :, z0]
            + w1 * volume[:, :, z1]
            + w2 * volume[:, :, z2]
            + w3 * volume[:, :, z3]
        )
        return result.astype(np.float32)
