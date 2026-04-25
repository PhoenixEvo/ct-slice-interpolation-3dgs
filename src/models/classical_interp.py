"""
Classical interpolation baselines for CT slice interpolation.
Provides nearest, linear, and cubic interpolation along the z-axis,
plus 3D methods that exploit x-y self-similarity (cubic + BM4D, sinc3d).
"""

import numpy as np
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from typing import Dict, List, Optional, Tuple


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


# ==========================================================================
# 3D methods that exploit x-y self-similarity (beyond per-z-axis cubic)
# ==========================================================================


def _build_cubic_dense_volume(
    volume: np.ndarray,
    observed_indices: np.ndarray,
    all_indices: np.ndarray,
) -> np.ndarray:
    """Build a dense reconstructed volume using cubic interpolation along z.

    Values at observed z equal the original slices; values at target z are
    Catmull-Rom cubic interpolations using all observed slices as controls.
    Returns a volume of shape (H, W, len(all_indices)).
    """
    H, W, _ = volume.shape
    sorted_obs = np.sort(observed_indices)
    all_sorted = np.sort(all_indices)
    dense = np.zeros((H, W, len(all_sorted)), dtype=np.float32)
    for i, z_idx in enumerate(all_sorted):
        dense[:, :, i] = ClassicalInterpolator.interpolate_target_slice(
            volume, sorted_obs, int(z_idx), "cubic"
        )
    return dense, all_sorted


def interpolate_cubic_bm4d(
    volume: np.ndarray,
    observed_indices: np.ndarray,
    all_indices: np.ndarray,
    sigma_psd: float = 0.015,
    profile: str = "lc",
) -> Tuple[np.ndarray, np.ndarray]:
    """Cubic interpolation along z followed by BM4D denoising.

    Rationale: cubic exploits only z-axis correlation. BM4D (Block-Matching
    4D filter, Maggioni et al.) exploits 3D non-local self-similarity in
    the x-y plane and across slices, cleaning up the residual errors that
    cubic leaves behind around organ boundaries.

    Falls back gracefully to scikit-image non-local means (3D) if the
    `bm4d` package is not installed (pure numpy fallback).

    Args:
        volume: Full CT volume (H, W, D), float32 in [0, 1].
        observed_indices: Indices of observed slices.
        all_indices: All z-indices to reconstruct (usually observed + target).
        sigma_psd: Noise-like residual std for the denoiser (tuned small
            because the "noise" here is cubic interpolation error, not
            actual imaging noise).
        profile: BM4D profile. "lc" is kept as a backward-compatible alias
            and mapped to "np" for bm4d versions that only accept
            {"np", "refilter"} or a BM4DProfile object.

    Returns:
        (dense_volume, sorted_indices) where dense_volume has shape
        (H, W, len(all_indices)) with BM4D-denoised cubic interpolation.
    """
    dense, all_sorted = _build_cubic_dense_volume(
        volume, observed_indices, all_indices
    )

    # Ensure C-contiguous float32 for denoiser backends
    dense = np.ascontiguousarray(dense, dtype=np.float32)

    denoised = None
    try:
        from bm4d import bm4d  # type: ignore
    except ImportError:
        print("  [cubic_bm4d] bm4d not installed; falling back to skimage non-local means 3D")
        try:
            from skimage.restoration import denoise_nl_means, estimate_sigma

            est_sigma = float(
                np.mean(estimate_sigma(dense, channel_axis=None))
            )
            # Use explicit sigma to avoid over-smoothing constant regions
            h_val = max(sigma_psd, 0.5 * est_sigma)
            denoised = denoise_nl_means(
                dense,
                patch_size=3,
                patch_distance=5,
                h=h_val,
                fast_mode=True,
                sigma=est_sigma,
                channel_axis=None,
            ).astype(np.float32)
            print(f"  [cubic_bm4d] NLM3D applied (h={h_val:.4f}, sigma_est={est_sigma:.4f})")
        except Exception as exc2:
            print(
                f"  [cubic_bm4d] skimage NLM3D also failed ({type(exc2).__name__}); "
                f"returning plain cubic as fallback"
            )
            denoised = dense
    else:
        # BM4D expects (D1, D2, D3); use as-is since it is shape-agnostic
        bm4d_profile = "np" if profile == "lc" else profile
        denoised = bm4d(
            dense.astype("float32", copy=False),
            sigma_psd=sigma_psd,
            profile=bm4d_profile,
        ).astype(np.float32)
        print(
            f"  [cubic_bm4d] BM4D denoise applied "
            f"(sigma={sigma_psd}, profile={bm4d_profile})"
        )

    # Critical: preserve observed slices exactly (they are ground truth,
    # we must not denoise them away). Only target z positions get denoised.
    obs_set = set(int(z) for z in observed_indices)
    for i, z_idx in enumerate(all_sorted):
        if int(z_idx) in obs_set:
            denoised[:, :, i] = volume[:, :, int(z_idx)].astype(np.float32)

    denoised = np.clip(denoised, 0.0, 1.0)
    return denoised, all_sorted


def interpolate_sinc3d(
    volume: np.ndarray,
    observed_indices: np.ndarray,
    all_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sinc interpolation in frequency domain via FFT zero-padding along z.

    Treats the observed slices as uniformly sampled signal along z and
    performs ideal band-limited reconstruction by zero-padding its DFT.
    Naturally exploits full 3D signal statistics for smooth but sharp
    reconstruction in regions where cubic blurs.

    Assumes uniform spacing of observed indices (i.e. sparse simulator
    dropped every R-th slice). If spacing is non-uniform, falls back to
    cubic interpolation.

    Args:
        volume: Full CT volume (H, W, D).
        observed_indices: Indices of observed slices (sorted).
        all_indices: All z-indices to reconstruct.

    Returns:
        (dense_volume, sorted_indices).
    """
    H, W, _ = volume.shape
    sorted_obs = np.sort(observed_indices).astype(np.int64)
    all_sorted = np.sort(all_indices).astype(np.int64)

    if len(sorted_obs) < 4:
        return _build_cubic_dense_volume(volume, observed_indices, all_indices)

    gaps = np.diff(sorted_obs)
    if not np.all(gaps == gaps[0]):
        # Non-uniform sampling breaks the DFT zero-pad assumption
        print("  [sinc3d] non-uniform gaps detected, falling back to cubic")
        return _build_cubic_dense_volume(volume, observed_indices, all_indices)

    R = int(gaps[0])
    N_obs = len(sorted_obs)
    N_target = N_obs * R  # Upsampled length

    # Stack observed slices along z
    obs_stack = volume[:, :, sorted_obs].astype(np.float32)  # (H, W, N_obs)

    # FFT along z, zero-pad, IFFT. Preserve energy by scaling by R.
    fft_obs = np.fft.fft(obs_stack, axis=-1)
    # Zero-pad symmetrically around Nyquist
    pad_total = N_target - N_obs
    # Split spectrum around Nyquist: keep low positive freqs, insert zeros at high-freq center
    half_low = (N_obs + 1) // 2
    half_high = N_obs - half_low
    fft_padded = np.zeros((H, W, N_target), dtype=np.complex64)
    fft_padded[:, :, :half_low] = fft_obs[:, :, :half_low]
    fft_padded[:, :, -half_high:] = fft_obs[:, :, -half_high:]
    fft_padded *= R  # compensate for length change

    dense_full = np.fft.ifft(fft_padded, axis=-1).real.astype(np.float32)
    # dense_full covers z in [sorted_obs[0], sorted_obs[0] + N_target - 1]

    # Map all_sorted into dense_full's local coordinate
    z_start = int(sorted_obs[0])
    dense = np.zeros((H, W, len(all_sorted)), dtype=np.float32)
    fallback_used = False
    for i, z_idx in enumerate(all_sorted):
        local_idx = int(z_idx) - z_start
        if 0 <= local_idx < N_target:
            dense[:, :, i] = dense_full[:, :, local_idx]
        else:
            # Out of interpolable region (beyond last observed); fall back to cubic
            dense[:, :, i] = ClassicalInterpolator.interpolate_target_slice(
                volume, sorted_obs, int(z_idx), "cubic"
            )
            fallback_used = True

    # Preserve observed slices exactly (sinc can oscillate; keep GT at knots)
    obs_set = set(int(z) for z in observed_indices)
    for i, z_idx in enumerate(all_sorted):
        if int(z_idx) in obs_set:
            dense[:, :, i] = volume[:, :, int(z_idx)].astype(np.float32)

    dense = np.clip(dense, 0.0, 1.0)
    if fallback_used:
        print("  [sinc3d] some slices used cubic fallback (out-of-range)")
    return dense, all_sorted


def interpolate_bm4d_standalone(
    volume: np.ndarray,
    observed_indices: np.ndarray,
    target_indices: np.ndarray,
    sigma_psd: float = 0.015,
    profile: str = "lc",
) -> np.ndarray:
    """Standalone BM4D baseline for the same evaluation protocol as cubic.

    Wraps `interpolate_cubic_bm4d` and returns only the target slices in
    the same (H, W, N_target) format used by cubic/nearest/linear so it
    plugs directly into `evaluate_volume`.

    This is a classical (non-learned, no external data) baseline that
    exploits 3D non-local self-similarity, positioned between cubic
    (1D only) and 3DGS/tri-plane (per-volume learned).

    Args:
        volume: Full CT volume (H, W, D).
        observed_indices: Indices of observed slices.
        target_indices: Indices of target slices to output.
        sigma_psd: BM4D noise std.
        profile: BM4D profile.

    Returns:
        Predicted slices (H, W, N_target) aligned with target_indices order.
    """
    all_indices = np.concatenate([observed_indices, target_indices])
    dense, all_sorted = interpolate_cubic_bm4d(
        volume, observed_indices, all_indices,
        sigma_psd=sigma_psd, profile=profile,
    )
    # Map back to target order
    index_to_local = {int(z): i for i, z in enumerate(all_sorted)}
    H, W, _ = volume.shape
    results = np.zeros((H, W, len(target_indices)), dtype=np.float32)
    for i, z_idx in enumerate(target_indices):
        results[:, :, i] = dense[:, :, index_to_local[int(z_idx)]]
    return results


def interpolate_sinc3d_standalone(
    volume: np.ndarray,
    observed_indices: np.ndarray,
    target_indices: np.ndarray,
) -> np.ndarray:
    """Standalone sinc3d baseline matching the evaluation protocol.

    Wraps `interpolate_sinc3d` and returns only target slices.
    """
    all_indices = np.concatenate([observed_indices, target_indices])
    dense, all_sorted = interpolate_sinc3d(
        volume, observed_indices, all_indices
    )
    index_to_local = {int(z): i for i, z in enumerate(all_sorted)}
    H, W, _ = volume.shape
    results = np.zeros((H, W, len(target_indices)), dtype=np.float32)
    for i, z_idx in enumerate(target_indices):
        results[:, :, i] = dense[:, :, index_to_local[int(z_idx)]]
    return results


def interpolate_unet_blend(
    volume: np.ndarray,
    observed_indices: np.ndarray,
    all_indices: np.ndarray,
    unet_predictor,
    blend_alpha: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Blend cubic interpolation with U-Net 2D predictions.

    For each target z, take cubic(z) and U-Net(z_prev_obs, z_next_obs)
    and blend: base = (1-alpha)*cubic + alpha*unet. Preserves observed
    slices exactly. Works as a learned-base option when a U-Net checkpoint
    is available.

    Args:
        volume: Full CT volume (H, W, D).
        observed_indices: Sorted observed z-indices.
        all_indices: All z-indices to reconstruct.
        unet_predictor: Callable (slice_before, slice_after) -> slice_mid.
            Must accept numpy (H, W) float32 in [0, 1] and return same.
        blend_alpha: Weight for U-Net in [0, 1]. 0 = pure cubic, 1 = pure U-Net.

    Returns:
        (dense_volume, sorted_indices).
    """
    H, W, _ = volume.shape
    sorted_obs = np.sort(observed_indices).astype(np.int64)
    all_sorted = np.sort(all_indices).astype(np.int64)
    obs_set = set(int(z) for z in observed_indices)

    dense = np.zeros((H, W, len(all_sorted)), dtype=np.float32)

    for i, z_idx in enumerate(all_sorted):
        z_idx = int(z_idx)
        if z_idx in obs_set:
            dense[:, :, i] = volume[:, :, z_idx].astype(np.float32)
            continue

        cubic_slice = ClassicalInterpolator.interpolate_target_slice(
            volume, sorted_obs, z_idx, "cubic"
        )

        # Find bracketing observed slices for U-Net
        left_mask = sorted_obs <= z_idx
        right_mask = sorted_obs >= z_idx
        if not left_mask.any() or not right_mask.any():
            dense[:, :, i] = cubic_slice
            continue

        z_prev = int(sorted_obs[left_mask][-1])
        z_next = int(sorted_obs[right_mask][0])
        if z_prev == z_next:
            dense[:, :, i] = cubic_slice
            continue

        try:
            slice_before = volume[:, :, z_prev].astype(np.float32)
            slice_after = volume[:, :, z_next].astype(np.float32)
            unet_slice = unet_predictor(slice_before, slice_after)
            unet_slice = np.clip(unet_slice, 0.0, 1.0).astype(np.float32)
            blended = (1.0 - blend_alpha) * cubic_slice + blend_alpha * unet_slice
            dense[:, :, i] = blended
        except Exception as exc:
            # Graceful fallback: if U-Net fails, use cubic only
            print(
                f"  [unet_blend] predictor failed at z={z_idx} "
                f"({type(exc).__name__}); using cubic"
            )
            dense[:, :, i] = cubic_slice

    dense = np.clip(dense, 0.0, 1.0)
    return dense, all_sorted
