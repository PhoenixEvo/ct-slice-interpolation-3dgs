"""
Evaluation metrics for CT slice interpolation.
Includes PSNR, SSIM, and ROI-based metrics.
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from typing import Dict, List, Optional, Tuple

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# LPIPS is an optional dependency. We lazy-load it on first use because the
# model weights are downloaded at import time which is expensive.
_LPIPS_NET = None
_LPIPS_DEVICE = None


def _get_lpips_net(device: str = "cpu"):
    """Lazily construct and cache a LPIPS model."""
    global _LPIPS_NET, _LPIPS_DEVICE
    if not _HAS_TORCH:
        return None
    if _LPIPS_NET is not None and _LPIPS_DEVICE == device:
        return _LPIPS_NET
    try:
        import lpips  # type: ignore
    except ImportError:
        return None
    try:
        # AlexNet variant is the smallest/fastest.
        net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        for p in net.parameters():
            p.requires_grad_(False)
        _LPIPS_NET = net
        _LPIPS_DEVICE = device
        return net
    except Exception:
        return None


def compute_psnr(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    data_range: float = 1.0,
) -> float:
    """Compute Peak Signal-to-Noise Ratio.

    Args:
        prediction: Predicted image/slice.
        ground_truth: Ground truth image/slice.
        data_range: Data range of the images.

    Returns:
        PSNR value in dB.
    """
    return float(
        peak_signal_noise_ratio(ground_truth, prediction, data_range=data_range)
    )


def compute_ssim(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    data_range: float = 1.0,
) -> float:
    """Compute Structural Similarity Index.

    Args:
        prediction: Predicted image/slice.
        ground_truth: Ground truth image/slice.
        data_range: Data range of the images.

    Returns:
        SSIM value in [0, 1].
    """
    # Determine win_size based on smallest dimension
    min_dim = min(prediction.shape[0], prediction.shape[1])
    win_size = min(7, min_dim)
    if win_size % 2 == 0:
        win_size -= 1
    win_size = max(win_size, 3)

    return float(
        structural_similarity(
            ground_truth,
            prediction,
            data_range=data_range,
            win_size=win_size,
        )
    )


def compute_roi_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    mask: np.ndarray,
    organ_labels: Optional[Dict[str, int]] = None,
    data_range: float = 1.0,
) -> Dict[str, Dict[str, float]]:
    """Compute metrics on Region-of-Interest defined by organ masks.

    Args:
        prediction: Predicted slice (H, W).
        ground_truth: Ground truth slice (H, W).
        mask: Segmentation mask (H, W) with integer labels.
        organ_labels: Dictionary mapping organ name to label value.
        data_range: Data range of the images.

    Returns:
        Dictionary mapping organ names to their PSNR/SSIM metrics.
    """
    if organ_labels is None:
        organ_labels = {
            "liver": 1,
            "bladder": 2,
            "lungs": 3,
            "kidneys": 4,
            "bone": 5,
        }

    results = {}
    for organ_name, label_val in organ_labels.items():
        organ_mask = (mask == label_val)

        if organ_mask.sum() < 100:
            # Skip organs with too few pixels
            continue

        # Compute masked metrics
        pred_masked = prediction[organ_mask]
        gt_masked = ground_truth[organ_mask]

        # PSNR on ROI
        mse = np.mean((pred_masked - gt_masked) ** 2)
        if mse < 1e-10:
            roi_psnr = 100.0
        else:
            roi_psnr = float(10 * np.log10(data_range ** 2 / mse))

        # SSIM requires 2D patches; compute on bounding box instead
        roi_ssim = _compute_bbox_ssim(
            prediction, ground_truth, organ_mask, data_range
        )

        results[organ_name] = {
            "psnr": roi_psnr,
            "ssim": roi_ssim,
            "num_pixels": int(organ_mask.sum()),
        }

    return results


def _compute_bbox_ssim(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    mask: np.ndarray,
    data_range: float = 1.0,
) -> float:
    """Compute SSIM on the bounding box of a mask region.

    Args:
        prediction: Full predicted slice.
        ground_truth: Full ground truth slice.
        mask: Binary mask for the organ.
        data_range: Data range.

    Returns:
        SSIM on the bounding box region.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return 0.0

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Ensure minimum size for SSIM computation
    h = rmax - rmin + 1
    w = cmax - cmin + 1
    if h < 7 or w < 7:
        return 0.0

    pred_crop = prediction[rmin:rmax + 1, cmin:cmax + 1]
    gt_crop = ground_truth[rmin:rmax + 1, cmin:cmax + 1]

    return compute_ssim(pred_crop, gt_crop, data_range)


def compute_lpips(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    data_range: float = 1.0,
    device: str = "cpu",
) -> float:
    """Compute LPIPS (Learned Perceptual Image Patch Similarity).

    Lower is better. Returns NaN if lpips/torch not available.

    Args:
        prediction: Predicted slice (H, W), values in [0, data_range].
        ground_truth: Ground truth slice (H, W).
        data_range: Data range of the images.
        device: 'cpu' or 'cuda'.

    Returns:
        LPIPS score (lower better) or float('nan') if unavailable.
    """
    net = _get_lpips_net(device=device)
    if net is None:
        return float("nan")

    def _to_tensor(x):
        # LPIPS expects (N, 3, H, W) in range [-1, 1].
        arr = np.asarray(x, dtype=np.float32) / max(float(data_range), 1e-8)
        arr = np.clip(arr, 0.0, 1.0)
        arr = arr * 2.0 - 1.0
        t = torch.from_numpy(arr).float()
        if t.dim() == 2:
            t = t.unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1)
        elif t.dim() == 3:
            if t.shape[0] == 1:
                t = t.expand(3, -1, -1)
            t = t.unsqueeze(0)
        return t.to(device)

    with torch.no_grad():
        p = _to_tensor(prediction)
        g = _to_tensor(ground_truth)
        try:
            val = net(p, g).view(-1).mean().item()
        except Exception:
            return float("nan")
    return float(val)


def compute_hfen(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    sigma: float = 1.5,
) -> float:
    """Compute High-Frequency Error Norm (HFEN).

    Standard metric in medical imaging for edge/detail fidelity. Applies a
    Laplacian-of-Gaussian (LoG) to both images and measures the L2 norm of
    the difference, normalized by the norm of the ground-truth LoG.

    Lower is better.

    Args:
        prediction: Predicted slice (H, W).
        ground_truth: Ground truth slice (H, W).
        sigma: Gaussian sigma for the LoG filter.

    Returns:
        HFEN scalar (>=0).
    """
    try:
        from scipy.ndimage import gaussian_laplace
    except ImportError:
        return float("nan")

    log_pred = gaussian_laplace(prediction.astype(np.float64), sigma=sigma)
    log_gt = gaussian_laplace(ground_truth.astype(np.float64), sigma=sigma)
    diff = log_pred - log_gt
    denom = np.sqrt(np.sum(log_gt ** 2)) + 1e-12
    return float(np.sqrt(np.sum(diff ** 2)) / denom)


def compute_gmsd(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    data_range: float = 1.0,
    c: float = 0.0026,
) -> float:
    """Compute Gradient Magnitude Similarity Deviation (GMSD).

    Efficient full-reference quality metric that correlates well with human
    perception on local structural distortions. Lower is better.

    Reference: Xue et al., "Gradient Magnitude Similarity Deviation: A Highly
    Efficient Perceptual Image Quality Index" (2013).

    Args:
        prediction: Predicted slice (H, W).
        ground_truth: Ground truth slice (H, W).
        data_range: Data range (used to scale the stability constant c).
        c: Stability constant for GMS (scaled by data_range ** 2).

    Returns:
        GMSD scalar in [0, ~0.3]; lower is better.
    """
    from scipy.ndimage import convolve

    # Prewitt gradient kernels.
    dx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float64) / 3.0
    dy = dx.T

    pred = prediction.astype(np.float64)
    gt = ground_truth.astype(np.float64)

    gx_p = convolve(pred, dx, mode="reflect")
    gy_p = convolve(pred, dy, mode="reflect")
    gx_g = convolve(gt, dx, mode="reflect")
    gy_g = convolve(gt, dy, mode="reflect")

    gm_p = np.sqrt(gx_p ** 2 + gy_p ** 2)
    gm_g = np.sqrt(gx_g ** 2 + gy_g ** 2)

    c_scaled = c * (max(float(data_range), 1e-8) ** 2)
    gms = (2.0 * gm_p * gm_g + c_scaled) / (gm_p ** 2 + gm_g ** 2 + c_scaled)
    return float(np.std(gms))


def compute_mae(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """Compute Mean Absolute Error.

    Args:
        prediction: Predicted image/slice.
        ground_truth: Ground truth image/slice.

    Returns:
        MAE value.
    """
    return float(np.mean(np.abs(prediction - ground_truth)))


def evaluate_slice(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    mask: Optional[np.ndarray] = None,
    organ_labels: Optional[Dict[str, int]] = None,
    data_range: float = 1.0,
    compute_perceptual: bool = False,
    lpips_device: str = "cpu",
) -> Dict:
    """Evaluate a single interpolated slice.

    Args:
        prediction: Predicted slice (H, W).
        ground_truth: Ground truth slice (H, W).
        mask: Optional segmentation mask (H, W).
        organ_labels: Optional organ label mapping.
        data_range: Data range.
        compute_perceptual: If True, also computes LPIPS/HFEN/GMSD.
        lpips_device: Device string for LPIPS ('cpu' or 'cuda').

    Returns:
        Dictionary with full-image and per-organ metrics.
    """
    result = {
        "psnr": compute_psnr(prediction, ground_truth, data_range),
        "ssim": compute_ssim(prediction, ground_truth, data_range),
        "mae": compute_mae(prediction, ground_truth),
    }

    if compute_perceptual:
        result["hfen"] = compute_hfen(prediction, ground_truth)
        result["gmsd"] = compute_gmsd(prediction, ground_truth, data_range)
        result["lpips"] = compute_lpips(
            prediction, ground_truth, data_range, device=lpips_device
        )

    if mask is not None:
        result["roi"] = compute_roi_metrics(
            prediction, ground_truth, mask, organ_labels, data_range
        )

    return result


def evaluate_volume(
    predictions: np.ndarray,
    ground_truths: np.ndarray,
    target_indices: np.ndarray,
    labels: Optional[np.ndarray] = None,
    organ_labels: Optional[Dict[str, int]] = None,
    data_range: float = 1.0,
    compute_perceptual: bool = False,
    lpips_device: str = "cpu",
) -> Dict:
    """Evaluate all interpolated slices for a volume.

    Args:
        predictions: Predicted slices (H, W, N).
        ground_truths: Ground truth slices (H, W, N).
        target_indices: Z-indices of target slices.
        labels: Optional label volume (H, W, D).
        organ_labels: Optional organ label mapping.
        data_range: Data range.

    Returns:
        Dictionary with per-slice and aggregated metrics.
    """
    per_slice = []
    for i in range(predictions.shape[2]):
        pred = predictions[:, :, i]
        gt = ground_truths[:, :, i]

        mask = None
        if labels is not None:
            z_idx = target_indices[i]
            if z_idx < labels.shape[2]:
                mask = labels[:, :, z_idx]

        metrics = evaluate_slice(
            pred, gt, mask, organ_labels, data_range,
            compute_perceptual=compute_perceptual,
            lpips_device=lpips_device,
        )
        metrics["z_idx"] = int(target_indices[i])
        per_slice.append(metrics)

    # Aggregate metrics
    psnr_values = [m["psnr"] for m in per_slice]
    ssim_values = [m["ssim"] for m in per_slice]
    mae_values = [m.get("mae", 0.0) for m in per_slice]

    summary = {
        "mean_psnr": float(np.mean(psnr_values)),
        "std_psnr": float(np.std(psnr_values)),
        "mean_ssim": float(np.mean(ssim_values)),
        "std_ssim": float(np.std(ssim_values)),
        "mean_mae": float(np.mean(mae_values)),
        "std_mae": float(np.std(mae_values)),
        "num_slices": len(per_slice),
    }

    # Perceptual metrics (skip NaN entries from unavailable backends).
    if compute_perceptual:
        for key in ("hfen", "gmsd", "lpips"):
            vals = [m[key] for m in per_slice if key in m and np.isfinite(m[key])]
            if vals:
                summary[f"mean_{key}"] = float(np.mean(vals))
                summary[f"std_{key}"] = float(np.std(vals))

    # Aggregate ROI metrics if available
    roi_psnr: Dict[str, List[float]] = {}
    roi_ssim: Dict[str, List[float]] = {}
    for m in per_slice:
        if "roi" in m:
            for organ, vals in m["roi"].items():
                if organ not in roi_psnr:
                    roi_psnr[organ] = []
                    roi_ssim[organ] = []
                roi_psnr[organ].append(vals["psnr"])
                roi_ssim[organ].append(vals["ssim"])

    if roi_psnr:
        summary["roi"] = {}
        for organ in roi_psnr:
            summary["roi"][organ] = {
                "mean_psnr": float(np.mean(roi_psnr[organ])),
                "mean_ssim": float(np.mean(roi_ssim[organ])),
            }

    return {
        "summary": summary,
        "per_slice": per_slice,
    }
