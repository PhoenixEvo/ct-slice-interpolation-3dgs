"""
Evaluation metrics for CT slice interpolation.
Includes PSNR, SSIM, and ROI-based metrics.
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from typing import Dict, List, Optional, Tuple


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


def evaluate_slice(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    mask: Optional[np.ndarray] = None,
    organ_labels: Optional[Dict[str, int]] = None,
    data_range: float = 1.0,
) -> Dict:
    """Evaluate a single interpolated slice.

    Args:
        prediction: Predicted slice (H, W).
        ground_truth: Ground truth slice (H, W).
        mask: Optional segmentation mask (H, W).
        organ_labels: Optional organ label mapping.
        data_range: Data range.

    Returns:
        Dictionary with full-image and per-organ metrics.
    """
    result = {
        "psnr": compute_psnr(prediction, ground_truth, data_range),
        "ssim": compute_ssim(prediction, ground_truth, data_range),
    }

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

        metrics = evaluate_slice(pred, gt, mask, organ_labels, data_range)
        metrics["z_idx"] = int(target_indices[i])
        per_slice.append(metrics)

    # Aggregate metrics
    psnr_values = [m["psnr"] for m in per_slice]
    ssim_values = [m["ssim"] for m in per_slice]

    summary = {
        "mean_psnr": float(np.mean(psnr_values)),
        "std_psnr": float(np.std(psnr_values)),
        "mean_ssim": float(np.mean(ssim_values)),
        "std_ssim": float(np.std(ssim_values)),
        "num_slices": len(per_slice),
    }

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
