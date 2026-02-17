"""
Visualization utilities for CT slice interpolation results.
Generates comparison figures, error maps, and metric summaries.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def plot_slice_comparison(
    gt_slice: np.ndarray,
    predictions: Dict[str, np.ndarray],
    z_idx: int = 0,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = None,
    zoom_region: Optional[Tuple[int, int, int, int]] = None,
) -> plt.Figure:
    """Plot comparison of ground truth vs multiple prediction methods.

    Args:
        gt_slice: Ground truth slice (H, W).
        predictions: Dict mapping method name to predicted slice (H, W).
        z_idx: Z-index for title.
        save_path: Optional path to save figure.
        figsize: Figure size.
        zoom_region: Optional (row_start, row_end, col_start, col_end) for zoom.

    Returns:
        Matplotlib figure.
    """
    num_methods = len(predictions)
    ncols = num_methods + 1  # +1 for GT
    if zoom_region is not None:
        nrows = 2  # Original + Zoomed
    else:
        nrows = 1

    if figsize is None:
        figsize = (4 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1:
        axes = axes.reshape(1, -1)

    # Plot ground truth
    axes[0, 0].imshow(gt_slice, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Ground Truth", fontsize=10)
    axes[0, 0].axis("off")

    if zoom_region is not None:
        r1, r2, c1, c2 = zoom_region
        axes[1, 0].imshow(gt_slice[r1:r2, c1:c2], cmap="gray", vmin=0, vmax=1)
        axes[1, 0].set_title("GT (Zoomed)", fontsize=10)
        axes[1, 0].axis("off")
        # Draw rectangle on original
        rect = plt.Rectangle(
            (c1, r1), c2 - c1, r2 - r1,
            linewidth=2, edgecolor="red", facecolor="none"
        )
        axes[0, 0].add_patch(rect)

    # Plot predictions
    for j, (method_name, pred) in enumerate(predictions.items()):
        from .metrics import compute_psnr, compute_ssim

        psnr = compute_psnr(pred, gt_slice)
        ssim = compute_ssim(pred, gt_slice)

        axes[0, j + 1].imshow(pred, cmap="gray", vmin=0, vmax=1)
        axes[0, j + 1].set_title(
            f"{method_name}\nPSNR: {psnr:.2f} | SSIM: {ssim:.4f}",
            fontsize=9,
        )
        axes[0, j + 1].axis("off")

        if zoom_region is not None:
            r1, r2, c1, c2 = zoom_region
            axes[1, j + 1].imshow(
                pred[r1:r2, c1:c2], cmap="gray", vmin=0, vmax=1
            )
            axes[1, j + 1].set_title(f"{method_name} (Zoomed)", fontsize=9)
            axes[1, j + 1].axis("off")
            rect = plt.Rectangle(
                (c1, r1), c2 - c1, r2 - r1,
                linewidth=2, edgecolor="red", facecolor="none"
            )
            axes[0, j + 1].add_patch(rect)

    fig.suptitle(f"Slice Interpolation Comparison (z={z_idx})", fontsize=12)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_error_map(
    gt_slice: np.ndarray,
    predictions: Dict[str, np.ndarray],
    z_idx: int = 0,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = None,
) -> plt.Figure:
    """Plot error maps (absolute difference) for each method.

    Args:
        gt_slice: Ground truth slice (H, W).
        predictions: Dict mapping method name to predicted slice.
        z_idx: Z-index for title.
        save_path: Optional path to save figure.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    num_methods = len(predictions)
    if figsize is None:
        figsize = (4 * num_methods, 4)

    fig, axes = plt.subplots(1, num_methods, figsize=figsize)
    if num_methods == 1:
        axes = [axes]

    max_error = 0
    errors = {}
    for method_name, pred in predictions.items():
        err = np.abs(pred - gt_slice)
        errors[method_name] = err
        max_error = max(max_error, err.max())

    for j, (method_name, err) in enumerate(errors.items()):
        im = axes[j].imshow(err, cmap="hot", vmin=0, vmax=max_error)
        mae = err.mean()
        axes[j].set_title(f"{method_name}\nMAE: {mae:.4f}", fontsize=10)
        axes[j].axis("off")

    fig.colorbar(im, ax=axes, shrink=0.8, label="Absolute Error")
    fig.suptitle(f"Error Maps (z={z_idx})", fontsize=12)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_metrics_summary(
    results: Dict[str, Dict],
    metric: str = "psnr",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plot bar chart summary of metrics across methods and cases.

    Args:
        results: Nested dict {method: {case_id: {metric: value}}}.
        metric: Which metric to plot ('psnr' or 'ssim').
        save_path: Optional save path.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    import seaborn as sns

    methods = list(results.keys())
    all_values = {m: [] for m in methods}

    for method in methods:
        for case_id, metrics in results[method].items():
            if metric in metrics:
                all_values[method].append(metrics[metric])

    fig, ax = plt.subplots(figsize=figsize)

    # Box plot
    data_for_plot = []
    labels = []
    for method in methods:
        data_for_plot.extend(all_values[method])
        labels.extend([method] * len(all_values[method]))

    import pandas as pd
    df = pd.DataFrame({"Method": labels, metric.upper(): data_for_plot})

    sns.boxplot(x="Method", y=metric.upper(), data=df, ax=ax)
    sns.stripplot(
        x="Method", y=metric.upper(), data=df, ax=ax,
        color="black", alpha=0.4, size=3
    )

    ax.set_title(
        f"{metric.upper()} Comparison Across Methods",
        fontsize=14,
    )
    ax.set_ylabel(f"{metric.upper()} {'(dB)' if metric == 'psnr' else ''}")

    # Add mean annotations
    for i, method in enumerate(methods):
        mean_val = np.mean(all_values[method])
        ax.annotate(
            f"{mean_val:.2f}",
            xy=(i, mean_val),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_training_curves(
    history: Dict[str, List],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
) -> plt.Figure:
    """Plot training loss curves and metrics over iterations.

    Args:
        history: Training history dictionary.
        save_path: Optional save path.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    iterations = history.get("iteration", [])

    # Loss curves
    if "loss_total" in history:
        axes[0, 0].plot(iterations, history["loss_total"], label="Total Loss")
    if "loss_rec" in history:
        axes[0, 0].plot(iterations, history["loss_rec"], label="Reconstruction")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training Losses")
    axes[0, 0].legend()
    axes[0, 0].set_yscale("log")
    axes[0, 0].grid(True, alpha=0.3)

    # Regularization losses
    if "loss_smooth" in history:
        axes[0, 1].plot(iterations, history["loss_smooth"], label="Smoothness")
    if "loss_edge" in history:
        axes[0, 1].plot(iterations, history["loss_edge"], label="Edge")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title("Regularization Losses")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # PSNR
    if "psnr_train" in history:
        axes[1, 0].plot(iterations, history["psnr_train"])
        axes[1, 0].set_xlabel("Iteration")
        axes[1, 0].set_ylabel("PSNR (dB)")
        axes[1, 0].set_title("Training PSNR")
        axes[1, 0].grid(True, alpha=0.3)

    # Number of Gaussians
    if "num_gaussians" in history:
        axes[1, 1].plot(iterations, history["num_gaussians"])
        axes[1, 1].set_xlabel("Iteration")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].set_title("Number of Gaussians")
        axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("3DGS Training Progress", fontsize=14)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_z_error_profile(
    per_slice_metrics: List[Dict],
    method_name: str = "3DGS",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 4),
) -> plt.Figure:
    """Plot error profile along the z-axis.

    Args:
        per_slice_metrics: List of per-slice metric dicts with 'z_idx' and 'psnr'.
        method_name: Name of the method for the title.
        save_path: Optional save path.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    z_indices = [m["z_idx"] for m in per_slice_metrics]
    psnr_values = [m["psnr"] for m in per_slice_metrics]
    ssim_values = [m["ssim"] for m in per_slice_metrics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.plot(z_indices, psnr_values, "b-o", markersize=2)
    ax1.set_xlabel("Z-position (slice index)")
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_title(f"{method_name} - PSNR vs Z-position")
    ax1.grid(True, alpha=0.3)

    ax2.plot(z_indices, ssim_values, "r-o", markersize=2)
    ax2.set_xlabel("Z-position (slice index)")
    ax2.set_ylabel("SSIM")
    ax2.set_title(f"{method_name} - SSIM vs Z-position")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
