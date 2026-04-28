"""
Per-volume 3DGS training loop for CT slice interpolation.
Optimizes a set of 3D Gaussians to represent a single CT volume,
then renders interpolated slices at unseen z-positions.

Supports:
- Residual learning: 3DGS predicts residual on top of cubic/linear base
- Error-map densification: densify based on reconstruction error
- FFT high-frequency loss: penalize high-freq discrepancies
- HU gradient-weighted loss: focus on organ boundaries
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import json

from ..models.gaussian_volume import GaussianVolume
from ..models.slice_renderer import SliceRenderer
from ..models.classical_interp import (
    ClassicalInterpolator,
    interpolate_cubic_bm4d,
    interpolate_sinc3d,
    interpolate_unet_blend,
)
from ..losses.regularization import TotalLoss


# Set of residual bases that require computing a dense 3D volume once,
# rather than interpolating per-slice with the z-axis-only cubic helper.
# These bases exploit x-y correlation / self-similarity or require a
# trained network to query.
VOLUME_LEVEL_BASES = {"cubic_bm4d", "sinc3d", "unet_blend"}


class Trainer3DGS:
    """Per-volume 3DGS trainer for CT slice interpolation.

    When residual_mode is enabled, the training target changes from
    the raw ground truth slice to (gt_slice - cubic_base), and the
    final prediction becomes (cubic_base + 3dgs_residual). This
    guarantees PSNR >= cubic baseline (since residual=0 gives cubic)
    and lets 3DGS focus on learning fine detail corrections.
    """

    def __init__(
        self,
        volume: np.ndarray,
        observed_indices: np.ndarray,
        target_indices: np.ndarray,
        config: Dict,
        labels: Optional[np.ndarray] = None,
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints/3dgs",
    ):
        """Initialize 3DGS trainer for a single CT volume.

        Args:
            volume: Preprocessed CT volume (H, W, D), values in [0, 1].
            observed_indices: Indices of observed (input) slices.
            target_indices: Indices of target (to interpolate) slices.
            config: Configuration dictionary with Gaussian/training params.
            labels: Optional organ segmentation labels (H, W, D).
            device: Computation device.
            checkpoint_dir: Directory for saving checkpoints.
        """
        self.volume = volume
        self.observed_indices = observed_indices
        self.target_indices = target_indices
        self.labels = labels
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        H, W, D = volume.shape
        self.H, self.W, self.D = H, W, D

        # Extract config
        gs_config = config.get("gaussian", {})
        loss_config = dict(config.get("loss", {}))  # local copy to allow rescaling
        train_config = config.get("training", {})
        self._gs_config = gs_config

        # ===== RESIDUAL MODE =====
        self.residual_mode = gs_config.get("residual_mode", False)
        self.residual_base = gs_config.get("residual_base", "cubic")
        # Extra params for non-cubic bases
        self.base_bm4d_sigma = float(gs_config.get("base_bm4d_sigma", 0.015))
        self.base_unet_alpha = float(gs_config.get("base_unet_alpha", 0.5))
        self.base_unet_predictor = gs_config.get("base_unet_predictor", None)
        self.loo_ratio = gs_config.get("loo_ratio", 0.0)
        self.loo_start = gs_config.get("loo_start", 0.6)  # Start LOO training at 60% progress
        if self.residual_mode:
            print(f"  *** RESIDUAL MODE ENABLED (base: {self.residual_base}) ***")
            print(f"  3DGS will predict residual on top of {self.residual_base} interpolation")
            if self.loo_ratio > 0:
                print(f"  Two-phase LOO: standard until {self.loo_start:.0%}, then LOO ramps to {self.loo_ratio:.0%}")
        else:
            # Default loss weights in configs/default.yaml are tuned for residual_mode=True
            # (residual magnitude ~10x smaller than absolute prediction). When residual_mode
            # is OFF, applying those tuned weights to absolute predictions causes loss to
            # blow up to NaN/Inf. Rescale to recover stable training behaviour.
            rescale_map = {
                "lambda_smooth": 10.0,
                "lambda_edge": 10.0,
                "lambda_tv": 10.0,
                "lambda_fft": 0.1,
            }
            for k, factor in rescale_map.items():
                if k in loss_config:
                    loss_config[k] = float(loss_config[k]) * factor
            # Residual penalty has no meaning when there is no residual to penalize
            loss_config["lambda_residual"] = 0.0
            # SSIM is suppressed in residual mode (residuals near zero); enable it back
            if loss_config.get("ssim_weight", 0.0) < 0.05:
                loss_config["ssim_weight"] = 0.1
            print(
                "  [non-residual rescale] smooth/edge/tv x10, fft x0.1, "
                "residual=0, ssim>=0.1"
            )

        # Error-map densification config
        self.use_error_map = gs_config.get("densify_use_error_map", False)
        self.error_percentile = gs_config.get("densify_error_percentile", 95.0)

        init_mode = gs_config.get("init_mode", "grid")
        max_gs = gs_config.get("max_gaussians", 500000)
        self.use_rotation = bool(gs_config.get("use_rotation", False))
        self.rotation_warmup_frac = float(gs_config.get("rotation_warmup_frac", 0.3))
        self.use_structure_tensor = bool(gs_config.get("use_structure_tensor", False))
        if self.use_rotation:
            print(
                f"  *** ORIENTED GAUSSIANS ENABLED (quaternion rotation) ***\n"
                f"      rotation_warmup_frac={self.rotation_warmup_frac:.2f}, "
                f"structure_tensor_init={self.use_structure_tensor}"
            )

        # Adaptive z-scale: cover inter-slice gaps based on sparse ratio
        self.sparse_ratio = self._infer_sparse_ratio(observed_indices)
        base_scale_z = gs_config.get("init_scale_z", 1.0)
        adaptive_scale_z = max(base_scale_z, self.sparse_ratio * 0.6)
        print(f"  Sparse ratio ~{self.sparse_ratio}, init_scale_z: {base_scale_z} -> {adaptive_scale_z:.2f}")

        if init_mode == "adaptive":
            self.gaussian_model = GaussianVolume.from_volume_adaptive(
                volume=volume,
                observed_indices=observed_indices,
                subsample_xy=gs_config.get("subsample_xy", 4),
                edge_boost=gs_config.get("edge_boost", 2.0),
                init_scale_xy=gs_config.get("init_scale_xy", 2.0),
                init_scale_z=adaptive_scale_z,
                init_opacity=gs_config.get("init_opacity", 0.8),
                max_gaussians=max_gs,
                device=device,
                use_rotation=self.use_rotation,
                use_structure_tensor=self.use_structure_tensor,
            )
        else:
            self.gaussian_model = GaussianVolume.from_volume_grid(
                volume=volume,
                observed_indices=observed_indices,
                subsample_xy=gs_config.get("subsample_xy", 4),
                init_scale_xy=gs_config.get("init_scale_xy", 2.0),
                init_scale_z=adaptive_scale_z,
                init_opacity=gs_config.get("init_opacity", 0.8),
                max_gaussians=max_gs,
                device=device,
                use_rotation=self.use_rotation,
            )

        self.renderer = SliceRenderer(
            image_height=H,
            image_width=W,
            tile_size=gs_config.get("tile_size", 64),
            z_threshold=gs_config.get("z_threshold", 3.0),
            render_mode=gs_config.get("render_mode", "weighted"),
        ).to(device)

        self.criterion = TotalLoss(
            lambda_smooth=loss_config.get("lambda_smooth", 0.01),
            lambda_edge=loss_config.get("lambda_edge", 0.005),
            lambda_tv=loss_config.get("lambda_tv", 0.001),
            lambda_fft=loss_config.get("lambda_fft", 0.0),
            fft_cutoff=loss_config.get("fft_cutoff", 0.3),
            lambda_residual=loss_config.get("lambda_residual", 0.0),
            lambda_crossview=loss_config.get("lambda_crossview", 0.0),
            l1_weight=loss_config.get("l1_weight", 1.0),
            ssim_weight=loss_config.get("ssim_weight", 0.1),
            hu_gradient_weight=loss_config.get("hu_gradient_weight", False),
            hu_weight_max=loss_config.get("hu_weight_max", 3.0),
            multiscale=loss_config.get("multiscale", True),
        ).to(device)
        self.crossview_interval = int(
            loss_config.get("crossview_interval", 5)
        )
        self.crossview_window = int(
            loss_config.get("crossview_window", 3)
        )
        # H3d: patch-based non-local prior (self-supervision at target z).
        # lambda_patch=0 disables. Expected small (~0.01-0.05).
        self.lambda_patch = float(loss_config.get("lambda_patch", 0.0))
        self.patch_prior_k = int(loss_config.get("patch_prior_k", 5))
        self.patch_prior_interval = int(
            loss_config.get("patch_prior_interval", 5)
        )

        # Training parameters with adaptive scaling for large volumes
        base_iters = gs_config.get("num_iterations", 5000)
        num_obs = len(observed_indices)
        iter_scale = max(1.0, (num_obs / 50) ** 0.5)
        self.num_iterations = int(base_iters * iter_scale)
        self.batch_slices = gs_config.get("batch_slices", 4)
        if self.num_iterations != base_iters:
            print(f"  Adaptive iterations: {base_iters} x {iter_scale:.2f} = {self.num_iterations} "
                  f"(volume has {num_obs} observed slices)")
        # Optional hard cap on iterations (applied AFTER adaptive scaling).
        # 0 (default) = no cap, preserves historical behavior.
        iters_cap = int(gs_config.get("num_iterations_cap", 0) or 0)
        if iters_cap > 0 and self.num_iterations > iters_cap:
            print(
                f"  Iterations capped: {self.num_iterations} -> {iters_cap} "
                f"(num_iterations_cap={iters_cap})"
            )
            self.num_iterations = iters_cap
        self.warmup_iterations = gs_config.get("warmup_iterations", 200)
        self.densify_interval = gs_config.get("densify_interval", 100)
        self.densify_grad_threshold = gs_config.get(
            "densify_grad_threshold", 0.0005
        )
        self.prune_opacity_threshold = gs_config.get(
            "prune_opacity_threshold", 0.01
        )
        self.max_gaussians = gs_config.get("max_gaussians", 500000)

        # LR scheduling (exponential decay on position LR, standard in 3DGS)
        self.lr_position_init = gs_config.get("lr_position", 0.01)
        self.lr_position_final = gs_config.get("lr_position_final", 0.0001)
        self.lr_decay_start = gs_config.get("lr_decay_start", 0)

        # Opacity reset schedule
        self.opacity_reset_interval = gs_config.get("opacity_reset_interval", 0)
        self.opacity_reset_value = gs_config.get("opacity_reset_value", 0.5)

        # Mixed precision
        self.mixed_precision = train_config.get("mixed_precision", True)
        self.scaler = GradScaler(enabled=self.mixed_precision)

        # Logging
        self.checkpoint_interval = train_config.get("checkpoint_interval", 500)
        self.log_interval = train_config.get("log_interval", 50)

        # Setup optimizer (with per-parameter learning rates)
        self._setup_optimizer(gs_config)

        # Prepare observed slice tensors (pre-loaded on GPU for fast training)
        self.observed_slices = {}
        for idx in observed_indices:
            s = torch.from_numpy(
                volume[:, :, idx]
            ).unsqueeze(0).float().to(device, non_blocking=True)
            self.observed_slices[int(idx)] = s
        if device == "cuda":
            torch.cuda.synchronize()

        # ===== PRECOMPUTE CUBIC/LINEAR BASE FOR RESIDUAL MODE =====
        self.cubic_cache = {}      # Full cubic base (all observed as control)
        self.loo_cache = {}        # LOO cubic base (leave-one-out per observed)
        self.patch_prior_cache = {}  # H3d: target-z -> non-local prior tensor
        if self.residual_mode:
            self._precompute_residual_base()
            if self.loo_ratio > 0:
                self._precompute_loo_base()
        if self.lambda_patch > 0:
            self._precompute_patch_priors()

        # For residual mode: initialize Gaussian intensities to zero
        # since we want the initial 3DGS output to be zero (pure cubic base)
        if self.residual_mode:
            with torch.no_grad():
                self.gaussian_model.intensity.data.zero_()
                print(f"  Residual mode: initialized intensities to zero")

        # Training history
        self.history = {
            "iteration": [],
            "loss_total": [],
            "loss_rec": [],
            "loss_smooth": [],
            "loss_edge": [],
            "loss_tv": [],
            "loss_fft": [],
            "loss_crossview": [],
            "loss_patch": [],
            "num_gaussians": [],
            "psnr_train": [],
            "lr_position": [],
        }

        # Counter for non-finite loss occurrences (used to detect divergence)
        self.nan_skips = 0
        self.aborted_early = False

    def _precompute_residual_base(self) -> None:
        """Precompute full residual base for all slices.

        Supports multiple base types:
        - 'cubic' / 'linear' / 'nearest': per-slice z-axis interpolation
          (uses ALL observed slices as control points; cubic = GT at
          observed positions by Catmull-Rom property).
        - 'cubic_bm4d': cubic dense volume + BM4D 3D denoising; exploits
          x-y non-local self-similarity beyond what cubic offers.
        - 'sinc3d': FFT zero-pad along z; band-limited reconstruction.
        - 'unet_blend': blend of cubic and a pretrained U-Net 2D predictor.
        """
        method = self.residual_base
        sorted_obs = np.sort(self.observed_indices)

        all_z = sorted(
            set(int(z) for z in self.observed_indices)
            | set(int(z) for z in self.target_indices)
        )

        print(f"  Precomputing FULL {method} base for {len(all_z)} slices "
              f"(using {len(sorted_obs)} observed as control points)...")

        t_start = time.time()

        if method in VOLUME_LEVEL_BASES:
            all_z_arr = np.array(all_z, dtype=np.int64)
            if method == "cubic_bm4d":
                dense, all_sorted = interpolate_cubic_bm4d(
                    self.volume, sorted_obs, all_z_arr,
                    sigma_psd=self.base_bm4d_sigma,
                )
            elif method == "sinc3d":
                dense, all_sorted = interpolate_sinc3d(
                    self.volume, sorted_obs, all_z_arr,
                )
            elif method == "unet_blend":
                if self.base_unet_predictor is None:
                    raise ValueError(
                        "residual_base='unet_blend' requires base_unet_predictor "
                        "(callable) in gaussian config. Got None."
                    )
                dense, all_sorted = interpolate_unet_blend(
                    self.volume, sorted_obs, all_z_arr,
                    unet_predictor=self.base_unet_predictor,
                    blend_alpha=self.base_unet_alpha,
                )
            else:
                raise ValueError(f"Unknown volume-level base: {method}")

            for i, z_idx in enumerate(all_sorted):
                self.cubic_cache[int(z_idx)] = torch.from_numpy(
                    dense[:, :, i]
                ).unsqueeze(0).float().to(self.device, non_blocking=True)
        else:
            for z_idx in all_z:
                base_slice = ClassicalInterpolator.interpolate_target_slice(
                    self.volume, sorted_obs, z_idx, method
                )
                self.cubic_cache[z_idx] = torch.from_numpy(
                    base_slice
                ).unsqueeze(0).float().to(self.device, non_blocking=True)

        if self.device == "cuda":
            torch.cuda.synchronize()

        elapsed = time.time() - t_start
        print(f"  Precomputed {len(self.cubic_cache)} full base slices in {elapsed:.1f}s")

        # Log residual statistics
        residual_mags = []
        for z_idx in self.observed_indices:
            z_idx = int(z_idx)
            gt = self.observed_slices[z_idx]
            base = self.cubic_cache[z_idx]
            residual = (gt - base).abs().mean().item()
            residual_mags.append(residual)
        avg_res = np.mean(residual_mags)
        max_res = np.max(residual_mags)
        print(f"  Full base residual stats: mean_abs={avg_res:.6f}, max_abs={max_res:.6f}")

    def _precompute_loo_base(self) -> None:
        """Precompute leave-one-out cubic base for each observed slice.

        For each observed z_obs, computes cubic interpolation using all
        OTHER observed slices as control points. This creates nonzero
        residual targets that teach 3DGS to predict corrections.

        The LOO residual at observed z approximates the cubic prediction
        error at target z, enabling 3DGS to learn corrections that
        can exceed cubic quality at inference.
        """
        # LOO is inherently per-slice; for volume-level bases (BM4D/sinc3d/
        # unet_blend) we would need N_obs dense reconstructions which is
        # too expensive. Fallback to plain cubic LOO which is still a
        # useful learning signal (the 3DGS still predicts corrections on
        # top of the volume-level base at inference time).
        method = "cubic" if self.residual_base in VOLUME_LEVEL_BASES else self.residual_base
        if self.residual_base in VOLUME_LEVEL_BASES:
            print(
                f"  [LOO] residual_base='{self.residual_base}' uses dense 3D "
                f"reconstruction; LOO fallback method = 'cubic'"
            )
        sorted_obs = np.sort(self.observed_indices)

        print(f"  Precomputing LOO {method} base for {len(sorted_obs)} observed slices...")

        t_start = time.time()

        loo_residual_mags = []
        for z_obs in sorted_obs:
            z_obs = int(z_obs)
            # Leave out this slice from control points
            loo_obs = sorted_obs[sorted_obs != z_obs]
            if len(loo_obs) < 2:
                # Can't compute meaningful cubic with <2 control points
                self.loo_cache[z_obs] = self.cubic_cache[z_obs]
                continue

            loo_slice = ClassicalInterpolator.interpolate_target_slice(
                self.volume, loo_obs, z_obs, method
            )
            self.loo_cache[z_obs] = torch.from_numpy(
                loo_slice
            ).unsqueeze(0).float().to(self.device, non_blocking=True)

            # Track LOO residual magnitude
            gt = self.observed_slices[z_obs]
            residual = (gt - self.loo_cache[z_obs]).abs().mean().item()
            loo_residual_mags.append(residual)

        if self.device == "cuda":
            torch.cuda.synchronize()

        elapsed = time.time() - t_start
        avg_loo = np.mean(loo_residual_mags) if loo_residual_mags else 0
        max_loo = np.max(loo_residual_mags) if loo_residual_mags else 0
        print(f"  Precomputed {len(self.loo_cache)} LOO base slices in {elapsed:.1f}s")
        print(f"  LOO residual stats: mean_abs={avg_loo:.6f}, max_abs={max_loo:.6f}")
        print(f"  (LOO residuals are nonzero -> 3DGS learns meaningful corrections)")

    def _precompute_patch_priors(self) -> None:
        """H3d: precompute non-local patch prior for each target slice.

        For every target z, find the top-k most similar observed slices
        (similarity = -L1 distance on downsampled grayscale, combined with
        inverse z-distance weight). The prior is the softmax-weighted mean
        of those neighbors. This captures non-local self-similarity (a la
        NL-means / BM4D) without requiring BM4D on the full volume.

        The prior is used as an auxiliary self-supervision target during
        training: predictions at target z are regularized to match it with
        weight `lambda_patch`. This injects x-y correlation information
        that the base cubic interpolation cannot provide.
        """
        if len(self.target_indices) == 0:
            return
        sorted_obs = np.sort(self.observed_indices)
        if len(sorted_obs) < 2:
            return
        H, W = self.volume.shape[1], self.volume.shape[2]
        # Downsample to 64x64 for cheap similarity scoring.
        ds = 64 if min(H, W) > 64 else min(H, W)
        stride_h = max(1, H // ds)
        stride_w = max(1, W // ds)
        obs_tensor = torch.stack(
            [self.observed_slices[int(z)] for z in sorted_obs], dim=0
        ).squeeze(1)  # (N_obs, H, W)
        obs_ds = obs_tensor[:, ::stride_h, ::stride_w].reshape(
            len(sorted_obs), -1
        )  # (N_obs, D)
        k = min(self.patch_prior_k, len(sorted_obs))
        z_obs_t = torch.from_numpy(sorted_obs).float().to(self.device)

        print(
            f"  [H3d] Precomputing patch priors for {len(self.target_indices)} "
            f"target slices (k={k})..."
        )
        t_start = time.time()

        # For a robust similarity feature, blur slightly.
        obs_ds_norm = obs_ds - obs_ds.mean(dim=1, keepdim=True)
        obs_ds_norm = obs_ds_norm / (
            obs_ds_norm.std(dim=1, keepdim=True).clamp(min=1e-6)
        )

        for z_tgt in self.target_indices:
            z_tgt_i = int(z_tgt)
            # z-gap based prior: closer obs slices get higher weight.
            z_gaps = (z_obs_t - float(z_tgt_i)).abs()
            # Use a soft feature target: cubic base at target z (already cached
            # for residual mode) or linear average of nearest neighbors.
            if z_tgt_i in self.cubic_cache:
                anchor = self.cubic_cache[z_tgt_i].squeeze(0)
            else:
                # Nearest-neighbor average fallback.
                nearest = torch.argsort(z_gaps)[:2]
                anchor = obs_tensor[nearest].mean(dim=0)
            anchor_ds = anchor[::stride_h, ::stride_w].reshape(-1)
            anchor_ds = anchor_ds - anchor_ds.mean()
            anchor_ds = anchor_ds / anchor_ds.std().clamp(min=1e-6)
            # Cosine similarity (higher = more similar).
            sim = (obs_ds_norm * anchor_ds.unsqueeze(0)).mean(dim=1)
            # Distance penalty: exp(-gap / sigma)
            sigma = max(2.0, float(self.sparse_ratio) * 2.0)
            z_weight = torch.exp(-z_gaps / sigma)
            score = sim * z_weight
            topk_idx = torch.topk(score, k).indices
            topk_w = torch.softmax(score[topk_idx] * 4.0, dim=0)  # temperature
            prior = (obs_tensor[topk_idx] * topk_w.view(-1, 1, 1)).sum(dim=0)
            self.patch_prior_cache[z_tgt_i] = prior.unsqueeze(0).to(
                self.device, non_blocking=True
            )

        if self.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t_start
        print(
            f"  [H3d] Patch priors cached for {len(self.patch_prior_cache)} "
            f"targets in {elapsed:.1f}s"
        )

    @staticmethod
    def _infer_sparse_ratio(observed_indices: np.ndarray) -> int:
        """Infer the sparse ratio from observed slice indices."""
        if len(observed_indices) < 2:
            return 2
        sorted_idx = np.sort(observed_indices)
        gaps = np.diff(sorted_idx)
        return int(np.median(gaps)) if len(gaps) > 0 else 2

    def _get_position_lr(self, iteration: int) -> float:
        """Compute position learning rate with exponential decay."""
        if self.lr_position_init == self.lr_position_final:
            return self.lr_position_init
        if iteration < self.lr_decay_start:
            return self.lr_position_init
        t = max(0, iteration - self.lr_decay_start)
        total = max(1, self.num_iterations - self.lr_decay_start)
        lr = self.lr_position_final * (
            self.lr_position_init / self.lr_position_final
        ) ** (1 - t / total)
        return lr

    def _update_learning_rate(self, iteration: int) -> None:
        """Update position LR based on schedule."""
        new_lr = self._get_position_lr(iteration)
        for param_group in self.optimizer.param_groups:
            if param_group.get("name") == "positions":
                param_group["lr"] = new_lr

    def _setup_optimizer(self, gs_config: Dict) -> None:
        """Setup optimizer with per-parameter learning rates."""
        groups = [
            {
                "params": [self.gaussian_model.positions],
                "lr": gs_config.get("lr_position", 0.01),
                "name": "positions",
            },
            {
                "params": [self.gaussian_model.log_scales],
                "lr": gs_config.get("lr_scale", 0.005),
                "name": "scales",
            },
            {
                "params": [self.gaussian_model.raw_opacity],
                "lr": gs_config.get("lr_opacity", 0.01),
                "name": "opacity",
            },
            {
                "params": [self.gaussian_model.intensity],
                "lr": gs_config.get("lr_intensity", 0.005),
                "name": "intensity",
            },
        ]
        if self.use_rotation:
            groups.append({
                "params": [self.gaussian_model.rotation],
                # Rotation LR starts at 0 and ramps up after warmup
                "lr": 0.0,
                "name": "rotation",
            })
        self.optimizer = torch.optim.Adam(groups)

    def _rebuild_optimizer(self, gs_config: Dict = None) -> None:
        """Rebuild optimizer after densification/pruning changes parameters."""
        if gs_config is None:
            # Preserve current learning rates
            gs_config = {}
            for group in self.optimizer.param_groups:
                gs_config[f"lr_{group['name']}"] = group["lr"]

        groups = [
            {
                "params": [self.gaussian_model.positions],
                "lr": gs_config.get("lr_position", gs_config.get("lr_positions", 0.01)),
                "name": "positions",
            },
            {
                "params": [self.gaussian_model.log_scales],
                "lr": gs_config.get("lr_scale", gs_config.get("lr_scales", 0.005)),
                "name": "scales",
            },
            {
                "params": [self.gaussian_model.raw_opacity],
                "lr": gs_config.get("lr_opacity", 0.01),
                "name": "opacity",
            },
            {
                "params": [self.gaussian_model.intensity],
                "lr": gs_config.get("lr_intensity", 0.005),
                "name": "intensity",
            },
        ]
        if self.use_rotation:
            groups.append({
                "params": [self.gaussian_model.rotation],
                "lr": gs_config.get("lr_rotation", 0.0),
                "name": "rotation",
            })
        self.optimizer = torch.optim.Adam(groups)

    def _update_rotation_lr(self, iteration: int) -> None:
        """Enable rotation learning only after warmup phase.

        Initially Gaussians train with frozen identity rotation; after
        `rotation_warmup_frac` of total iterations, the rotation LR is
        ramped up so position/scale/opacity have first converged.
        """
        if not self.use_rotation:
            return
        warmup_iter = int(self.rotation_warmup_frac * self.num_iterations)
        target_lr = float(self._gs_config.get("lr_rotation", 0.001))
        if iteration < warmup_iter:
            new_lr = 0.0
        else:
            ramp = min(
                1.0,
                (iteration - warmup_iter) / max(1, warmup_iter),
            )
            new_lr = target_lr * ramp
        for group in self.optimizer.param_groups:
            if group.get("name") == "rotation":
                group["lr"] = new_lr

    def _get_training_target(self, z_idx: int) -> torch.Tensor:
        """Get the training target for a given slice index.

        Always returns the ground truth slice. In residual mode, the
        loss is computed on the FULL prediction (base + residual) vs GT,
        not on residual vs residual_target. This gives better gradient
        signal and avoids issues where observed slices have zero residual.

        Args:
            z_idx: Slice index.

        Returns:
            Training target tensor (1, H, W).
        """
        return self.observed_slices[z_idx]

    def _compose_prediction(
        self, rendered: torch.Tensor, z_idx: int, use_loo: bool = False
    ) -> torch.Tensor:
        """Compose final prediction from rendered output.

        In standard mode: returns rendered directly.
        In residual mode: returns (base + rendered_residual).

        Args:
            rendered: Raw renderer output (1, H, W).
            z_idx: Slice index.
            use_loo: If True, use leave-one-out cubic base (training only).

        Returns:
            Final prediction (1, H, W).
        """
        if self.residual_mode:
            if use_loo and z_idx in self.loo_cache:
                return self.loo_cache[z_idx] + rendered
            elif z_idx in self.cubic_cache:
                return self.cubic_cache[z_idx] + rendered
        return rendered

    def train(self) -> Dict:
        """Run the full training loop.

        Returns:
            Training history dictionary.
        """
        B = min(self.batch_slices, len(self.observed_indices))
        mode_str = "RESIDUAL" if self.residual_mode else "STANDARD"
        print(
            f"Starting 3DGS training [{mode_str}]: {self.gaussian_model.num_gaussians} "
            f"Gaussians, {len(self.observed_indices)} observed slices, "
            f"batch_slices={B}, volume shape {self.volume.shape}"
        )
        if self.use_error_map:
            print(f"  Error-map densification: ON (percentile={self.error_percentile})")

        t_start = time.time()
        obs_indices_arr = np.array(list(self.observed_indices))

        for iteration in range(self.num_iterations):
            self._update_learning_rate(iteration)
            self._update_rotation_lr(iteration)

            # Regularization annealing: coarse-to-fine
            progress = iteration / max(1, self.num_iterations - 1)
            self.criterion.set_progress(progress)

            self.optimizer.zero_grad()

            batch_z = np.random.choice(obs_indices_arr, B, replace=False)

            sum_loss = 0.0
            sum_rec = 0.0
            sum_smooth = 0.0
            sum_edge = 0.0
            sum_tv = 0.0
            sum_fft = 0.0
            sum_crossview = 0.0
            sum_patch = 0.0
            last_psnr = 0.0
            # Track if any slice in this iteration successfully produced a
            # scaled backward pass. GradScaler.step() asserts that at least
            # one backward was recorded for the optimizer; if every slice
            # was skipped by the NaN guard, we must skip the step as well.
            did_backward = False

            # Two-phase LOO scheduling:
            # Phase A (0 to loo_start): standard training, stable convergence
            # Phase B (loo_start to end): LOO ramps up, learn corrections
            use_loo_flags = {}
            if self.residual_mode and self.loo_ratio > 0 and len(self.loo_cache) > 0:
                if progress > self.loo_start:
                    # Ramp loo_ratio from 0 to max over Phase B
                    loo_progress = (progress - self.loo_start) / max(0.01, 1.0 - self.loo_start)
                    current_loo_ratio = self.loo_ratio * min(1.0, loo_progress)
                    for z in batch_z:
                        use_loo_flags[int(z)] = (np.random.random() < current_loo_ratio)
                    if iteration % 500 == 0:
                        print(f"  [LOO Phase B] progress={progress:.2f}, loo_ratio={current_loo_ratio:.2f}")

            for z_idx in batch_z:
                z_idx = int(z_idx)
                gt_slice = self._get_training_target(z_idx)
                slice_use_loo = use_loo_flags.get(z_idx, False)

                with autocast(enabled=self.mixed_precision):
                    params = self.gaussian_model.get_params()
                    rot_param = params.get("rotation")
                    rendered = self.renderer(
                        params["positions"],
                        params["scales"],
                        params["opacity"],
                        params["intensity"],
                        float(z_idx),
                        rotation=rot_param,
                    )

                    # In residual mode: compose full prediction for loss
                    # LOO base creates nonzero residual → 3DGS learns corrections
                    if self.residual_mode:
                        prediction = self._compose_prediction(
                            rendered, z_idx, use_loo=slice_use_loo
                        )
                    else:
                        prediction = rendered

                    adjacent_pred = None
                    adjacent_gt = None
                    if iteration % 3 == 0:
                        neighbor_z = z_idx + 1 if z_idx + 1 < self.D else z_idx - 1
                        if neighbor_z in self.observed_slices:
                            adjacent_gt = self.observed_slices[neighbor_z]
                            adj_rendered = self.renderer(
                                params["positions"],
                                params["scales"],
                                params["opacity"],
                                params["intensity"],
                                float(neighbor_z),
                                rotation=rot_param,
                            )
                            if self.residual_mode:
                                adjacent_pred = self._compose_prediction(
                                    adj_rendered, neighbor_z, use_loo=slice_use_loo
                                )
                            else:
                                adjacent_pred = adj_rendered

                    # Residual penalty: only apply on full-base slices (push to zero)
                    # LOO slices naturally need nonzero output, so skip penalty
                    res_out = None
                    if self.residual_mode and not slice_use_loo:
                        res_out = rendered

                    # Cross-view consistency (H3b): render a short z-stack
                    # around this slice and compare with base's z-stack. Only
                    # runs every `crossview_interval` iterations to control
                    # extra render cost.
                    cv_pred_stack = None
                    cv_base_stack = None
                    if (
                        self.criterion.lambda_crossview > 0
                        and iteration % self.crossview_interval == 0
                        and self.residual_mode
                        and len(self.cubic_cache) > 0
                    ):
                        K = max(1, self.crossview_window // 2)
                        stack_zs = [z_idx + k for k in range(-K, K + 1)]
                        stack_zs = [
                            z for z in stack_zs
                            if z in self.cubic_cache
                        ]
                        if len(stack_zs) >= 2:
                            pred_list = []
                            base_list = []
                            for zz in stack_zs:
                                if zz == z_idx:
                                    pred_list.append(prediction.squeeze(0))
                                else:
                                    # Detach neighbor renders to avoid OOM:
                                    # each extra render retains ~3 GB of
                                    # intermediates in the computation graph.
                                    # Gradient still flows through the current
                                    # slice's prediction in the stack.
                                    with torch.no_grad():
                                        ren = self.renderer(
                                            params["positions"],
                                            params["scales"],
                                            params["opacity"],
                                            params["intensity"],
                                            float(zz),
                                            rotation=rot_param,
                                        )
                                        pred_list.append(
                                            self._compose_prediction(
                                                ren, zz, use_loo=False
                                            ).squeeze(0)
                                        )
                                base_list.append(
                                    self.cubic_cache[zz].squeeze(0)
                                )
                            cv_pred_stack = torch.stack(pred_list, dim=0)
                            cv_base_stack = torch.stack(base_list, dim=0)

                    loss_dict = self.criterion(
                        prediction, gt_slice, adjacent_pred, adjacent_gt,
                        residual_output=res_out,
                        crossview_pred_stack=cv_pred_stack,
                        crossview_base_stack=cv_base_stack,
                    )
                    slice_loss = loss_dict["total"] / B

                # Guard: skip backward if loss is NaN/Inf (e.g. mismatched loss
                # weights for non-residual mode, or AMP overflow). Avoids polluting
                # parameters with NaN which would silently break the rest of training.
                if not torch.isfinite(slice_loss):
                    self.nan_skips += 1
                    if self.nan_skips <= 3 or self.nan_skips % 50 == 0:
                        print(
                            f"  [WARN] Non-finite loss at iter {iteration} "
                            f"(slice z={z_idx}), skipping. total_skips={self.nan_skips}"
                        )
                    if self.nan_skips >= 100:
                        print(
                            f"  [ABORT] >=100 non-finite losses encountered. "
                            f"Stopping training early at iter {iteration}."
                        )
                        self.aborted_early = True
                        break
                    continue

                if slice_loss.grad_fn is not None:
                    self.scaler.scale(slice_loss).backward()
                    did_backward = True

                # Detach scalars for logging (frees the computation graph)
                sum_loss += loss_dict["total"].item()
                sum_rec += loss_dict["reconstruction"].item()
                sum_smooth += loss_dict["smoothness"].item()
                sum_edge += loss_dict["edge"].item()
                sum_tv += loss_dict["tv"].item()
                sum_fft += loss_dict["fft"].item()
                if "crossview" in loss_dict:
                    sum_crossview = sum_crossview + loss_dict["crossview"].item()

                # Track error for error-map densification
                if self.use_error_map:
                    with torch.no_grad():
                        error_map = torch.abs(prediction.detach() - gt_slice.detach())
                        self.gaussian_model.accumulate_errors(
                            error_map,
                            params["positions"].detach(),
                            params["scales"].detach(),
                            float(z_idx),
                        )

                # PSNR computation on full prediction
                with torch.no_grad():
                    mse_i = ((prediction.detach() - gt_slice.detach()) ** 2).mean().item()
                    last_psnr = 10 * np.log10(1.0 / max(mse_i, 1e-10))

            if self.aborted_early:
                break

            # H3d: patch-prior self-supervision at target z.
            # Every `patch_prior_interval` iterations, render a random target
            # slice and push it toward the non-local patch prior with a small
            # weight. This injects x-y correlation information that the base
            # interpolation cannot provide.
            if (
                self.lambda_patch > 0
                and len(self.patch_prior_cache) > 0
                and iteration % self.patch_prior_interval == 0
                and did_backward
            ):
                try:
                    target_zs = list(self.patch_prior_cache.keys())
                    z_patch = int(np.random.choice(target_zs))
                    rot_param = (
                        self.gaussian_model.rotation if self.use_rotation else None
                    )
                    with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                        params_p = self.gaussian_model.get_params()
                        ren_p = self.renderer(
                            params_p["positions"],
                            params_p["scales"],
                            params_p["opacity"],
                            params_p["intensity"],
                            float(z_patch),
                            rotation=rot_param,
                        )
                        pred_p = self._compose_prediction(
                            ren_p, z_patch, use_loo=False
                        )
                        prior_p = self.patch_prior_cache[z_patch]
                        # L1 to the prior, weighted down.
                        patch_loss = self.lambda_patch * torch.nn.functional.l1_loss(
                            pred_p, prior_p
                        )
                    if torch.isfinite(patch_loss) and patch_loss.grad_fn is not None:
                        self.scaler.scale(patch_loss).backward()
                        sum_patch += patch_loss.item()
                except Exception as e:
                    if iteration % 200 == 0:
                        print(f"  [H3d] patch-prior step failed: {e}")

            # Skip optimizer step entirely if no backward was successfully
            # recorded this iteration (all slices produced NaN/Inf).
            # GradScaler would otherwise raise:
            #   "AssertionError: No inf checks were recorded for this optimizer"
            if not did_backward:
                continue

            self.gaussian_model.accumulate_gradients()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Keep quaternions unit-norm (prevents drift after Adam updates)
            if self.use_rotation and iteration % 20 == 0:
                self.gaussian_model.normalize_quaternions()

            # Sanitize parameters if AMP overflow / underflow leaked NaNs into them.
            # Resets affected entries to a benign value so subsequent iterations can
            # continue rather than silently produce NaN renderings.
            with torch.no_grad():
                sanitize_list = [
                    self.gaussian_model.positions,
                    self.gaussian_model.log_scales,
                    self.gaussian_model.raw_opacity,
                    self.gaussian_model.intensity,
                ]
                if self.use_rotation:
                    sanitize_list.append(self.gaussian_model.rotation)
                for p in sanitize_list:
                    if not torch.isfinite(p).all():
                        p.nan_to_num_(nan=0.0, posinf=1.0, neginf=-1.0)

            if iteration % self.log_interval == 0:
                cur_lr = self._get_position_lr(iteration)
                self.history["iteration"].append(iteration)
                self.history["loss_total"].append(sum_loss / B)
                self.history["loss_rec"].append(sum_rec / B)
                self.history["loss_smooth"].append(sum_smooth / B)
                self.history["loss_edge"].append(sum_edge / B)
                self.history["loss_tv"].append(sum_tv / B)
                self.history["loss_fft"].append(sum_fft / B)
                self.history["loss_crossview"].append(sum_crossview / B)
                self.history["loss_patch"].append(sum_patch)
                self.history["num_gaussians"].append(
                    self.gaussian_model.num_gaussians
                )
                self.history["psnr_train"].append(last_psnr)
                self.history["lr_position"].append(cur_lr)

                if iteration % (self.log_interval * 4) == 0:
                    elapsed = time.time() - t_start
                    fft_str = f"FFT: {sum_fft / B:.6f} | " if sum_fft > 0 else ""
                    print(
                        f"Iter {iteration}/{self.num_iterations} | "
                        f"Loss: {sum_loss / B:.6f} | "
                        f"Rec: {sum_rec / B:.6f} | "
                        f"{fft_str}"
                        f"PSNR: {last_psnr:.2f} dB | "
                        f"#GS: {self.gaussian_model.num_gaussians} | "
                        f"LR_pos: {cur_lr:.6f} | "
                        f"Time: {elapsed:.1f}s"
                    )

            # Progressive densification
            if (
                iteration > self.warmup_iterations
                and iteration % self.densify_interval == 0
                and iteration < self.num_iterations * 0.8
            ):
                phase = (iteration - self.warmup_iterations) / max(
                    1, self.num_iterations * 0.8 - self.warmup_iterations
                )
                # Threshold ramps from 0.5x to 1.5x of base
                adaptive_thresh = self.densify_grad_threshold * (0.5 + phase)

                stats = self.gaussian_model.densify_and_prune(
                    grad_threshold=adaptive_thresh,
                    opacity_threshold=self.prune_opacity_threshold,
                    max_gaussians=self.max_gaussians,
                    use_error_map=self.use_error_map,
                    error_percentile=self.error_percentile,
                )
                self._rebuild_optimizer()
                if self.device != "cpu":
                    torch.cuda.empty_cache()

                if stats["cloned"] > 0 or stats["pruned"] > 0:
                    densify_mode = "error-map" if self.use_error_map else "gradient"
                    print(
                        f"  Densify/Prune [{densify_mode}] at iter {iteration}: "
                        f"+{stats['cloned']} cloned, "
                        f"-{stats['pruned']} pruned, "
                        f"total: {stats['after']}"
                    )

            # Periodic opacity reset to allow pruning stale Gaussians
            if (
                self.opacity_reset_interval > 0
                and iteration > self.warmup_iterations
                and iteration % self.opacity_reset_interval == 0
                and iteration < self.num_iterations * 0.8
            ):
                self.gaussian_model.reset_opacity(self.opacity_reset_value)
                print(f"  Opacity reset at iter {iteration}")

            # Checkpoint (rolling: keep only latest to save disk)
            if (
                iteration > 0
                and iteration % self.checkpoint_interval == 0
            ):
                ckpt_name = f"iter_{iteration}.pt"
                self.save_checkpoint(ckpt_name)
                # Delete previous intermediate checkpoints
                for old in self.checkpoint_dir.glob("iter_*.pt"):
                    if old.name != ckpt_name:
                        old.unlink(missing_ok=True)

        total_time = time.time() - t_start
        self.training_time = total_time
        print(
            f"Training complete in {total_time:.1f}s. "
            f"Final #Gaussians: {self.gaussian_model.num_gaussians}"
        )

        # Save final checkpoint
        self.save_checkpoint("final.pt")
        self._save_history()

        return self.history

    @torch.no_grad()
    def render_interpolated_slices(self) -> np.ndarray:
        """Render all target (interpolated) slices.

        In residual mode, adds the precomputed cubic base to the
        rendered residual to produce the final prediction.

        Returns:
            Interpolated slices array (H, W, N_target).
        """
        self.gaussian_model.eval()
        params = self.gaussian_model.get_params()

        n_targets = len(self.target_indices)
        results = np.zeros((self.H, self.W, n_targets), dtype=np.float32)

        rot_param = params.get("rotation")
        batch_size = 16
        for start in range(0, n_targets, batch_size):
            end = min(start + batch_size, n_targets)
            batch_rendered = []
            for i in range(start, end):
                z_idx = int(self.target_indices[i])
                rendered = self.renderer(
                    params["positions"],
                    params["scales"],
                    params["opacity"],
                    params["intensity"],
                    float(z_idx),
                    rotation=rot_param,
                )
                # In residual mode: add cubic base
                if self.residual_mode and z_idx in self.cubic_cache:
                    rendered = self.cubic_cache[z_idx] + rendered

                batch_rendered.append(torch.clamp(rendered.squeeze(0), 0.0, 1.0))

            stacked = torch.stack(batch_rendered, dim=0).cpu().numpy()
            results[:, :, start:end] = stacked.transpose(1, 2, 0)
            del batch_rendered, stacked

        if self.device != "cpu":
            torch.cuda.empty_cache()
        return results

    @torch.no_grad()
    def render_all_slices(self) -> np.ndarray:
        """Render all slices (observed + target) for full volume reconstruction.

        Returns:
            Full reconstructed volume (H, W, D).
        """
        self.gaussian_model.eval()
        params = self.gaussian_model.get_params()

        volume = np.zeros((self.H, self.W, self.D), dtype=np.float32)
        rot_param = params.get("rotation")

        for z_idx in range(self.D):
            rendered = self.renderer(
                params["positions"],
                params["scales"],
                params["opacity"],
                params["intensity"],
                float(z_idx),
                rotation=rot_param,
            )
            # In residual mode: add cubic base if available
            if self.residual_mode and z_idx in self.cubic_cache:
                rendered = self.cubic_cache[z_idx] + rendered

            rendered = torch.clamp(rendered, 0.0, 1.0)
            volume[:, :, z_idx] = rendered.squeeze().cpu().numpy()

        return volume

    @torch.no_grad()
    def evaluate_on_targets(
        self,
        organ_labels: Optional[Dict[str, int]] = None,
        compute_perceptual: bool = False,
        lpips_device: str = "cpu",
    ) -> Dict:
        """Evaluate interpolation quality on target slices.

        Args:
            organ_labels: Optional organ label mapping for ROI metrics.
            compute_perceptual: If True, also computes H4 metrics
                (LPIPS/HFEN/GMSD) in evaluate_volume.
            lpips_device: Device for LPIPS network ("cpu" or "cuda").

        Returns:
            Dictionary with per-slice and aggregated metrics.
        """
        from ..evaluation.metrics import evaluate_volume

        t_start = time.time()
        predictions = self.render_interpolated_slices()
        inference_time = time.time() - t_start

        gt_slices = np.zeros(
            (self.H, self.W, len(self.target_indices)), dtype=np.float32
        )
        for i, z_idx in enumerate(self.target_indices):
            gt_slices[:, :, i] = self.volume[:, :, z_idx]

        result = evaluate_volume(
            predictions,
            gt_slices,
            self.target_indices,
            labels=self.labels,
            organ_labels=organ_labels,
            compute_perceptual=compute_perceptual,
            lpips_device=lpips_device,
        )

        result["summary"]["inference_time_s"] = inference_time
        result["summary"]["training_time_s"] = getattr(
            self, "training_time", 0.0
        )
        result["summary"]["num_gaussians_final"] = self.gaussian_model.num_gaussians
        result["summary"]["mae"] = float(
            np.mean(np.abs(predictions - gt_slices))
        )
        result["summary"]["residual_mode"] = self.residual_mode
        result["summary"]["training_nan_skips"] = int(getattr(self, "nan_skips", 0))
        result["summary"]["aborted_early"] = bool(getattr(self, "aborted_early", False))

        return result

    def save_checkpoint(self, filename: str) -> None:
        """Save training checkpoint."""
        path = self.checkpoint_dir / filename
        payload = {
            "positions": self.gaussian_model.positions.data.cpu(),
            "log_scales": self.gaussian_model.log_scales.data.cpu(),
            "raw_opacity": self.gaussian_model.raw_opacity.data.cpu(),
            "intensity": self.gaussian_model.intensity.data.cpu(),
            "rotation": self.gaussian_model.rotation.data.cpu(),
            "use_rotation": self.gaussian_model.use_rotation,
            "volume_shape": self.volume.shape,
            "observed_indices": self.observed_indices,
            "target_indices": self.target_indices,
            "num_gaussians": self.gaussian_model.num_gaussians,
            "history": self.history,
            "residual_mode": self.residual_mode,
            "residual_base": self.residual_base,
        }
        torch.save(payload, path)

    def load_checkpoint(self, filename: str) -> None:
        """Load training checkpoint."""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        n = checkpoint["positions"].shape[0]
        has_rotation = checkpoint.get("use_rotation", False)
        self.gaussian_model = GaussianVolume(
            n, checkpoint["volume_shape"], self.device,
            use_rotation=has_rotation,
        )
        with torch.no_grad():
            self.gaussian_model.positions.data = checkpoint["positions"].to(
                self.device
            )
            self.gaussian_model.log_scales.data = checkpoint["log_scales"].to(
                self.device
            )
            self.gaussian_model.raw_opacity.data = checkpoint[
                "raw_opacity"
            ].to(self.device)
            self.gaussian_model.intensity.data = checkpoint["intensity"].to(
                self.device
            )
            if "rotation" in checkpoint:
                self.gaussian_model.rotation.data = checkpoint["rotation"].to(
                    self.device
                )

        if "history" in checkpoint:
            self.history = checkpoint["history"]

        # Restore residual mode settings
        if "residual_mode" in checkpoint:
            self.residual_mode = checkpoint["residual_mode"]
        if "residual_base" in checkpoint:
            self.residual_base = checkpoint["residual_base"]

    def _save_history(self) -> None:
        """Save training history to JSON."""
        path = self.checkpoint_dir / "history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
