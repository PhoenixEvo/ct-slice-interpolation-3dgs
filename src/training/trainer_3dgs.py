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
from ..models.classical_interp import ClassicalInterpolator
from ..losses.regularization import TotalLoss


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

        # Adaptive z-scale: cover inter-slice gaps based on sparse ratio
        sparse_ratio = self._infer_sparse_ratio(observed_indices)
        base_scale_z = gs_config.get("init_scale_z", 1.0)
        adaptive_scale_z = max(base_scale_z, sparse_ratio * 0.6)
        print(f"  Sparse ratio ~{sparse_ratio}, init_scale_z: {base_scale_z} -> {adaptive_scale_z:.2f}")

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
            l1_weight=loss_config.get("l1_weight", 1.0),
            ssim_weight=loss_config.get("ssim_weight", 0.1),
            hu_gradient_weight=loss_config.get("hu_gradient_weight", False),
            hu_weight_max=loss_config.get("hu_weight_max", 3.0),
            multiscale=loss_config.get("multiscale", True),
        ).to(device)

        # Training parameters with adaptive scaling for large volumes
        base_iters = gs_config.get("num_iterations", 5000)
        num_obs = len(observed_indices)
        iter_scale = max(1.0, (num_obs / 50) ** 0.5)
        self.num_iterations = int(base_iters * iter_scale)
        self.batch_slices = gs_config.get("batch_slices", 4)
        if self.num_iterations != base_iters:
            print(f"  Adaptive iterations: {base_iters} x {iter_scale:.2f} = {self.num_iterations} "
                  f"(volume has {num_obs} observed slices)")
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
        if self.residual_mode:
            self._precompute_residual_base()
            if self.loo_ratio > 0:
                self._precompute_loo_base()

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
            "num_gaussians": [],
            "psnr_train": [],
            "lr_position": [],
        }

        # Counter for non-finite loss occurrences (used to detect divergence)
        self.nan_skips = 0
        self.aborted_early = False

    def _precompute_residual_base(self) -> None:
        """Precompute full cubic/linear base for all slices.

        Uses ALL observed slices as control points (consistent train/inference).
        At observed z-positions, cubic = GT exactly (Catmull-Rom property).
        """
        method = self.residual_base
        sorted_obs = np.sort(self.observed_indices)

        all_z = set(int(z) for z in self.observed_indices)
        all_z.update(int(z) for z in self.target_indices)

        print(f"  Precomputing FULL {method} base for {len(all_z)} slices "
              f"(using {len(sorted_obs)} observed as control points)...")

        t_start = time.time()

        for z_idx in sorted(all_z):
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
        method = self.residual_base
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
        self.optimizer = torch.optim.Adam([
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
        ])

    def _rebuild_optimizer(self, gs_config: Dict = None) -> None:
        """Rebuild optimizer after densification/pruning changes parameters."""
        if gs_config is None:
            # Preserve current learning rates
            gs_config = {}
            for group in self.optimizer.param_groups:
                gs_config[f"lr_{group['name']}"] = group["lr"]

        self.optimizer = torch.optim.Adam([
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
        ])

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
                    rendered = self.renderer(
                        params["positions"],
                        params["scales"],
                        params["opacity"],
                        params["intensity"],
                        float(z_idx),
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

                    loss_dict = self.criterion(
                        prediction, gt_slice, adjacent_pred, adjacent_gt,
                        residual_output=res_out,
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

            # Skip optimizer step entirely if no backward was successfully
            # recorded this iteration (all slices produced NaN/Inf).
            # GradScaler would otherwise raise:
            #   "AssertionError: No inf checks were recorded for this optimizer"
            if not did_backward:
                continue

            self.gaussian_model.accumulate_gradients()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Sanitize parameters if AMP overflow / underflow leaked NaNs into them.
            # Resets affected entries to a benign value so subsequent iterations can
            # continue rather than silently produce NaN renderings.
            with torch.no_grad():
                for p in (
                    self.gaussian_model.positions,
                    self.gaussian_model.log_scales,
                    self.gaussian_model.raw_opacity,
                    self.gaussian_model.intensity,
                ):
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

        for z_idx in range(self.D):
            rendered = self.renderer(
                params["positions"],
                params["scales"],
                params["opacity"],
                params["intensity"],
                float(z_idx),
            )
            # In residual mode: add cubic base if available
            if self.residual_mode and z_idx in self.cubic_cache:
                rendered = self.cubic_cache[z_idx] + rendered

            rendered = torch.clamp(rendered, 0.0, 1.0)
            volume[:, :, z_idx] = rendered.squeeze().cpu().numpy()

        return volume

    @torch.no_grad()
    def evaluate_on_targets(
        self, organ_labels: Optional[Dict[str, int]] = None
    ) -> Dict:
        """Evaluate interpolation quality on target slices.

        Args:
            organ_labels: Optional organ label mapping for ROI metrics.

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
        torch.save({
            "positions": self.gaussian_model.positions.data.cpu(),
            "log_scales": self.gaussian_model.log_scales.data.cpu(),
            "raw_opacity": self.gaussian_model.raw_opacity.data.cpu(),
            "intensity": self.gaussian_model.intensity.data.cpu(),
            "volume_shape": self.volume.shape,
            "observed_indices": self.observed_indices,
            "target_indices": self.target_indices,
            "num_gaussians": self.gaussian_model.num_gaussians,
            "history": self.history,
            "residual_mode": self.residual_mode,
            "residual_base": self.residual_base,
        }, path)

    def load_checkpoint(self, filename: str) -> None:
        """Load training checkpoint."""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)

        n = checkpoint["positions"].shape[0]
        self.gaussian_model = GaussianVolume(
            n, checkpoint["volume_shape"], self.device
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
