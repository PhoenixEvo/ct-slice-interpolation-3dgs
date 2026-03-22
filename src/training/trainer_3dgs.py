"""
Per-volume 3DGS training loop for CT slice interpolation.
Optimizes a set of 3D Gaussians to represent a single CT volume,
then renders interpolated slices at unseen z-positions.
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
from ..losses.regularization import TotalLoss


class Trainer3DGS:
    """Per-volume 3DGS trainer for CT slice interpolation."""

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
        loss_config = config.get("loss", {})
        train_config = config.get("training", {})
        self._gs_config = gs_config

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
            l1_weight=loss_config.get("l1_weight", 1.0),
            ssim_weight=loss_config.get("ssim_weight", 0.1),
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

        # Training history
        self.history = {
            "iteration": [],
            "loss_total": [],
            "loss_rec": [],
            "loss_smooth": [],
            "loss_edge": [],
            "loss_tv": [],
            "num_gaussians": [],
            "psnr_train": [],
            "lr_position": [],
        }

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
                "lr": gs_config.get("lr_intensity", 0.01),
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
                "lr": gs_config.get("lr_intensity", 0.01),
                "name": "intensity",
            },
        ])

    def train(self) -> Dict:
        """Run the full training loop.

        Returns:
            Training history dictionary.
        """
        B = min(self.batch_slices, len(self.observed_indices))
        print(
            f"Starting 3DGS training: {self.gaussian_model.num_gaussians} "
            f"Gaussians, {len(self.observed_indices)} observed slices, "
            f"batch_slices={B}, volume shape {self.volume.shape}"
        )

        t_start = time.time()
        obs_indices_arr = np.array(list(self.observed_indices))

        for iteration in range(self.num_iterations):
            self._update_learning_rate(iteration)

            # Regularization annealing: coarse-to-fine
            progress = iteration / max(1, self.num_iterations - 1)
            self.criterion.set_progress(progress)

            self.optimizer.zero_grad()

            batch_z = np.random.choice(obs_indices_arr, B, replace=False)

            with autocast(enabled=self.mixed_precision):
                params = self.gaussian_model.get_params()

                total_loss = torch.tensor(0.0, device=self.device)
                total_rec = torch.tensor(0.0, device=self.device)
                total_smooth = torch.tensor(0.0, device=self.device)
                total_edge = torch.tensor(0.0, device=self.device)
                total_tv = torch.tensor(0.0, device=self.device)
                last_rendered = None
                last_gt = None

                for z_idx in batch_z:
                    z_idx = int(z_idx)
                    gt_slice = self.observed_slices[z_idx]
                    rendered = self.renderer(
                        params["positions"],
                        params["scales"],
                        params["opacity"],
                        params["intensity"],
                        float(z_idx),
                    )

                    adjacent_pred = None
                    adjacent_gt = None
                    if iteration % 3 == 0:
                        neighbor_z = z_idx + 1 if z_idx + 1 < self.D else z_idx - 1
                        if neighbor_z in self.observed_slices:
                            adjacent_gt = self.observed_slices[neighbor_z]
                            adjacent_pred = self.renderer(
                                params["positions"],
                                params["scales"],
                                params["opacity"],
                                params["intensity"],
                                float(neighbor_z),
                            )

                    loss_dict = self.criterion(
                        rendered, gt_slice, adjacent_pred, adjacent_gt
                    )
                    total_loss = total_loss + loss_dict["total"]
                    total_rec = total_rec + loss_dict["reconstruction"]
                    total_smooth = total_smooth + loss_dict["smoothness"]
                    total_edge = total_edge + loss_dict["edge"]
                    total_tv = total_tv + loss_dict["tv"]
                    last_rendered = rendered
                    last_gt = gt_slice

                loss = total_loss / B

            if loss.grad_fn is None:
                # Safety: skip iteration if grad graph is broken
                # (can happen when all Gaussians are pruned from a z-region)
                continue

            self.scaler.scale(loss).backward()
            self.gaussian_model.accumulate_gradients()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if iteration % self.log_interval == 0:
                with torch.no_grad():
                    mse = ((last_rendered - last_gt) ** 2).mean().item()
                    psnr = 10 * np.log10(1.0 / max(mse, 1e-10))

                cur_lr = self._get_position_lr(iteration)
                self.history["iteration"].append(iteration)
                self.history["loss_total"].append((total_loss / B).item())
                self.history["loss_rec"].append((total_rec / B).item())
                self.history["loss_smooth"].append((total_smooth / B).item())
                self.history["loss_edge"].append((total_edge / B).item())
                self.history["loss_tv"].append((total_tv / B).item())
                self.history["num_gaussians"].append(
                    self.gaussian_model.num_gaussians
                )
                self.history["psnr_train"].append(psnr)
                self.history["lr_position"].append(cur_lr)

                if iteration % (self.log_interval * 4) == 0:
                    elapsed = time.time() - t_start
                    print(
                        f"Iter {iteration}/{self.num_iterations} | "
                        f"Loss: {(total_loss / B).item():.6f} | "
                        f"Rec: {(total_rec / B).item():.6f} | "
                        f"PSNR: {psnr:.2f} dB | "
                        f"#GS: {self.gaussian_model.num_gaussians} | "
                        f"LR_pos: {cur_lr:.6f} | "
                        f"Time: {elapsed:.1f}s"
                    )

            # Progressive densification: lower threshold early (more aggressive),
            # higher threshold late (conservative refinement)
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
                )
                self._rebuild_optimizer()

                if stats["cloned"] > 0 or stats["pruned"] > 0:
                    print(
                        f"  Densify/Prune at iter {iteration}: "
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

            # Checkpoint
            if (
                iteration > 0
                and iteration % self.checkpoint_interval == 0
            ):
                self.save_checkpoint(f"iter_{iteration}.pt")

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
                rendered = self.renderer(
                    params["positions"],
                    params["scales"],
                    params["opacity"],
                    params["intensity"],
                    float(self.target_indices[i]),
                )
                batch_rendered.append(torch.clamp(rendered.squeeze(0), 0.0, 1.0))

            stacked = torch.stack(batch_rendered, dim=0).cpu().numpy()
            results[:, :, start:end] = stacked.transpose(1, 2, 0)

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

        return result

    def save_checkpoint(self, filename: str) -> None:
        """Save training checkpoint.

        Args:
            filename: Checkpoint filename.
        """
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
        }, path)

    def load_checkpoint(self, filename: str) -> None:
        """Load training checkpoint.

        Args:
            filename: Checkpoint filename.
        """
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

    def _save_history(self) -> None:
        """Save training history to JSON."""
        path = self.checkpoint_dir / "history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
