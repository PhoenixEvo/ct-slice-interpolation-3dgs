"""
Per-volume trainer for the Tri-plane INR model.

Mirrors the interface of Trainer3DGS so benchmarking notebooks can swap
the two backends with minimal changes. Supports:
- Residual mode with cubic / cubic_bm4d / sinc3d / unet_blend bases
- Multi-scale and FFT losses (reusing regularization.TotalLoss)
- Per-volume self-supervised training on observed slices only
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from ..models.classical_interp import (
    ClassicalInterpolator,
    interpolate_cubic_bm4d,
    interpolate_sinc3d,
    interpolate_unet_blend,
)
from ..models.triplane_inr import TriPlaneINR
from ..losses.regularization import TotalLoss


VOLUME_LEVEL_BASES = {"cubic_bm4d", "sinc3d", "unet_blend"}


class TrainerTriPlane:
    """Per-volume Tri-plane INR trainer for CT slice interpolation."""

    def __init__(
        self,
        volume: np.ndarray,
        observed_indices: np.ndarray,
        target_indices: np.ndarray,
        config: Dict,
        labels: Optional[np.ndarray] = None,
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints/triplane",
    ) -> None:
        self.volume = volume
        self.observed_indices = observed_indices
        self.target_indices = target_indices
        self.labels = labels
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        H, W, D = volume.shape
        self.H, self.W, self.D = H, W, D

        tri_cfg = dict(config.get("triplane", {}))
        loss_cfg = dict(config.get("loss", {}))
        train_cfg = dict(config.get("training", {}))
        self._tri_cfg = tri_cfg

        # Residual mode
        self.residual_mode = tri_cfg.get("residual_mode", True)
        self.residual_base = tri_cfg.get("residual_base", "cubic")
        self.base_bm4d_sigma = float(tri_cfg.get("base_bm4d_sigma", 0.015))
        self.base_unet_alpha = float(tri_cfg.get("base_unet_alpha", 0.5))
        self.base_unet_predictor = tri_cfg.get("base_unet_predictor", None)

        # Model
        feat_dim = tri_cfg.get("feat_dim", 32)
        res_xy = tri_cfg.get("plane_res_xy", 128)
        res_xz = tri_cfg.get("plane_res_xz", 128)
        res_yz = tri_cfg.get("plane_res_yz", 128)
        mlp_hidden = tri_cfg.get("mlp_hidden", 64)
        mlp_layers = tri_cfg.get("mlp_layers", 3)
        num_freq = tri_cfg.get("num_freq", 4)

        self.model = TriPlaneINR(
            volume_shape=(H, W, D),
            feat_dim=feat_dim,
            plane_res=(res_xy, res_xz, res_yz),
            mlp_hidden=mlp_hidden,
            mlp_layers=mlp_layers,
            num_freq=num_freq,
            use_residual_base=self.residual_mode,
        ).to(device)

        print(
            f"  [TriPlaneINR] feat_dim={feat_dim}, "
            f"planes=({res_xy},{res_xz},{res_yz}), "
            f"MLP {mlp_layers}x{mlp_hidden}, num_freq={num_freq}, "
            f"residual={self.residual_mode} (base={self.residual_base})"
        )

        # Loss
        self.criterion = TotalLoss(
            lambda_smooth=loss_cfg.get("lambda_smooth", 0.002),
            lambda_edge=loss_cfg.get("lambda_edge", 0.001),
            lambda_tv=loss_cfg.get("lambda_tv", 0.0002),
            lambda_fft=loss_cfg.get("lambda_fft", 0.05),
            fft_cutoff=loss_cfg.get("fft_cutoff", 0.25),
            lambda_residual=loss_cfg.get("lambda_residual", 0.0),
            l1_weight=loss_cfg.get("l1_weight", 1.0),
            ssim_weight=loss_cfg.get("ssim_weight", 0.0),
            hu_gradient_weight=loss_cfg.get("hu_gradient_weight", True),
            hu_weight_max=loss_cfg.get("hu_weight_max", 5.0),
            multiscale=loss_cfg.get("multiscale", True),
        ).to(device)

        # Training hyperparameters
        self.num_iterations = int(tri_cfg.get("num_iterations", 3000))
        self.samples_per_iter = int(tri_cfg.get("samples_per_iter", 16384))
        self.slice_batch = int(tri_cfg.get("slice_batch", 2))
        self.lr = float(tri_cfg.get("lr", 5e-3))
        self.lr_mlp = float(tri_cfg.get("lr_mlp", 1e-3))
        self.weight_decay = float(tri_cfg.get("weight_decay", 0.0))
        # Iterations between full-slice losses (FFT/edge/TV require images).
        # Point-based L1 is cheap and runs every iteration.
        self.slice_loss_interval = int(tri_cfg.get("slice_loss_interval", 4))
        self.chunk_render = int(tri_cfg.get("chunk_render", 65536))

        # Optimizer: separate LR for planes vs MLP (planes need larger LR)
        plane_params = [
            self.model.plane_xy,
            self.model.plane_xz,
            self.model.plane_yz,
        ]
        mlp_params = [p for p in self.model.mlp.parameters()] + [
            p for p in self.model.head.parameters()
        ]
        self.optimizer = torch.optim.Adam([
            {"params": plane_params, "lr": self.lr, "name": "planes"},
            {"params": mlp_params, "lr": self.lr_mlp, "name": "mlp"},
        ], weight_decay=self.weight_decay)

        self.mixed_precision = train_cfg.get("mixed_precision", True)
        self.scaler = GradScaler(enabled=self.mixed_precision)
        self.log_interval = train_cfg.get("log_interval", 100)
        self.checkpoint_interval = train_cfg.get("checkpoint_interval", 1000)

        # Pre-load observed slices on GPU
        self.observed_slices: Dict[int, torch.Tensor] = {}
        for idx in observed_indices:
            s = torch.from_numpy(
                volume[:, :, int(idx)]
            ).float().to(device, non_blocking=True)
            self.observed_slices[int(idx)] = s
        if device == "cuda":
            torch.cuda.synchronize()

        # Precompute base volume if in residual mode
        self.base_cache: Dict[int, torch.Tensor] = {}
        if self.residual_mode:
            self._precompute_residual_base()

        self.history = {
            "iteration": [],
            "loss_total": [],
            "loss_rec": [],
            "loss_fft": [],
            "psnr_train": [],
        }

        self.training_time = 0.0
        self.aborted_early = False
        self.nan_skips = 0

    # ------------------------------------------------------------------
    # Base precomputation
    # ------------------------------------------------------------------
    def _precompute_residual_base(self) -> None:
        method = self.residual_base
        sorted_obs = np.sort(self.observed_indices)
        all_z = sorted(
            set(int(z) for z in self.observed_indices)
            | set(int(z) for z in self.target_indices)
        )
        print(
            f"  [TriPlaneINR] Precomputing {method} base for "
            f"{len(all_z)} slices..."
        )
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
                        "residual_base='unet_blend' requires base_unet_predictor"
                    )
                dense, all_sorted = interpolate_unet_blend(
                    self.volume, sorted_obs, all_z_arr,
                    unet_predictor=self.base_unet_predictor,
                    blend_alpha=self.base_unet_alpha,
                )
            else:
                raise ValueError(f"Unknown volume-level base: {method}")
            for i, z_idx in enumerate(all_sorted):
                self.base_cache[int(z_idx)] = torch.from_numpy(
                    dense[:, :, i]
                ).float().to(self.device, non_blocking=True)
        else:
            for z_idx in all_z:
                base_slice = ClassicalInterpolator.interpolate_target_slice(
                    self.volume, sorted_obs, z_idx, method
                )
                self.base_cache[z_idx] = torch.from_numpy(
                    base_slice
                ).float().to(self.device, non_blocking=True)

        if self.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t_start
        print(
            f"  [TriPlaneINR] Precomputed {len(self.base_cache)} base slices "
            f"in {elapsed:.1f}s"
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _sample_points(self, z_idx: int) -> torch.Tensor:
        """Sample random (x, y) positions on a given z slice.

        Returns (N, 3) voxel coords and (N,) target intensities.
        """
        H, W = self.H, self.W
        N = self.samples_per_iter
        xs = torch.randint(0, H, (N,), device=self.device)
        ys = torch.randint(0, W, (N,), device=self.device)
        zs = torch.full((N,), float(z_idx), device=self.device)
        coords = torch.stack([xs.float(), ys.float(), zs], dim=-1)

        slice_tensor = self.observed_slices[int(z_idx)]
        targets = slice_tensor[xs, ys]
        if self.residual_mode and int(z_idx) in self.base_cache:
            base = self.base_cache[int(z_idx)][xs, ys]
            targets = targets - base
        return coords, targets

    def _render_full_slice(self, z_idx: int) -> torch.Tensor:
        """Render full slice as (1, H, W) with base added in residual mode."""
        base = None
        if self.residual_mode and int(z_idx) in self.base_cache:
            base = self.base_cache[int(z_idx)].unsqueeze(0)
        slice_img = self.model.render_slice(
            z_value=float(z_idx),
            image_height=self.H,
            image_width=self.W,
            base_slice=None,  # Add base outside to keep residual output accessible
            chunk=self.chunk_render,
        )
        if base is not None:
            slice_img = slice_img + base
        return slice_img

    def train(self) -> Dict:
        print(
            f"Starting TriPlane training: {self.num_iterations} iters, "
            f"{self.samples_per_iter} pts/iter x {self.slice_batch} slices, "
            f"volume {self.volume.shape}"
        )
        t_start = time.time()
        obs_list = np.array(list(self.observed_indices), dtype=np.int64)

        for iteration in range(self.num_iterations):
            progress = iteration / max(1, self.num_iterations - 1)
            self.criterion.set_progress(progress)

            self.optimizer.zero_grad(set_to_none=True)

            batch_z = np.random.choice(
                obs_list, size=min(self.slice_batch, len(obs_list)), replace=False
            )

            sum_loss = 0.0
            sum_rec = 0.0
            sum_fft = 0.0
            last_psnr = 0.0
            did_backward = False

            # Fast per-point L1 loss every iteration
            for z_idx in batch_z:
                z_idx = int(z_idx)
                coords, targets = self._sample_points(z_idx)

                with autocast(enabled=self.mixed_precision):
                    pred = self.model(coords)
                    loss_pt = F.l1_loss(pred, targets)

                if not torch.isfinite(loss_pt):
                    self.nan_skips += 1
                    continue

                scaled = loss_pt / len(batch_z)
                self.scaler.scale(scaled).backward()
                did_backward = True

                sum_loss += float(loss_pt.item())
                sum_rec += float(loss_pt.item())
                with torch.no_grad():
                    mse_i = (pred.detach() - targets.detach()).pow(2).mean().item()
                    last_psnr = 10 * np.log10(1.0 / max(mse_i, 1e-10))

            # Periodic full-slice loss: FFT, edge, TV, HU-weighted multi-scale
            if iteration % self.slice_loss_interval == 0:
                z_pick = int(np.random.choice(obs_list))
                with autocast(enabled=self.mixed_precision):
                    pred_slice = self._render_full_slice(z_pick)
                    gt_slice = self.observed_slices[z_pick].unsqueeze(0)
                    loss_dict = self.criterion(
                        pred_slice,
                        gt_slice,
                        adjacent_pred=None,
                        adjacent_target=None,
                        residual_output=None,
                    )
                    # Only take non-rec regularization terms to avoid
                    # double-counting reconstruction; per-point L1 above
                    # already provides the main reconstruction signal.
                    slice_reg = (
                        loss_dict["total"] - loss_dict["reconstruction"]
                    )

                if torch.isfinite(slice_reg):
                    scaled_reg = slice_reg
                    self.scaler.scale(scaled_reg).backward()
                    did_backward = True
                    sum_fft = float(loss_dict["fft"].item())

            if not did_backward:
                continue

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if iteration % self.log_interval == 0:
                self.history["iteration"].append(iteration)
                self.history["loss_total"].append(sum_loss)
                self.history["loss_rec"].append(sum_rec)
                self.history["loss_fft"].append(sum_fft)
                self.history["psnr_train"].append(last_psnr)

                if iteration % (self.log_interval * 4) == 0:
                    elapsed = time.time() - t_start
                    print(
                        f"Iter {iteration}/{self.num_iterations} | "
                        f"L_pt: {sum_rec:.6f} | FFT: {sum_fft:.6f} | "
                        f"PSNR(pt): {last_psnr:.2f} dB | "
                        f"Time: {elapsed:.1f}s"
                    )

            if iteration > 0 and iteration % self.checkpoint_interval == 0:
                self.save_checkpoint(f"iter_{iteration}.pt")
                for old in self.checkpoint_dir.glob("iter_*.pt"):
                    if old.name != f"iter_{iteration}.pt":
                        old.unlink(missing_ok=True)

        self.training_time = time.time() - t_start
        print(
            f"TriPlane training complete in {self.training_time:.1f}s. "
            f"NaN skips: {self.nan_skips}"
        )
        self.save_checkpoint("final.pt")
        self._save_history()
        return self.history

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def render_interpolated_slices(self) -> np.ndarray:
        self.model.eval()
        n_targets = len(self.target_indices)
        results = np.zeros((self.H, self.W, n_targets), dtype=np.float32)
        for i, z_idx in enumerate(self.target_indices):
            slice_img = self._render_full_slice(int(z_idx))
            slice_img = torch.clamp(slice_img.squeeze(0), 0.0, 1.0)
            results[:, :, i] = slice_img.cpu().numpy()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return results

    @torch.no_grad()
    def evaluate_on_targets(
        self, organ_labels: Optional[Dict[str, int]] = None
    ) -> Dict:
        from ..evaluation.metrics import evaluate_volume

        t_start = time.time()
        predictions = self.render_interpolated_slices()
        inference_time = time.time() - t_start

        gt_slices = np.zeros(
            (self.H, self.W, len(self.target_indices)), dtype=np.float32
        )
        for i, z_idx in enumerate(self.target_indices):
            gt_slices[:, :, i] = self.volume[:, :, int(z_idx)]

        result = evaluate_volume(
            predictions,
            gt_slices,
            self.target_indices,
            labels=self.labels,
            organ_labels=organ_labels,
        )
        result["summary"]["inference_time_s"] = inference_time
        result["summary"]["training_time_s"] = float(self.training_time)
        result["summary"]["num_parameters"] = int(
            sum(p.numel() for p in self.model.parameters())
        )
        result["summary"]["mae"] = float(np.mean(np.abs(predictions - gt_slices)))
        result["summary"]["residual_mode"] = self.residual_mode
        result["summary"]["residual_base"] = self.residual_base
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_checkpoint(self, filename: str) -> None:
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "volume_shape": self.volume.shape,
                "observed_indices": self.observed_indices,
                "target_indices": self.target_indices,
                "residual_mode": self.residual_mode,
                "residual_base": self.residual_base,
                "history": self.history,
            },
            path,
        )

    def load_checkpoint(self, filename: str) -> None:
        path = self.checkpoint_dir / filename
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        if "history" in ckpt:
            self.history = ckpt["history"]

    def _save_history(self) -> None:
        path = self.checkpoint_dir / "history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
