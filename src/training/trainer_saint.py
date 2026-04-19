"""
Trainer for SAINT (Peng et al., CVPR 2020).

One SAINT model per sparse ratio R. Training:
    * Randomly pick a training volume and a sparse ratio R (fixed per model).
    * Pick a random coronal slice (shape (D, W)) or sagittal slice ((D, H)).
    * Downsample along z to (D_lr, W) / (D_lr, H) by R, then train the
      corresponding branch as a z-axis SR network.
    * For fusion training (after branches are warmed up), pick an axial
      target slice, produce coronal and sagittal HR estimates at that z,
      and train the fusion to match the axial GT.

Inference on a test volume:
    * For each coronal y: LR coronal slice -> HR via coronal_net.
    * For each sagittal x: LR sagittal slice -> HR via sagittal_net.
    * For each HR target z: fuse (coronal_HR[:, :, z], sagittal_HR[:, :, z]).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from ..losses.reconstruction import CombinedReconstructionLoss
from ..models.saint import SAINT


def _load_volume_lazy(
    volume_path: str,
    hu_min: float = -1000.0,
    hu_max: float = 1000.0,
) -> np.ndarray:
    """Lazy NIfTI load with HU clip+normalize. Returns float32 (H, W, D)."""
    nii = nib.load(volume_path)
    vol = np.asarray(nii.dataobj, dtype=np.float32)
    nii.uncache()
    del nii
    np.clip(vol, hu_min, hu_max, out=vol)
    vol -= hu_min
    vol /= (hu_max - hu_min)
    return vol


def _z_downsample(slice_2d: np.ndarray, R: int) -> np.ndarray:
    """Downsample a 2D slice (L, D) along its last axis by R using average
    pooling-like stride (take every R-th column). This mirrors the sparse
    simulation convention in `SparseSimulator`.
    """
    return slice_2d[:, ::R].copy()


class SAINTSliceSampler:
    """Randomly sample coronal/sagittal training pairs from volume files."""

    def __init__(
        self,
        volume_paths: List[str],
        sparse_ratio: int,
        patch_l: int = 256,
        patch_z_hr: int = 64,
        hu_min: float = -1000.0,
        hu_max: float = 1000.0,
        cache_size: int = 2,
    ):
        from collections import OrderedDict

        self.volume_paths = list(volume_paths)
        self.R = sparse_ratio
        self.patch_l = patch_l
        self.patch_z_hr = patch_z_hr
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.cache_size = cache_size
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def _get_volume(self, path: str) -> np.ndarray:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        if len(self._cache) >= self.cache_size:
            self._cache.popitem(last=False)
        vol = _load_volume_lazy(path, self.hu_min, self.hu_max)
        self._cache[path] = vol
        return vol

    def sample_branch_pair(
        self, rng: np.random.Generator, branch: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lr_slice, hr_slice) for training one branch.

        branch='coronal' : HR slice shape (H_patch, D_patch_hr) at fixed y.
                           LR is (H_patch, D_patch_hr // R).
        branch='sagittal': HR slice shape (W_patch, D_patch_hr) at fixed x.

        Both: last axis is z (length R * ceil of D_lr); the SR model
        upsamples the z-axis.
        """
        L, Zhr = self.patch_l, self.patch_z_hr
        for _ in range(20):
            path = self.volume_paths[int(rng.integers(len(self.volume_paths)))]
            vol = self._get_volume(path)
            H, W, D = vol.shape
            if D < Zhr:
                continue
            if branch == "coronal":
                if H < 1 or W < L:
                    continue
                y = int(rng.integers(0, H))
                x = int(rng.integers(0, W - L + 1))
                z = int(rng.integers(0, D - Zhr + 1))
                hr = vol[y, x:x + L, z:z + Zhr].copy()  # shape (L, Zhr)
            elif branch == "sagittal":
                if H < L or W < 1:
                    continue
                x = int(rng.integers(0, W))
                y = int(rng.integers(0, H - L + 1))
                z = int(rng.integers(0, D - Zhr + 1))
                hr = vol[y:y + L, x, z:z + Zhr].copy()
            else:
                raise ValueError(branch)
            break
        else:
            raise RuntimeError("Could not sample a valid slice patch")

        lr = _z_downsample(hr, self.R)
        return lr, hr

    def sample_fusion_target(
        self, rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Return data needed for a fusion-training step.

        Produces the LR coronal and sagittal STRIPS needed to reconstruct a
        single axial target slice, plus the axial HR GT.

        Returns:
            cor_lr:  (H, Z_lr) coronal LR plane stacked for H y-positions?
                     -> no; we need a FULL LR coronal slice per y, which is
                     (1, H_y_rows=1, W, Z_lr). To get an axial slice of shape
                     (H, W), we need H coronal slices (one per y) and W
                     sagittal slices (one per x). That is too expensive per
                     fusion step; instead we sample a small window:
                     a random (y0:y0+hp, x0:x0+wp) axial patch and produce
                     coronal LR slices only for y in [y0, y0+hp) and sagittal
                     LR slices only for x in [x0, x0+wp). Shapes are:
                - cor_lr: (hp, wp, Z_lr)  -> hp coronal rows, each (wp, Z_lr).
                - sag_lr: (hp, wp, Z_lr)  -> wp sagittal columns, each
                  (hp, Z_lr), transposed to align with coronal layout.
                - axial_hr: (hp, wp) GT axial slice at some z_hr.
                - z_rel: the HR z-index inside [0, R * Z_lr) to reconstruct.
        """
        hp, wp = 64, 64
        Zlr_target = max(4, self.patch_z_hr // self.R)
        Zhr_target = Zlr_target * self.R

        for _ in range(20):
            path = self.volume_paths[int(rng.integers(len(self.volume_paths)))]
            vol = self._get_volume(path)
            H, W, D = vol.shape
            if H < hp or W < wp or D < Zhr_target:
                continue
            y0 = int(rng.integers(0, H - hp + 1))
            x0 = int(rng.integers(0, W - wp + 1))
            z0 = int(rng.integers(0, D - Zhr_target + 1))

            # LR coronal planes: for each y in [y0, y0+hp), extract (W, Zhr)
            # but we only need columns x0..x0+wp, so shape per y: (wp, Zhr).
            # Stack to (hp, wp, Zhr) then downsample z by R.
            cor_hr = vol[y0:y0 + hp, x0:x0 + wp, z0:z0 + Zhr_target]
            # cor_hr shape: (hp, wp, Zhr). LR along z:
            cor_lr = cor_hr[:, :, ::self.R]  # (hp, wp, Zlr)

            # Sagittal "strips": same volume region (hp, wp, Zhr), but the
            # sagittal branch should see LR in a different orientation. Since
            # the LR is created by z-downsampling, the LR axial information
            # stored in both views is identical in this window (the LR 3D
            # sub-volume). What differs is the SR model: coronal_net sees
            # "rows" along x, while sagittal_net sees "rows" along y.
            #
            # For simplicity we feed coronal_net a batch of (hp) coronal
            # slices of shape (wp, Zlr) and sagittal_net a batch of (wp)
            # sagittal slices of shape (hp, Zlr). Both upsample Zlr -> Zhr.
            sag_lr = cor_lr  # same LR content; branches differ by weights

            # Pick which z to fuse
            z_rel = int(rng.integers(0, Zhr_target))
            axial_hr = cor_hr[:, :, z_rel]  # (hp, wp)
            break
        else:
            raise RuntimeError("Could not sample fusion window")

        return cor_lr, sag_lr, axial_hr, z_rel


class TrainerSAINT:
    """Train SAINT end-to-end for one sparse ratio R."""

    def __init__(
        self,
        model: SAINT,
        train_volume_paths: List[str],
        val_volume_paths: Optional[List[str]] = None,
        sparse_ratio: int = 2,
        num_iterations: int = 80000,
        branch_phase_ratio: float = 0.6,
        patches_per_iter: int = 4,
        patch_l: int = 256,
        patch_z_hr: int = 64,
        lr: float = 1e-4,
        lr_final: float = 1e-5,
        mixed_precision: bool = True,
        checkpoint_interval: int = 5000,
        log_interval: int = 200,
        hu_min: float = -1000.0,
        hu_max: float = 1000.0,
        checkpoint_dir: str = "checkpoints/saint",
        device: str = "cuda",
    ):
        if device == "cuda" and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        self.device = device
        self.model = model.to(device)
        self.R = int(sparse_ratio)
        self.num_iterations = num_iterations
        self.branch_iterations = int(num_iterations * branch_phase_ratio)
        self.fusion_iterations = num_iterations - self.branch_iterations
        self.patches_per_iter = patches_per_iter
        self.mixed_precision = mixed_precision
        self.checkpoint_interval = checkpoint_interval
        self.log_interval = log_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.hu_min = hu_min
        self.hu_max = hu_max

        self.train_sampler = SAINTSliceSampler(
            train_volume_paths, sparse_ratio,
            patch_l=patch_l, patch_z_hr=patch_z_hr,
            hu_min=hu_min, hu_max=hu_max,
        )
        self.val_volume_paths = list(val_volume_paths or [])
        self.val_sampler = (
            SAINTSliceSampler(
                val_volume_paths, sparse_ratio,
                patch_l=patch_l, patch_z_hr=patch_z_hr,
                hu_min=hu_min, hu_max=hu_max,
            )
            if val_volume_paths
            else None
        )

        self.lr_init = lr
        self.lr_final = lr_final
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scaler = GradScaler(enabled=mixed_precision)

        # Branch loss: L1 + 0.1*SSIM (2D, same as UNet baseline for parity)
        self.criterion = CombinedReconstructionLoss(
            l1_weight=1.0, ssim_weight=0.1
        ).to(device)

        self.history: Dict[str, List[float]] = {
            "iter": [], "loss": [], "phase": [], "lr": [],
        }
        self.start_iter = 0

    def _set_lr(self, it: int) -> None:
        t = it / max(1, self.num_iterations - 1)
        lr = self.lr_final + 0.5 * (self.lr_init - self.lr_final) * (
            1.0 + float(np.cos(np.pi * t))
        )
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def _train_branch_step(
        self, branch: str, rng: np.random.Generator,
    ) -> torch.Tensor:
        """One gradient step on a single SR branch."""
        lr_list, hr_list = [], []
        for _ in range(self.patches_per_iter):
            lr_np, hr_np = self.train_sampler.sample_branch_pair(rng, branch)
            lr_list.append(lr_np)
            hr_list.append(hr_np)

        # Stack to (B, 1, L, Z)
        lr_batch = torch.from_numpy(np.stack(lr_list, axis=0)).unsqueeze(1).to(self.device)
        hr_batch = torch.from_numpy(np.stack(hr_list, axis=0)).unsqueeze(1).to(self.device)

        net = self.model.coronal_net if branch == "coronal" else self.model.sagittal_net
        with autocast(enabled=self.mixed_precision):
            pred = net(lr_batch)
            if pred.shape[-1] != hr_batch.shape[-1]:
                pred = F.interpolate(
                    pred, size=(pred.shape[-2], hr_batch.shape[-1]),
                    mode="bilinear", align_corners=False,
                )
            loss = self.criterion(pred, hr_batch)
        return loss

    def _train_fusion_step(self, rng: np.random.Generator) -> torch.Tensor:
        """One gradient step on the fusion module.

        We reconstruct a small (hp x wp) axial patch using coronal and
        sagittal branches (detached up to fusion stage) and backprop through
        the full pipeline via the fusion module.
        """
        cor_lr, sag_lr, axial_hr, z_rel = self.train_sampler.sample_fusion_target(rng)
        hp, wp, Zlr = cor_lr.shape
        Zhr = Zlr * self.R

        # Prepare batch for coronal branch:
        # hp coronal LR slices, each of shape (wp, Zlr) -> (hp, 1, wp, Zlr)
        cor_lr_t = torch.from_numpy(cor_lr).permute(0, 1, 2).contiguous()  # (hp, wp, Zlr)
        cor_lr_t = cor_lr_t.unsqueeze(1).to(self.device)  # (hp, 1, wp, Zlr)

        # sagittal branch input: for each x, slice (hp, Zlr) -> (wp, 1, hp, Zlr)
        sag_lr_t = torch.from_numpy(sag_lr).permute(1, 0, 2).contiguous()  # (wp, hp, Zlr)
        sag_lr_t = sag_lr_t.unsqueeze(1).to(self.device)  # (wp, 1, hp, Zlr)

        with autocast(enabled=self.mixed_precision):
            cor_hr = self.model.coronal_net(cor_lr_t)  # (hp, 1, wp, Zhr)
            sag_hr = self.model.sagittal_net(sag_lr_t)  # (wp, 1, hp, Zhr)

            # Extract axial estimates at z_rel
            # cor_hr[:, :, :, z_rel] has shape (hp, 1, wp) = axial at this z
            cor_axial = cor_hr[:, :, :, z_rel].unsqueeze(0)  # (1, hp, 1, wp)
            cor_axial = cor_axial.squeeze(2).unsqueeze(1)    # (1, 1, hp, wp)

            sag_axial = sag_hr[:, :, :, z_rel]  # (wp, 1, hp)
            sag_axial = sag_axial.squeeze(1).transpose(0, 1).contiguous()  # (hp, wp)
            sag_axial = sag_axial.unsqueeze(0).unsqueeze(0)  # (1, 1, hp, wp)

            fused = self.model.fusion(cor_axial, sag_axial)

            gt = torch.from_numpy(axial_hr).unsqueeze(0).unsqueeze(0).to(self.device)
            loss = self.criterion(fused, gt)
        return loss

    def train(self) -> Dict[str, List[float]]:
        """Training loop: branch-only phase, then fusion end-to-end."""
        if self.start_iter >= self.num_iterations:
            print(f"  SAINT R={self.R} training already complete "
                  f"({self.start_iter} / {self.num_iterations})")
            return self.history

        if self.start_iter > 0:
            print(f"  Resuming SAINT R={self.R} from iter {self.start_iter}")

        rng = np.random.default_rng(42 + self.R)
        t0 = time.time()
        running_loss = 0.0
        running_count = 0

        for it in range(self.start_iter, self.num_iterations):
            self._set_lr(it)
            self.optimizer.zero_grad(set_to_none=True)

            if it < self.branch_iterations:
                phase = "branch"
                branch = "coronal" if (it % 2 == 0) else "sagittal"
                loss = self._train_branch_step(branch, rng)
            else:
                phase = "fusion"
                loss = self._train_fusion_step(rng)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += float(loss.detach().cpu())
            running_count += 1

            if (it + 1) % self.log_interval == 0:
                avg = running_loss / max(running_count, 1)
                cur_lr = self.optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t0
                it_per_sec = (it + 1 - self.start_iter) / max(elapsed, 1e-3)
                eta = (self.num_iterations - it - 1) / max(it_per_sec, 1e-6)
                print(
                    f"  [SAINT R={self.R}] iter {it + 1}/{self.num_iterations} | "
                    f"phase={phase} | loss={avg:.4f} | lr={cur_lr:.2e} | "
                    f"it/s={it_per_sec:.2f} | eta={eta / 60:.1f}min"
                )
                self.history["iter"].append(it + 1)
                self.history["loss"].append(avg)
                self.history["phase"].append(0 if phase == "branch" else 1)
                self.history["lr"].append(cur_lr)
                running_loss = 0.0
                running_count = 0

            if (it + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint("last.pt", it + 1)
                self._save_history()

        self._save_checkpoint("final.pt", self.num_iterations)
        self._save_history()
        return self.history

    # ------------------------- Inference -------------------------

    @torch.no_grad()
    def infer_volume(
        self,
        lr_volume: np.ndarray,
        target_z_indices: np.ndarray,
        y_tile: int = 16,
        x_tile: int = 16,
    ) -> np.ndarray:
        """Reconstruct HR axial target slices from a LR volume.

        Args:
            lr_volume: (H, W, D_lr) float32 in [0, 1]. D_lr = D_hr / R.
            target_z_indices: HR z-indices to produce.
            y_tile / x_tile: batch-size tiles along y and x to avoid OOM.

        Returns:
            (H, W, len(target_z_indices)) predictions.
        """
        self.model.eval()
        H, W, D_lr = lr_volume.shape
        Zhr = D_lr * self.R
        device = self.device

        # Map HR target indices to [0, Zhr-1] inside the reconstructed volume
        target_z = np.asarray(target_z_indices, dtype=np.int64)
        target_z = np.clip(target_z, 0, Zhr - 1)

        # We only need HR output at the target z-indices, so we extract those
        # columns during batched branch inference to avoid holding full (H,W,Zhr)
        # float32 volumes (~300MB each) in RAM.
        Ntgt = len(target_z)
        cor_at_tgt = np.zeros((H, W, Ntgt), dtype=np.float32)
        sag_at_tgt = np.zeros((H, W, Ntgt), dtype=np.float32)

        tgt_idx_t = torch.from_numpy(target_z).long().to(device)

        # Phase 1: coronal branch
        for y0 in range(0, H, y_tile):
            y1 = min(y0 + y_tile, H)
            cor_lr = lr_volume[y0:y1, :, :]  # (B, W, D_lr)
            cor_lr_t = torch.from_numpy(cor_lr).unsqueeze(1).to(device)  # (B, 1, W, D_lr)
            with autocast(enabled=self.mixed_precision):
                cor_hr = self.model.coronal_net(cor_lr_t)  # (B, 1, W, Zhr)
            cor_hr_tgt = cor_hr.index_select(dim=-1, index=tgt_idx_t)  # (B, 1, W, Ntgt)
            cor_hr_tgt = cor_hr_tgt.squeeze(1).cpu().float().numpy()  # (B, W, Ntgt)
            cor_at_tgt[y0:y1, :, :] = cor_hr_tgt
            del cor_lr_t, cor_hr, cor_hr_tgt
            torch.cuda.empty_cache()

        # Phase 2: sagittal branch
        for x0 in range(0, W, x_tile):
            x1 = min(x0 + x_tile, W)
            sag_lr = lr_volume[:, x0:x1, :]  # (H, B, D_lr)
            sag_lr = sag_lr.transpose(1, 0, 2)  # (B, H, D_lr)
            sag_lr_t = torch.from_numpy(sag_lr).unsqueeze(1).to(device)
            with autocast(enabled=self.mixed_precision):
                sag_hr = self.model.sagittal_net(sag_lr_t)  # (B, 1, H, Zhr)
            sag_hr_tgt = sag_hr.index_select(dim=-1, index=tgt_idx_t)  # (B, 1, H, Ntgt)
            sag_hr_tgt = sag_hr_tgt.squeeze(1).cpu().float().numpy()  # (B, H, Ntgt)
            sag_hr_tgt = sag_hr_tgt.transpose(1, 0, 2)  # (H, B, Ntgt)
            sag_at_tgt[:, x0:x1, :] = sag_hr_tgt
            del sag_lr_t, sag_hr, sag_hr_tgt
            torch.cuda.empty_cache()

        # Phase 3: fuse per target z
        out = np.zeros((H, W, Ntgt), dtype=np.float32)
        for i in range(Ntgt):
            cor_axial = torch.from_numpy(cor_at_tgt[:, :, i]).unsqueeze(0).unsqueeze(0).to(device)
            sag_axial = torch.from_numpy(sag_at_tgt[:, :, i]).unsqueeze(0).unsqueeze(0).to(device)
            with autocast(enabled=self.mixed_precision):
                fused = self.model.fusion(cor_axial, sag_axial)
            out[:, :, i] = fused.squeeze(0).squeeze(0).cpu().float().numpy()

        np.clip(out, 0.0, 1.0, out=out)
        return out

    # ------------------------- Checkpoints -------------------------

    def _unwrap_model(self) -> nn.Module:
        if isinstance(self.model, nn.DataParallel):
            return self.model.module
        return self.model

    def _save_checkpoint(self, filename: str, iteration: int) -> None:
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "iteration": iteration,
                "sparse_ratio": self.R,
                "model_state_dict": self._unwrap_model().state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": self.history,
            },
            path,
        )

    def load_checkpoint(self, filename: str) -> None:
        path = self.checkpoint_dir / filename
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self._unwrap_model().load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception:
                pass
        if "history" in ckpt:
            self.history = ckpt["history"]
        self.start_iter = int(ckpt.get("iteration", 0))

    def _save_history(self) -> None:
        path = self.checkpoint_dir / "history.json"
        with open(path, "w") as f:
            json.dump(
                {k: [float(v) for v in vals] for k, vals in self.history.items()},
                f, indent=2,
            )
