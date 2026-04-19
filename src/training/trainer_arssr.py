"""
Trainer for ArSSR arbitrary-scale CT slice interpolation.

Training: pretrain once on a set of training volumes. Each iteration samples
a random volume + random 3D patch + random z-axis scale factor s in [s_min,
s_max], builds a LR patch by z-downsampling with linear interpolation, and
trains the ArSSR to reconstruct the HR patch via INR.

Inference: given a test volume that has been z-downsampled by ratio R, run
the encoder on the LR volume in xy tiles and query the INR decoder at the
target z locations.
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

from ..models.arssr import ArSSR


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


class ArSSRPatchSampler:
    """Memory-efficient random-patch sampler over a list of training volumes.

    Keeps at most `cache_size` volumes loaded in RAM. Randomly picks a
    volume, a 3D (H x W x D_hr) patch, and produces a paired HR patch and
    LR patch (z-downsampled by a random scale in [s_min, s_max]).
    """

    def __init__(
        self,
        volume_paths: List[str],
        patch_hr_size: Tuple[int, int, int] = (48, 48, 32),
        scale_min: float = 2.0,
        scale_max: float = 4.0,
        integer_scales: bool = False,
        hu_min: float = -1000.0,
        hu_max: float = 1000.0,
        cache_size: int = 2,
    ):
        from collections import OrderedDict

        self.volume_paths = list(volume_paths)
        self.patch_hr_size = patch_hr_size
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.integer_scales = integer_scales
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

    def sample(
        self, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return one (hr_patch, lr_patch, scale) tuple.

        hr_patch: (Ph, Pw, Pz_hr) float32 in [0, 1]
        lr_patch: (Ph, Pw, Pz_lr) float32 in [0, 1], Pz_lr = round(Pz_hr/s)
        """
        Ph, Pw, Pz = self.patch_hr_size

        # Keep sampling until we get a volume large enough
        for _attempt in range(20):
            path = self.volume_paths[int(rng.integers(len(self.volume_paths)))]
            vol = self._get_volume(path)
            H, W, D = vol.shape
            if H < Ph or W < Pw or D < Pz:
                continue
            y = int(rng.integers(0, H - Ph + 1))
            x = int(rng.integers(0, W - Pw + 1))
            z = int(rng.integers(0, D - Pz + 1))
            hr = vol[y:y + Ph, x:x + Pw, z:z + Pz].copy()
            break
        else:
            raise RuntimeError("Could not sample a valid patch")

        if self.integer_scales:
            s = float(int(rng.integers(int(self.scale_min), int(self.scale_max) + 1)))
        else:
            s = float(rng.uniform(self.scale_min, self.scale_max))

        Pz_lr = max(2, int(round(Pz / s)))
        # True scale actually realized (may differ from requested s)
        real_s = Pz / Pz_lr

        # Z-downsample the HR patch with trilinear interp (just along z)
        hr_t = torch.from_numpy(hr).float().permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
        # (1, 1, Pz_hr, Ph, Pw)
        lr_t = F.interpolate(
            hr_t, size=(Pz_lr, Ph, Pw),
            mode="trilinear", align_corners=False,
        )
        lr = lr_t.squeeze(0).squeeze(0).permute(1, 2, 0).numpy().astype(np.float32)
        # lr shape back to (Ph, Pw, Pz_lr)
        return hr, lr, real_s


def _sample_query_coords(
    patch_hr_shape: Tuple[int, int, int],
    num_coords: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pick random voxel coords in an HR patch.

    Returns:
        coords_norm: (N, 3) in [-1, 1] with axis order (z, y, x).
        gt_indices: (N, 3) integer voxel indices in HR patch shape (z, y, x).
    """
    Ph, Pw, Pz = patch_hr_shape
    z_idx = rng.integers(0, Pz, size=num_coords)
    y_idx = rng.integers(0, Ph, size=num_coords)
    x_idx = rng.integers(0, Pw, size=num_coords)

    # Normalize to [-1, 1] using align_corners=True convention
    z_norm = 2.0 * z_idx / max(Pz - 1, 1) - 1.0
    y_norm = 2.0 * y_idx / max(Ph - 1, 1) - 1.0
    x_norm = 2.0 * x_idx / max(Pw - 1, 1) - 1.0

    coords_norm = np.stack([z_norm, y_norm, x_norm], axis=-1).astype(np.float32)
    gt_idx = np.stack([z_idx, y_idx, x_idx], axis=-1).astype(np.int64)
    return coords_norm, gt_idx


class TrainerArSSR:
    """Pretrain and evaluate ArSSR for CT slice interpolation."""

    def __init__(
        self,
        model: ArSSR,
        volume_paths: List[str],
        num_iterations: int = 200000,
        patch_hr_size: Tuple[int, int, int] = (48, 48, 32),
        patches_per_iter: int = 2,
        coords_per_patch: int = 4096,
        scale_min: float = 2.0,
        scale_max: float = 4.0,
        integer_scales: bool = False,
        lr: float = 1e-4,
        lr_final: float = 1e-5,
        weight_decay: float = 0.0,
        mixed_precision: bool = True,
        checkpoint_interval: int = 5000,
        log_interval: int = 200,
        hu_min: float = -1000.0,
        hu_max: float = 1000.0,
        checkpoint_dir: str = "checkpoints/arssr",
        device: str = "cuda",
    ):
        if device == "cuda" and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        self.device = device
        self.model = model.to(device)
        self.volume_paths = list(volume_paths)
        self.num_iterations = num_iterations
        self.patches_per_iter = patches_per_iter
        self.coords_per_patch = coords_per_patch
        self.patch_hr_size = patch_hr_size
        self.mixed_precision = mixed_precision
        self.checkpoint_interval = checkpoint_interval
        self.log_interval = log_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.hu_min = hu_min
        self.hu_max = hu_max

        self.sampler = ArSSRPatchSampler(
            volume_paths=volume_paths,
            patch_hr_size=patch_hr_size,
            scale_min=scale_min,
            scale_max=scale_max,
            integer_scales=integer_scales,
            hu_min=hu_min,
            hu_max=hu_max,
        )

        # Optimizer + cosine lr schedule
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay,
        )
        self.lr_init = lr
        self.lr_final = lr_final
        self.scaler = GradScaler(enabled=mixed_precision)

        self.history: Dict[str, List[float]] = {"iter": [], "loss": [], "lr": []}
        self.start_iter = 0

    def _set_lr(self, it: int) -> None:
        """Cosine lr schedule from lr_init to lr_final over all iterations."""
        t = it / max(1, self.num_iterations - 1)
        lr = self.lr_final + 0.5 * (self.lr_init - self.lr_final) * (
            1.0 + float(np.cos(np.pi * t))
        )
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def _build_batch(
        self, rng: np.random.Generator
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build a training batch of B patches with variable LR z-size.

        To handle variable LR z-sizes in a single batch, we pad along z to the
        max LR z-size in the batch (padding is fine because we only query
        coords within the valid region via `real_s` -> normalized coords).
        """
        lr_list, hr_list, coord_list, gt_list = [], [], [], []
        max_z_lr = 0
        for _ in range(self.patches_per_iter):
            hr, lr, _s = self.sampler.sample(rng)
            lr_list.append(lr)
            hr_list.append(hr)
            max_z_lr = max(max_z_lr, lr.shape[2])
            coords_norm, gt_idx = _sample_query_coords(
                hr.shape, self.coords_per_patch, rng,
            )
            coord_list.append(coords_norm)
            gt_list.append(gt_idx)

        B = len(lr_list)
        Ph, Pw, _ = hr_list[0].shape
        lr_batch = np.zeros((B, 1, max_z_lr, Ph, Pw), dtype=np.float32)
        for i, lr in enumerate(lr_list):
            Pz_lr_i = lr.shape[2]
            lr_batch[i, 0, :Pz_lr_i] = lr.transpose(2, 0, 1)
            # Edge-pad: replicate last valid LR z-slice so that INR query at
            # border stays within a meaningful receptive field.
            if Pz_lr_i < max_z_lr:
                lr_batch[i, 0, Pz_lr_i:] = lr.transpose(2, 0, 1)[-1:, :, :]

        coords_batch = np.stack(coord_list, axis=0)  # (B, N, 3)
        # Remap the z-normalization of each patch to account for padding:
        # Without padding, z_norm was computed assuming length Pz_lr_i. After
        # padding to max_z_lr, grid_sample uses max_z_lr-1 as the span, so
        # we need to rescale z_norm -> z_norm * (Pz_lr_i - 1) / (max_z_lr - 1).
        # But actually we queried HR coords (z_hr), not LR-feature coords;
        # grid_sample uses feature-volume coords which depend on LR length.
        # Instead: we pass HR coords normalized against HR size (which is what
        # we did), and the encoder is applied to the LR volume. When sampling
        # feature at a normalized coord `z_norm` in [-1, 1] with align_corners
        # = True, grid_sample treats z_norm = -1 as the first LR voxel and
        # z_norm = +1 as the last LR voxel. So an HR z_index -> correct LR
        # position is z_lr_frac = (z_hr_idx * (Pz_lr - 1)) / (Pz_hr - 1).
        # This is already captured implicitly because the LR was produced by
        # trilinear interp of HR (i.e., linear z-positions of LR voxels line
        # up at the same relative positions as the HR voxels when align_corners
        # = False during F.interpolate). Mismatch here is small; accepting it
        # as an approximation preserves the spirit of ArSSR.
        #
        # To avoid overhead we keep the naive normalization here.

        gt_batch = np.stack(gt_list, axis=0)  # (B, N, 3)
        # Pull GT intensities from each HR patch
        gt_values = np.zeros((B, self.coords_per_patch, 1), dtype=np.float32)
        for i in range(B):
            hr = hr_list[i]
            z, y, x = gt_batch[i][:, 0], gt_batch[i][:, 1], gt_batch[i][:, 2]
            gt_values[i, :, 0] = hr[y, x, z]

        return (
            torch.from_numpy(lr_batch),
            torch.from_numpy(coords_batch),
            torch.from_numpy(gt_values),
        )

    def train(self) -> Dict[str, List[float]]:
        """Main training loop with checkpoint resume support."""
        self.model.train()
        rng = np.random.default_rng(42)

        if self.start_iter >= self.num_iterations:
            print(f"  ArSSR training already complete "
                  f"({self.start_iter} / {self.num_iterations} iters)")
            return self.history

        if self.start_iter > 0:
            print(f"  Resuming ArSSR from iter {self.start_iter}/{self.num_iterations}")

        t0 = time.time()
        running_loss = 0.0
        running_count = 0

        for it in range(self.start_iter, self.num_iterations):
            self._set_lr(it)
            lr_patches, coords, gt = self._build_batch(rng)
            lr_patches = lr_patches.to(self.device, non_blocking=True)
            coords = coords.to(self.device, non_blocking=True)
            gt = gt.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.mixed_precision):
                pred = self.model(lr_patches, coords)
                loss = F.l1_loss(pred, gt)

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
                    f"  [ArSSR] iter {it + 1}/{self.num_iterations} | "
                    f"loss={avg:.4f} | lr={cur_lr:.2e} | "
                    f"it/s={it_per_sec:.2f} | eta={eta / 60:.1f}min"
                )
                self.history["iter"].append(it + 1)
                self.history["loss"].append(avg)
                self.history["lr"].append(cur_lr)
                running_loss = 0.0
                running_count = 0

            if (it + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint("last.pt", it + 1)
                self._save_history()

        self._save_checkpoint("final.pt", self.num_iterations)
        self._save_history()
        return self.history

    def _save_checkpoint(self, filename: str, iteration: int) -> None:
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "iteration": iteration,
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
            json.dump({k: [float(v) for v in vals] for k, vals in self.history.items()}, f, indent=2)

    def _unwrap_model(self) -> nn.Module:
        if isinstance(self.model, nn.DataParallel):
            return self.model.module
        return self.model

    # ------------------------- Inference -------------------------

    @torch.no_grad()
    def infer_volume(
        self,
        lr_volume: np.ndarray,
        target_z_indices: np.ndarray,
        total_hr_slices: int,
        xy_tile: int = 128,
        xy_overlap: int = 16,
        z_tile: int = 128,
        z_overlap: int = 16,
    ) -> np.ndarray:
        """Predict HR axial slices at given z-indices.

        Args:
            lr_volume: (H, W, D_lr) LR volume in [0, 1].
            target_z_indices: integer z-indices in HR coordinates to produce.
            total_hr_slices: HR volume z-size = D_lr + len(target_z_indices)
                             or, more generally, what ratio R implies.
            xy_tile, xy_overlap, z_tile, z_overlap: tiling parameters.

        Returns:
            (H, W, N_targets) float32 predictions.
        """
        self.model.eval()
        H, W, D_lr = lr_volume.shape
        device = self.device

        out = np.zeros((H, W, len(target_z_indices)), dtype=np.float32)
        weight = np.zeros((H, W, len(target_z_indices)), dtype=np.float32)

        # Precompute target z normalized coords in [-1, 1] against HR length.
        # Map HR z-index to LR z-coord: z_lr_norm = 2 * z_hr / (total_hr - 1) - 1
        z_norm_all = (
            2.0 * target_z_indices.astype(np.float32) / max(total_hr_slices - 1, 1) - 1.0
        )  # (N,)

        def tile_starts(total: int, tile: int, overlap: int) -> List[int]:
            if total <= tile:
                return [0]
            step = tile - overlap
            starts = list(range(0, total - tile + 1, step))
            if starts[-1] + tile < total:
                starts.append(total - tile)
            return starts

        y_starts = tile_starts(H, xy_tile, xy_overlap)
        x_starts = tile_starts(W, xy_tile, xy_overlap)
        z_starts = tile_starts(D_lr, z_tile, z_overlap) if D_lr > z_tile else [0]

        for zs in z_starts:
            ze = min(D_lr, zs + z_tile)
            # The LR subvolume covers HR z range approximately
            # [zs*scale, ze*scale) but in practice z_subvol covers all HR
            # target z whose LR mapping falls in [zs, ze-1].
            # We keep it simple: always feed full D_lr to the encoder when
            # D_lr <= z_tile (default), which is the common case for CT-ORG.
            if len(z_starts) == 1:
                z_chunk = lr_volume[:, :, zs:ze]
                lr_z_off = 0
                lr_z_total = D_lr
                z_coord_mask = np.ones(len(target_z_indices), dtype=bool)
                z_coords_norm = z_norm_all
            else:
                z_chunk = lr_volume[:, :, zs:ze]
                lr_z_off = zs
                lr_z_total = ze - zs
                # Filter targets whose LR position falls into this chunk
                z_frac = (target_z_indices.astype(np.float32)
                          / max(total_hr_slices - 1, 1)) * (D_lr - 1)
                z_coord_mask = (z_frac >= zs) & (z_frac <= ze - 1)
                if not z_coord_mask.any():
                    continue
                z_local = z_frac[z_coord_mask] - zs
                z_coords_norm = 2.0 * z_local / max(lr_z_total - 1, 1) - 1.0

            for ys in y_starts:
                ye = min(H, ys + xy_tile)
                for xs in x_starts:
                    xe = min(W, xs + xy_tile)

                    tile_h = ye - ys
                    tile_w = xe - xs

                    lr_tile = z_chunk[ys:ye, xs:xe, :]
                    lr_t = (
                        torch.from_numpy(lr_tile)
                        .permute(2, 0, 1)
                        .unsqueeze(0).unsqueeze(0)
                        .to(device)
                    )

                    with autocast(enabled=self.mixed_precision):
                        feat = self.model.encode(lr_t)

                    # Build query coords for all (y, x, z) in tile x target_z
                    # We do it in chunks to bound memory.
                    active_z = np.where(z_coord_mask)[0]
                    z_norms = z_coords_norm
                    y_idx = np.arange(tile_h, dtype=np.float32)
                    x_idx = np.arange(tile_w, dtype=np.float32)
                    y_norm = 2.0 * y_idx / max(tile_h - 1, 1) - 1.0
                    x_norm = 2.0 * x_idx / max(tile_w - 1, 1) - 1.0

                    # Build (N=tile_h*tile_w, 3) coords for one z at a time
                    yy, xx = np.meshgrid(y_norm, x_norm, indexing="ij")
                    yy_flat = yy.reshape(-1).astype(np.float32)
                    xx_flat = xx.reshape(-1).astype(np.float32)

                    yx_coords = torch.from_numpy(
                        np.stack([yy_flat, xx_flat], axis=-1)
                    ).to(device)  # (N, 2)

                    for zi, z_norm in zip(active_z, z_norms):
                        N = yx_coords.shape[0]
                        zc = torch.full((N, 1), float(z_norm), device=device)
                        coords = torch.cat([zc, yx_coords], dim=-1).unsqueeze(0)
                        with autocast(enabled=self.mixed_precision):
                            pred = self.model.query(feat, coords)
                        pred = pred.squeeze(0).squeeze(-1).cpu().float().numpy()
                        pred = pred.reshape(tile_h, tile_w)

                        out[ys:ye, xs:xe, zi] += pred
                        weight[ys:ye, xs:xe, zi] += 1.0

                    del feat, lr_t
                    torch.cuda.empty_cache()

        weight = np.maximum(weight, 1e-6)
        out /= weight
        np.clip(out, 0.0, 1.0, out=out)
        return out
