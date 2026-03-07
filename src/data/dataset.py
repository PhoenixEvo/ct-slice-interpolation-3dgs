"""
PyTorch Dataset classes for slice interpolation experiments.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class SliceInterpolationDataset(Dataset):
    """Dataset for per-volume 3DGS slice interpolation.

    Returns observed slices and their z-positions for training,
    or target slices and positions for evaluation.
    """

    def __init__(
        self,
        volume: np.ndarray,
        observed_indices: np.ndarray,
        target_indices: Optional[np.ndarray] = None,
        mode: str = "train",
    ):
        """Initialize dataset.

        Args:
            volume: Full preprocessed volume (H, W, D).
            observed_indices: Indices of observed (input) slices.
            target_indices: Indices of target slices (for eval mode).
            mode: 'train' returns observed slices, 'eval' returns targets.
        """
        self.volume = volume
        self.observed_indices = observed_indices
        self.target_indices = target_indices
        self.mode = mode
        self.num_slices = volume.shape[2]

    def __len__(self) -> int:
        if self.mode == "train":
            return len(self.observed_indices)
        else:
            return len(self.target_indices) if self.target_indices is not None else 0

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.mode == "train":
            z_idx = self.observed_indices[idx]
        else:
            z_idx = self.target_indices[idx]

        slice_2d = self.volume[:, :, z_idx]
        # Convert to tensor (1, H, W) - single channel
        slice_tensor = torch.from_numpy(slice_2d).unsqueeze(0).float()
        # Normalize z position to [0, 1]
        z_pos = z_idx / max(self.num_slices - 1, 1)

        return {
            "slice": slice_tensor,
            "z_idx": z_idx,
            "z_pos": torch.tensor(z_pos, dtype=torch.float32),
        }

    def get_all_observed(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get all observed slices as a batch.

        Returns:
            Tuple of (slices [N, 1, H, W], z_positions [N]).
        """
        slices = []
        z_positions = []
        for idx in self.observed_indices:
            s = self.volume[:, :, idx]
            slices.append(torch.from_numpy(s).unsqueeze(0).float())
            z_positions.append(idx / max(self.num_slices - 1, 1))

        return torch.stack(slices), torch.tensor(z_positions, dtype=torch.float32)


class UNetSliceDataset(Dataset):
    """Dataset for U-Net 2D baseline training.

    Returns pairs of adjacent observed slices as input
    and the middle slice as target.
    """

    def __init__(
        self,
        volumes: List[np.ndarray],
        sparse_ratio: int = 2,
        augment: bool = False,
    ):
        """Initialize U-Net dataset.

        Args:
            volumes: List of preprocessed volumes (H, W, D).
            sparse_ratio: Sparsity ratio R.
            augment: Whether to apply data augmentation.
        """
        self.samples = []
        self.augment = augment

        for vol in volumes:
            H, W, D = vol.shape
            R = sparse_ratio

            # Get observed indices
            observed = list(range(0, D, R))

            # Create pairs
            for i in range(len(observed) - 1):
                idx_before = observed[i]
                idx_after = observed[i + 1]

                # For R=2: one target (midpoint)
                # For R=3: two targets, etc.
                for t in range(idx_before + 1, idx_after):
                    # Relative position of target between before and after
                    alpha = (t - idx_before) / (idx_after - idx_before)
                    self.samples.append({
                        "vol_ref": vol,
                        "idx_before": idx_before,
                        "idx_after": idx_after,
                        "idx_target": t,
                        "alpha": alpha,
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        vol = sample["vol_ref"]

        slice_before = vol[:, :, sample["idx_before"]]
        slice_after = vol[:, :, sample["idx_after"]]
        slice_target = vol[:, :, sample["idx_target"]]

        # Stack input as 2-channel image
        input_tensor = torch.from_numpy(
            np.stack([slice_before, slice_after], axis=0)
        ).float()

        target_tensor = torch.from_numpy(slice_target).unsqueeze(0).float()
        alpha = torch.tensor(sample["alpha"], dtype=torch.float32)

        if self.augment:
            input_tensor, target_tensor = self._augment(input_tensor, target_tensor)

        return {
            "input": input_tensor,       # (2, H, W)
            "target": target_tensor,     # (1, H, W)
            "alpha": alpha,              # scalar
        }

    @staticmethod
    def _augment(
        input_tensor: torch.Tensor, target_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random augmentations (flip, rotation).

        Args:
            input_tensor: Input tensor (2, H, W).
            target_tensor: Target tensor (1, H, W).

        Returns:
            Augmented (input, target) pair.
        """
        # Random horizontal flip
        if torch.rand(1).item() > 0.5:
            input_tensor = torch.flip(input_tensor, dims=[2])
            target_tensor = torch.flip(target_tensor, dims=[2])

        # Random vertical flip
        if torch.rand(1).item() > 0.5:
            input_tensor = torch.flip(input_tensor, dims=[1])
            target_tensor = torch.flip(target_tensor, dims=[1])

        return input_tensor, target_tensor


class LazyUNetSliceDataset(Dataset):
    """Memory-efficient U-Net dataset with tiny LRU cache.

    Designed for Kaggle 30GB RAM. Uses cache_size=2 by default
    (~1GB for 2 volumes). MUST be paired with VolumeGroupedSampler
    to avoid cache thrashing (shuffle=True with random access would
    constantly evict/reload volumes, causing OOM from load spikes).
    """

    def __init__(
        self,
        case_entries: List[Dict],
        sparse_ratio: int = 2,
        augment: bool = False,
        hu_min: float = -1000.0,
        hu_max: float = 1000.0,
        cache_size: int = 2,
    ):
        from collections import OrderedDict

        self.augment = augment
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.cache_size = cache_size
        self._cache: OrderedDict = OrderedDict()
        self.samples: List[Dict] = []
        self._vol_to_sample_indices: Dict[str, List[int]] = {}

        idx = 0
        for entry in case_entries:
            vol_path = str(Path(entry["volume_path"]))
            D = entry["num_slices"]
            R = sparse_ratio

            observed = list(range(0, D, R))
            for i in range(len(observed) - 1):
                idx_before = observed[i]
                idx_after = observed[i + 1]
                for t in range(idx_before + 1, idx_after):
                    alpha = (t - idx_before) / (idx_after - idx_before)
                    self.samples.append({
                        "volume_path": vol_path,
                        "idx_before": idx_before,
                        "idx_after": idx_after,
                        "idx_target": t,
                        "alpha": alpha,
                    })
                    if vol_path not in self._vol_to_sample_indices:
                        self._vol_to_sample_indices[vol_path] = []
                    self._vol_to_sample_indices[vol_path].append(idx)
                    idx += 1

    def _get_volume(self, volume_path: str) -> np.ndarray:
        """Load volume with LRU cache. Zero-copy preprocessing."""
        if volume_path in self._cache:
            self._cache.move_to_end(volume_path)
            return self._cache[volume_path]

        import gc
        import nibabel as nib

        if len(self._cache) >= self.cache_size:
            self._cache.popitem(last=False)
            gc.collect()

        nii = nib.load(volume_path)
        vol = np.asarray(nii.dataobj, dtype=np.float32)
        nii.uncache()
        del nii

        np.clip(vol, self.hu_min, self.hu_max, out=vol)
        vol -= self.hu_min
        vol /= (self.hu_max - self.hu_min)

        self._cache[volume_path] = vol
        return vol

    def get_volume_groups(self) -> List[List[int]]:
        """Return sample indices grouped by volume (for VolumeGroupedSampler)."""
        return list(self._vol_to_sample_indices.values())

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        vol = self._get_volume(sample["volume_path"])

        slice_before = vol[:, :, sample["idx_before"]]
        slice_after = vol[:, :, sample["idx_after"]]
        slice_target = vol[:, :, sample["idx_target"]]

        input_tensor = torch.from_numpy(
            np.stack([slice_before, slice_after], axis=0)
        ).float()
        target_tensor = torch.from_numpy(slice_target).unsqueeze(0).float()
        alpha = torch.tensor(sample["alpha"], dtype=torch.float32)

        if self.augment:
            input_tensor, target_tensor = UNetSliceDataset._augment(
                input_tensor, target_tensor
            )

        return {
            "input": input_tensor,
            "target": target_tensor,
            "alpha": alpha,
        }


class VolumeGroupedSampler(torch.utils.data.Sampler):
    """Sampler that groups samples by source volume for cache-friendly access.

    Without this, shuffle=True causes the DataLoader to request samples from
    random volumes, thrashing the LRU cache and spiking RAM to OOM.

    With this sampler, consecutive batches come from the same 1-2 volumes,
    so cache hit rate is ~99% and only 1-2 volumes are in RAM at once.

    Randomization is preserved at two levels:
    1. Volume order is shuffled each epoch
    2. Sample order within each volume is shuffled
    """

    def __init__(self, volume_groups: List[List[int]], shuffle: bool = True):
        self.volume_groups = volume_groups
        self.shuffle = shuffle
        self._total = sum(len(g) for g in volume_groups)

    def __iter__(self):
        groups = [g.copy() for g in self.volume_groups]

        if self.shuffle:
            group_order = np.random.permutation(len(groups))
            for g in groups:
                np.random.shuffle(g)
        else:
            group_order = np.arange(len(groups))

        for gi in group_order:
            yield from groups[gi]

    def __len__(self) -> int:
        return self._total
