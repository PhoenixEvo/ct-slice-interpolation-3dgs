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
    """Memory-efficient U-Net dataset that loads slices on demand from NIfTI.

    Instead of holding all volumes in RAM, this dataset stores file paths
    and loads only the needed slices when __getitem__ is called.
    Suitable for Kaggle (16GB RAM) with large CT-ORG dataset.
    """

    def __init__(
        self,
        case_entries: List[Dict],
        sparse_ratio: int = 2,
        augment: bool = False,
        hu_min: float = -1000.0,
        hu_max: float = 1000.0,
    ):
        """Initialize lazy U-Net dataset.

        Args:
            case_entries: List of dicts with keys:
                - 'volume_path': Path to NIfTI volume file
                - 'num_slices': Number of slices in the volume (D)
            sparse_ratio: Sparsity ratio R.
            augment: Whether to apply data augmentation.
            hu_min: HU clipping minimum.
            hu_max: HU clipping maximum.
        """
        self.augment = augment
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.samples: List[Dict] = []

        for entry in case_entries:
            vol_path = Path(entry["volume_path"])
            D = entry["num_slices"]
            R = sparse_ratio

            observed = list(range(0, D, R))
            for i in range(len(observed) - 1):
                idx_before = observed[i]
                idx_after = observed[i + 1]
                for t in range(idx_before + 1, idx_after):
                    alpha = (t - idx_before) / (idx_after - idx_before)
                    self.samples.append({
                        "volume_path": str(vol_path),
                        "idx_before": idx_before,
                        "idx_after": idx_after,
                        "idx_target": t,
                        "alpha": alpha,
                    })

    def _load_slice(self, volume_path: str, z_idx: int) -> np.ndarray:
        """Load and preprocess a single slice from NIfTI."""
        import nibabel as nib
        nii = nib.load(volume_path)
        raw = nii.dataobj[..., z_idx].astype(np.float32)
        clipped = np.clip(raw, self.hu_min, self.hu_max)
        normalized = (clipped - self.hu_min) / (self.hu_max - self.hu_min)
        return normalized

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        vol_path = sample["volume_path"]

        slice_before = self._load_slice(vol_path, sample["idx_before"])
        slice_after = self._load_slice(vol_path, sample["idx_after"])
        slice_target = self._load_slice(vol_path, sample["idx_target"])

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
