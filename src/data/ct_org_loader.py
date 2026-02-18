"""
CT-ORG dataset loader with preprocessing utilities.
Handles NIfTI loading, HU normalization, and optional resampling.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class CTORGLoader:
    """Loader for CT-ORG dataset in NIfTI format."""

    def __init__(
        self,
        dataset_root: str,
        hu_min: float = -1000.0,
        hu_max: float = 1000.0,
        normalize_range: Tuple[float, float] = (0.0, 1.0),
    ):
        """Initialize CT-ORG loader.

        Args:
            dataset_root: Path to the OrganSegmentations directory.
            hu_min: Minimum HU value for clipping.
            hu_max: Maximum HU value for clipping.
            normalize_range: Target normalization range (min, max).
        """
        self.dataset_root = Path(dataset_root)
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.normalize_range = normalize_range

        # Verify dataset path
        if not self.dataset_root.exists():
            raise FileNotFoundError(
                f"Dataset root not found: {self.dataset_root}"
            )

    def get_available_cases(self) -> List[int]:
        """Get list of available case indices.

        Returns:
            Sorted list of case indices found in the dataset.
        """
        cases = []
        # Try both .nii.gz and .nii (Kaggle auto-extracts .gz files)
        for pattern in ["volume-*.nii.gz", "volume-*.nii"]:
            for f in self.dataset_root.glob(pattern):
                try:
                    # Handle both .nii.gz and .nii
                    stem = f.stem
                    if stem.endswith(".nii"):
                        stem = stem[:-4]  # Remove .nii
                    idx = int(stem.replace("volume-", ""))
                    if idx not in cases:
                        cases.append(idx)
                except ValueError:
                    continue
        return sorted(cases)

    def load_volume(self, case_idx: int) -> Tuple[np.ndarray, Dict]:
        """Load a CT volume and return with metadata.

        Args:
            case_idx: Case index number.

        Returns:
            Tuple of (volume_array, metadata_dict).
            volume_array shape: (H, W, D) in float32.
        """
        # Try .nii.gz first, then fallback to .nii (Kaggle auto-extracts)
        volume_path = self.dataset_root / f"volume-{case_idx}.nii.gz"
        if not volume_path.exists():
            volume_path = self.dataset_root / f"volume-{case_idx}.nii"
        if not volume_path.exists():
            raise FileNotFoundError(
                f"Volume not found: volume-{case_idx}.nii.gz or volume-{case_idx}.nii"
            )

        nii = nib.load(str(volume_path))
        volume = nii.get_fdata().astype(np.float32)

        metadata = {
            "case_idx": case_idx,
            "shape": volume.shape,
            "affine": nii.affine,
            "header": nii.header,
            "voxel_size": tuple(nii.header.get_zooms()),
        }

        return volume, metadata

    def load_labels(self, case_idx: int) -> Optional[np.ndarray]:
        """Load organ segmentation labels for a case.

        Args:
            case_idx: Case index number.

        Returns:
            Label array (H, W, D) in int, or None if not found.
        """
        # Try .nii.gz first, then fallback to .nii (Kaggle auto-extracts)
        label_path = self.dataset_root / f"labels-{case_idx}.nii.gz"
        if not label_path.exists():
            label_path = self.dataset_root / f"labels-{case_idx}.nii"
        if not label_path.exists():
            return None

        nii = nib.load(str(label_path))
        labels = nii.get_fdata().astype(np.int32)
        return labels

    def preprocess_volume(self, volume: np.ndarray) -> np.ndarray:
        """Apply HU clipping and normalization.

        Args:
            volume: Raw CT volume in HU.

        Returns:
            Normalized volume in target range.
        """
        # Clip HU values
        volume = np.clip(volume, self.hu_min, self.hu_max)

        # Normalize to target range
        lo, hi = self.normalize_range
        volume = (volume - self.hu_min) / (self.hu_max - self.hu_min)
        volume = volume * (hi - lo) + lo

        return volume.astype(np.float32)

    def load_and_preprocess(
        self, case_idx: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
        """Load and preprocess a volume with optional labels.

        Args:
            case_idx: Case index number.

        Returns:
            Tuple of (preprocessed_volume, labels_or_None, metadata).
        """
        volume, metadata = self.load_volume(case_idx)
        volume = self.preprocess_volume(volume)
        labels = self.load_labels(case_idx)

        metadata["preprocessed"] = True
        metadata["hu_range"] = (self.hu_min, self.hu_max)
        metadata["normalize_range"] = self.normalize_range

        return volume, labels, metadata

    def get_volume_info(self, case_idx: int) -> Dict:
        """Get volume metadata without loading the full array.

        Args:
            case_idx: Case index number.

        Returns:
            Dictionary with volume metadata.
        """
        # Try .nii.gz first, then fallback to .nii (Kaggle auto-extracts)
        volume_path = self.dataset_root / f"volume-{case_idx}.nii.gz"
        if not volume_path.exists():
            volume_path = self.dataset_root / f"volume-{case_idx}.nii"
        if not volume_path.exists():
            raise FileNotFoundError(
                f"Volume not found: volume-{case_idx}.nii.gz or volume-{case_idx}.nii"
            )

        nii = nib.load(str(volume_path))
        header = nii.header

        return {
            "case_idx": case_idx,
            "shape": tuple(header.get_data_shape()),
            "voxel_size": tuple(header.get_zooms()),
            "dtype": header.get_data_dtype(),
        }

    @staticmethod
    def get_split(
        available_cases: List[int],
        test_cases: List[int],
        val_cases: List[int],
    ) -> Dict[str, List[int]]:
        """Split cases into train/val/test sets.

        Args:
            available_cases: All available case indices.
            test_cases: Case indices for test set.
            val_cases: Case indices for validation set.

        Returns:
            Dictionary with 'train', 'val', 'test' keys.
        """
        available = set(available_cases)
        test_set = sorted(set(test_cases) & available)
        val_set = sorted(set(val_cases) & available)
        train_set = sorted(available - set(test_set) - set(val_set))

        return {
            "train": train_set,
            "val": val_set,
            "test": test_set,
        }
