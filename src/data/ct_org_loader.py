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
        cases: List[int] = []
        for pattern in ["volume-*.nii.gz", "volume-*.nii"]:
            for f in self.dataset_root.rglob(pattern):
                # Only count actual files (Kaggle may create folders like volume-17.nii/)
                if not f.is_file():
                    continue
                try:
                    name = f.name
                    if name.endswith(".nii.gz"):
                        core = name[:-7]
                    elif name.endswith(".nii"):
                        core = name[:-4]
                    else:
                        continue
                    idx = int(core.replace("volume-", ""))
                    if idx not in cases:
                        cases.append(idx)
                except ValueError:
                    continue
        return sorted(cases)

    def _resolve_case_path(self, prefix: str, case_idx: int) -> Optional[Path]:
        """Resolve a NIfTI path for a given case.

        This is robust to Kaggle's extraction behavior where:
        - `.nii.gz` may become `.nii`
        - extra folders may be created and files placed inside unexpected directories

        Args:
            prefix: "volume" or "labels"
            case_idx: Case index

        Returns:
            Path to an existing file, or None if not found.
        """
        # Prefer flat layout first
        candidates = [
            self.dataset_root / f"{prefix}-{case_idx}.nii.gz",
            self.dataset_root / f"{prefix}-{case_idx}.nii",
        ]
        for p in candidates:
            if p.exists() and p.is_file():
                return p

        # Recursively search for files by name
        file_matches: List[Path] = []
        for pat in [f"{prefix}-{case_idx}.nii.gz", f"{prefix}-{case_idx}.nii"]:
            for p in self.dataset_root.rglob(pat):
                if p.exists() and p.is_file():
                    file_matches.append(p)

        if file_matches:
            # Pick the closest match to dataset_root (stable + avoids deep weird paths)
            file_matches = sorted(file_matches, key=lambda x: (len(x.parts), str(x)))
            return file_matches[0]

        # Some Kaggle extractions create a directory named like the file (e.g., volume-17.nii/)
        # In that case, search for directories and then look for the file inside.
        dir_matches: List[Path] = []
        for pat in [f"{prefix}-{case_idx}.nii", f"{prefix}-{case_idx}.nii.gz"]:
            for p in self.dataset_root.rglob(pat):
                if p.exists() and p.is_dir():
                    dir_matches.append(p)

        if dir_matches:
            dir_matches = sorted(dir_matches, key=lambda x: (len(x.parts), str(x)))
            for d in dir_matches:
                inside = d / d.name
                if inside.exists() and inside.is_file():
                    return inside
                # Also try the expected names inside this dir
                for inside_name in [f"{prefix}-{case_idx}.nii", f"{prefix}-{case_idx}.nii.gz"]:
                    inside2 = d / inside_name
                    if inside2.exists() and inside2.is_file():
                        return inside2

        return None

    def load_volume(self, case_idx: int) -> Tuple[np.ndarray, Dict]:
        """Load a CT volume and return with metadata.

        Args:
            case_idx: Case index number.

        Returns:
            Tuple of (volume_array, metadata_dict).
            volume_array shape: (H, W, D) in float32.
        """
        volume_path = self._resolve_case_path("volume", case_idx)
        if volume_path is None:
            raise FileNotFoundError(
                f"Volume not found for case {case_idx} (.nii.gz or .nii)"
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
        label_path = self._resolve_case_path("labels", case_idx)
        if label_path is None:
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
        volume_path = self._resolve_case_path("volume", case_idx)
        if volume_path is None:
            raise FileNotFoundError(
                f"Volume not found for case {case_idx} (.nii.gz or .nii)"
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
        """Split cases into train/val/test sets.1

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
