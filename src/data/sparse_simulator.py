"""
Sparse slice simulation for CT volume interpolation experiments.
Simulates the scenario where only every R-th slice is available.
"""

import numpy as np
from typing import Dict, List, Tuple


class SparseSimulator:
    """Simulate sparse slice acquisition from a full CT volume."""

    def __init__(self, sparse_ratio: int = 2):
        """Initialize sparse simulator.

        Args:
            sparse_ratio: Keep every R-th slice. R=2 keeps 0,2,4,...
                          R=3 keeps 0,3,6,...
        """
        if sparse_ratio < 2:
            raise ValueError("sparse_ratio must be >= 2")
        self.sparse_ratio = sparse_ratio

    def simulate(
        self, volume: np.ndarray, axis: int = 2
    ) -> Dict[str, np.ndarray]:
        """Create sparse volume and ground truth targets.

        Args:
            volume: Full CT volume, shape (H, W, D).
            axis: Axis along which to simulate sparsity (default=2 for z).

        Returns:
            Dictionary containing:
                - 'observed_slices': Array of observed (kept) slices
                - 'observed_indices': Indices of kept slices
                - 'target_slices': Array of target (removed) slices
                - 'target_indices': Indices of removed slices
                - 'full_volume': Original volume reference
        """
        num_slices = volume.shape[axis]
        R = self.sparse_ratio

        # Observed slice indices: 0, R, 2R, ...
        observed_indices = list(range(0, num_slices, R))

        # Target slice indices: all non-observed
        all_indices = set(range(num_slices))
        target_indices = sorted(all_indices - set(observed_indices))

        # Extract slices
        observed_slices = np.take(volume, observed_indices, axis=axis)
        target_slices = np.take(volume, target_indices, axis=axis)

        return {
            "observed_slices": observed_slices,
            "observed_indices": np.array(observed_indices),
            "target_slices": target_slices,
            "target_indices": np.array(target_indices),
            "full_volume": volume,
            "sparse_ratio": R,
        }

    def get_interpolation_pairs(
        self, volume: np.ndarray, axis: int = 2
    ) -> List[Dict[str, np.ndarray]]:
        """Get pairs of adjacent observed slices and their interpolation targets.

        For R=2: pairs are (0,2)->1, (2,4)->3, ...
        For R=3: pairs are (0,3)->[1,2], (3,6)->[4,5], ...

        Args:
            volume: Full CT volume, shape (H, W, D).
            axis: Sparsity axis.

        Returns:
            List of dictionaries, each containing:
                - 'slice_before': Previous observed slice
                - 'slice_after': Next observed slice
                - 'idx_before': Index of previous slice
                - 'idx_after': Index of next slice
                - 'targets': List of target slices between them
                - 'target_indices': Indices of target slices
                - 'positions': Normalized positions [0,1] of targets between pair
        """
        num_slices = volume.shape[axis]
        R = self.sparse_ratio
        observed_indices = list(range(0, num_slices, R))

        pairs = []
        for i in range(len(observed_indices) - 1):
            idx_before = observed_indices[i]
            idx_after = observed_indices[i + 1]

            slice_before = np.take(volume, idx_before, axis=axis)
            slice_after = np.take(volume, idx_after, axis=axis)

            # Target slices between the pair
            target_idx = list(range(idx_before + 1, idx_after))
            targets = [
                np.take(volume, t, axis=axis) for t in target_idx
            ]

            # Normalized positions between [0, 1]
            positions = [
                (t - idx_before) / (idx_after - idx_before)
                for t in target_idx
            ]

            pairs.append({
                "slice_before": slice_before,
                "slice_after": slice_after,
                "idx_before": idx_before,
                "idx_after": idx_after,
                "targets": targets,
                "target_indices": target_idx,
                "positions": positions,
            })

        return pairs

    @staticmethod
    def reconstruct_volume(
        observed_slices: np.ndarray,
        observed_indices: np.ndarray,
        interpolated_slices: np.ndarray,
        interpolated_indices: np.ndarray,
        total_slices: int,
        axis: int = 2,
    ) -> np.ndarray:
        """Reconstruct full volume from observed + interpolated slices.

        Args:
            observed_slices: Array of observed slices.
            observed_indices: Indices of observed slices.
            interpolated_slices: Array of interpolated slices.
            interpolated_indices: Indices of interpolated slices.
            total_slices: Total number of slices in the full volume.
            axis: Reconstruction axis.

        Returns:
            Reconstructed full volume.
        """
        # Determine output shape
        sample_slice = np.take(observed_slices, 0, axis=axis)
        shape = list(sample_slice.shape)
        shape.insert(axis, total_slices)
        reconstructed = np.zeros(shape, dtype=np.float32)

        # Place observed slices
        for i, idx in enumerate(observed_indices):
            slc = [slice(None)] * len(shape)
            slc[axis] = idx
            src_slc = [slice(None)] * len(observed_slices.shape)
            src_slc[axis] = i
            reconstructed[tuple(slc)] = observed_slices[tuple(src_slc)]

        # Place interpolated slices
        for i, idx in enumerate(interpolated_indices):
            slc = [slice(None)] * len(shape)
            slc[axis] = idx
            src_slc = [slice(None)] * len(interpolated_slices.shape)
            src_slc[axis] = i
            reconstructed[tuple(slc)] = interpolated_slices[tuple(src_slc)]

        return reconstructed
