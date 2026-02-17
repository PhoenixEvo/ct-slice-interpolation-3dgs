"""
Training loop for U-Net 2D baseline.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from typing import Dict, Optional
import time
import json

from ..models.unet2d import UNet2D
from ..losses.reconstruction import CombinedReconstructionLoss


class TrainerUNet:
    """Trainer for U-Net 2D slice interpolation baseline."""

    def __init__(
        self,
        model: UNet2D,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 1e-4,
        num_epochs: int = 50,
        scheduler_step: int = 20,
        scheduler_gamma: float = 0.5,
        mixed_precision: bool = True,
        checkpoint_dir: str = "checkpoints/unet",
        device: str = "cuda",
    ):
        """Initialize U-Net trainer.

        Args:
            model: U-Net model instance.
            train_loader: Training data loader.
            val_loader: Optional validation data loader.
            lr: Learning rate.
            num_epochs: Number of training epochs.
            scheduler_step: LR scheduler step size.
            scheduler_gamma: LR scheduler decay factor.
            mixed_precision: Whether to use mixed precision training.
            checkpoint_dir: Directory to save checkpoints.
            device: Device to train on.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.mixed_precision = mixed_precision
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        # Loss
        self.criterion = CombinedReconstructionLoss(
            l1_weight=1.0, ssim_weight=0.1
        ).to(device)

        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=scheduler_step, gamma=scheduler_gamma
        )

        # Mixed precision
        self.scaler = GradScaler(enabled=mixed_precision)

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_psnr": [],
            "lr": [],
        }

    def train(self) -> Dict:
        """Run full training loop.

        Returns:
            Training history dictionary.
        """
        best_val_loss = float("inf")

        for epoch in range(self.num_epochs):
            t0 = time.time()

            # Train
            train_loss = self._train_epoch()
            self.history["train_loss"].append(train_loss)
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

            # Validate
            val_loss = 0.0
            val_psnr = 0.0
            if self.val_loader is not None:
                val_loss, val_psnr = self._validate()
                self.history["val_loss"].append(val_loss)
                self.history["val_psnr"].append(val_psnr)

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint("best.pt", epoch)

            self.scheduler.step()

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch + 1}/{self.num_epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val PSNR: {val_psnr:.2f} dB | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

            # Periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(f"epoch_{epoch + 1}.pt", epoch)

        # Save final model and history
        self._save_checkpoint("final.pt", self.num_epochs - 1)
        self._save_history()

        return self.history

    def _train_epoch(self) -> float:
        """Train for one epoch.

        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            input_slices = batch["input"].to(self.device)
            target_slices = batch["target"].to(self.device)

            self.optimizer.zero_grad()

            with autocast(enabled=self.mixed_precision):
                predictions = self.model(input_slices)
                loss = self.criterion(predictions, target_slices)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate(self):
        """Validate on validation set.

        Returns:
            Tuple of (average_loss, average_psnr).
        """
        self.model.eval()
        total_loss = 0.0
        total_psnr = 0.0
        num_batches = 0

        for batch in self.val_loader:
            input_slices = batch["input"].to(self.device)
            target_slices = batch["target"].to(self.device)

            with autocast(enabled=self.mixed_precision):
                predictions = self.model(input_slices)
                loss = self.criterion(predictions, target_slices)

            total_loss += loss.item()

            # Compute PSNR
            pred_np = predictions.cpu().float().numpy()
            gt_np = target_slices.cpu().float().numpy()
            mse = ((pred_np - gt_np) ** 2).mean()
            if mse > 1e-10:
                psnr = 10 * __import__("numpy").log10(1.0 / mse)
            else:
                psnr = 100.0
            total_psnr += psnr
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_psnr = total_psnr / max(num_batches, 1)
        return avg_loss, avg_psnr

    def predict_slice(
        self, slice_before: torch.Tensor, slice_after: torch.Tensor
    ) -> torch.Tensor:
        """Predict a single interpolated slice.

        Args:
            slice_before: Previous slice tensor (1, H, W) or (H, W).
            slice_after: Next slice tensor (1, H, W) or (H, W).

        Returns:
            Predicted slice tensor (1, H, W).
        """
        self.model.eval()
        if slice_before.dim() == 2:
            slice_before = slice_before.unsqueeze(0)
        if slice_after.dim() == 2:
            slice_after = slice_after.unsqueeze(0)

        input_tensor = torch.stack(
            [slice_before.squeeze(0), slice_after.squeeze(0)]
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            with autocast(enabled=self.mixed_precision):
                pred = self.model(input_tensor)

        return pred.squeeze(0).cpu().float()

    def _save_checkpoint(self, filename: str, epoch: int) -> None:
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "history": self.history,
        }, path)

    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint."""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "history" in checkpoint:
            self.history = checkpoint["history"]

    def _save_history(self) -> None:
        """Save training history to JSON."""
        path = self.checkpoint_dir / "history.json"
        
        # Convert numpy types to Python native types
        history_serializable = {}
        for key, values in self.history.items():
            history_serializable[key] = [float(v) for v in values]
        
        with open(path, "w") as f:
            json.dump(history_serializable, f, indent=2)

