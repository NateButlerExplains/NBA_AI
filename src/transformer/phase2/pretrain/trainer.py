"""
Pre-training Trainer for Phase 3 Experiment 2.

MSE reconstruction loss on masked positions.
Same infrastructure as Phase2Trainer: EMA, AMP, cosine schedule, early stopping.
"""

import collections
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.transformer.phase2.config import Phase2ExperimentConfig
from src.transformer.phase2.pretrain.model import PretrainModel
from src.transformer.phase2.training.ema import EMA

logger = logging.getLogger(__name__)


@dataclass
class PretrainState:
    """Tracks pre-training progress."""
    epoch: int = 0
    global_step: int = 0
    best_val_mse: float = float("inf")
    patience_counter: int = 0
    train_losses: list = field(default_factory=list)
    val_losses: list = field(default_factory=list)


class PretrainTrainer:
    """Pre-training trainer with MSE reconstruction loss."""

    def __init__(
        self,
        model: PretrainModel,
        config: Phase2ExperimentConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Device
        self.device = self._setup_device(config.training.device)
        self.model = self.model.to(self.device)

        # Loss
        self.criterion = nn.MSELoss()

        # Optimizer
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()

        # AMP
        self.use_amp = config.training.use_amp and self.device.type == "cuda"
        self.scaler = GradScaler("cuda") if self.use_amp else None
        self.amp_dtype = getattr(torch, config.training.amp_dtype) if self.use_amp else None

        # EMA
        self.ema = None
        if config.training.use_ema:
            self.ema = EMA(self.model, decay=config.training.ema_decay)

        # Early stopping
        self.smoothing_window = collections.deque(
            maxlen=config.training.smoothing_window
        )

        # Checkpointing
        self.checkpoint_dir = (
            Path(config.training.checkpoint_dir) / config.training.experiment_name
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.state = PretrainState()

    def _setup_device(self, device_str: str) -> torch.device:
        if device_str == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            return torch.device("cpu")
        return torch.device(device_str)

    def _setup_optimizer(self) -> AdamW:
        opt_config = self.config.optimizer
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "embedding" in name or "emb" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": opt_config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        return AdamW(
            param_groups,
            lr=opt_config.learning_rate,
            betas=tuple(opt_config.betas),
            eps=opt_config.eps,
        )

    def _setup_scheduler(self):
        opt_config = self.config.optimizer
        train_config = self.config.training

        steps_per_epoch = len(self.train_loader)
        accum_steps = train_config.gradient_accumulation_steps
        optimizer_steps_per_epoch = max(steps_per_epoch // accum_steps, 1)
        total_steps = optimizer_steps_per_epoch * train_config.max_epochs

        if opt_config.warmup_steps > 0:
            warmup_steps = opt_config.warmup_steps
        elif opt_config.warmup_ratio > 0:
            warmup_steps = int(total_steps * opt_config.warmup_ratio)
        else:
            warmup_steps = 0

        warmup_scheduler = LinearLR(
            self.optimizer, start_factor=0.1, end_factor=1.0,
            total_iters=max(warmup_steps, 1),
        )

        main_steps = max(total_steps - warmup_steps, 1)
        min_lr = opt_config.learning_rate * opt_config.min_lr_ratio

        main_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=main_steps, eta_min=min_lr,
        )

        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )

    def train(self) -> float:
        """Run full pre-training loop. Returns best val MSE."""
        logger.info(f"Starting pre-training: {self.config.training.experiment_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Train samples: {len(self.train_loader.dataset)}")
        logger.info(f"Val samples: {len(self.val_loader.dataset)}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        self.config.to_yaml(self.checkpoint_dir / "config.yaml")

        try:
            for epoch in range(self.state.epoch, self.config.training.max_epochs):
                self.state.epoch = epoch
                epoch_start = time.time()

                train_loss = self._train_epoch(epoch)
                self.state.train_losses.append(train_loss)

                val_loss = self._validate()
                self.state.val_losses.append(val_loss)

                epoch_time = time.time() - epoch_start

                logger.info(
                    f"Epoch {epoch + 1}: "
                    f"train_mse={train_loss:.6f}, "
                    f"val_mse={val_loss:.6f}, "
                    f"lr={self.optimizer.param_groups[0]['lr']:.2e}, "
                    f"time={epoch_time:.0f}s"
                )

                # Early stopping on smoothed val MSE
                self.smoothing_window.append(val_loss)
                smoothed = sum(self.smoothing_window) / len(self.smoothing_window)

                if smoothed < self.state.best_val_mse - self.config.training.min_delta:
                    self.state.best_val_mse = smoothed
                    self.state.patience_counter = 0
                    self._save_checkpoint("best.pt")
                    self._save_transferable_weights()
                    logger.info(f"  New best! Smoothed val_mse: {smoothed:.6f}")
                else:
                    self.state.patience_counter += 1

                if self.state.patience_counter >= self.config.training.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        except KeyboardInterrupt:
            logger.info("Pre-training interrupted")
            self._save_checkpoint("interrupted.pt")

        logger.info(f"Pre-training complete. Best val MSE: {self.state.best_val_mse:.6f}")
        return self.state.best_val_mse

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        accum_steps = self.config.training.gradient_accumulation_steps

        pbar = tqdm(self.train_loader, desc=f"Pretrain Epoch {epoch + 1}", leave=False)
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            batch = self._batch_to_device(batch)

            if self.use_amp:
                with autocast("cuda", dtype=self.amp_dtype):
                    output = self.model(batch)
                    if output["predictions"].shape[0] == 0:
                        continue
                    loss = self.criterion(output["predictions"], output["targets"])
                    scaled_loss = loss / accum_steps
                self.scaler.scale(scaled_loss).backward()
            else:
                output = self.model(batch)
                if output["predictions"].shape[0] == 0:
                    continue
                loss = self.criterion(output["predictions"], output["targets"])
                scaled_loss = loss / accum_steps
                scaled_loss.backward()

            total_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % accum_steps == 0:
                self._optimizer_step()

            pbar.set_postfix({"mse": f"{loss.item():.4f}"})
            self.state.global_step += 1

        if n_batches % accum_steps != 0:
            self._optimizer_step()

        return total_loss / max(n_batches, 1)

    def _optimizer_step(self):
        if self.use_amp:
            self.scaler.unscale_(self.optimizer)

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.optimizer.max_grad_norm,
        )

        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.scheduler.step()
        self.optimizer.zero_grad()

        if self.ema is not None:
            self.ema.update()

    def _validate(self) -> float:
        if self.ema is not None:
            with self.ema.apply():
                return self._validate_inner()
        return self._validate_inner()

    def _validate_inner(self) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Val", leave=False):
                batch = self._batch_to_device(batch)

                if self.use_amp:
                    with autocast("cuda", dtype=self.amp_dtype):
                        output = self.model(batch)
                else:
                    output = self.model(batch)

                if output["predictions"].shape[0] == 0:
                    continue

                loss = self.criterion(output["predictions"], output["targets"])
                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def _batch_to_device(self, batch: dict) -> dict:
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _save_checkpoint(self, filename: str):
        path = self.checkpoint_dir / filename
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "state": {
                "epoch": self.state.epoch,
                "global_step": self.state.global_step,
                "best_val_mse": self.state.best_val_mse,
                "patience_counter": self.state.patience_counter,
            },
            "config": self.config.to_dict(),
        }
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        if self.ema is not None:
            checkpoint["ema_state_dict"] = self.ema.state_dict()
        torch.save(checkpoint, path)
        logger.debug(f"Saved checkpoint: {path}")

    def _save_transferable_weights(self):
        """Save only the weights that transfer to Phase2Model."""
        if self.ema is not None:
            with self.ema.apply():
                state = self.model.get_transferable_state_dict()
        else:
            state = self.model.get_transferable_state_dict()

        path = self.checkpoint_dir / "transferable_weights.pt"
        torch.save(state, path)
        logger.info(f"Saved transferable weights: {len(state)} tensors to {path}")
