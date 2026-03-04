"""
Phase 2 Trainer.

Adapts Phase 1 trainer pattern with:
- EMA weight averaging
- val_spread_mae early stopping with 3-epoch smoothing
- Phase2CombinedLoss (with consistency term)
- Forward pass via batch dict (not separate home/away history)
"""

import collections
import logging
import time
import warnings
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

from src.transformer.models.prediction_heads import GamePrediction
from src.transformer.training.metrics import MetricsCalculator, MetricResults
from src.transformer.phase2.config import Phase2ExperimentConfig
from src.transformer.phase2.models.phase2_model import Phase2Model
from src.transformer.phase2.training.loss import Phase2CombinedLoss
from src.transformer.phase2.training.ema import EMA

logger = logging.getLogger(__name__)


@dataclass
class Phase2TrainingState:
    """Tracks training progress for checkpointing and resumption."""

    epoch: int = 0
    global_step: int = 0
    best_metric: float = float("inf")
    patience_counter: int = 0
    train_losses: list = field(default_factory=list)
    val_losses: list = field(default_factory=list)
    val_spread_maes: list = field(default_factory=list)


class Phase2Trainer:
    """
    Trainer for Phase 2 model with EMA and val_spread_mae early stopping.
    """

    def __init__(
        self,
        model: Phase2Model,
        config: Phase2ExperimentConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        pretrained_params: set[str] | None = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Pre-trained weight tracking (must be set before _setup_optimizer)
        self.pretrained_params = pretrained_params or set()
        self._current_unfreeze_phase = 0  # 0=frozen, 1=top, 2=all
        self.freeze_pretrained_epochs = config.training.freeze_pretrained_epochs
        self.unfreeze_top_epochs = config.training.unfreeze_top_epochs
        self.lr_decay_factor = config.training.lr_decay_factor

        # Device setup
        self.device = self._setup_device(config.training.device)
        self.model = self.model.to(self.device)

        # Loss function
        self.criterion = Phase2CombinedLoss(
            spread_weight=config.training.spread_loss_weight,
            score_weight=config.training.score_loss_weight,
            win_prob_weight=config.training.win_prob_loss_weight,
            consistency_weight=config.training.consistency_loss_weight,
            nll_weight=config.training.nll_weight,
            mse_weight=config.training.mse_weight,
            spread_huber_delta=config.training.spread_huber_delta,
            score_huber_delta=config.training.score_huber_delta,
        )

        # Optimizer
        self.optimizer = self._setup_optimizer()

        # Scheduler
        self.scheduler = self._setup_scheduler()

        # AMP
        self.use_amp = config.training.use_amp and self.device.type == "cuda"
        self.scaler = GradScaler("cuda") if self.use_amp else None
        self.amp_dtype = getattr(torch, config.training.amp_dtype) if self.use_amp else None

        # EMA
        self.ema = None
        if config.training.use_ema:
            self.ema = EMA(self.model, decay=config.training.ema_decay)

        # Apply initial freeze if we have pre-trained params
        if self.pretrained_params and self.freeze_pretrained_epochs > 0:
            self._apply_freeze_phase(0)

        # Metrics
        self.metrics_calculator = MetricsCalculator()

        # Early stopping with smoothing
        self.smoothing_window = collections.deque(
            maxlen=config.training.smoothing_window
        )

        # Checkpointing
        self.checkpoint_dir = (
            Path(config.training.checkpoint_dir) / config.training.experiment_name
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.state = Phase2TrainingState()

        # Wandb
        self.wandb_run = None
        if config.training.use_wandb:
            self._setup_wandb()

    def _setup_device(self, device_str: str) -> torch.device:
        if device_str == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            return torch.device("cpu")
        return torch.device(device_str)

    def _setup_optimizer(self) -> AdamW:
        opt_config = self.config.optimizer

        # Check if we need discriminative LR
        use_discriminative = (
            self.pretrained_params
            and self.lr_decay_factor < 1.0
            and self._current_unfreeze_phase >= 2
        )

        if use_discriminative:
            return self._setup_discriminative_optimizer()

        # Standard optimizer
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

    def _setup_discriminative_optimizer(self) -> AdamW:
        """Set up optimizer with per-layer discriminative learning rates."""
        opt_config = self.config.optimizer
        base_lr = opt_config.learning_rate
        decay = self.lr_decay_factor

        # Define layer groups from top (highest LR) to bottom (lowest LR)
        # Layer ordering: prediction_heads > fusion > temporal layers (top to bottom) > per_game_encoder > player_embed
        n_temporal = self.config.model.temporal_layers

        layer_prefixes = []
        # Top layers (highest LR)
        layer_prefixes.append(("prediction_heads.", "prediction_heads"))
        layer_prefixes.append(("fusion.", "fusion"))
        layer_prefixes.append(("team_combine.", "team_combine"))
        layer_prefixes.append(("roster_encoder.", "roster_encoder"))
        layer_prefixes.append(("rest_embed.", "rest_embed"))
        layer_prefixes.append(("gs_encoder.", "gs_encoder"))

        # Temporal layers top to bottom
        for i in range(n_temporal - 1, -1, -1):
            layer_prefixes.append((f"temporal_attention.layers.{i}.", f"temporal_layer_{i}"))

        # Also include temporal_attention components not in layers
        layer_prefixes.append(("temporal_attention.norm.", "temporal_norm"))
        layer_prefixes.append(("temporal_attention.pool.", "temporal_pool"))
        layer_prefixes.append(("temporal_attention.pos_encoder.", "temporal_pos"))

        # Bottom layers (lowest LR)
        layer_prefixes.append(("per_game_encoder.", "per_game_encoder"))
        layer_prefixes.append(("player_embed.", "player_embed"))

        # If form_encoder exists
        if hasattr(self.model, 'form_encoder') and self.model.form_encoder is not None:
            layer_prefixes.append(("form_encoder.", "form_encoder"))

        # Assign parameters to groups with decaying LR
        param_groups = []
        assigned = set()

        for depth, (prefix, group_name) in enumerate(layer_prefixes):
            lr = base_lr * (decay ** depth)
            decay_group = []
            no_decay_group = []

            for name, param in self.model.named_parameters():
                if not param.requires_grad or name in assigned:
                    continue
                if name.startswith(prefix):
                    assigned.add(name)
                    if "bias" in name or "norm" in name or "embedding" in name or "emb" in name:
                        no_decay_group.append(param)
                    else:
                        decay_group.append(param)

            if decay_group:
                param_groups.append({
                    "params": decay_group,
                    "lr": lr,
                    "weight_decay": opt_config.weight_decay,
                })
            if no_decay_group:
                param_groups.append({
                    "params": no_decay_group,
                    "lr": lr,
                    "weight_decay": 0.0,
                })

        # Any remaining unassigned parameters get base LR
        remaining_decay = []
        remaining_no_decay = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad or name in assigned:
                continue
            if "bias" in name or "norm" in name or "embedding" in name or "emb" in name:
                remaining_no_decay.append(param)
            else:
                remaining_decay.append(param)

        if remaining_decay:
            param_groups.append({
                "params": remaining_decay,
                "lr": base_lr,
                "weight_decay": opt_config.weight_decay,
            })
        if remaining_no_decay:
            param_groups.append({
                "params": remaining_no_decay,
                "lr": base_lr,
                "weight_decay": 0.0,
            })

        # Log LR assignment
        for depth, (prefix, group_name) in enumerate(layer_prefixes):
            lr = base_lr * (decay ** depth)
            logger.info(f"  Discriminative LR: {group_name} = {lr:.2e}")

        return AdamW(
            param_groups,
            lr=base_lr,  # default LR (overridden per group)
            betas=tuple(opt_config.betas),
            eps=opt_config.eps,
        )

    def _apply_freeze_phase(self, phase: int):
        """Apply freezing based on phase: 0=frozen, 1=top unfrozen, 2=all unfrozen."""
        if not self.pretrained_params:
            return

        if phase == 0:
            # Freeze all pre-trained parameters
            frozen_count = 0
            for name, param in self.model.named_parameters():
                if name in self.pretrained_params:
                    param.requires_grad = False
                    frozen_count += 1
            logger.info(f"Freeze phase 0: froze {frozen_count} pre-trained parameters")

        elif phase == 1:
            # Unfreeze top temporal block only
            unfrozen = 0
            # Find the top layer index
            n_layers = self.config.model.temporal_layers
            top_layer_prefix = f"temporal_attention.layers.{n_layers - 1}."

            for name, param in self.model.named_parameters():
                if name in self.pretrained_params:
                    if name.startswith(top_layer_prefix) or name.startswith("temporal_attention.norm."):
                        param.requires_grad = True
                        unfrozen += 1
                    else:
                        param.requires_grad = False
            logger.info(f"Freeze phase 1: unfroze top temporal block ({unfrozen} params)")

        elif phase == 2:
            # Unfreeze all
            for name, param in self.model.named_parameters():
                param.requires_grad = True
            logger.info("Freeze phase 2: all parameters unfrozen")

        self._current_unfreeze_phase = phase
        # Rebuild optimizer with new param groups
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()

    def _setup_scheduler(self):
        opt_config = self.config.optimizer
        train_config = self.config.training

        steps_per_epoch = len(self.train_loader)
        accum_steps = train_config.gradient_accumulation_steps
        optimizer_steps_per_epoch = max(steps_per_epoch // accum_steps, 1)
        total_steps = (
            train_config.max_steps
            if train_config.max_steps
            else optimizer_steps_per_epoch * train_config.max_epochs
        )

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

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Detected call of `lr_scheduler.step\\(\\)` before")
            return SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_steps],
            )

    def _setup_wandb(self):
        try:
            import wandb
            self.wandb_run = wandb.init(
                project=self.config.training.wandb_project,
                name=self.config.training.wandb_run_name or self.config.training.experiment_name,
                config=self.config.to_dict(),
            )
        except ImportError:
            logger.warning("wandb not installed, skipping")

    def train(self) -> MetricResults:
        """Run the full training loop."""
        logger.info(f"Starting Phase 2 training: {self.config.training.experiment_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Train samples: {len(self.train_loader.dataset)}")
        logger.info(f"Val samples: {len(self.val_loader.dataset)}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Resolved configuration:\n{self.config}")

        self.config.to_yaml(self.checkpoint_dir / "config.yaml")

        try:
            for epoch in range(self.state.epoch, self.config.training.max_epochs):
                self.state.epoch = epoch

                # Check for unfreezing phase transitions
                if self.pretrained_params:
                    if (self._current_unfreeze_phase == 0
                            and epoch >= self.freeze_pretrained_epochs):
                        self._apply_freeze_phase(1)
                    elif (self._current_unfreeze_phase == 1
                            and epoch >= self.freeze_pretrained_epochs + self.unfreeze_top_epochs):
                        self._apply_freeze_phase(2)

                epoch_start = time.time()

                # Train
                train_loss = self._train_epoch(epoch)
                self.state.train_losses.append(train_loss)

                # Validate (with EMA weights if enabled)
                val_loss, val_metrics = self._validate()
                self.state.val_losses.append(val_loss)
                self.state.val_spread_maes.append(val_metrics.spread_mae)

                epoch_time = time.time() - epoch_start

                # Log epoch summary
                self._log_epoch(epoch, train_loss, val_loss, val_metrics, epoch_time)

                # Early stopping on smoothed val_spread_mae
                self.smoothing_window.append(val_metrics.spread_mae)
                smoothed_mae = sum(self.smoothing_window) / len(self.smoothing_window)

                if smoothed_mae < self.state.best_metric - self.config.training.min_delta:
                    self.state.best_metric = smoothed_mae
                    self.state.patience_counter = 0
                    self._save_checkpoint("best.pt")
                    logger.info(
                        f"New best! Smoothed val_spread_mae: {smoothed_mae:.4f} "
                        f"(raw: {val_metrics.spread_mae:.4f})"
                    )
                else:
                    self.state.patience_counter += 1

                if (epoch + 1) % self.config.training.save_every_n_epochs == 0:
                    self._save_checkpoint(f"epoch_{epoch + 1}.pt")

                if self.state.patience_counter >= self.config.training.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

                if (
                    self.config.training.max_steps
                    and self.state.global_step >= self.config.training.max_steps
                ):
                    logger.info(f"Reached max steps: {self.config.training.max_steps}")
                    break

        except KeyboardInterrupt:
            logger.info("Training interrupted")
            self._save_checkpoint("interrupted.pt")

        # Final evaluation
        self._load_checkpoint("best.pt")
        final_loader = self.test_loader if self.test_loader else self.val_loader
        final_metrics = self._evaluate(final_loader, desc="Final Evaluation")

        logger.info(f"Final Results:")
        logger.info(f"  Spread MAE: {final_metrics.spread_mae:.2f}")
        logger.info(f"  Spread RMSE: {final_metrics.spread_rmse:.2f}")
        logger.info(f"  Win Accuracy: {final_metrics.win_accuracy:.3f}")
        logger.info(f"  Win AUC: {final_metrics.win_auc:.3f}")
        logger.info(f"  Brier Score: {final_metrics.brier_score:.4f}")

        if self.wandb_run:
            self.wandb_run.finish()

        return final_metrics

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        accum_steps = self.config.training.gradient_accumulation_steps

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}", leave=False)
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue

            loss = self._accumulate_step(batch, accum_steps)
            total_loss += loss
            n_batches += 1

            if (batch_idx + 1) % accum_steps == 0:
                self._optimizer_step()

            pbar.set_postfix({
                "loss": f"{loss:.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })

            if self.state.global_step % self.config.training.log_every_n_steps == 0:
                self._log_step(loss)

            self.state.global_step += 1

            if (
                self.config.training.max_steps
                and self.state.global_step >= self.config.training.max_steps
            ):
                break

        if n_batches % accum_steps != 0:
            self._optimizer_step()

        return total_loss / max(n_batches, 1)

    def _accumulate_step(self, batch: dict, accum_steps: int) -> float:
        """Forward + backward pass for one micro-batch."""
        batch = self._batch_to_device(batch)

        if self.use_amp:
            with autocast("cuda", dtype=self.amp_dtype):
                predictions = self.model(batch)
                losses = self.criterion(
                    predictions,
                    batch["target_home_scores"],
                    batch["target_away_scores"],
                )
                loss = losses["total"] / accum_steps

            self.scaler.scale(loss).backward()
        else:
            predictions = self.model(batch)
            losses = self.criterion(
                predictions,
                batch["target_home_scores"],
                batch["target_away_scores"],
            )
            loss = losses["total"] / accum_steps
            loss.backward()

        return losses["total"].item()

    def _optimizer_step(self):
        """Clip gradients, step optimizer, step scheduler, update EMA, zero grads."""
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

        # EMA update after each optimizer step
        if self.ema is not None:
            self.ema.update()

    def _validate(self) -> tuple[float, MetricResults]:
        """Validate with EMA weights if enabled."""
        if self.ema is not None:
            with self.ema.apply():
                return self._validate_inner()
        else:
            return self._validate_inner()

    def _validate_inner(self) -> tuple[float, MetricResults]:
        """Run validation pass."""
        self.model.eval()
        self.metrics_calculator.reset()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                if batch is None:
                    continue

                batch = self._batch_to_device(batch)

                if self.use_amp:
                    with autocast("cuda", dtype=self.amp_dtype):
                        predictions = self.model(batch)
                        losses = self.criterion(
                            predictions,
                            batch["target_home_scores"],
                            batch["target_away_scores"],
                        )
                else:
                    predictions = self.model(batch)
                    losses = self.criterion(
                        predictions,
                        batch["target_home_scores"],
                        batch["target_away_scores"],
                    )

                total_loss += losses["total"].item()
                n_batches += 1

                self.metrics_calculator.update(
                    predictions,
                    batch["target_home_scores"],
                    batch["target_away_scores"],
                )

        avg_loss = total_loss / max(n_batches, 1)
        metrics = self.metrics_calculator.compute()

        return avg_loss, metrics

    def _evaluate(self, loader: DataLoader, desc: str = "Evaluation") -> MetricResults:
        """Evaluate with EMA weights if enabled."""
        if self.ema is not None:
            with self.ema.apply():
                return self._evaluate_inner(loader, desc)
        else:
            return self._evaluate_inner(loader, desc)

    def _evaluate_inner(self, loader: DataLoader, desc: str) -> MetricResults:
        self.model.eval()
        self.metrics_calculator.reset()

        with torch.no_grad():
            for batch in tqdm(loader, desc=desc, leave=False):
                if batch is None:
                    continue

                batch = self._batch_to_device(batch)

                if self.use_amp:
                    with autocast("cuda", dtype=self.amp_dtype):
                        predictions = self.model(batch)
                else:
                    predictions = self.model(batch)

                self.metrics_calculator.update(
                    predictions,
                    batch["target_home_scores"],
                    batch["target_away_scores"],
                )

        return self.metrics_calculator.compute()

    def _batch_to_device(self, batch: dict) -> dict:
        """Move all tensors in batch to device."""
        result = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(self.device)
            elif isinstance(value, dict):
                result[key] = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in value.items()
                }
            else:
                result[key] = value
        return result

    def _log_step(self, loss: float):
        if self.wandb_run:
            import wandb
            wandb.log({
                "train/loss": loss,
                "train/lr": self.optimizer.param_groups[0]["lr"],
                "train/step": self.state.global_step,
            })

    def _log_epoch(self, epoch: int, train_loss: float, val_loss: float,
                   val_metrics: MetricResults, epoch_time: float):
        logger.info(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"spread_mae={val_metrics.spread_mae:.2f}, "
            f"win_acc={val_metrics.win_accuracy:.3f}, "
            f"win_auc={val_metrics.win_auc:.3f}, "
            f"time={epoch_time:.0f}s"
        )

        if self.wandb_run:
            import wandb
            wandb.log({
                "epoch": epoch + 1,
                "train/epoch_loss": train_loss,
                "val/loss": val_loss,
                "val/spread_mae": val_metrics.spread_mae,
                "val/spread_rmse": val_metrics.spread_rmse,
                "val/home_mae": val_metrics.home_mae,
                "val/away_mae": val_metrics.away_mae,
                "val/win_accuracy": val_metrics.win_accuracy,
                "val/win_auc": val_metrics.win_auc,
                "val/brier_score": val_metrics.brier_score,
                "val/ece": val_metrics.ece,
                "val/spread_coverage_90": val_metrics.spread_coverage_90,
                "epoch_time": epoch_time,
            })

    def _save_checkpoint(self, filename: str):
        path = self.checkpoint_dir / filename

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "state": {
                "epoch": self.state.epoch,
                "global_step": self.state.global_step,
                "best_metric": self.state.best_metric,
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

        self._cleanup_checkpoints()

    def _load_checkpoint(self, filename: str):
        path = self.checkpoint_dir / filename

        if not path.exists():
            logger.warning(f"Checkpoint not found: {path}")
            return

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except ValueError as e:
            # Parameter groups may differ if checkpoint was saved during a different
            # freeze phase. Model weights are loaded correctly; skip optimizer/scheduler.
            logger.warning(f"Skipping optimizer/scheduler restore (param group mismatch): {e}")

        state = checkpoint["state"]
        self.state.epoch = state["epoch"]
        self.state.global_step = state["global_step"]
        self.state.best_metric = state["best_metric"]
        self.state.patience_counter = state["patience_counter"]

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        if self.ema is not None and "ema_state_dict" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state_dict"])

        logger.info(f"Loaded checkpoint: {path}")

    def _cleanup_checkpoints(self):
        if not self.config.training.save_best_only:
            return

        keep_n = self.config.training.keep_n_checkpoints
        checkpoints = sorted(
            self.checkpoint_dir.glob("epoch_*.pt"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

        for ckpt in checkpoints[keep_n:]:
            ckpt.unlink()

    def resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from a specific checkpoint."""
        self._load_checkpoint(checkpoint_path)
        logger.info(f"Resuming from epoch {self.state.epoch + 1}")
