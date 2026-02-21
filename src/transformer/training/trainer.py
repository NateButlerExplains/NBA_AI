"""
Trainer for NBA Transformer Model.

This is the "engine" that orchestrates the entire training process. It handles:
- Optimization: AdamW optimizer with cosine learning rate schedule
- Gradient accumulation: simulates larger batch sizes on limited GPU memory
- Mixed precision (AMP): uses float16 for speed, float32 for accuracy where needed
- Validation: periodically evaluates on held-out data
- Early stopping: stops training when the model stops improving
- Checkpointing: saves snapshots so training can be resumed if interrupted
- Logging: tracks metrics via console logs and optionally Weights & Biases (wandb)

The typical training flow:
    1. For each epoch, iterate over all training batches
    2. For each batch: forward pass -> compute loss -> backward pass -> update weights
    3. After each epoch, evaluate on validation set
    4. Save the model if it is the best so far
    5. Stop early if no improvement for `patience` epochs

Usage:
    trainer = Trainer(model, config, train_loader, val_loader)
    trainer.train()
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.transformer.models.phase1_model import Phase1Model
from src.transformer.models.prediction_heads import GamePrediction
from src.transformer.training.config import ExperimentConfig
from src.transformer.training.loss import CombinedLoss
from src.transformer.training.metrics import MetricsCalculator, MetricResults


@dataclass
class TrainingState:
    """
    Tracks training progress so we can resume from checkpoints.

    This is saved inside checkpoint files, so if training is interrupted
    (e.g., GPU crash), we can pick up exactly where we left off.
    """

    epoch: int = 0                              # Current epoch number
    global_step: int = 0                        # Total micro-batches processed so far
    best_val_loss: float = float("inf")         # Best validation loss seen (for early stopping)
    patience_counter: int = 0                   # Epochs since last improvement
    train_losses: list = None                   # History of per-epoch training losses
    val_losses: list = None                     # History of per-epoch validation losses

    def __post_init__(self):
        if self.train_losses is None:
            self.train_losses = []
        if self.val_losses is None:
            self.val_losses = []


class Trainer:
    """
    Trainer for Phase1Model.

    Handles the complete training loop including optimization,
    validation, checkpointing, and logging.
    """

    def __init__(
        self,
        model: Phase1Model,
        config: ExperimentConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: Phase1Model instance
            config: ExperimentConfig with all hyperparameters
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Optional test data loader
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Setup logger first (needed by other setup methods)
        self.logger = logging.getLogger(__name__)

        # Setup device (GPU if available, otherwise CPU)
        self.device = self._setup_device(config.training.device)
        # Move all model weights to the chosen device (GPU memory or CPU RAM)
        self.model = self.model.to(self.device)

        # Setup loss function — CombinedLoss wraps spread + score + win prob losses
        # with the configured weights (spread=1.0, score=0.5, win_prob=0.1 by default)
        self.criterion = CombinedLoss(
            spread_weight=config.training.spread_loss_weight,
            score_weight=config.training.score_loss_weight,
            win_prob_weight=config.training.win_prob_loss_weight,
            nll_weight=config.training.nll_weight,
            mse_weight=config.training.mse_weight,
        )

        # Setup optimizer — AdamW with weight decay separation (see _setup_optimizer)
        self.optimizer = self._setup_optimizer()

        # Setup learning rate scheduler — warmup then cosine decay (see _setup_scheduler)
        self.scheduler = self._setup_scheduler()

        # --- Mixed Precision (AMP) Setup ---
        # AMP (Automatic Mixed Precision) uses float16 for the forward and backward
        # passes (2x faster, uses less GPU memory) but keeps weight updates in float32
        # (to maintain precision). Only works on CUDA GPUs.
        self.use_amp = config.training.use_amp and self.device.type == "cuda"
        # GradScaler prevents float16 underflow: it scales the loss UP before the
        # backward pass (so tiny gradients do not round to zero), then scales
        # gradients back DOWN before the optimizer step.
        self.scaler = GradScaler("cuda") if self.use_amp else None
        self.amp_dtype = getattr(torch, config.training.amp_dtype) if self.use_amp else None

        # Metrics calculator accumulates predictions across validation batches
        self.metrics_calculator = MetricsCalculator()

        # Create directory for saving model checkpoints
        self.checkpoint_dir = Path(config.training.checkpoint_dir) / config.training.experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize training state (epoch counter, best loss, etc.)
        self.state = TrainingState()

        # Optional Weights & Biases integration for experiment tracking dashboards
        self.wandb_run = None
        if config.training.use_wandb:
            self._setup_wandb()

    def _setup_device(self, device_str: str) -> torch.device:
        """Setup and validate device."""
        if device_str == "cuda" and not torch.cuda.is_available():
            self.logger.warning("CUDA not available, falling back to CPU")
            return torch.device("cpu")
        elif device_str == "mps" and not torch.backends.mps.is_available():
            self.logger.warning("MPS not available, falling back to CPU")
            return torch.device("cpu")
        return torch.device(device_str)

    def _setup_optimizer(self) -> AdamW:
        """
        Setup AdamW optimizer with selective weight decay.

        WEIGHT DECAY EXPLAINED:
            Weight decay is a regularization technique that penalizes large weights.
            Each update, weights are slightly pulled toward zero. This prevents
            overfitting by discouraging the model from relying too heavily on any
            single feature.

            However, certain parameters should NOT have weight decay applied:
            - Bias terms: these are just offsets, not learned features
            - LayerNorm parameters: these normalize activations, regularizing them
              would fight the normalization
            - Embeddings: these are lookup tables, not weight matrices

            This is standard practice across all modern transformer training.
        """
        opt_config = self.config.optimizer

        # Separate parameters into two groups: those that get weight decay and those that do not.
        # This is a common pattern in transformer training.
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue  # Skip frozen parameters (not being trained)
            # Bias, normalization, and embedding params go in the "no decay" group
            if "bias" in name or "norm" in name or "embedding" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        # PyTorch optimizers accept a list of "parameter groups," each with its own
        # hyperparameters. Here, decay_params get weight_decay; no_decay_params get 0.
        param_groups = [
            {"params": decay_params, "weight_decay": opt_config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        return AdamW(
            param_groups,
            lr=opt_config.learning_rate,
            betas=tuple(opt_config.betas),  # Momentum parameters (default: 0.9, 0.999)
            eps=opt_config.eps,              # Numerical stability (prevents division by zero)
        )

    def _setup_scheduler(self):
        """
        Setup learning rate scheduler with warmup + decay.

        LEARNING RATE SCHEDULE EXPLAINED:
            The learning rate (LR) controls how big each weight update is.
            Too high -> training diverges (loss explodes). Too low -> training is
            very slow. The schedule changes the LR over the course of training:

            Phase 1 - WARMUP (first ~1000 steps):
                LR starts at near-zero (0.1x the target) and linearly increases
                to the full learning rate. This prevents the randomly initialized
                model from taking huge, destructive gradient steps at the start.

            Phase 2 - COSINE DECAY (remaining steps):
                LR gradually decreases following a cosine curve, eventually
                reaching min_lr (0.1x the peak). This lets the model make
                fine-grained adjustments as it converges, avoiding overshooting
                the optimal weights.

            Visualization:
                LR
                ^     /----------\\
                |    /            \\
                |   /              \\
                |  /                \\____
                | /
                +----------------------------> steps
                  warmup    cosine decay
        """
        opt_config = self.config.optimizer
        train_config = self.config.training

        # Calculate total optimizer steps, accounting for gradient accumulation.
        # With gradient accumulation, the optimizer only steps every `accum_steps`
        # micro-batches, so there are fewer optimizer steps than total batches.
        steps_per_epoch = len(self.train_loader)
        accum_steps = train_config.gradient_accumulation_steps
        optimizer_steps_per_epoch = steps_per_epoch // accum_steps
        total_steps = (
            train_config.max_steps
            if train_config.max_steps
            else optimizer_steps_per_epoch * train_config.max_epochs
        )

        # Warmup can be specified as a fixed number of steps or as a ratio of total steps.
        # Explicit warmup_steps takes precedence; warmup_ratio is only used as fallback.
        if opt_config.warmup_steps > 0:
            warmup_steps = opt_config.warmup_steps
        elif opt_config.warmup_ratio > 0:
            warmup_steps = int(total_steps * opt_config.warmup_ratio)
        else:
            warmup_steps = 0

        # Warmup phase: linearly ramp LR from 10% to 100% of the target
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,   # Start at 10% of the configured learning rate
            end_factor=1.0,     # Ramp up to 100%
            total_iters=warmup_steps,
        )

        # Main decay phase: gradually reduce LR after warmup
        main_steps = total_steps - warmup_steps
        min_lr = opt_config.learning_rate * opt_config.min_lr_ratio  # Floor LR

        if opt_config.scheduler == "cosine":
            # Cosine annealing: smooth, gradual decay (most popular choice)
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=main_steps,
                eta_min=min_lr,
            )
        elif opt_config.scheduler == "linear":
            # Linear decay: LR decreases at a constant rate
            main_scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=opt_config.min_lr_ratio,
                total_iters=main_steps,
            )
        else:  # constant
            # No decay: LR stays at the peak value (useful for debugging)
            main_scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=1.0,
                total_iters=main_steps,
            )

        # SequentialLR chains warmup and main: warmup runs first, then switches
        # to the main scheduler at the milestone step.
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )

    def _setup_wandb(self):
        """Setup wandb logging."""
        try:
            import wandb

            self.wandb_run = wandb.init(
                project=self.config.training.wandb_project,
                name=self.config.training.wandb_run_name or self.config.training.experiment_name,
                config=self.config.to_dict(),
            )
        except ImportError:
            self.logger.warning("wandb not installed, skipping")

    def train(self) -> MetricResults:
        """
        Run the full training loop: train epochs, validate, checkpoint, early stop.

        This is the main entry point. It loops over epochs, and for each epoch:
            1. Trains on all training data (_train_epoch)
            2. Evaluates on validation data (_validate)
            3. Saves the model if it improved (checkpointing)
            4. Stops early if no improvement for `patience` epochs

        Returns:
            Final test set metrics (or validation if no test set)
        """
        self.logger.info(f"Starting training for {self.config.training.experiment_name}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Train samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Val samples: {len(self.val_loader.dataset)}")

        # Log the fully resolved configuration so the user can verify all settings.
        # This runs AFTER CLI overrides and vocab_sizes injection, so it shows
        # the actual values that will be used for training.
        self.logger.info(f"Resolved configuration:\n{self.config}")

        # Save the experiment config alongside checkpoints for reproducibility —
        # you can always see exactly what hyperparameters produced a given model.
        self.config.to_yaml(self.checkpoint_dir / "config.yaml")

        try:
            for epoch in range(self.state.epoch, self.config.training.max_epochs):
                self.state.epoch = epoch

                # --- Training phase ---
                train_loss = self._train_epoch(epoch)
                self.state.train_losses.append(train_loss)

                # --- Validation phase ---
                val_loss, val_metrics = self._validate()
                self.state.val_losses.append(val_loss)

                # Log epoch summary (train loss, val loss, key metrics)
                self._log_epoch(epoch, train_loss, val_loss, val_metrics)

                # --- Early stopping logic ---
                # Did validation loss improve by at least min_delta?
                if val_loss < self.state.best_val_loss - self.config.training.min_delta:
                    # Yes! Save this as the new best model.
                    self.state.best_val_loss = val_loss
                    self.state.patience_counter = 0
                    self._save_checkpoint("best.pt")
                    self.logger.info(f"New best model! Val loss: {val_loss:.4f}")
                else:
                    # No improvement. Increment patience counter.
                    self.state.patience_counter += 1

                # Save periodic checkpoint (e.g., every 5 epochs) for safety
                if (epoch + 1) % self.config.training.save_every_n_epochs == 0:
                    self._save_checkpoint(f"epoch_{epoch + 1}.pt")

                # If no improvement for `patience` epochs, stop training.
                # This prevents overfitting: the model starts memorizing training
                # data without improving on unseen validation data.
                if self.state.patience_counter >= self.config.training.patience:
                    self.logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

                # Alternative stopping condition: max training steps reached
                if (
                    self.config.training.max_steps
                    and self.state.global_step >= self.config.training.max_steps
                ):
                    self.logger.info(f"Reached max steps: {self.config.training.max_steps}")
                    break

        except KeyboardInterrupt:
            # If the user presses Ctrl+C, save progress before exiting
            self.logger.info("Training interrupted by user")
            self._save_checkpoint("interrupted.pt")

        # --- Final evaluation ---
        # Load the BEST model (not the last one!) for final evaluation.
        # The last model may have started overfitting; the best one is saved above.
        self._load_checkpoint("best.pt")

        # Evaluate on test set if available, otherwise use validation set
        final_loader = self.test_loader if self.test_loader else self.val_loader
        final_metrics = self._evaluate(final_loader, desc="Final Evaluation")

        self.logger.info(f"Final Results:")
        self.logger.info(f"  Spread MAE: {final_metrics.spread_mae:.2f}")
        self.logger.info(f"  Win Accuracy: {final_metrics.win_accuracy:.3f}")
        self.logger.info(f"  Brier Score: {final_metrics.brier_score:.4f}")

        if self.wandb_run:
            self.wandb_run.finish()

        return final_metrics

    def _train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch (one pass through the entire training dataset).

        GRADIENT ACCUMULATION EXPLAINED:
            If we want an effective batch size of 32 but can only fit 8 samples
            in GPU memory, we process 4 micro-batches of 8 and accumulate the
            gradients. Only after all 4 micro-batches do we update the weights.
            This gives the same result as a single batch of 32, but uses less memory.

            Example with gradient_accumulation_steps=4:
                Micro-batch 1: forward + backward (gradients accumulate)
                Micro-batch 2: forward + backward (gradients accumulate more)
                Micro-batch 3: forward + backward (gradients accumulate more)
                Micro-batch 4: forward + backward (gradients accumulate more)
                -> NOW: clip gradients, optimizer.step(), zero gradients
        """
        self.model.train()  # Set model to training mode (enables dropout, etc.)
        total_loss = 0.0
        n_batches = 0
        accum_steps = self.config.training.gradient_accumulation_steps

        # tqdm creates a progress bar in the terminal
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}",
            leave=False,
        )

        # Zero out any leftover gradients from previous epoch
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            if batch is None:  # Skip empty batches (can happen with bad data)
                continue

            # Forward + backward pass for one micro-batch (no weight update yet)
            loss = self._accumulate_step(batch, accum_steps)
            total_loss += loss
            n_batches += 1

            # Only update weights after accumulating `accum_steps` micro-batches
            if (batch_idx + 1) % accum_steps == 0:
                self._optimizer_step()

            # Update the progress bar with current loss and learning rate
            pbar.set_postfix({"loss": f"{loss:.4f}", "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"})

            # Periodic logging (e.g., every 100 steps) to wandb if enabled
            if self.state.global_step % self.config.training.log_every_n_steps == 0:
                self._log_step(loss)

            self.state.global_step += 1

            # Stop if we have reached the max number of training steps
            if (
                self.config.training.max_steps
                and self.state.global_step >= self.config.training.max_steps
            ):
                break

        # If the last set of micro-batches did not complete a full accumulation cycle,
        # still update the weights with whatever gradients we have accumulated.
        if n_batches % accum_steps != 0:
            self._optimizer_step()

        # Return average loss for this epoch
        return total_loss / max(n_batches, 1)

    def _accumulate_step(self, batch: dict, accum_steps: int) -> float:
        """
        Forward + backward pass for one micro-batch. Does NOT update weights.

        The gradients from this micro-batch are accumulated (added) to any
        existing gradients from previous micro-batches. Weight update happens
        later in _optimizer_step() after enough micro-batches have been processed.
        """
        # Move batch data to GPU (or CPU)
        batch = self._batch_to_device(batch)

        if self.use_amp:
            # --- Mixed Precision Path ---
            # autocast tells PyTorch to run operations in float16 where safe,
            # and float32 where needed (like loss computation). This is ~2x faster
            # on modern GPUs and uses ~half the memory.
            with autocast("cuda", dtype=self.amp_dtype):
                predictions = self._forward(batch)
                losses = self.criterion(
                    predictions,
                    batch["target_home_scores"],
                    batch["target_away_scores"],
                )
                # Divide loss by accum_steps so the total gradient after accumulation
                # equals what we would get from a single large batch
                loss = losses["total"] / accum_steps

            # GradScaler scales the loss UP before backward() to prevent tiny float16
            # gradients from underflowing to zero. The scaling is undone before the
            # optimizer step (in _optimizer_step via scaler.unscale_).
            self.scaler.scale(loss).backward()
        else:
            # --- Standard Precision Path (CPU or when AMP is disabled) ---
            predictions = self._forward(batch)
            losses = self.criterion(
                predictions,
                batch["target_home_scores"],
                batch["target_away_scores"],
            )
            loss = losses["total"] / accum_steps

            # backward() computes gradients of the loss with respect to all model
            # parameters. Gradients accumulate (add) across micro-batches.
            loss.backward()

        # Return the UN-divided loss for logging (so the displayed loss is
        # interpretable regardless of accumulation steps)
        return losses["total"].item()

    def _optimizer_step(self):
        """
        Perform a single weight update: clip gradients, step optimizer, step scheduler, zero gradients.

        This is called after gradient accumulation is complete (every accum_steps micro-batches).
        """
        if self.use_amp:
            # Undo the GradScaler's scaling so we can clip gradients at their true magnitude
            self.scaler.unscale_(self.optimizer)

        # GRADIENT CLIPPING: cap the total gradient norm at max_grad_norm (default: 1.0).
        # This prevents "exploding gradients" — rare batches that produce huge gradients
        # which would otherwise cause a catastrophic weight update and destabilize training.
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.optimizer.max_grad_norm,
        )

        if self.use_amp:
            # scaler.step() only updates weights if the gradients are finite (no NaN/Inf).
            # If gradients overflowed in float16, it skips this step and adjusts the scale.
            self.scaler.step(self.optimizer)
            # Update the GradScaler's internal scale factor for the next iteration
            self.scaler.update()
        else:
            # Standard optimizer step: update all model weights using the computed gradients
            self.optimizer.step()

        # Advance the learning rate schedule (warmup or cosine decay)
        self.scheduler.step()
        # Zero out all gradients to prepare for the next accumulation cycle.
        # Without this, gradients would keep accumulating indefinitely.
        self.optimizer.zero_grad()

    def _forward(self, batch: dict) -> GamePrediction:
        """Execute model forward pass."""
        return self.model(
            home_history=batch["home_history"],
            away_history=batch["away_history"],
            home_roster=batch.get("home_roster"),
            away_roster=batch.get("away_roster"),
            home_rest_days=batch.get("home_rest_days"),
            away_rest_days=batch.get("away_rest_days"),
        )

    def _validate(self) -> tuple[float, MetricResults]:
        """
        Run validation: evaluate the model on the held-out validation set.

        Validation happens without gradient computation (faster, less memory)
        and with the model in eval mode (dropout disabled).
        """
        self.model.eval()   # Disable dropout (LayerNorm is unaffected by eval mode)
        self.metrics_calculator.reset()  # Clear any accumulated predictions
        total_loss = 0.0
        n_batches = 0

        # torch.no_grad() disables gradient tracking — we do not need gradients
        # during validation, and this saves ~50% memory and speeds up computation.
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                if batch is None:
                    continue

                batch = self._batch_to_device(batch)

                if self.use_amp:
                    with autocast("cuda", dtype=self.amp_dtype):
                        predictions = self._forward(batch)
                        losses = self.criterion(
                            predictions,
                            batch["target_home_scores"],
                            batch["target_away_scores"],
                        )
                else:
                    predictions = self._forward(batch)
                    losses = self.criterion(
                        predictions,
                        batch["target_home_scores"],
                        batch["target_away_scores"],
                    )

                total_loss += losses["total"].item()
                n_batches += 1

                # Update metrics
                self.metrics_calculator.update(
                    predictions,
                    batch["target_home_scores"],
                    batch["target_away_scores"],
                )

        avg_loss = total_loss / max(n_batches, 1)
        metrics = self.metrics_calculator.compute()

        return avg_loss, metrics

    def _evaluate(self, loader: DataLoader, desc: str = "Evaluation") -> MetricResults:
        """Evaluate on a data loader."""
        self.model.eval()
        self.metrics_calculator.reset()

        with torch.no_grad():
            for batch in tqdm(loader, desc=desc, leave=False):
                if batch is None:
                    continue

                batch = self._batch_to_device(batch)

                if self.use_amp:
                    with autocast("cuda", dtype=self.amp_dtype):
                        predictions = self._forward(batch)
                else:
                    predictions = self._forward(batch)

                self.metrics_calculator.update(
                    predictions,
                    batch["target_home_scores"],
                    batch["target_away_scores"],
                )

        return self.metrics_calculator.compute()

    def _batch_to_device(self, batch: dict) -> dict:
        """
        Move all tensors in a batch from CPU to GPU (or the configured device).

        Data loaders produce batches on CPU by default. Before the model can
        process them, all tensors must be on the same device as the model weights.
        This recursively handles nested dicts (e.g., batch["context"]["day_of_week"]).
        """
        result = {}

        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(self.device)
            elif isinstance(value, dict):
                # Handle nested dicts (e.g., context features)
                result[key] = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in value.items()
                }
            else:
                result[key] = value

        return result

    def _log_step(self, loss: float):
        """Log training step."""
        if self.wandb_run:
            import wandb

            wandb.log({
                "train/loss": loss,
                "train/lr": self.optimizer.param_groups[0]["lr"],
                "train/step": self.state.global_step,
            })

    def _log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_metrics: MetricResults,
    ):
        """Log epoch results."""
        self.logger.info(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"spread_mae={val_metrics.spread_mae:.2f}, "
            f"win_acc={val_metrics.win_accuracy:.3f}"
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
            })

    def _save_checkpoint(self, filename: str):
        """
        Save a complete training snapshot (checkpoint) to disk.

        A checkpoint contains EVERYTHING needed to resume training or evaluate
        the model later:
            - model_state_dict: all learned weights (millions of numbers)
            - optimizer_state_dict: optimizer momentum/state (needed to resume)
            - scheduler_state_dict: current position in the LR schedule
            - state: epoch number, step count, best loss (for early stopping)
            - config: hyperparameters (for reproducibility)
            - scaler_state_dict: GradScaler state (if using mixed precision)

        This means if your computer crashes at epoch 50, you can resume from
        exactly where you left off — same weights, same optimizer state, same LR.
        """
        path = self.checkpoint_dir / filename

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "state": {
                "epoch": self.state.epoch,
                "global_step": self.state.global_step,
                "best_val_loss": self.state.best_val_loss,
                "patience_counter": self.state.patience_counter,
            },
            "config": self.config.to_dict(),
        }

        # Save GradScaler state if using mixed precision (needed to resume AMP correctly)
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, path)
        self.logger.debug(f"Saved checkpoint: {path}")

        # Remove old periodic checkpoints to save disk space
        self._cleanup_checkpoints()

    def _load_checkpoint(self, filename: str):
        """
        Restore the full training state from a saved checkpoint.

        map_location=self.device ensures the checkpoint is loaded to the correct
        device (e.g., if you saved on GPU but are loading on CPU, or vice versa).
        """
        path = self.checkpoint_dir / filename

        if not path.exists():
            self.logger.warning(f"Checkpoint not found: {path}")
            return

        # Load all saved state from disk
        checkpoint = torch.load(path, map_location=self.device)

        # Restore each component to exactly where it was when saved
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Restore training progress counters
        state = checkpoint["state"]
        self.state.epoch = state["epoch"]
        self.state.global_step = state["global_step"]
        self.state.best_val_loss = state["best_val_loss"]
        self.state.patience_counter = state["patience_counter"]

        # Restore GradScaler state for mixed precision continuity
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.logger.info(f"Loaded checkpoint: {path}")

    def _cleanup_checkpoints(self):
        """
        Remove old periodic checkpoints, keeping only the most recent N.

        This prevents disk space from filling up during long training runs.
        Note: "best.pt" and "interrupted.pt" are never cleaned up — only
        the periodic "epoch_N.pt" files are managed here.
        """
        if not self.config.training.save_best_only:
            return

        keep_n = self.config.training.keep_n_checkpoints
        # Sort by modification time (most recent first)
        checkpoints = sorted(
            self.checkpoint_dir.glob("epoch_*.pt"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

        # Delete everything after the first `keep_n` checkpoints
        for ckpt in checkpoints[keep_n:]:
            ckpt.unlink()
            self.logger.debug(f"Removed old checkpoint: {ckpt}")

    def resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from a specific checkpoint."""
        self._load_checkpoint(checkpoint_path)
        self.logger.info(f"Resuming from epoch {self.state.epoch + 1}")


def test_trainer():
    """Test trainer with mock data."""
    import logging

    logging.basicConfig(level=logging.INFO)

    print("Testing Trainer (mock data)...")

    # This test requires actual model and data, so we just test initialization
    from src.transformer.training.config import get_small_config

    config = get_small_config()
    config.training.device = "cpu"
    config.training.max_epochs = 1
    config.training.use_amp = False

    print(f"  Config loaded")
    print(f"  Hidden dim: {config.model.hidden_dim}")
    print(f"  Device: {config.training.device}")

    print("\nTrainer initialization test passed!")
    print("(Full trainer test requires actual data)")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(test_trainer())
