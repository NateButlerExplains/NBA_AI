"""Training loop for the Phase 4 generative model.

Features
--------
- AMP (float16 on CUDA)
- EMA weight averaging
- Gradient accumulation (default 8 micro-batches -> effective batch 32)
- Gradient clipping (max_norm configurable, default 1.0)
- Cosine LR schedule with linear warmup
- Early stopping on ``val_loss`` with 3-epoch smoothing window
- Checkpointing (save best + keep N periodic)
- Per-class score accuracy logging
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

from src.generative.config import GenerativeExperimentConfig
from src.generative.training.ema import EMA
from src.generative.training.loss import GenerativeLoss

logger = logging.getLogger(__name__)


# ---- Score-event class names (for logging) ---------------------------------
SCORE_CLASS_NAMES = [
    "no_score",
    "home+1",
    "home+2",
    "home+3",
    "away+1",
    "away+2",
    "away+3",
]


@dataclass
class TrainingState:
    """Tracks training progress for checkpointing and resumption."""

    epoch: int = 0
    global_step: int = 0
    best_metric: float = float("inf")
    patience_counter: int = 0
    train_losses: list = field(default_factory=list)
    val_losses: list = field(default_factory=list)


class GenerativeTrainer:
    """Training loop for the generative model.

    Parameters
    ----------
    model : nn.Module
        The ``GenerativeModel`` instance (not yet on device).
    config : GenerativeExperimentConfig
        Full experiment configuration.
    train_loader, val_loader : DataLoader
        Training and validation data loaders.
    test_loader : DataLoader, optional
        Held-out test loader for final evaluation.
    """

    def __init__(
        self,
        model: nn.Module,
        config: GenerativeExperimentConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
    ) -> None:
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # ---- Device --------------------------------------------------------
        self.device = self._setup_device(config.training.device)
        self.model = model.to(self.device)

        # ---- Loss ----------------------------------------------------------
        class_weights = torch.tensor(
            config.model.score_class_weights, dtype=torch.float32, device=self.device
        )
        self.criterion = GenerativeLoss(
            score_weight=config.training.score_loss_weight,
            clock_weight=config.training.clock_loss_weight,
            context_weight=config.training.context_loss_weight,
            pre_margin_weight=config.training.pre_margin_weight,
            pre_win_weight=config.training.pre_win_weight,
            class_weights=class_weights,
        )

        # ---- Optimizer -----------------------------------------------------
        self.optimizer = self._setup_optimizer()

        # ---- Scheduler -----------------------------------------------------
        self.scheduler = self._setup_scheduler()

        # ---- AMP -----------------------------------------------------------
        self.use_amp = config.training.use_amp and self.device.type == "cuda"
        self.scaler = GradScaler("cuda") if self.use_amp else None
        self.amp_dtype = (
            getattr(torch, config.training.amp_dtype) if self.use_amp else None
        )

        # ---- EMA -----------------------------------------------------------
        self.ema: Optional[EMA] = None
        if config.training.use_ema:
            self.ema = EMA(self.model, decay=config.training.ema_decay)

        # ---- Early stopping (smoothed val_loss) ----------------------------
        self.smoothing_window: collections.deque = collections.deque(
            maxlen=config.training.smoothing_window,
        )

        # ---- Checkpointing ------------------------------------------------
        self.checkpoint_dir = (
            Path(config.training.checkpoint_dir) / config.training.experiment_name
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # ---- State ---------------------------------------------------------
        self.state = TrainingState()

    # ---- Setup helpers -----------------------------------------------------

    @staticmethod
    def _setup_device(device_str: str) -> torch.device:
        if device_str == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            return torch.device("cpu")
        return torch.device(device_str)

    def _setup_optimizer(self) -> AdamW:
        opt = self.config.optimizer
        decay_params: list[torch.Tensor] = []
        no_decay_params: list[torch.Tensor] = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "embedding" in name or "emb" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": opt.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        return AdamW(
            param_groups,
            lr=opt.learning_rate,
            betas=tuple(opt.betas),
            eps=opt.eps,
        )

    def _setup_scheduler(self):
        opt = self.config.optimizer
        train_cfg = self.config.training

        steps_per_epoch = len(self.train_loader)
        accum = train_cfg.gradient_accumulation_steps
        opt_steps_per_epoch = max(steps_per_epoch // accum, 1)
        total_steps = opt_steps_per_epoch * train_cfg.max_epochs

        warmup_steps = max(int(total_steps * opt.warmup_ratio), 1)

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        main_steps = max(total_steps - warmup_steps, 1)
        min_lr = opt.learning_rate * opt.min_lr_ratio

        main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=main_steps,
            eta_min=min_lr,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "Detected call of `lr_scheduler.step\\(\\)` before"
            )
            return SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_steps],
            )

    # ---- Main training loop ------------------------------------------------

    def train(self) -> dict:
        """Run the full training loop.

        Returns
        -------
        dict
            Final validation metrics from the best checkpoint.
        """
        logger.info(
            f"Starting generative training: {self.config.training.experiment_name}"
        )
        logger.info(f"Device: {self.device}")
        logger.info(f"Train batches: {len(self.train_loader)}")
        logger.info(f"Val batches: {len(self.val_loader)}")
        logger.info(
            f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}"
        )

        # Save config alongside checkpoints
        self.config.to_yaml(self.checkpoint_dir / "config.yaml")

        try:
            for epoch in range(self.state.epoch, self.config.training.max_epochs):
                self.state.epoch = epoch
                epoch_start = time.time()

                # -- Train --
                train_metrics = self._train_epoch(epoch)
                self.state.train_losses.append(train_metrics["loss"])

                # -- Validate --
                val_metrics = self._validate(epoch)
                self.state.val_losses.append(val_metrics["loss"])

                epoch_time = time.time() - epoch_start

                # -- Log --
                self._log_epoch(epoch, train_metrics, val_metrics, epoch_time)

                # -- Early stopping on smoothed val_loss --
                self.smoothing_window.append(val_metrics["loss"])
                smoothed = sum(self.smoothing_window) / len(self.smoothing_window)

                if smoothed < self.state.best_metric - self.config.training.min_delta:
                    self.state.best_metric = smoothed
                    self.state.patience_counter = 0
                    self._save_checkpoint("best.pt")
                    logger.info(
                        f"  New best! Smoothed val_loss: {smoothed:.4f} "
                        f"(raw: {val_metrics['loss']:.4f})"
                    )
                else:
                    self.state.patience_counter += 1

                if self.state.patience_counter >= self.config.training.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        except KeyboardInterrupt:
            logger.info("Training interrupted — saving checkpoint")
            self._save_checkpoint("interrupted.pt")

        # -- Final evaluation with best weights --
        best_path = self.checkpoint_dir / "best.pt"
        if best_path.exists():
            self._load_checkpoint("best.pt")

        final_loader = self.test_loader if self.test_loader else self.val_loader
        final_metrics = self._validate_full(final_loader, desc="Final Evaluation")

        logger.info("Final Results:")
        for k, v in final_metrics.items():
            logger.info(f"  {k}: {v:.4f}")

        return final_metrics

    # ---- Train epoch -------------------------------------------------------

    def _train_epoch(self, epoch: int) -> dict:
        """Single training epoch with gradient accumulation."""
        self.model.train()
        accum = self.config.training.gradient_accumulation_steps
        log_interval = self.config.training.log_every_n_steps

        running_loss = 0.0
        running_score_loss = 0.0
        running_clock_loss = 0.0
        running_context_loss = 0.0
        running_pre_margin_loss = 0.0
        running_pre_win_loss = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}", leave=False)
        self.optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            if batch is None:
                continue

            batch = self._to_device(batch)
            losses = self._accumulate_step(batch, accum)

            running_loss += losses["total"]
            running_score_loss += losses["score_loss"]
            running_clock_loss += losses["clock_loss"]
            running_context_loss += losses["context_loss"]
            running_pre_margin_loss += losses.get("pre_margin_loss", 0.0)
            running_pre_win_loss += losses.get("pre_win_loss", 0.0)
            n_batches += 1

            if (step + 1) % accum == 0:
                self._optimizer_step()

            pbar.set_postfix(
                {
                    "loss": f"{losses['total']:.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                }
            )

            if (
                self.state.global_step > 0
                and self.state.global_step % log_interval == 0
            ):
                logger.debug(
                    f"  step {self.state.global_step}: "
                    f"loss={losses['total']:.4f}, "
                    f"score={losses['score_loss']:.4f}, "
                    f"clock={losses['clock_loss']:.4f}, "
                    f"ctx={losses['context_loss']:.4f}"
                )

            self.state.global_step += 1

        # Flush remaining accumulated gradients
        if n_batches % accum != 0:
            self._optimizer_step()

        n = max(n_batches, 1)
        return {
            "loss": running_loss / n,
            "score_loss": running_score_loss / n,
            "clock_loss": running_clock_loss / n,
            "context_loss": running_context_loss / n,
            "pre_margin_loss": running_pre_margin_loss / n,
            "pre_win_loss": running_pre_win_loss / n,
        }

    def _accumulate_step(self, batch: dict, accum_steps: int) -> dict:
        """Forward + backward for one micro-batch.  Returns raw loss values."""
        if self.use_amp:
            with autocast("cuda", dtype=self.amp_dtype):
                predictions = self.model(batch)
                targets = self._extract_targets(batch)
                losses = self.criterion(predictions, targets)
                scaled = losses["total"] / accum_steps

            # Skip NaN losses to prevent poisoning optimizer state
            if torch.isnan(scaled) or torch.isinf(scaled):
                logger.warning(
                    f"NaN/Inf loss detected at step {self.state.global_step}, skipping batch"
                )
                return {k: 0.0 for k in losses}

            self.scaler.scale(scaled).backward()
        else:
            predictions = self.model(batch)
            targets = self._extract_targets(batch)
            losses = self.criterion(predictions, targets)
            scaled = losses["total"] / accum_steps

            if torch.isnan(scaled) or torch.isinf(scaled):
                logger.warning(
                    f"NaN/Inf loss detected at step {self.state.global_step}, skipping batch"
                )
                return {k: 0.0 for k in losses}

            scaled.backward()

        return {k: v.item() for k, v in losses.items()}

    def _optimizer_step(self) -> None:
        """Clip grads, step optimizer/scheduler, update EMA, zero grads."""
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

    # ---- Validation --------------------------------------------------------

    def _validate(self, epoch: int) -> dict:
        """Validation pass (with EMA weights if enabled).

        Uses per-step CE/clock/context losses — cheaper than full rollout.
        Full rollout evaluation is available via ``_validate_full``.
        """
        if self.ema is not None:
            with self.ema.apply():
                return self._validate_inner()
        return self._validate_inner()

    def _validate_inner(self) -> dict:
        """Core validation: per-step loss and per-class accuracy."""
        self.model.eval()

        total_loss = 0.0
        total_score = 0.0
        total_clock = 0.0
        total_context = 0.0
        total_pre_margin = 0.0
        total_pre_win = 0.0
        n_batches = 0

        # Per-class accuracy tracking
        n_classes = self.config.model.n_score_classes
        class_correct = torch.zeros(n_classes, device=self.device)
        class_total = torch.zeros(n_classes, device=self.device)

        # Context margin tracking
        margin_abs_errors: list[float] = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                if batch is None:
                    continue

                batch = self._to_device(batch)

                if self.use_amp:
                    with autocast("cuda", dtype=self.amp_dtype):
                        predictions = self.model(batch)
                        targets = self._extract_targets(batch)
                        losses = self.criterion(predictions, targets)
                else:
                    predictions = self.model(batch)
                    targets = self._extract_targets(batch)
                    losses = self.criterion(predictions, targets)

                total_loss += losses["total"].item()
                total_score += losses["score_loss"].item()
                total_clock += losses["clock_loss"].item()
                total_context += losses["context_loss"].item()
                total_pre_margin += losses["pre_margin_loss"].item()
                total_pre_win += losses["pre_win_loss"].item()
                n_batches += 1

                # Per-class accuracy
                score_logits = predictions["score_logits"]  # (B, T, 7)
                score_targets = targets["score_events"]  # (B, T)
                mask = targets["state_mask"]  # (B, T)

                preds = score_logits.argmax(dim=-1)  # (B, T)
                valid = mask.bool()

                for c in range(n_classes):
                    c_mask = (score_targets == c) & valid
                    class_total[c] += c_mask.sum()
                    class_correct[c] += ((preds == c) & c_mask).sum()

                # Context margin MAE (in raw points: pred * 50 vs actual * 50)
                ctx_pred = predictions["context_margin_pred"]  # (B,) normalized
                ctx_true = targets["final_margin"]  # (B,) normalized
                margin_abs_errors.extend((ctx_pred - ctx_true).abs().mul(50.0).tolist())

        n = max(n_batches, 1)
        metrics = {
            "loss": total_loss / n,
            "score_loss": total_score / n,
            "clock_loss": total_clock / n,
            "context_loss": total_context / n,
            "pre_margin_loss": total_pre_margin / n,
            "pre_win_loss": total_pre_win / n,
        }

        # Context margin MAE (raw points)
        if margin_abs_errors:
            metrics["context_margin_mae"] = sum(margin_abs_errors) / len(
                margin_abs_errors
            )

        # Per-class accuracy
        for c in range(n_classes):
            ct = class_total[c].item()
            acc = class_correct[c].item() / ct if ct > 0 else 0.0
            metrics[f"acc_{SCORE_CLASS_NAMES[c]}"] = acc

        overall_correct = class_correct.sum().item()
        overall_total = class_total.sum().item()
        metrics["score_accuracy"] = (
            overall_correct / overall_total if overall_total > 0 else 0.0
        )

        return metrics

    def _validate_full(self, loader: DataLoader, desc: str = "Evaluation") -> dict:
        """Full validation pass (can be used on any loader)."""
        if self.ema is not None:
            with self.ema.apply():
                return self._validate_full_inner(loader, desc)
        return self._validate_full_inner(loader, desc)

    def _validate_full_inner(self, loader: DataLoader, desc: str) -> dict:
        """Core full-evaluation (loss + per-class accuracy)."""
        self.model.eval()

        total_loss = 0.0
        total_score = 0.0
        total_clock = 0.0
        total_context = 0.0
        n_batches = 0

        n_classes = self.config.model.n_score_classes
        class_correct = torch.zeros(n_classes, device=self.device)
        class_total = torch.zeros(n_classes, device=self.device)

        with torch.no_grad():
            for batch in tqdm(loader, desc=desc, leave=False):
                if batch is None:
                    continue

                batch = self._to_device(batch)

                if self.use_amp:
                    with autocast("cuda", dtype=self.amp_dtype):
                        predictions = self.model(batch)
                        targets = self._extract_targets(batch)
                        losses = self.criterion(predictions, targets)
                else:
                    predictions = self.model(batch)
                    targets = self._extract_targets(batch)
                    losses = self.criterion(predictions, targets)

                total_loss += losses["total"].item()
                total_score += losses["score_loss"].item()
                total_clock += losses["clock_loss"].item()
                total_context += losses["context_loss"].item()
                n_batches += 1

                score_logits = predictions["score_logits"]
                score_targets = targets["score_events"]
                mask = targets["state_mask"]
                preds = score_logits.argmax(dim=-1)
                valid = mask.bool()

                for c in range(n_classes):
                    c_mask = (score_targets == c) & valid
                    class_total[c] += c_mask.sum()
                    class_correct[c] += ((preds == c) & c_mask).sum()

        n = max(n_batches, 1)
        metrics = {
            "loss": total_loss / n,
            "score_loss": total_score / n,
            "clock_loss": total_clock / n,
            "context_loss": total_context / n,
        }

        for c in range(n_classes):
            ct = class_total[c].item()
            acc = class_correct[c].item() / ct if ct > 0 else 0.0
            metrics[f"acc_{SCORE_CLASS_NAMES[c]}"] = acc

        overall_correct = class_correct.sum().item()
        overall_total = class_total.sum().item()
        metrics["score_accuracy"] = (
            overall_correct / overall_total if overall_total > 0 else 0.0
        )

        return metrics

    # ---- Helpers -----------------------------------------------------------

    def _extract_targets(self, batch: dict) -> dict:
        """Extract target tensors from a batch dict."""
        final_margin = batch["final_margin"]
        return {
            "score_events": batch["score_events"],
            "clock_targets": batch["clock_targets"],
            "final_margin": final_margin / 50.0,  # normalize to match margin_norm scale
            "state_mask": batch["state_mask"],
            "home_win": (final_margin > 0).float(),  # derived for pre-decoder BCE
        }

    def _to_device(self, batch: dict) -> dict:
        """Move all tensors in *batch* to ``self.device``."""
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device)
            elif isinstance(v, dict):
                out[k] = {
                    dk: dv.to(self.device) if isinstance(dv, torch.Tensor) else dv
                    for dk, dv in v.items()
                }
            else:
                out[k] = v
        return out

    # ---- Logging -----------------------------------------------------------

    def _log_epoch(
        self,
        epoch: int,
        train_metrics: dict,
        val_metrics: dict,
        epoch_time: float,
    ) -> None:
        per_class = "  ".join(
            f"{SCORE_CLASS_NAMES[c]}={val_metrics.get(f'acc_{SCORE_CLASS_NAMES[c]}', 0):.3f}"
            for c in range(self.config.model.n_score_classes)
        )
        logger.info(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_metrics['loss']:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, "
            f"score_acc={val_metrics.get('score_accuracy', 0):.3f}, "
            f"time={epoch_time:.0f}s"
        )
        ctx_margin_mae = val_metrics.get("context_margin_mae", 0)
        logger.info(
            f"  Score: {train_metrics['score_loss']:.4f} / {val_metrics['score_loss']:.4f}  "
            f"Clock: {train_metrics['clock_loss']:.4f} / {val_metrics['clock_loss']:.4f}  "
            f"Ctx: {train_metrics['context_loss']:.4f} / {val_metrics['context_loss']:.4f}  "
            f"CtxMAE: {ctx_margin_mae:.1f}pts"
        )
        logger.info(
            f"  PreMargin: {train_metrics.get('pre_margin_loss', 0):.4f} / "
            f"{val_metrics.get('pre_margin_loss', 0):.4f}  "
            f"PreWin: {train_metrics.get('pre_win_loss', 0):.4f} / "
            f"{val_metrics.get('pre_win_loss', 0):.4f}"
        )
        logger.info(f"  Per-class acc: {per_class}")

    # ---- Checkpointing -----------------------------------------------------

    def _save_checkpoint(self, filename: str) -> None:
        path = self.checkpoint_dir / filename

        # For best checkpoint with EMA, save EMA weights as the primary model
        # weights (since validation used EMA weights to select this checkpoint)
        if filename == "best.pt" and self.ema is not None:
            with self.ema.apply():
                model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        else:
            model_state = self.model.state_dict()

        checkpoint = {
            "model_state_dict": model_state,
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

    def _load_checkpoint(self, filename: str) -> None:
        path = self.checkpoint_dir / filename
        if not path.exists():
            logger.warning(f"Checkpoint not found: {path}")
            return

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except (ValueError, KeyError) as e:
            logger.warning(f"Skipping optimizer/scheduler restore: {e}")

        state = checkpoint["state"]
        self.state.epoch = state["epoch"]
        self.state.global_step = state["global_step"]
        self.state.best_metric = state["best_metric"]
        self.state.patience_counter = state["patience_counter"]

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if self.ema is not None and "ema_state_dict" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state_dict"])

        logger.info(f"Loaded checkpoint: {path} (epoch {self.state.epoch + 1})")

    def load_checkpoint(self, path: str) -> None:
        """Public interface for resuming from an arbitrary checkpoint path."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except (ValueError, KeyError) as e:
            logger.warning(f"Skipping optimizer/scheduler restore: {e}")

        state = checkpoint.get("state", {})
        self.state.epoch = state.get("epoch", 0) + 1  # resume from next epoch
        self.state.global_step = state.get("global_step", 0)
        self.state.best_metric = state.get("best_metric", float("inf"))
        self.state.patience_counter = state.get("patience_counter", 0)

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if self.ema is not None and "ema_state_dict" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state_dict"])

        logger.info(f"Resumed from {path} (will start at epoch {self.state.epoch + 1})")

    def _cleanup_checkpoints(self) -> None:
        """Remove old periodic checkpoints beyond ``keep_n``."""
        if not self.config.training.save_best_only:
            return

        keep_n = self.config.training.keep_n_checkpoints
        checkpoints = sorted(
            self.checkpoint_dir.glob("epoch_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for ckpt in checkpoints[keep_n:]:
            ckpt.unlink()
