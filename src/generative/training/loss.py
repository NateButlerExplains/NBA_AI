"""Multi-task loss for generative model (Exp 2: adaLN-Zero).

L = w_score * CE(score_logits, score_targets)          [class-weighted, masked]
  + w_clock * MSE(clock_pred, clock_target)              [masked]
  + w_context * MSE(context_margin_pred, final_margin)   [post-decoder, full batch]
  + w_pre_margin * MSE(pre_margin_pred, final_margin)    [pre-decoder, full batch]
  + w_pre_win * BCE(pre_win_pred, home_win)              [pre-decoder, full batch]
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class GenerativeLoss(nn.Module):
    """Multi-task loss with pre-decoder auxiliary losses for adaLN-Zero."""

    def __init__(
        self,
        score_weight: float = 1.0,
        clock_weight: float = 0.3,
        context_weight: float = 1.0,
        pre_margin_weight: float = 1.0,
        pre_win_weight: float = 0.5,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.score_weight = score_weight
        self.clock_weight = clock_weight
        self.context_weight = context_weight
        self.pre_margin_weight = pre_margin_weight
        self.pre_win_weight = pre_win_weight

        self.score_loss = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
        self.clock_loss = nn.MSELoss(reduction="none")
        self.context_loss = nn.MSELoss()
        self.pre_margin_loss_fn = nn.MSELoss()
        self.pre_win_loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, predictions: dict, targets: dict) -> dict:
        """Compute multi-task loss.

        Args:
            predictions: dict with keys
                ``score_logits``          (B, T, 7)
                ``clock_preds``           (B, T)
                ``context_margin_pred``   (B,)
                ``pre_margin_pred``       (B,) — optional
                ``pre_win_pred``          (B,) — optional
            targets: dict with keys
                ``score_events``   (B, T) int64 class indices 0-6
                ``clock_targets``  (B, T) float32 normalised clock
                ``final_margin``   (B,) float32
                ``state_mask``     (B, T) bool — True for valid positions
                ``home_win``       (B,) float32 — optional, for pre-decoder win loss

        Returns:
            dict with per-task losses and ``total`` (all scalar tensors).
        """
        score_logits = predictions["score_logits"]
        clock_preds = predictions["clock_preds"]
        context_margin = predictions["context_margin_pred"]

        score_targets = targets["score_events"]
        clock_targets = targets["clock_targets"]
        final_margin = targets["final_margin"]
        mask = targets["state_mask"]

        B, T, C = score_logits.shape

        # --- Score cross-entropy (masked) ---
        score_ce = self.score_loss(
            score_logits.reshape(B * T, C),
            score_targets.reshape(B * T),
        ).reshape(B, T)
        score_ce = score_ce * mask.float()
        n_valid = mask.sum().clamp(min=1)
        score_loss = score_ce.sum() / n_valid

        # --- Clock MSE (masked) ---
        clock_mse = self.clock_loss(clock_preds, clock_targets)
        clock_mse = clock_mse * mask.float()
        clock_loss = clock_mse.sum() / n_valid

        # --- Post-decoder context margin MSE ---
        context_loss = self.context_loss(context_margin, final_margin)

        # --- Pre-decoder auxiliary losses ---
        pre_margin_loss = torch.tensor(0.0, device=score_logits.device)
        pre_win_loss = torch.tensor(0.0, device=score_logits.device)

        if "pre_margin_pred" in predictions:
            pre_margin_loss = self.pre_margin_loss_fn(
                predictions["pre_margin_pred"], final_margin
            )

        if "pre_win_pred" in predictions and "home_win" in targets:
            pre_win_loss = self.pre_win_loss_fn(
                predictions["pre_win_pred"], targets["home_win"]
            )

        # --- Combined loss ---
        total = (
            self.score_weight * score_loss
            + self.clock_weight * clock_loss
            + self.context_weight * context_loss
            + self.pre_margin_weight * pre_margin_loss
            + self.pre_win_weight * pre_win_loss
        )

        return {
            "score_loss": score_loss,
            "clock_loss": clock_loss,
            "context_loss": context_loss,
            "pre_margin_loss": pre_margin_loss,
            "pre_win_loss": pre_win_loss,
            "total": total,
        }
