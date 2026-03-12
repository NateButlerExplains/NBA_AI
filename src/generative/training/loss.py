"""Multi-task loss for generative model.

L = w_score * CE(score_logits, score_targets)          [class-weighted, masked]
  + w_clock * MSE(clock_pred, clock_target)              [masked]
  + w_context * MSE(context_margin_pred, final_margin)   [full batch]
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class GenerativeLoss(nn.Module):
    """Multi-task loss combining score classification, clock regression, and context margin.

    The score CE and clock MSE are masked by ``state_mask`` to ignore padding
    positions.  The context margin MSE is computed over the full batch (it is
    a single prediction per sample, not per-position).
    """

    def __init__(
        self,
        score_weight: float = 1.0,
        clock_weight: float = 0.3,
        context_weight: float = 1.0,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.score_weight = score_weight
        self.clock_weight = clock_weight
        self.context_weight = context_weight

        # Score CE: per-element (masked later), with optional class weights
        self.score_loss = nn.CrossEntropyLoss(
            weight=class_weights, reduction="none"
        )
        # Clock MSE: per-element (masked later)
        self.clock_loss = nn.MSELoss(reduction="none")
        # Context margin MSE: full-batch mean
        self.context_loss = nn.MSELoss()

    def forward(self, predictions: dict, targets: dict) -> dict:
        """Compute multi-task loss.

        Args:
            predictions: dict with keys
                ``score_logits``          (B, T, 7)
                ``clock_preds``           (B, T)
                ``context_margin_pred``   (B,)
            targets: dict with keys
                ``score_events``   (B, T) int64 class indices 0-6
                ``clock_targets``  (B, T) float32 normalised clock
                ``final_margin``   (B,) float32
                ``state_mask``     (B, T) bool — True for valid positions

        Returns:
            dict with ``score_loss``, ``clock_loss``, ``context_loss``, ``total``
            (all scalar tensors).
        """
        score_logits = predictions["score_logits"]       # (B, T, 7)
        clock_preds = predictions["clock_preds"]         # (B, T)
        context_margin = predictions["context_margin_pred"]  # (B,)

        score_targets = targets["score_events"]          # (B, T)
        clock_targets = targets["clock_targets"]         # (B, T)
        final_margin = targets["final_margin"]           # (B,)
        mask = targets["state_mask"]                     # (B, T)

        B, T, C = score_logits.shape

        # --- Score cross-entropy (masked) ---
        # CrossEntropyLoss expects (N, C) and (N,)
        score_ce = self.score_loss(
            score_logits.reshape(B * T, C),
            score_targets.reshape(B * T),
        )  # (B*T,)
        score_ce = score_ce.reshape(B, T)
        # Mask: zero out padded positions
        score_ce = score_ce * mask.float()
        n_valid = mask.sum().clamp(min=1)
        score_loss = score_ce.sum() / n_valid

        # --- Clock MSE (masked) ---
        clock_mse = self.clock_loss(clock_preds, clock_targets)  # (B, T)
        clock_mse = clock_mse * mask.float()
        clock_loss = clock_mse.sum() / n_valid

        # --- Context margin MSE (full batch) ---
        context_loss = self.context_loss(context_margin, final_margin)

        # --- Combined loss ---
        total = (
            self.score_weight * score_loss
            + self.clock_weight * clock_loss
            + self.context_weight * context_loss
        )

        return {
            "score_loss": score_loss,
            "clock_loss": clock_loss,
            "context_loss": context_loss,
            "total": total,
        }
