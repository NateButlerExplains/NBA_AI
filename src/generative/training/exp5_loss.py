"""Multi-task loss for Exp 5: full-context generative model.

L = score_weight   * CE(score_logits, score_events)           # CORE: all valid positions
  + clock_weight   * MSE(clock_pred, clock_target)             # CORE: all valid positions
  + outcome_weight * GaussianNLL(mu, sigma, final_margin)      # GUIDANCE: weighted by game_progress
  + pre_margin_weight * MSE(pre_margin, final_margin)          # AUXILIARY
  + pre_win_weight    * BCE(pre_win, home_win)                 # AUXILIARY
  + context_weight    * MSE(ctx_margin, final_margin)          # AUXILIARY

Key differences from GenerativeLoss:
1. Outcome loss (NEW): Gaussian NLL at every sequence position, with target =
   final_margin (same for all positions). Position-weighted by game_progress
   (linear ramp) so late-game positions get stronger signal.
2. Score CE uses 6 classes (no game_end — termination is deterministic).
3. All losses at all valid positions (no prefix masking).
"""

import math
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Exp5Loss(nn.Module):
    """Multi-task loss with endpoint guidance for Exp 5."""

    def __init__(
        self,
        score_weight: float = 1.0,
        clock_weight: float = 0.3,
        outcome_weight: float = 0.2,
        pre_margin_weight: float = 0.5,
        pre_win_weight: float = 0.3,
        context_weight: float = 0.3,
        class_weights: torch.Tensor | None = None,
        use_clock_delta: bool = False,
    ) -> None:
        super().__init__()
        self.score_weight = score_weight
        self.clock_weight = clock_weight
        self.outcome_weight = outcome_weight
        self.pre_margin_weight = pre_margin_weight
        self.pre_win_weight = pre_win_weight
        self.context_weight = context_weight

        # ignore_index=6 so game_end sentinel targets (class 6) are ignored in 6-class mode
        n_classes = len(class_weights) if class_weights is not None else 6
        ignore_idx = (
            n_classes  # targets == n_classes will be ignored (game_end sentinel)
        )
        self.score_loss = nn.CrossEntropyLoss(
            weight=class_weights, reduction="none", ignore_index=ignore_idx
        )
        self.clock_loss = nn.MSELoss(reduction="none")
        self.context_loss = nn.MSELoss()
        self.pre_margin_loss_fn = nn.MSELoss()
        self.pre_win_loss_fn = nn.BCEWithLogitsLoss()
        self.use_clock_delta = use_clock_delta

        self._log2pi = math.log(2.0 * math.pi)

    def forward(self, predictions: dict, targets: dict) -> dict:
        """Compute multi-task loss with endpoint guidance.

        Args:
            predictions: dict with keys:
                score_logits          (B, T, 6)
                clock_preds           (B, T)
                context_margin_pred   (B,)
                pre_margin_pred       (B,)
                pre_win_pred          (B,)
                outcome_mu            (B, T)
                outcome_sigma         (B, T)
            targets: dict with keys:
                score_events    (B, T) int64 class indices 0-5
                clock_targets   (B, T) float32 normalized clock/progress
                final_margin    (B,) float32
                state_mask      (B, T) bool — True for valid positions
                home_win        (B,) float32
                game_progress   (B, T) float32 — for outcome position weighting

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
        n_valid = mask.sum().clamp(min=1)
        mask_float = mask.float()

        # --- Score cross-entropy (masked, 6-class) ---
        score_ce = self.score_loss(
            score_logits.reshape(B * T, C),
            score_targets.reshape(B * T),
        ).reshape(B, T)
        score_ce = score_ce * mask_float
        score_loss = score_ce.sum() / n_valid

        # --- Clock MSE (masked) ---
        if self.use_clock_delta and "game_progress" in targets:
            # Delta mode: target is the positive increment to next event
            delta_targets = (clock_targets - targets["game_progress"]).clamp(min=0)
            clock_mse = self.clock_loss(clock_preds, delta_targets)
        else:
            clock_mse = self.clock_loss(clock_preds, clock_targets)
        clock_mse = clock_mse * mask_float
        clock_loss = clock_mse.sum() / n_valid

        # --- Outcome Gaussian NLL (endpoint guidance, position-weighted) ---
        outcome_loss = torch.tensor(0.0, device=score_logits.device)
        if "outcome_mu" in predictions and "outcome_sigma" in predictions:
            outcome_mu = predictions["outcome_mu"]  # (B, T)
            outcome_sigma = predictions["outcome_sigma"]  # (B, T)
            game_progress = targets["game_progress"]  # (B, T)

            # Expand final_margin to match sequence dim: (B,) -> (B, T)
            target_margin = final_margin.unsqueeze(1).expand_as(outcome_mu)

            # Gaussian NLL per position:
            # NLL = 0.5 * log(2*pi*sigma^2) + (target - mu)^2 / (2*sigma^2)
            variance = outcome_sigma.square()
            nll = 0.5 * (
                self._log2pi
                + variance.log()
                + (target_margin - outcome_mu).square() / variance
            )

            # Position-weight by game_progress (linear ramp)
            # Late positions (high progress) get stronger gradient
            weighted_nll = nll * game_progress

            # Apply padding mask
            weighted_nll = weighted_nll * mask_float

            # Average over valid positions
            outcome_loss = weighted_nll.sum() / n_valid

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
            + self.outcome_weight * outcome_loss
            + self.pre_margin_weight * pre_margin_loss
            + self.pre_win_weight * pre_win_loss
            + self.context_weight * context_loss
        )

        return {
            "score_loss": score_loss,
            "clock_loss": clock_loss,
            "outcome_loss": outcome_loss,
            "context_loss": context_loss,
            "pre_margin_loss": pre_margin_loss,
            "pre_win_loss": pre_win_loss,
            "total": total,
        }
