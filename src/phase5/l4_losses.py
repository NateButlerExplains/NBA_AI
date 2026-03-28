"""
Loss functions for L4 Game Prediction training.

Multi-task composite loss:
  Primary (w=1.0): Gaussian NLL for spread prediction
  Auxiliary (w=0.3): BCE for win probability
  Auxiliary (w=0.3): Gaussian NLL for total points
  Consistency (w=0.1): MSE between Phi(mu/sigma) and win_prob
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .l4_config import L4Config

# Standard normal CDF via error function
_LOG_SQRT_2 = 0.5 * math.log(2.0)
_INV_SQRT_2 = 1.0 / math.sqrt(2.0)


def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF: Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))."""
    return 0.5 * (1.0 + torch.erf(x * _INV_SQRT_2))


def gaussian_nll(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Gaussian negative log-likelihood: -log N(target | mu, sigma^2).

    Args:
        mu: (B,) predicted mean
        sigma: (B,) predicted standard deviation (must be > 0)
        target: (B,) actual value
    Returns:
        scalar mean loss
    """
    # -log N(x|mu,sigma^2) = 0.5*log(2*pi) + log(sigma) + 0.5*((x-mu)/sigma)^2
    variance = sigma.pow(2)
    nll = (
        0.5 * math.log(2.0 * math.pi)
        + torch.log(sigma)
        + 0.5 * (target - mu).pow(2) / variance
    )
    return nll.mean()


def spread_loss(
    spread_mu: torch.Tensor,
    spread_sigma: torch.Tensor,
    actual_margin: torch.Tensor,
) -> torch.Tensor:
    """
    Gaussian NLL for spread prediction.

    Args:
        spread_mu: (B,) predicted margin (positive = home advantage)
        spread_sigma: (B,) predicted uncertainty
        actual_margin: (B,) actual home margin (home_score - away_score)
    Returns:
        scalar loss
    """
    return gaussian_nll(spread_mu, spread_sigma, actual_margin)


def win_loss(
    win_logit: torch.Tensor,
    actual_home_win: torch.Tensor,
) -> torch.Tensor:
    """
    Binary cross-entropy for win probability.

    Args:
        win_logit: (B,) raw logit (before sigmoid)
        actual_home_win: (B,) 1.0 if home won, 0.0 otherwise
    Returns:
        scalar loss
    """
    return F.binary_cross_entropy_with_logits(win_logit, actual_home_win)


def total_loss(
    total_mu: torch.Tensor,
    total_sigma: torch.Tensor,
    actual_total: torch.Tensor,
) -> torch.Tensor:
    """
    Gaussian NLL for total points prediction.

    Args:
        total_mu: (B,) predicted total points
        total_sigma: (B,) predicted uncertainty
        actual_total: (B,) actual total points (home_score + away_score)
    Returns:
        scalar loss
    """
    return gaussian_nll(total_mu, total_sigma, actual_total)


def consistency_loss(
    spread_mu: torch.Tensor,
    spread_sigma: torch.Tensor,
    win_prob: torch.Tensor,
) -> torch.Tensor:
    """
    Consistency regularization between Gaussian-derived and direct win probabilities.

    The Gaussian spread model implies P(home_win) = Phi(spread_mu / spread_sigma).
    This should agree with the directly predicted win_prob. We stop gradient on
    the Gaussian side so it acts as a soft target for the win head.

    Args:
        spread_mu: (B,) predicted margin
        spread_sigma: (B,) predicted uncertainty
        win_prob: (B,) directly predicted win probability (sigmoid output)
    Returns:
        scalar MSE loss
    """
    gaussian_win_prob = _normal_cdf(spread_mu / spread_sigma)
    return F.mse_loss(win_prob, gaussian_win_prob.detach())


def ats_loss(
    ats_logit: torch.Tensor,
    actual_margin: torch.Tensor,
    vegas_spread: torch.Tensor,
) -> torch.Tensor:
    """
    BCE loss for ATS (against the spread) prediction.

    Target: did home cover? i.e., actual_margin > -vegas_spread
    Excludes pushes (|ats_result| < 0.25) since those have no payout.

    Args:
        ats_logit: (B,) raw logit for P(home covers)
        actual_margin: (B,) home_score - away_score
        vegas_spread: (B,) Vegas closing spread (negative = home favored)
    Returns:
        scalar loss (0 if no valid games)
    """
    # ats_result > 0 means home covered
    ats_result = actual_margin - (-vegas_spread)  # = actual_margin + vegas_spread
    target = (ats_result > 0).float()

    # Exclude pushes
    valid = ats_result.abs() > 0.25
    if not valid.any():
        return torch.tensor(0.0, device=ats_logit.device)

    return F.binary_cross_entropy_with_logits(ats_logit[valid], target[valid])


def game_prediction_loss(
    predictions: dict[str, torch.Tensor],
    actual_margin: torch.Tensor,
    actual_home_win: torch.Tensor,
    actual_total: torch.Tensor,
    cfg: L4Config | None = None,
) -> dict[str, torch.Tensor]:
    """
    Combined L4 multi-task loss.

    Args:
        predictions: dict from GamePredictor.forward() with keys:
            spread_mu, spread_sigma, win_logit, win_prob, total_mu, total_sigma
        actual_margin: (B,) home_score - away_score
        actual_home_win: (B,) 1.0 if home won, 0.0 otherwise
        actual_total: (B,) home_score + away_score
        cfg: L4Config for loss weights (uses defaults if None)
    Returns:
        dict with individual components and total:
            spread, win, total, consistency, total_loss
    """
    if cfg is None:
        cfg = L4Config()

    L_spread = spread_loss(
        predictions["spread_mu"],
        predictions["spread_sigma"],
        actual_margin,
    )

    L_win = win_loss(
        predictions["win_logit"],
        actual_home_win,
    )

    L_total = total_loss(
        predictions["total_mu"],
        predictions["total_sigma"],
        actual_total,
    )

    L_consist = consistency_loss(
        predictions["spread_mu"],
        predictions["spread_sigma"],
        predictions["win_prob"],
    )

    L_combined = (
        cfg.w_spread * L_spread
        + cfg.w_win * L_win
        + cfg.w_total * L_total
        + cfg.w_consistency * L_consist
    )

    return {
        "total_loss": L_combined,
        "spread": L_spread,
        "win": L_win,
        "total": L_total,
        "consistency": L_consist,
    }
