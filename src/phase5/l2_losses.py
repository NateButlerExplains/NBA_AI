"""
Loss functions for L2 Player Synergy Network training.

Hierarchical multi-resolution loss:
  Primary (w=1.0): 2-man WOWY residual prediction
  Secondary (w=0.3): 5-man lineup net rating prediction
  Auxiliary (w=0.1): Additive consistency regularization
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .l2_config import L2Config


def pairwise_loss(
    pred_synergy: torch.Tensor,
    target_wowy_residual: torch.Tensor,
    shared_possessions: torch.Tensor,
    delta: float = 8.0,
) -> torch.Tensor:
    """
    Primary loss: predict 2-man WOWY residual.

    WOWY residual = net_rtg(both_on) - net_rtg(A_only) - net_rtg(B_only) + net_rtg(neither)
    This isolates the pure interaction effect controlling for individual quality.

    Args:
        pred_synergy: (B,) predicted pairwise synergy score
        target_wowy_residual: (B,) observed WOWY residual (per 100 possessions)
        shared_possessions: (B,) number of shared possessions (for weighting)
        delta: Huber loss delta
    Returns:
        scalar loss
    """
    # Weight by sqrt(possessions) — more data = more trust
    weights = shared_possessions.sqrt().clamp(min=1.0)
    weights = weights / weights.mean()  # normalize to mean 1

    raw_loss = F.huber_loss(
        pred_synergy, target_wowy_residual, reduction="none", delta=delta
    )
    return (raw_loss * weights).mean()


def lineup_loss(
    pred_lineup_net_rtg: torch.Tensor,
    target_lineup_net_rtg: torch.Tensor,
    lineup_possessions: torch.Tensor,
    min_possessions: float = 50.0,
    delta: float = 12.0,
) -> torch.Tensor:
    """
    Secondary loss: predict 5-man lineup net rating.

    Args:
        pred_lineup_net_rtg: (B,) predicted net rating for lineup
        target_lineup_net_rtg: (B,) observed net rating per 100 possessions
        lineup_possessions: (B,) number of possessions for this lineup
        min_possessions: minimum possessions to include (filter noise)
        delta: Huber loss delta (wider than pairwise — noisier targets)
    Returns:
        scalar loss
    """
    # Filter low-possession lineups
    valid = lineup_possessions >= min_possessions
    if not valid.any():
        return torch.tensor(0.0, device=pred_lineup_net_rtg.device)

    pred = pred_lineup_net_rtg[valid]
    target = target_lineup_net_rtg[valid]
    poss = lineup_possessions[valid]

    weights = poss.sqrt().clamp(min=1.0)
    weights = weights / weights.mean()

    raw_loss = F.huber_loss(pred, target, reduction="none", delta=delta)
    return (raw_loss * weights).mean()


def consistency_loss(
    pred_lineup_net_rtg: torch.Tensor,
    sum_individual_abilities: torch.Tensor,
    sum_pairwise_synergies: torch.Tensor,
) -> torch.Tensor:
    """
    Auxiliary loss: regularize the higher-order residual to be small.

    The 5-man prediction should approximately equal:
        sum(individual_abilities) + sum(pairwise_synergies)
    Any remaining gap is the "higher-order residual" from GATv2.
    We penalize this to enforce an additive prior.

    Args:
        pred_lineup_net_rtg: (B,) full model prediction
        sum_individual_abilities: (B,) sum of individual player contributions
        sum_pairwise_synergies: (B,) sum of all C(5,2) pairwise synergy scores
    Returns:
        scalar loss (MSE of the higher-order residual)
    """
    higher_order = pred_lineup_net_rtg - (
        sum_individual_abilities + sum_pairwise_synergies
    )
    return higher_order.pow(2).mean()


def fm_regularization_loss(model: torch.nn.Module) -> torch.Tensor:
    """
    L2 regularization on FM residual embeddings.
    Encourages residuals to stay near zero (MLP-dominated) unless data pushes them away.
    """
    return model.fm_synergy.residual.weight.pow(2).mean()


def l2_total_loss(
    pred_pair_synergy: torch.Tensor,
    target_wowy: torch.Tensor,
    pair_possessions: torch.Tensor,
    pred_lineup: torch.Tensor | None = None,
    target_lineup: torch.Tensor | None = None,
    lineup_poss: torch.Tensor | None = None,
    sum_individual: torch.Tensor | None = None,
    sum_pairwise: torch.Tensor | None = None,
    model: torch.nn.Module | None = None,
    cfg: L2Config | None = None,
) -> dict[str, torch.Tensor]:
    """
    Combined L2 loss with hierarchical weighting.

    Returns dict with individual components and total.
    """
    if cfg is None:
        cfg = L2Config()

    device = pred_pair_synergy.device

    # Primary: 2-man WOWY
    L_2man = pairwise_loss(pred_pair_synergy, target_wowy, pair_possessions)

    # Secondary: 5-man lineup (if available)
    if (
        pred_lineup is not None
        and target_lineup is not None
        and lineup_poss is not None
    ):
        L_5man = lineup_loss(pred_lineup, target_lineup, lineup_poss)
    else:
        L_5man = torch.tensor(0.0, device=device)

    # Consistency (if lineup data available)
    if (
        pred_lineup is not None
        and sum_individual is not None
        and sum_pairwise is not None
    ):
        L_consist = consistency_loss(pred_lineup, sum_individual, sum_pairwise)
    else:
        L_consist = torch.tensor(0.0, device=device)

    # FM regularization
    if model is not None:
        L_fm_reg = fm_regularization_loss(model)
    else:
        L_fm_reg = torch.tensor(0.0, device=device)

    L_total = (
        cfg.w_2man * L_2man
        + cfg.w_5man * L_5man
        + cfg.w_consistency * L_consist
        + cfg.w_fm_l2 * L_fm_reg
    )

    return {
        "total": L_total,
        "pairwise": L_2man,
        "lineup": L_5man,
        "consistency": L_consist,
        "fm_reg": L_fm_reg,
    }
