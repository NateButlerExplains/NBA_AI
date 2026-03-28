"""
Loss functions for L3 Team Model training.

Phase B loss: predict team performance residual after L1+L2.
L3's loss is intentionally simple — the hard work is defining the target
(the residual after L1+L2). With right-sized capacity (~5K-20K params),
MSE with weight decay provides sufficient regularization.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .l3_config import L3Config


def team_residual_loss(
    pred_residual: torch.Tensor,
    true_residual: torch.Tensor,
) -> torch.Tensor:
    """
    Primary Phase B loss: MSE on the team performance residual.

    The residual is: actual_team_net_rtg - team_pred_from_players
    where team_pred_from_players = sum(L1 ability * proj_minutes) + L2 synergy.

    L3 learns to predict this residual from coaching/system/org features.

    Args:
        pred_residual: (B,) predicted residual from L3
        true_residual: (B,) actual residual (actual - L1_pred - L2_pred)
    Returns:
        scalar MSE loss
    """
    return F.mse_loss(pred_residual, true_residual)


def l3_total_loss(
    pred_residual: torch.Tensor,
    true_residual: torch.Tensor,
    cfg: L3Config | None = None,
) -> dict[str, torch.Tensor]:
    """
    Combined L3 loss (Phase B).

    Currently just weighted MSE on the team residual. The structure supports
    adding auxiliary losses later (e.g., L1-decorrelation regularizer to prevent
    L3 from recapturing player-level signal).

    Args:
        pred_residual: (B,) predicted residual from L3
        true_residual: (B,) actual residual
        cfg: L3Config for loss weights
    Returns:
        dict with individual components and total
    """
    if cfg is None:
        cfg = L3Config()

    L_residual = team_residual_loss(pred_residual, true_residual)

    L_total = cfg.w_team_residual * L_residual

    return {
        "total": L_total,
        "residual": L_residual,
    }
