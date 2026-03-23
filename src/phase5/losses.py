"""
Loss functions for NKE-H training.

Phase 1: Multi-task single-game losses (stat recon, next-game, DPM, archetype, VICReg)
Phase 2: Sequential Kalman losses (same heads but applied per-timestep with masking)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import NKEHConfig
from .model import vicreg_loss


def archetype_entropy_loss(archetype_weights: torch.Tensor) -> torch.Tensor:
    """
    Penalize archetype collapse by maximizing entropy of the marginal
    (batch-averaged) assignment distribution.

    This encourages all K archetypes to be used roughly equally across the
    batch, while still allowing individual players to have peaked assignments.

    Args:
        archetype_weights: (B, K) soft assignment probabilities
    Returns:
        Scalar loss. Minimizing this maximizes marginal entropy (uniform usage).
        Range: [-log(K), 0] where -log(K) is perfect uniformity.
    """
    # Marginal distribution: average assignment across batch
    marginal = archetype_weights.mean(dim=0)  # (K,)
    # Negative entropy (minimize to maximize entropy)
    log_marginal = torch.log(marginal + 1e-8)
    neg_entropy = (marginal * log_marginal).sum()
    return neg_entropy


# Stat weights: inversely proportional to stabilization rate
# High weight = fast stabilizing (minutes, usage), low weight = slow (3PM, plus_minus)
STAT_WEIGHTS = torch.tensor(
    [
        0.8,  # min (stabilizes ~10 games)
        1.0,  # pts
        0.6,  # oreb
        0.8,  # dreb
        0.9,  # ast
        0.7,  # stl
        0.7,  # blk
        0.8,  # tov
        0.5,  # pf
        0.9,  # fga
        0.9,  # fgm
        0.7,  # fg3a
        0.5,  # fg3m (stabilizes ~240 games)
        0.8,  # fta
        0.8,  # ftm
        0.3,  # plus_minus (extremely noisy)
    ],
    dtype=torch.float32,
)


def phase1_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    cfg: NKEHConfig,
) -> dict[str, torch.Tensor]:
    """
    Compute Phase 1 (single-game pre-training) losses.

    Returns dict with individual loss components and total.
    """
    device = outputs["ability"].device
    sw = STAT_WEIGHTS.to(device)

    # --- Stat reconstruction loss ---
    stat_recon = outputs["stat_recon"]
    stat_target = batch["stat_target"].to(device)
    L_recon = (sw * (stat_recon - stat_target).pow(2)).mean()

    # --- Next-game prediction loss ---
    next_pred = outputs["next_game"]
    next_target = batch["next_game_target"].to(device)
    has_next = batch["has_next"].to(device)
    if has_next.any():
        next_diff = sw * (next_pred - next_target).pow(2)
        L_next = (next_diff * has_next.unsqueeze(-1).float()).sum() / (
            has_next.sum() * stat_recon.shape[-1] + 1e-8
        )
    else:
        L_next = torch.tensor(0.0, device=device)

    # --- DPM impact prediction loss ---
    dpm_pred = outputs["dpm"]
    dpm_target = batch["dpm_target"].to(device)
    has_dpm = batch["has_dpm"].to(device)
    if has_dpm.any():
        dpm_diff = F.huber_loss(dpm_pred, dpm_target, reduction="none", delta=2.0)
        L_dpm = (dpm_diff * has_dpm.unsqueeze(-1).float()).sum() / (
            has_dpm.sum() * dpm_pred.shape[-1] + 1e-8
        )
        # Auxiliary D-DPM head loss (dedicated defense pathway)
        d_dpm_target = dpm_target[:, 1:2]  # (B, 1) — D-DPM only
        d_dpm_aux_pred = outputs["d_dpm_aux"]  # (B, 1) from defense_trunk
        d_dpm_aux_diff = F.huber_loss(
            d_dpm_aux_pred, d_dpm_target, reduction="none", delta=2.0
        )
        L_dpm_def = (d_dpm_aux_diff.squeeze(-1) * has_dpm.float()).sum() / (
            has_dpm.sum() + 1e-8
        )
    else:
        L_dpm = torch.tensor(0.0, device=device)
        L_dpm_def = torch.tensor(0.0, device=device)

    # --- Archetype consistency loss ---
    # Decoder archetype logits should agree with prior archetype weights (soft targets)
    arch_logits = outputs["archetype_logits"]
    arch_weights = outputs["archetype_weights"].detach()
    # Use KL divergence for soft target distribution
    log_probs = F.log_softmax(arch_logits, dim=-1)
    L_arch = F.kl_div(log_probs, arch_weights, reduction="batchmean")

    # --- Archetype entropy regularization ---
    # Maximize entropy of marginal assignment → encourage all archetypes to be used
    L_arch_entropy = archetype_entropy_loss(outputs["archetype_weights"])

    # --- VICReg covariance + variance loss ---
    L_cov, L_var = vicreg_loss(outputs["ability"], gamma=cfg.vicreg_gamma)

    # --- Total ---
    L_total = (
        cfg.w_reconstruction * L_recon
        + cfg.w_next_game * L_next
        + cfg.w_dpm * L_dpm
        + cfg.w_dpm_defense * L_dpm_def
        + cfg.w_archetype * L_arch
        + cfg.w_archetype_entropy * L_arch_entropy
        + cfg.w_covariance * L_cov
        + cfg.w_variance * L_var
    )

    return {
        "total": L_total,
        "recon": L_recon,
        "next_game": L_next,
        "dpm": L_dpm,
        "dpm_defense": L_dpm_def,
        "archetype": L_arch,
        "archetype_entropy": L_arch_entropy,
        "covariance": L_cov,
        "variance": L_var,
    }


def phase2_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    cfg: NKEHConfig,
) -> dict[str, torch.Tensor]:
    """
    Compute Phase 2 (sequential Kalman training) losses.
    All losses are computed per-timestep and masked.

    outputs have shape (B, T, ...) for sequence outputs.
    """
    device = outputs["ability"].device
    sw = STAT_WEIGHTS.to(device)
    mask = batch["mask"].to(device)  # (B, T)
    n_valid = mask.sum().clamp(min=1)

    # --- Stat reconstruction (per-timestep) ---
    stat_recon = outputs["stat_recon"]  # (B, T, 16)
    stat_target = batch["stat_target"].to(device)  # (B, T, 16)
    recon_diff = (sw * (stat_recon - stat_target).pow(2)).mean(dim=-1)  # (B, T)
    L_recon = (recon_diff * mask.float()).sum() / n_valid

    # --- Next-game prediction (per-timestep) ---
    next_pred = outputs["next_game"]  # (B, T, 16)
    next_target = batch["next_game_target"].to(device)  # (B, T, 16)
    has_next = batch["has_next"].to(device)  # (B, T)
    next_mask = mask & has_next
    n_next = next_mask.sum().clamp(min=1)
    next_diff = (sw * (next_pred - next_target).pow(2)).mean(dim=-1)  # (B, T)
    L_next = (next_diff * next_mask.float()).sum() / n_next

    # --- DPM impact (per-timestep) ---
    dpm_pred = outputs["dpm"]  # (B, T, 3)
    dpm_target = batch["dpm_target"].to(device)  # (B, T, 3)
    has_dpm = batch["has_dpm"].to(device)  # (B, T)
    dpm_mask = mask & has_dpm
    n_dpm = dpm_mask.sum().clamp(min=1)
    dpm_diff = F.huber_loss(dpm_pred, dpm_target, reduction="none", delta=2.0).mean(
        dim=-1
    )  # (B, T)
    L_dpm = (dpm_diff * dpm_mask.float()).sum() / n_dpm

    # Auxiliary D-DPM head loss (dedicated defense pathway)
    d_dpm_target = dpm_target[:, :, 1:2]  # (B, T, 1) — D-DPM only
    d_dpm_aux_pred = outputs["d_dpm_aux"]  # (B, T, 1) from defense_trunk
    d_dpm_aux_diff = F.huber_loss(
        d_dpm_aux_pred, d_dpm_target, reduction="none", delta=2.0
    ).squeeze(
        -1
    )  # (B, T)
    L_dpm_def = (d_dpm_aux_diff * dpm_mask.float()).sum() / n_dpm

    # --- Archetype entropy regularization ---
    L_arch_entropy = archetype_entropy_loss(outputs["archetype_weights"])

    # --- VICReg covariance + variance on final timestep ability vectors ---
    # Use the ability at each player's last valid timestep
    seq_len = batch["seq_len"].to(device)  # (B,)
    last_idx = (seq_len - 1).clamp(min=0)  # (B,)
    B = outputs["ability"].shape[0]
    final_ability = outputs["ability"][
        torch.arange(B, device=device), last_idx
    ]  # (B, d)
    L_cov, L_var = vicreg_loss(final_ability, gamma=cfg.vicreg_gamma)

    # --- Total ---
    L_total = (
        cfg.w_reconstruction_seq * L_recon
        + cfg.w_next_game_seq * L_next
        + cfg.w_dpm_seq * L_dpm
        + cfg.w_dpm_defense_seq * L_dpm_def
        + cfg.w_archetype_entropy_seq * L_arch_entropy
        + cfg.w_covariance_seq * L_cov
        + cfg.w_variance_seq * L_var
    )

    return {
        "total": L_total,
        "recon": L_recon,
        "next_game": L_next,
        "dpm": L_dpm,
        "dpm_defense": L_dpm_def,
        "archetype_entropy": L_arch_entropy,
        "covariance": L_cov,
        "variance": L_var,
    }
