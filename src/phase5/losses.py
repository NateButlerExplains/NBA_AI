"""
Loss functions for NKE-H training.

Phase 1: Multi-task single-game losses (stat recon, next-game, DPM, archetype, decorrelation)
Phase 2: Sequential Kalman losses (same heads but applied per-timestep with masking)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import NKEHConfig
from .model import decorrelation_loss

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
    else:
        L_dpm = torch.tensor(0.0, device=device)

    # --- Archetype consistency loss ---
    # Decoder archetype logits should agree with prior archetype weights (soft targets)
    arch_logits = outputs["archetype_logits"]
    arch_weights = outputs["archetype_weights"].detach()
    # Use KL divergence for soft target distribution
    log_probs = F.log_softmax(arch_logits, dim=-1)
    L_arch = F.kl_div(log_probs, arch_weights, reduction="batchmean")

    # --- Decorrelation loss ---
    L_decorr = decorrelation_loss(outputs["ability"])

    # --- Total ---
    L_total = (
        cfg.w_reconstruction * L_recon
        + cfg.w_next_game * L_next
        + cfg.w_dpm * L_dpm
        + cfg.w_archetype * L_arch
        + cfg.w_decorrelation * L_decorr
    )

    return {
        "total": L_total,
        "recon": L_recon,
        "next_game": L_next,
        "dpm": L_dpm,
        "archetype": L_arch,
        "decorrelation": L_decorr,
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

    # --- Decorrelation on final timestep ability vectors ---
    # Use the ability at each player's last valid timestep
    seq_len = batch["seq_len"].to(device)  # (B,)
    last_idx = (seq_len - 1).clamp(min=0)  # (B,)
    B = outputs["ability"].shape[0]
    final_ability = outputs["ability"][
        torch.arange(B, device=device), last_idx
    ]  # (B, d)
    L_decorr = decorrelation_loss(final_ability)

    # --- Total ---
    L_total = (
        cfg.w_reconstruction_seq * L_recon
        + cfg.w_next_game_seq * L_next
        + cfg.w_dpm_seq * L_dpm
        + cfg.w_decorrelation * L_decorr
    )

    return {
        "total": L_total,
        "recon": L_recon,
        "next_game": L_next,
        "dpm": L_dpm,
        "decorrelation": L_decorr,
    }
