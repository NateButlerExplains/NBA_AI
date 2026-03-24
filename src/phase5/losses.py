"""
Loss functions for NKE-H training.

Phase 1: Multi-task single-game losses (stat recon, next-game, DPM, RAPM, archetype, VICReg)
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

    # --- RAPM impact prediction loss ---
    rapm_pred = outputs["rapm"]
    rapm_target = batch["rapm_target"].to(device)
    has_rapm = batch["has_rapm"].to(device)
    if has_rapm.any():
        rapm_diff = F.huber_loss(rapm_pred, rapm_target, reduction="none", delta=2.0)
        L_rapm = (rapm_diff * has_rapm.unsqueeze(-1).float()).sum() / (
            has_rapm.sum() * rapm_pred.shape[-1] + 1e-8
        )
    else:
        L_rapm = torch.tensor(0.0, device=device)

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
        + cfg.w_rapm * L_rapm
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
        "rapm": L_rapm,
        "dpm_defense": L_dpm_def,
        "archetype": L_arch,
        "archetype_entropy": L_arch_entropy,
        "covariance": L_cov,
        "variance": L_var,
    }


def trade_consistency_loss(
    abilities: torch.Tensor,
    trade_mask: torch.Tensor,
    seq_mask: torch.Tensor,
    window: int = 5,
) -> torch.Tensor:
    """
    Encourage ability vectors to be similar across trade boundaries.

    At each trade (first game on new team at position t), compare the
    ability at t-1 (last game on old team) with the ability at
    t + window - 1 (after settling on new team). Loss = 1 - cosine_sim.

    Args:
        abilities: (B, T, d) ability vectors from Kalman filter
        trade_mask: (B, T) True at first game on a new team
        seq_mask: (B, T) True for valid (non-padded) timesteps
        window: number of games to allow for settling (default 5)

    Returns:
        Scalar loss (mean of 1 - cosine_sim), or 0.0 if no valid trades.
    """
    B, T, d = abilities.shape
    device = abilities.device

    # trade_mask[:, 0] should always be False (need t >= 1), but enforce it
    # We need: t >= 1, t + window - 1 < T, and both endpoints valid in seq_mask
    if T < window + 1:
        return torch.tensor(0.0, device=device)

    # Positions where a trade can be valid: t in [1, T - window]
    # Build a candidate mask over these positions
    # trade_mask[:, t] is True at trade boundaries
    valid_t_end = T - window + 1  # exclusive upper bound for t
    if valid_t_end <= 1:
        return torch.tensor(0.0, device=device)

    # Slice trade_mask to candidate range [1, valid_t_end)
    candidate_trades = trade_mask[:, 1:valid_t_end]  # (B, valid_t_end - 1)

    # Check seq_mask validity at t-1 and t + window - 1
    # For t in [1, valid_t_end): t-1 in [0, valid_t_end - 1), t+window-1 in [window, valid_t_end + window - 1)
    pre_trade_valid = seq_mask[:, 0 : valid_t_end - 1]  # (B, valid_t_end - 1)
    post_settle_indices = (
        torch.arange(1, valid_t_end, device=device) + window - 1
    )  # (valid_t_end - 1,)
    post_trade_valid = seq_mask[:, post_settle_indices]  # (B, valid_t_end - 1)

    # Combined validity: trade exists AND both endpoints are valid
    valid = (
        candidate_trades & pre_trade_valid & post_trade_valid
    )  # (B, valid_t_end - 1)

    n_valid = valid.sum()
    if n_valid == 0:
        return torch.tensor(0.0, device=device)

    # Gather ability vectors at pre-trade (t-1) and post-settle (t + window - 1)
    pre_indices = torch.arange(
        0, valid_t_end - 1, device=device
    )  # t - 1 for t in [1, valid_t_end)
    post_indices = post_settle_indices  # t + window - 1

    # Expand indices for all batches: (B, valid_t_end - 1)
    pre_abilities = abilities[:, pre_indices]  # (B, valid_t_end - 1, d)
    post_abilities = abilities[:, post_indices]  # (B, valid_t_end - 1, d)

    # Cosine similarity at each candidate position
    cos_sim = F.cosine_similarity(
        pre_abilities, post_abilities, dim=-1
    )  # (B, valid_t_end - 1)

    # Masked mean of (1 - cos_sim)
    loss_per_pos = (1.0 - cos_sim) * valid.float()  # (B, valid_t_end - 1)
    return loss_per_pos.sum() / n_valid


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

    # --- RAPM impact (per-timestep) ---
    rapm_pred = outputs["rapm"]  # (B, T, 2)
    rapm_target = batch["rapm_target"].to(device)  # (B, T, 2)
    has_rapm = batch["has_rapm"].to(device)  # (B, T)
    rapm_mask = mask & has_rapm
    n_rapm = rapm_mask.sum().clamp(min=1)
    rapm_diff = F.huber_loss(rapm_pred, rapm_target, reduction="none", delta=2.0).mean(
        dim=-1
    )  # (B, T)
    L_rapm = (rapm_diff * rapm_mask.float()).sum() / n_rapm

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

    # --- Trade consistency loss ---
    trade_mask = batch.get("trade_mask")
    if trade_mask is not None:
        trade_mask = trade_mask.to(device)
        L_trade = trade_consistency_loss(outputs["ability"], trade_mask, mask)
    else:
        L_trade = torch.tensor(0.0, device=device)

    # --- Total ---
    L_total = (
        cfg.w_reconstruction_seq * L_recon
        + cfg.w_next_game_seq * L_next
        + cfg.w_dpm_seq * L_dpm
        + cfg.w_rapm_seq * L_rapm
        + cfg.w_dpm_defense_seq * L_dpm_def
        + cfg.w_archetype_entropy_seq * L_arch_entropy
        + cfg.w_covariance_seq * L_cov
        + cfg.w_variance_seq * L_var
        + cfg.w_trade_consistency * L_trade
    )

    return {
        "total": L_total,
        "recon": L_recon,
        "next_game": L_next,
        "dpm": L_dpm,
        "rapm": L_rapm,
        "dpm_defense": L_dpm_def,
        "archetype_entropy": L_arch_entropy,
        "covariance": L_cov,
        "variance": L_var,
        "trade_consistency": L_trade,
    }
