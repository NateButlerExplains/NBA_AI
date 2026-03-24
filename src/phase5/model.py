"""
NKE-H: Neural Kalman Encoder with Hierarchical Prior.

Architecture:
  1. Prior Network: physical profile → initial state (mu_0, P_0)
  2. Archetype Network: K=10 soft mixture of learned prototypes
  3. Game Encoder: single-game stats + context → observation (mu_obs, sigma_obs)
  4. Kalman Update: sequential state estimation with aging drift
  5. Decoder: multi-head supervision (stat recon, next-game, DPM, archetype)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NKEHConfig


class PopulationPrior(nn.Module):
    """
    Learns the population-level prior mean for a 'generic NBA player'
    conditioned on game context. Produces mu_pop only.

    Note: log_sigma_head was removed (2026-03-24) because sigma_pop became
    dead code when P_0 was changed to a learned parameter. Old checkpoints
    that contain population_prior.log_sigma_head.{weight,bias} will need
    strict=False when loading.
    """

    def __init__(self, cfg: NKEHConfig):
        super().__init__()
        d = cfg.d_ability
        h = cfg.prior_hidden

        self.net = nn.Sequential(
            nn.Linear(cfg.n_context, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.GELU(),
        )
        self.mu_head = nn.Linear(h, d)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: (B, n_context) game context features
        Returns:
            mu_pop: (B, d_ability)
        """
        h = self.net(context)
        return self.mu_head(h)


class ArchetypeNetwork(nn.Module):
    """
    Learns K archetype prototypes and produces soft assignment + archetype mean.

    Takes player physical profile and produces:
    - Soft archetype weights (K probabilities)
    - Archetype-weighted mu

    Note: prototype_log_sigma was removed (2026-03-24) because sigma_arch
    became dead code when P_0 was changed to a learned parameter. Old
    checkpoints that contain archetype_network.prototype_log_sigma will need
    strict=False when loading.
    """

    def __init__(self, cfg: NKEHConfig):
        super().__init__()
        d = cfg.d_ability
        K = cfg.n_archetypes
        h = cfg.archetype_hidden

        # Input: population prior sample (d) + physical profile (n_profile)
        input_dim = d + cfg.n_profile

        self.net = nn.Sequential(
            nn.Linear(input_dim, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.GELU(),
        )

        # Soft archetype assignment
        self.assignment_head = nn.Linear(h, K)

        # Per-archetype parameters: K prototypes of dimension d
        self.prototype_mu = nn.Parameter(torch.randn(K, d) * 0.1)

        # Learnable temperature for softmax (prevents premature sharpening)
        self.use_learnable_temp = cfg.archetype_temperature_learnable
        self.min_temperature = cfg.archetype_min_temperature
        if self.use_learnable_temp:
            # Initialize at log(1.0) so initial temperature = softplus(0) + min ≈ 0.79
            self.log_temperature = nn.Parameter(torch.tensor(0.0))

    def initialize_from_centroids(self, centroids):
        """
        Initialize prototype_mu from pre-computed k-means centroids.
        Projects from centroid_dim to d_ability via a learned-style projection.

        If centroid_dim > d_ability, uses PCA (top-d components).
        If centroid_dim < d_ability, pads with small random noise.
        The SVD rank is min(K, C), so if K < d_ability we must pad.

        Args:
            centroids: (K, C) numpy array of centroid vectors
        """
        import numpy as np

        if isinstance(centroids, np.ndarray):
            centroids_t = torch.tensor(centroids, dtype=torch.float32)
        else:
            centroids_t = centroids.float()

        K, C = centroids_t.shape
        d = self.prototype_mu.shape[1]

        if C == d:
            self.prototype_mu.data = centroids_t
        else:
            # Project via SVD of the centroid matrix
            U, S, Vt = torch.linalg.svd(centroids_t, full_matrices=False)
            # Effective rank is min(K, C); project centroids into this space
            rank = min(K, C)
            projected = U[:, :rank] * S[:rank].unsqueeze(0)  # (K, rank)

            if rank >= d:
                # More components than needed: take first d (PCA)
                projected = projected[:, :d]
            else:
                # Fewer components than d: pad remaining dims with small noise
                pad = torch.randn(K, d - rank) * 0.05
                projected = torch.cat([projected, pad], dim=-1)

            # Scale to reasonable range
            projected = projected / (projected.std() + 1e-8) * 0.3
            self.prototype_mu.data = projected

    @property
    def temperature(self) -> torch.Tensor:
        """Current softmax temperature."""
        if self.use_learnable_temp:
            return F.softplus(self.log_temperature) + self.min_temperature
        return torch.tensor(1.0, device=self.prototype_mu.device)

    def forward(
        self,
        mu_pop: torch.Tensor,
        profile: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mu_pop: (B, d_ability) population prior mean
            profile: (B, n_profile) static player features
        Returns:
            mu_arch: (B, d_ability) archetype-weighted mean
            archetype_weights: (B, K) soft assignment probabilities
        """
        x = torch.cat([mu_pop, profile], dim=-1)
        h = self.net(x)

        # Soft assignment with temperature scaling
        logits = self.assignment_head(h)
        weights = F.softmax(logits / self.temperature, dim=-1)  # (B, K)

        # Weighted combination of prototypes
        # mu_arch = sum_k weights_k * prototype_mu_k
        mu_arch = torch.einsum("bk,kd->bd", weights, self.prototype_mu)

        return mu_arch, weights


class GameEncoder(nn.Module):
    """
    Encodes a single game's stats into an observation for the Kalman update.
    Produces mu_obs and log_sigma_obs.
    """

    def __init__(self, cfg: NKEHConfig):
        super().__init__()
        d = cfg.d_ability
        h = cfg.encoder_hidden
        # Input: box stats + pbp stats + context + current state
        input_dim = cfg.n_box_stats + cfg.n_pbp_stats + cfg.n_context + d

        layers = []
        prev_dim = input_dim
        for i in range(cfg.encoder_layers):
            layers.extend(
                [
                    nn.Linear(prev_dim, h),
                    nn.LayerNorm(h),
                    nn.GELU(),
                    nn.Dropout(cfg.encoder_dropout),
                ]
            )
            prev_dim = h

        self.net = nn.Sequential(*layers)
        self.mu_head = nn.Linear(h, d)
        self.log_sigma_head = nn.Linear(h, d)

        # Initialize log_sigma to produce moderate observation noise
        nn.init.constant_(self.log_sigma_head.bias, 0.0)

    def forward(
        self,
        box_stats: torch.Tensor,
        pbp_stats: torch.Tensor,
        context: torch.Tensor,
        current_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            box_stats: (B, n_box_stats)
            pbp_stats: (B, n_pbp_stats)
            context: (B, n_context)
            current_state: (B, d_ability) current Kalman state
        Returns:
            mu_obs: (B, d_ability)
            log_sigma_obs: (B, d_ability)
        """
        x = torch.cat([box_stats, pbp_stats, context, current_state], dim=-1)
        h = self.net(x)
        return self.mu_head(h), self.log_sigma_head(h)


class AgingModel(nn.Module):
    """
    Learns age-dependent drift for the Kalman state transition.
    F(age) = I + diag(drift), where drift is bounded by max_drift.
    """

    def __init__(self, cfg: NKEHConfig):
        super().__init__()
        d = cfg.d_ability
        h = cfg.aging_hidden
        self.max_drift = cfg.aging_max_drift

        self.net = nn.Sequential(
            nn.Linear(1, h),
            nn.GELU(),
            nn.Linear(h, d),
        )

    def forward(self, age: torch.Tensor) -> torch.Tensor:
        """
        Args:
            age: (B, 1) player age in years
        Returns:
            drift: (B, d_ability) per-dimension drift values in (-max_drift, max_drift)
        """
        return torch.tanh(self.net(age)) * self.max_drift


class Decoder(nn.Module):
    """
    Multi-head decoder for training supervision.

    Heads:
    1. Stat reconstruction: predict current game's box stats
    2. Next-game prediction: predict next game's box stats
    3. DPM prediction: predict o_dpm, d_dpm, dpm (shared trunk)
    4. RAPM prediction: predict off_rapm, def_rapm (season-level impact)
    5. D-DPM auxiliary head: separate deeper pathway for defensive impact
    6. Archetype classification: predict archetype weights
    """

    def __init__(self, cfg: NKEHConfig):
        super().__init__()
        d = cfg.d_ability
        h = cfg.decoder_hidden
        # Stat heads get profile (height/weight/position) to differentiate player types
        trunk_in = d + cfg.n_context + cfg.n_profile

        # Shared trunk: ability + context + profile
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.GELU(),
        )

        # Grouped decoder heads
        self.stat_recon_head = nn.Linear(h, cfg.n_stat_targets)
        self.next_game_head = nn.Linear(h, cfg.n_stat_targets)
        self.dpm_head = nn.Sequential(
            nn.Linear(h, 64),
            nn.GELU(),
            nn.Linear(64, cfg.n_dpm_targets),
        )
        self.rapm_head = nn.Sequential(
            nn.Linear(h, 32),
            nn.GELU(),
            nn.Linear(32, cfg.n_rapm_targets),
        )
        self.archetype_head = nn.Linear(h, cfg.n_archetypes)

        # Separate defensive impact head
        self.defense_trunk = nn.Sequential(
            nn.Linear(trunk_in, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, h // 2),
            nn.LayerNorm(h // 2),
            nn.GELU(),
        )
        self.d_dpm_head = nn.Linear(h // 2, 1)

    def forward(
        self,
        ability: torch.Tensor,
        context: torch.Tensor,
        profile: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            ability: (B, d_ability)
            context: (B, n_context)
            profile: (B, n_profile) static player features (height, weight, position, etc.)
        Returns:
            dict with keys: stat_recon, next_game, dpm, rapm, d_dpm_aux, archetype_logits
        """
        x = torch.cat([ability, context, profile], dim=-1)
        h = self.trunk(x)

        h_def = self.defense_trunk(x)

        return {
            "stat_recon": self.stat_recon_head(h),
            "next_game": self.next_game_head(h),
            "dpm": self.dpm_head(h),
            "rapm": self.rapm_head(h),
            "d_dpm_aux": self.d_dpm_head(h_def),
            "archetype_logits": self.archetype_head(h),
        }


class NKEH(nn.Module):
    """
    Neural Kalman Encoder with Hierarchical Prior.

    Full model combining:
    - Population prior + Archetype network → initial state
    - Game encoder → per-game observations
    - Kalman update with aging drift → sequential state estimation
    - Multi-head decoder → training supervision
    """

    def __init__(self, cfg: NKEHConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_ability

        # Components
        self.population_prior = PopulationPrior(cfg)
        self.archetype_network = ArchetypeNetwork(cfg)
        self.game_encoder = GameEncoder(cfg)
        self.aging_model = AgingModel(cfg)
        self.decoder = Decoder(cfg)

        # Learned Kalman parameters (diagonal, per-dimension)
        self.log_process_noise = nn.Parameter(
            torch.full((d,), cfg.initial_log_process_noise)
        )
        self.log_obs_noise = nn.Parameter(torch.full((d,), cfg.initial_log_obs_noise))

        # Learned initial covariance (softplus(0) ≈ 0.69)
        self.log_P_0 = nn.Parameter(torch.full((d,), 0.0))

    @property
    def process_noise(self) -> torch.Tensor:
        """Q: process noise covariance (diagonal)."""
        return F.softplus(self.log_process_noise)

    @property
    def obs_noise(self) -> torch.Tensor:
        """R: base observation noise (diagonal)."""
        return F.softplus(self.log_obs_noise)

    def initialize_state(
        self,
        profile: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Initialize Kalman state from player profile via hierarchical prior.

        Args:
            profile: (B, n_profile) static player features
            context: (B, n_context) initial game context

        Returns:
            mu_0: (B, d_ability) initial state mean
            P_0: (B, d_ability) initial state variance (diagonal)
            archetype_weights: (B, K) soft archetype assignment
        """
        # Population prior
        mu_pop = self.population_prior(context)

        # Archetype prior
        mu_arch, arch_weights = self.archetype_network(mu_pop, profile)

        # Combine via residual addition
        mu_0 = mu_pop + mu_arch
        # P_0 is a learned constant, not derived from prior spread
        P_0 = F.softplus(self.log_P_0).unsqueeze(0).expand(mu_0.shape[0], -1)

        return mu_0, P_0, arch_weights

    def kalman_predict(
        self,
        mu: torch.Tensor,
        P: torch.Tensor,
        age: torch.Tensor,
        days_gap: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Kalman prediction step: apply aging drift and add process noise.

        Both drift and process noise are scaled by the time gap between games.
        A typical inter-game gap is ~2 days (scale=1.0). Offseason gaps (~150
        days) produce larger drift and more uncertainty, clamped at 20x.

        Args:
            mu: (B, d) current state mean
            P: (B, d) current state variance
            age: (B, 1) player age
            days_gap: (B, 1) calendar days since previous game (raw, unnormalized).
                      If None, defaults to scale 1.0 for backward compat.

        Returns:
            mu_pred: (B, d) predicted state mean
            P_pred: (B, d) predicted state variance
        """
        drift = self.aging_model(age)  # (B, d)

        # Scale drift and process noise by normalized time gap
        # 2 days = typical inter-game spacing -> scale 1.0
        if days_gap is not None:
            gap_scale = (days_gap / 2.0).clamp(min=0.5, max=20.0)  # (B, 1)
        else:
            gap_scale = 1.0

        drift_scaled = drift * gap_scale
        F_diag = 1.0 + drift_scaled  # F = I + drift_scaled

        Q_scaled = self.process_noise.unsqueeze(0) * gap_scale  # (B, d) or (1, d)

        mu_pred = F_diag * mu
        P_pred = F_diag.pow(2) * P + Q_scaled

        return mu_pred, P_pred

    def kalman_update(
        self,
        mu_pred: torch.Tensor,
        P_pred: torch.Tensor,
        mu_obs: torch.Tensor,
        sigma_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Kalman update step: incorporate new observation.

        Args:
            mu_pred: (B, d) predicted state mean
            P_pred: (B, d) predicted state variance
            mu_obs: (B, d) observation mean from game encoder
            sigma_obs: (B, d) observation std from game encoder

        Returns:
            mu_updated: (B, d) updated state mean
            P_updated: (B, d) updated state variance
        """
        obs_var = sigma_obs.pow(2) + self.obs_noise.unsqueeze(0)
        K = P_pred / (P_pred + obs_var + 1e-8)  # Kalman gain, element-wise

        mu_updated = mu_pred + K * (mu_obs - mu_pred)
        P_updated = (1.0 - K) * P_pred

        return mu_updated, P_updated

    def forward_single_game(
        self,
        box_stats: torch.Tensor,
        pbp_stats: torch.Tensor,
        context: torch.Tensor,
        profile: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Phase 1 forward pass: single game, no Kalman sequencing.
        Used for hierarchy pre-training.

        Args:
            box_stats: (B, n_box_stats)
            pbp_stats: (B, n_pbp_stats)
            context: (B, n_context)
            profile: (B, n_profile)

        Returns:
            dict with: ability, P, archetype_weights, decoder outputs
        """
        # Initialize from prior
        mu, P, arch_weights = self.initialize_state(profile, context)

        # Encode this game
        mu_obs, log_sigma_obs = self.game_encoder(box_stats, pbp_stats, context, mu)
        sigma_obs = F.softplus(log_sigma_obs)

        # Single Kalman update
        mu_updated, P_updated = self.kalman_update(mu, P, mu_obs, sigma_obs)

        # Decode
        decoder_out = self.decoder(mu_updated, context, profile)

        return {
            "ability": mu_updated,
            "P": P_updated,
            "archetype_weights": arch_weights,
            **decoder_out,
        }

    def forward_sequence(
        self,
        box_stats_seq: torch.Tensor,
        pbp_stats_seq: torch.Tensor,
        context_seq: torch.Tensor,
        profile: torch.Tensor,
        age_seq: torch.Tensor,
        seq_mask: torch.Tensor | None = None,
        days_gap_seq: torch.Tensor | None = None,
        init_context: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Phase 2 forward pass: sequential Kalman updates over a career.

        Args:
            box_stats_seq: (B, T, n_box_stats)
            pbp_stats_seq: (B, T, n_pbp_stats)
            context_seq: (B, T, n_context)
            profile: (B, n_profile)
            age_seq: (B, T, 1)
            seq_mask: (B, T) boolean, True = valid game
            days_gap_seq: (B, T, 1) raw days since last game (unnormalized).
                          If None, kalman_predict uses default scale 1.0.
            init_context: (B, n_context) career-start context for prior init.
                          If None, falls back to context_seq[:, 0].

        Returns:
            dict with per-timestep outputs stacked along dim 1
        """
        B, T, _ = box_stats_seq.shape

        # Initialize state from prior using career-start context
        # (not truncation-start, which has wrong career_games_played for long careers)
        ctx_for_init = init_context if init_context is not None else context_seq[:, 0]
        mu, P, arch_weights = self.initialize_state(profile, ctx_for_init)

        # Accumulate per-timestep outputs
        abilities = []
        Ps = []
        decoder_outputs = {
            "stat_recon": [],
            "next_game": [],
            "dpm": [],
            "rapm": [],
            "d_dpm_aux": [],
            "archetype_logits": [],
        }

        for t in range(T):
            # Prediction step (aging drift, scaled by time gap)
            age_t = age_seq[:, t]  # (B, 1)
            days_gap_t = days_gap_seq[:, t] if days_gap_seq is not None else None
            mu_pred, P_pred = self.kalman_predict(mu, P, age_t, days_gap_t)

            # Observation from game encoder
            mu_obs, log_sigma_obs = self.game_encoder(
                box_stats_seq[:, t],
                pbp_stats_seq[:, t],
                context_seq[:, t],
                mu_pred,
            )
            sigma_obs = F.softplus(log_sigma_obs)

            # Kalman update
            mu, P = self.kalman_update(mu_pred, P_pred, mu_obs, sigma_obs)

            # Apply mask: if this timestep is invalid, keep previous state
            if seq_mask is not None:
                valid = seq_mask[:, t].unsqueeze(-1)  # (B, 1)
                mu = torch.where(valid, mu, mu_pred)
                P = torch.where(valid, P, P_pred)

            abilities.append(mu)
            Ps.append(P)

            # Decode at this timestep (profile is static across time)
            dec = self.decoder(mu, context_seq[:, t], profile)
            for key in decoder_outputs:
                decoder_outputs[key].append(dec[key])

        return {
            "ability": torch.stack(abilities, dim=1),  # (B, T, d)
            "P": torch.stack(Ps, dim=1),  # (B, T, d)
            "archetype_weights": arch_weights,  # (B, K)
            "stat_recon": torch.stack(decoder_outputs["stat_recon"], dim=1),
            "next_game": torch.stack(decoder_outputs["next_game"], dim=1),
            "dpm": torch.stack(decoder_outputs["dpm"], dim=1),
            "rapm": torch.stack(decoder_outputs["rapm"], dim=1),
            "d_dpm_aux": torch.stack(decoder_outputs["d_dpm_aux"], dim=1),
            "archetype_logits": torch.stack(decoder_outputs["archetype_logits"], dim=1),
        }

    def get_ability_vector(
        self,
        box_stats_seq: torch.Tensor,
        pbp_stats_seq: torch.Tensor,
        context_seq: torch.Tensor,
        profile: torch.Tensor,
        age_seq: torch.Tensor,
        seq_mask: torch.Tensor | None = None,
        days_gap_seq: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inference: run Kalman filter over game sequence, return final state.

        Args:
            Same as forward_sequence

        Returns:
            mu: (B, d_ability) final ability vector
            P: (B, d_ability) final uncertainty
        """
        with torch.no_grad():
            out = self.forward_sequence(
                box_stats_seq,
                pbp_stats_seq,
                context_seq,
                profile,
                age_seq,
                seq_mask,
                days_gap_seq,
            )
        return out["ability"][:, -1], out["P"][:, -1]


def vicreg_loss(
    ability: torch.Tensor, gamma: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    VICReg-style regularization: covariance + variance terms.

    Args:
        ability: (B, d) ability vectors for a batch
        gamma: target std per dimension (variance hinge threshold)

    Returns:
        cov_loss: scalar penalizing off-diagonal correlations
        var_loss: scalar penalizing dimensional collapse (std < gamma)
    """
    B, d = ability.shape
    if B < 2:
        zero = torch.tensor(0.0, device=ability.device)
        return zero, zero

    # Per-dimension std (before standardization)
    std = ability.std(dim=0)  # (d,)

    # Covariance loss (off-diagonal correlations)
    z = ability - ability.mean(dim=0, keepdim=True)
    z_normed = z / (std.unsqueeze(0) + 1e-4)
    corr = (z_normed.T @ z_normed) / (B - 1)
    off_diag = corr.pow(2)
    off_diag.fill_diagonal_(0.0)
    cov_loss = off_diag.sum() / d

    # Variance loss (hinge: penalize dimensions with std < gamma)
    var_loss = F.relu(gamma - std).mean()

    return cov_loss, var_loss


def decorrelation_loss(ability: torch.Tensor) -> torch.Tensor:
    """
    Legacy VICReg-style decorrelation: penalize off-diagonal correlations.
    Kept for backward compatibility. Prefer vicreg_loss() for new code.

    Args:
        ability: (B, d) ability vectors for a batch

    Returns:
        Scalar loss penalizing correlated dimensions.
    """
    B, d = ability.shape
    if B < 2:
        return torch.tensor(0.0, device=ability.device)

    # Standardize
    z = ability - ability.mean(dim=0, keepdim=True)
    std = z.std(dim=0, keepdim=True) + 1e-4
    z = z / std

    # Correlation matrix
    corr = (z.T @ z) / (B - 1)  # (d, d)

    # Off-diagonal penalty
    off_diag = corr.pow(2)
    off_diag.fill_diagonal_(0.0)

    return off_diag.sum() / d


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
