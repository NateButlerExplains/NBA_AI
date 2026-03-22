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
    Learns the distribution of a 'generic NBA player' conditioned on context.
    Produces mu_pop and log_sigma_pop for the population-level prior.
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
        self.log_sigma_head = nn.Linear(h, d)

        # Initialize log_sigma to produce moderate uncertainty
        nn.init.constant_(self.log_sigma_head.bias, 0.5)

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            context: (B, n_context) game context features
        Returns:
            mu_pop: (B, d_ability)
            log_sigma_pop: (B, d_ability)
        """
        h = self.net(context)
        return self.mu_head(h), self.log_sigma_head(h)


class ArchetypeNetwork(nn.Module):
    """
    Learns K archetype prototypes and produces soft assignment + archetype prior.

    Takes player physical profile and produces:
    - Soft archetype weights (K probabilities)
    - Archetype-weighted mu and sigma
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
        self.prototype_log_sigma = nn.Parameter(torch.zeros(K, d))

    def forward(
        self,
        mu_pop: torch.Tensor,
        profile: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            mu_pop: (B, d_ability) population prior mean
            profile: (B, n_profile) static player features
        Returns:
            mu_arch: (B, d_ability) archetype-weighted mean
            log_sigma_arch: (B, d_ability) archetype-weighted log sigma
            archetype_weights: (B, K) soft assignment probabilities
        """
        x = torch.cat([mu_pop, profile], dim=-1)
        h = self.net(x)

        # Soft assignment
        logits = self.assignment_head(h)
        weights = F.softmax(logits, dim=-1)  # (B, K)

        # Weighted combination of prototypes
        # mu_arch = sum_k weights_k * prototype_mu_k
        mu_arch = torch.einsum("bk,kd->bd", weights, self.prototype_mu)
        log_sigma_arch = torch.einsum("bk,kd->bd", weights, self.prototype_log_sigma)

        return mu_arch, log_sigma_arch, weights


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
    3. DPM prediction: predict o_dpm, d_dpm, dpm
    4. Archetype classification: predict archetype weights
    """

    def __init__(self, cfg: NKEHConfig):
        super().__init__()
        d = cfg.d_ability
        h = cfg.decoder_hidden

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(d + cfg.n_context, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.GELU(),
        )

        # Grouped decoder heads (per plan: scoring, playmaking, rebounding, defense, activity)
        self.stat_recon_head = nn.Linear(h, cfg.n_stat_targets)
        self.next_game_head = nn.Linear(h, cfg.n_stat_targets)
        self.dpm_head = nn.Sequential(
            nn.Linear(h, 64),
            nn.GELU(),
            nn.Linear(64, cfg.n_dpm_targets),
        )
        self.archetype_head = nn.Linear(h, cfg.n_archetypes)

    def forward(
        self,
        ability: torch.Tensor,
        context: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            ability: (B, d_ability)
            context: (B, n_context)
        Returns:
            dict with keys: stat_recon, next_game, dpm, archetype_logits
        """
        x = torch.cat([ability, context], dim=-1)
        h = self.trunk(x)

        return {
            "stat_recon": self.stat_recon_head(h),
            "next_game": self.next_game_head(h),
            "dpm": self.dpm_head(h),
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
        mu_pop, log_sigma_pop = self.population_prior(context)
        sigma_pop = F.softplus(log_sigma_pop)

        # Archetype prior
        mu_arch, log_sigma_arch, arch_weights = self.archetype_network(mu_pop, profile)
        sigma_arch = F.softplus(log_sigma_arch)

        # Combine via residual addition (simpler than precision-weighted merge)
        mu_0 = mu_pop + mu_arch
        P_0 = sigma_pop.pow(2) + sigma_arch.pow(2)

        return mu_0, P_0, arch_weights

    def kalman_predict(
        self,
        mu: torch.Tensor,
        P: torch.Tensor,
        age: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Kalman prediction step: apply aging drift and add process noise.

        Args:
            mu: (B, d) current state mean
            P: (B, d) current state variance
            age: (B, 1) player age

        Returns:
            mu_pred: (B, d) predicted state mean
            P_pred: (B, d) predicted state variance
        """
        drift = self.aging_model(age)  # (B, d)
        F_diag = 1.0 + drift  # F = I + drift

        mu_pred = F_diag * mu
        P_pred = F_diag.pow(2) * P + self.process_noise.unsqueeze(0)

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
        decoder_out = self.decoder(mu_updated, context)

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

        Returns:
            dict with per-timestep outputs stacked along dim 1
        """
        B, T, _ = box_stats_seq.shape

        # Initialize state from prior (using first game's context)
        mu, P, arch_weights = self.initialize_state(profile, context_seq[:, 0])

        # Accumulate per-timestep outputs
        abilities = []
        Ps = []
        decoder_outputs = {
            "stat_recon": [],
            "next_game": [],
            "dpm": [],
            "archetype_logits": [],
        }

        for t in range(T):
            # Prediction step (aging drift)
            age_t = age_seq[:, t]  # (B, 1)
            mu_pred, P_pred = self.kalman_predict(mu, P, age_t)

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

            # Decode at this timestep
            dec = self.decoder(mu, context_seq[:, t])
            for key in decoder_outputs:
                decoder_outputs[key].append(dec[key])

        return {
            "ability": torch.stack(abilities, dim=1),  # (B, T, d)
            "P": torch.stack(Ps, dim=1),  # (B, T, d)
            "archetype_weights": arch_weights,  # (B, K)
            "stat_recon": torch.stack(decoder_outputs["stat_recon"], dim=1),
            "next_game": torch.stack(decoder_outputs["next_game"], dim=1),
            "dpm": torch.stack(decoder_outputs["dpm"], dim=1),
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
            )
        return out["ability"][:, -1], out["P"][:, -1]


def decorrelation_loss(ability: torch.Tensor) -> torch.Tensor:
    """
    VICReg-style decorrelation: penalize off-diagonal correlations.

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
