"""
NKE-H model configuration.
"""

from dataclasses import dataclass, field


@dataclass
class NKEHConfig:
    """Configuration for the Neural Kalman Encoder with Hierarchical Prior."""

    # --- Dimensions ---
    d_ability: int = 32  # ability vector dimensionality
    n_archetypes: int = 10  # K soft archetypes

    # --- Input dimensions (set by cache metadata) ---
    n_box_stats: int = 16  # PlayerBox features
    n_pbp_stats: int = 42  # PBP enriched features
    n_context: int = 12  # game context features
    n_profile: int = 9  # static player profile features

    # --- Prior Network ---
    prior_hidden: int = 128
    prior_layers: int = 2

    # --- Archetype Network ---
    archetype_hidden: int = 128
    archetype_prototype_dim: int = 32  # same as d_ability

    # --- Game Encoder ---
    encoder_hidden: int = 256
    encoder_layers: int = 3
    encoder_dropout: float = 0.1

    # --- Aging Model ---
    aging_hidden: int = 32
    aging_max_drift: float = 0.01  # tanh * max_drift per game step

    # --- Kalman parameters ---
    initial_log_process_noise: float = -2.0  # log(Q) init → Q ≈ 0.14
    initial_log_obs_noise: float = 0.0  # log(R) init → R ≈ 1.0

    # --- Decoder ---
    decoder_hidden: int = 256
    decoder_layers: int = 2

    # --- Decoder heads ---
    n_stat_targets: int = 16  # stat reconstruction (box stats)
    n_dpm_targets: int = 3  # o_dpm, d_dpm, dpm
    n_rapm_targets: int = 2  # off_rapm, def_rapm

    # --- Decorrelation regularization ---
    decorrelation_weight: float = 0.01

    # --- Training ---
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 1024  # player-game samples (Phase 1)
    seq_batch_size: int = 128  # player careers (Phase 2)
    max_career_len: int = 300  # truncated BPTT window for Phase 2
    gradient_clip: float = 1.0

    # --- Loss weights (Phase 1: hierarchy pre-training) ---
    w_reconstruction: float = 0.5
    w_next_game: float = 0.3
    w_dpm: float = 0.3
    w_archetype: float = 0.1
    w_decorrelation: float = 0.01

    # --- Loss weights (Phase 2: sequential Kalman training) ---
    w_reconstruction_seq: float = 0.3
    w_next_game_seq: float = 0.3
    w_dpm_seq: float = 0.4

    # --- Data splits ---
    pretrain_seasons_end: str = "2017-2018"
    pretrain_val_season: str = "2017-2018"
    train_seasons_end: str = "2023-2024"
    val_season: str = "2023-2024"
