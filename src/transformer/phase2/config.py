"""
Configuration System for Phase 2 NBA Transformer Training.

Four dataclasses bundled into Phase2ExperimentConfig with YAML serialization.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class Phase2DataConfig:
    """Data loading and preprocessing configuration."""

    train_seasons: list[str] = field(default_factory=lambda: [
        "2018-2019", "2019-2020", "2020-2021", "2021-2022", "2022-2023",
    ])
    val_seasons: list[str] = field(default_factory=lambda: ["2023-2024"])
    test_seasons: list[str] = field(default_factory=lambda: ["2024-2025", "2025-2026"])

    cache_dir: str = "data/phase2_cache"
    n_recent_games: int = 5
    max_context_games: int = 82
    min_history_games: int = 3
    max_players_per_game: int = 15
    max_roster_size: int = 15

    batch_size: int = 4
    num_workers: int = 2
    pin_memory: bool = True
    enable_augmentation: bool = True
    mask_ratio: float = 0.0  # 0.0 = disabled (supervised), 0.4 = pre-training
    n_player_stats: int = 0       # 0 = legacy, 16 = full (must match cache)
    n_efficiency_features: int = 0  # 0 = disabled, 8 = full (must match model)


@dataclass
class Phase2ModelConfig:
    """Neural network architecture configuration."""

    hidden_dim: int = 512
    player_embed_dim: int = 128
    n_players: int = 4000
    n_teams: int = 30

    # Per-game encoder
    score_dim: int = 128
    opponent_dim: int = 64
    location_dim: int = 32
    player_contribution_dim: int = 256
    player_contribution_heads: int = 4
    player_contribution_dropout: float = 0.2

    # Player stats (Phase 3+)
    n_player_stats: int = 0       # 0 = legacy (points only), 16 = full box score
    stat_hidden_dim: int = 64     # Stat MLP output dimension
    n_positions: int = 4          # G/F/C/UNK
    position_dim: int = 8         # Position embedding dimension

    # Team efficiency features (Phase 3 Exp 7+)
    n_efficiency_features: int = 0    # 0 = disabled, 8 = full
    efficiency_hidden_dim: int = 64
    gs_summary_dim: int = 32
    flag_dim: int = 16
    season_efficiency_dim: int = 64

    # Player interaction (Phase 3 Exp 4+)
    player_interaction_layers: int = 0       # 0 = disabled (backward compat)
    player_interaction_heads: int = 4
    player_interaction_ff_dim: int = 1024
    player_interaction_dropout: float = 0.2
    player_contribution_n_pool_queries: int = 1  # 1 = single query (default), 4 = multi-query

    # Roster-conditioned temporal (Phase 3 Exp 5+)
    enable_roster_context: bool = False
    roster_context_heads: int = 8
    roster_context_dropout: float = 0.1

    # Team interaction GAT (Phase 3 Exp 6+)
    enable_team_gat: bool = False
    team_gat_hidden: int = 64
    team_gat_layers: int = 3
    team_gat_heads: int = 4
    team_gat_dropout: float = 0.1
    h2h_edge_features: int = 3

    # GameStates encoder
    gs_embed_dim: int = 16
    gs_encoder_layers: int = 4
    gs_encoder_heads: int = 8
    gs_encoder_ff_dim: int = 2048
    gs_encoder_dropout: float = 0.1
    gs_max_seq_len: int = 1000

    # Temporal module
    temporal_type: str = "transformer"  # "transformer" or "gru"
    temporal_layers: int = 3
    temporal_heads: int = 8
    temporal_ff_dim: int = 2048
    temporal_dropout: float = 0.1
    temporal_max_days: int = 180
    temporal_n_pool_queries: int = 4
    temporal_pos_encoding: str = "days_before"  # "days_before" or "ordinal"

    # GRU temporal (used when temporal_type == "gru")
    temporal_gru_layers: int = 2
    temporal_gru_hidden: int = 256       # Per direction (bidir doubles to match hidden_dim)
    temporal_gru_dropout: float = 0.1
    temporal_gru_time_dim: int = 64      # Time embedding dimension

    # Roster encoder
    roster_layers: int = 2
    roster_heads: int = 8
    roster_ff_dim: int = 2048
    roster_dropout: float = 0.2

    # Player form encoder
    enable_player_form: bool = False
    player_form_dim: int = 64
    player_form_days_dim: int = 32
    player_form_heads: int = 4
    player_form_ff_dim: int = 256
    player_form_dropout: float = 0.2
    max_player_appearances: int = 40

    # Rest days
    rest_embed_dim: int = 64
    max_rest_days: int = 30

    # PLE (Piecewise Linear Encoding) for score features
    use_ple: bool = False
    n_ple_bins: int = 16

    # Fusion
    fusion_dropout: float = 0.3
    use_cross_attention_fusion: bool = False
    fusion_heads: int = 8

    # Prediction heads
    prediction_hidden_dim: int = 256
    prediction_dropout: float = 0.3
    spread_min_std: float = 1.0
    spread_max_std: Optional[float] = None  # Cap spread sigma (e.g., 8.0)
    score_min_std: float = 5.0
    derive_spread_from_scores: bool = False  # Derive spread = home - away (no separate head)


@dataclass
class Phase2OptimizerConfig:
    """Optimizer and learning rate schedule configuration."""

    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    betas: list[float] = field(default_factory=lambda: [0.9, 0.98])
    eps: float = 1e-8

    scheduler: str = "cosine"
    warmup_steps: int = 0
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.1

    max_grad_norm: float = 1.0


@dataclass
class Phase2TrainingConfig:
    """Training loop behavior configuration."""

    max_epochs: int = 100
    max_steps: Optional[int] = None

    patience: int = 15
    early_stopping_metric: str = "val_spread_mae"
    smoothing_window: int = 3
    min_delta: float = 1e-4

    # EMA
    ema_decay: float = 0.999
    use_ema: bool = True

    # Loss weights
    spread_loss_weight: float = 1.0
    score_loss_weight: float = 0.5
    win_prob_loss_weight: float = 0.3
    consistency_loss_weight: float = 0.1
    nll_weight: float = 1.0
    mse_weight: float = 0.1
    spread_huber_delta: Optional[float] = None  # Use Huber instead of MSE for spread
    score_huber_delta: Optional[float] = None   # Use Huber instead of MSE for scores

    # Gradient accumulation
    gradient_accumulation_steps: int = 8

    # Checkpointing
    checkpoint_dir: str = "checkpoints/phase2"
    save_every_n_epochs: int = 5
    save_best_only: bool = True
    keep_n_checkpoints: int = 3

    # Logging
    log_every_n_steps: int = 50
    use_wandb: bool = False
    wandb_project: str = "nba-phase2"
    wandb_run_name: Optional[str] = None
    experiment_name: str = "phase2_baseline"

    # Pre-training / fine-tuning
    pretrained_checkpoint: str = ""  # Path to transferable_weights.pt
    freeze_pretrained_epochs: int = 0  # Epochs to freeze pre-trained components
    unfreeze_top_epochs: int = 0  # Epochs with only top block unfrozen
    lr_decay_factor: float = 1.0  # Per-layer LR decay (1.0 = uniform, 0.9 = discriminative)

    # Reproducibility
    seed: int = 42

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "float16"

    # Device
    device: str = "cuda"


@dataclass
class Phase2ExperimentConfig:
    """Complete Phase 2 experiment configuration."""

    data: Phase2DataConfig = field(default_factory=Phase2DataConfig)
    model: Phase2ModelConfig = field(default_factory=Phase2ModelConfig)
    optimizer: Phase2OptimizerConfig = field(default_factory=Phase2OptimizerConfig)
    training: Phase2TrainingConfig = field(default_factory=Phase2TrainingConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Phase2ExperimentConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Phase2ExperimentConfig":
        """Create configuration from dictionary."""
        expected_sections = {"data", "model", "optimizer", "training"}
        unknown_sections = set(config_dict.keys()) - expected_sections
        if unknown_sections:
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Unknown config sections ignored: {sorted(unknown_sections)}. "
                f"Expected: {sorted(expected_sections)}"
            )

        return cls(
            data=Phase2DataConfig(**config_dict.get("data", {})),
            model=Phase2ModelConfig(**config_dict.get("model", {})),
            optimizer=Phase2OptimizerConfig(**config_dict.get("optimizer", {})),
            training=Phase2TrainingConfig(**config_dict.get("training", {})),
        )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "data": asdict(self.data),
            "model": asdict(self.model),
            "optimizer": asdict(self.optimizer),
            "training": asdict(self.training),
        }

    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def to_json(self, path: str):
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def __str__(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
