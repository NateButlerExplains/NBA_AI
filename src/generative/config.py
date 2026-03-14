"""
Configuration for Phase 4 Generative Model.

Four dataclasses bundled into GenerativeExperimentConfig with YAML serialization.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class GenerativeDataConfig:
    """Data loading and preprocessing configuration."""

    train_seasons: list[str] = field(
        default_factory=lambda: [
            "2008-2009",
            "2009-2010",
            "2010-2011",
            "2011-2012",
            "2012-2013",
            "2013-2014",
            "2014-2015",
            "2015-2016",
            "2016-2017",
            "2017-2018",
            "2018-2019",
            "2019-2020",
            "2020-2021",
            "2021-2022",
            "2022-2023",
        ]
    )
    val_seasons: list[str] = field(default_factory=lambda: ["2023-2024"])
    test_seasons: list[str] = field(default_factory=lambda: ["2024-2025", "2025-2026"])

    cache_dir: str = "data/generative_cache"
    db_path: str = "data/NBA_AI_full.sqlite"

    max_context_games: int = 82
    min_history_games: int = 3
    max_players_per_game: int = 13
    max_seq_len: int = 700
    n_player_stats: int = 16

    batch_size: int = 4
    num_workers: int = 2
    pin_memory: bool = True

    # Score jitter for exposure bias mitigation (raw score space)
    score_jitter_std: float = 0.5


@dataclass
class GenerativeModelConfig:
    """Neural network architecture configuration."""

    # Global
    hidden_dim: int = 512
    dropout: float = 0.1

    # Player encoder
    n_players: int = 5000
    player_embed_dim: int = 128
    player_stat_dim: int = 64
    player_hidden_dim: int = 256
    player_pool_heads: int = 4

    # Per-game encoder
    score_input_dim: int = 4  # [team_score, opp_score, margin, total]
    score_dim: int = 128
    opponent_dim: int = 64
    location_dim: int = 32
    n_teams: int = 30

    # Temporal encoder
    temporal_layers: int = 3
    temporal_heads: int = 8
    temporal_ff_dim: int = 2048
    temporal_dropout: float = 0.1
    temporal_max_days: int = 180
    temporal_n_pool_queries: int = 4

    # Rest days
    max_rest_days: int = 30
    rest_dim: int = 64

    # State embedder
    state_input_dim: int = 7
    state_hidden_dim: int = 256

    # Causal decoder
    decoder_layers: int = 6
    decoder_heads: int = 8
    decoder_ff_dim: int = 2048
    decoder_dropout: float = 0.1
    decoder_max_seq_len: int = 800  # 2 context + up to 700 states + margin

    # Prediction heads
    n_score_classes: int = 7
    head_hidden_dim: int = 128

    # Context dropout (anti-collapse)
    context_dropout: float = 0.1

    # Simplified context encoder (Exp 3)
    use_simplified_context: bool = False
    n_rolling_features: int = 24
    simple_context_hidden: int = 256
    simple_team_embed_dim: int = 64

    # Scoring-event compression (Exp 4)
    use_scoring_events_only: bool = False
    # Max scoring events per game (covers >99% of games)
    max_scoring_events: int = 200

    # Score event class weights (inverse frequency)
    # Full mode: {no_score, home+1, home+2, home+3, away+1, away+2, away+3}
    # Compressed mode: {home+1, home+2, home+3, away+1, away+2, away+3, game_end}
    score_class_weights: list[float] = field(
        default_factory=lambda: [1.0, 4.0, 4.5, 5.0, 4.0, 4.5, 5.0]
    )


@dataclass
class GenerativeOptimizerConfig:
    """Optimizer and learning rate schedule configuration."""

    optimizer: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    betas: list[float] = field(default_factory=lambda: [0.9, 0.98])
    eps: float = 1e-8

    scheduler: str = "cosine"
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.01

    max_grad_norm: float = 1.0


@dataclass
class GenerativeTrainingConfig:
    """Training loop behavior configuration."""

    max_epochs: int = 50
    patience: int = 10
    early_stopping_metric: str = "val_spread_mae"
    smoothing_window: int = 3
    min_delta: float = 1e-4

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.999

    # Loss weights
    score_loss_weight: float = 1.0
    clock_loss_weight: float = 0.3
    context_loss_weight: float = 1.0
    pre_margin_weight: float = 1.0
    pre_win_weight: float = 0.5

    # Gradient accumulation
    gradient_accumulation_steps: int = 8

    # Checkpointing
    checkpoint_dir: str = "checkpoints/generative"
    save_best_only: bool = True
    keep_n_checkpoints: int = 3

    # Logging
    log_every_n_steps: int = 50
    experiment_name: str = "gen_exp1_baseline"

    # Reproducibility
    seed: int = 42

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "float16"

    # Device
    device: str = "cuda"

    # Scheduled sampling (Exp 3)
    use_scheduled_sampling: bool = False
    ss_start_ratio: float = 1.0
    ss_end_ratio: float = 0.8
    ss_warmup_epochs: int = 5
    ss_anneal_epochs: int = 20

    # Inference
    n_rollouts: int = 100
    rollout_temperature: float = 1.0
    max_rollout_steps: int = 700
    guidance_scale: float = 1.5


@dataclass
class GenerativeExperimentConfig:
    """Complete generative experiment configuration."""

    data: GenerativeDataConfig = field(default_factory=GenerativeDataConfig)
    model: GenerativeModelConfig = field(default_factory=GenerativeModelConfig)
    optimizer: GenerativeOptimizerConfig = field(
        default_factory=GenerativeOptimizerConfig
    )
    training: GenerativeTrainingConfig = field(default_factory=GenerativeTrainingConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "GenerativeExperimentConfig":
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "GenerativeExperimentConfig":
        expected_sections = {"data", "model", "optimizer", "training"}
        unknown_sections = set(config_dict.keys()) - expected_sections
        if unknown_sections:
            logger.warning(
                f"Unknown config sections ignored: {sorted(unknown_sections)}. "
                f"Expected: {sorted(expected_sections)}"
            )
        return cls(
            data=GenerativeDataConfig(**config_dict.get("data", {})),
            model=GenerativeModelConfig(**config_dict.get("model", {})),
            optimizer=GenerativeOptimizerConfig(**config_dict.get("optimizer", {})),
            training=GenerativeTrainingConfig(**config_dict.get("training", {})),
        )

    def to_dict(self) -> dict:
        return {
            "data": asdict(self.data),
            "model": asdict(self.model),
            "optimizer": asdict(self.optimizer),
            "training": asdict(self.training),
        }

    def to_yaml(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def __str__(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
