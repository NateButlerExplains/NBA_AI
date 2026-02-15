"""
Configuration System for NBA Transformer Training.

This module defines all hyperparameters and settings for training experiments
as structured Python dataclasses. The configuration is organized into four groups:

    DataConfig      - What data to load and how to process it
    ModelConfig     - Neural network architecture (layers, dimensions, etc.)
    OptimizerConfig - Training optimization (learning rate, weight decay, etc.)
    TrainingConfig  - Training loop behavior (epochs, early stopping, logging, etc.)

These are bundled into ExperimentConfig, which can be serialized to/from YAML
files. This makes experiments fully reproducible — you can always see exactly
what hyperparameters produced a given model by looking at its config file.

Usage:
    config = ExperimentConfig.from_yaml("configs/transformer/baseline.yaml")
    config = ExperimentConfig.from_dict({...})

    # Or use defaults
    config = ExperimentConfig()
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Any
import yaml


@dataclass
class DataConfig:
    """
    Configuration for data loading and preprocessing.

    Controls WHAT data is loaded (which seasons), HOW sequences are constructed
    (how many past games to look at), and DataLoader settings (batch size, workers).
    """

    # --- Temporal train/val/test split ---
    # We split by SEASON to prevent data leakage: the model trains on older seasons
    # and is tested on newer ones. This simulates real-world usage where you train
    # on historical data and predict future games.
    train_seasons: list[str] = field(default_factory=lambda: [
        "2018-2019", "2019-2020", "2020-2021", "2021-2022", "2022-2023",
    ])
    val_seasons: list[str] = field(default_factory=lambda: ["2023-2024"])     # Validation: tune hyperparameters
    test_seasons: list[str] = field(default_factory=lambda: ["2024-2025", "2025-2026"])  # Test: final evaluation only

    # --- Sequence parameters ---
    # How many recent games to feed the model for each team (context window)
    n_history_games: int = 5
    # Minimum games required (early-season teams may not have 5 games yet)
    min_history_games: int = 3

    # --- Tokenizer ---
    # Path to the saved vocabulary file (maps play-by-play text to token IDs)
    tokenizer_path: str = "data/tokenized/tokenizer.json"
    # Minimum token frequency — rare tokens (seen fewer than this many times) are
    # replaced with <UNK> to keep the vocabulary manageable
    min_token_count: int = 5

    # --- DataLoader settings ---
    batch_size: int = 4          # Games per GPU batch (small due to long sequences)
    num_workers: int = 2         # Parallel data loading threads (speeds up I/O)
    cache_size: int = 2000       # Number of processed games to cache in memory
    pin_memory: bool = True      # Pin CPU memory for faster GPU transfer (CUDA only)


@dataclass
class ModelConfig:
    """
    Configuration for the neural network architecture.

    This is the SINGLE ModelConfig used by both the training pipeline and the model.
    Controls the SIZE and STRUCTURE of the transformer model. Larger values
    (more layers, wider dimensions) give the model more capacity to learn
    complex patterns, but also require more memory and training time.

    `vocab_sizes` is injected at runtime from the tokenizer (not from YAML).
    """

    # --- Vocabulary (injected at runtime from tokenizer, not from YAML) ---
    vocab_sizes: dict = field(default_factory=dict)

    # --- Core dimensions ---
    hidden_dim: int = 256          # Width of transformer layers (all internal representations)

    # --- Sequence limits ---
    max_pbp_length: int = 1000     # Maximum plays per game (longer games are truncated)
    max_history_games: int = 10    # Maximum historical games per team

    # --- Event Encoder (processes play-by-play sequences within each game) ---
    event_encoder_layers: int = 4    # Number of transformer layers (depth)
    event_encoder_heads: int = 8     # Number of attention heads (parallel attention patterns)
    event_encoder_ff_dim: Optional[int] = None  # Feed-forward dim (default: 4 * hidden_dim)

    # --- Temporal Attention (attends across multiple games over time) ---
    temporal_attention_layers: int = 2   # Layers for cross-game attention
    temporal_attention_heads: int = 4    # Attention heads for temporal patterns
    temporal_positional_encoding: str = "learned"  # "learned" or "sinusoidal" position encoding
    temporal_pooling: str = "attention"  # How to aggregate games: "attention", "mean", or "first"

    # --- Regularization ---
    dropout: float = 0.1  # Probability of dropping neurons during training (prevents overfitting)

    def __post_init__(self):
        """Validate vocab_sizes if provided (injected at runtime, not from YAML)."""
        if self.vocab_sizes:
            required_vocab_keys = [
                "action_type", "sub_type", "player", "period",
                "clock_bucket", "team_indicator", "score_diff_bucket", "shot_result"
            ]
            missing = [k for k in required_vocab_keys if k not in self.vocab_sizes]
            if missing:
                raise ValueError(f"Missing required vocabulary keys: {missing}")


@dataclass
class OptimizerConfig:
    """
    Configuration for the optimizer and learning rate schedule.

    Controls HOW the model learns: how big the weight updates are (learning rate),
    how fast to decay unused features (weight decay), and how the learning rate
    changes over training (schedule).
    """

    # --- Optimizer selection ---
    optimizer: str = "adamw"         # AdamW: Adam with decoupled weight decay (standard for transformers)
    learning_rate: float = 1e-4      # Peak learning rate (0.0001). Too high -> diverge, too low -> slow
    weight_decay: float = 0.01       # L2 regularization strength (penalizes large weights)
    # Betas control the optimizer's "momentum":
    #   beta1 (0.9): how much to smooth the gradient direction (like a rolling average)
    #   beta2 (0.999): how much to smooth the gradient magnitude (adaptive step sizes)
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8                # Prevents division by zero in Adam's denominator

    # --- Learning rate schedule ---
    scheduler: str = "cosine"    # "cosine" (smooth decay), "linear" (straight line), or "constant"
    warmup_steps: int = 0       # Fixed warmup steps (0 = use warmup_ratio instead)
    warmup_ratio: float = 0.1   # Warmup as a fraction of total steps (scales with dataset size)
    min_lr_ratio: float = 0.1   # Floor LR = learning_rate * min_lr_ratio (LR never drops below this)

    # --- Gradient clipping ---
    # Caps the total gradient norm to prevent exploding gradients from destabilizing training
    max_grad_norm: float = 1.0


@dataclass
class TrainingConfig:
    """
    Configuration for the training loop behavior.

    Controls HOW LONG to train, WHEN to stop, HOW to save progress,
    and various training tricks (gradient accumulation, mixed precision).
    """

    # --- Training duration ---
    max_epochs: int = 50                     # Maximum number of passes through the training data
    max_steps: Optional[int] = None          # If set, stop after this many steps (overrides epochs)

    # --- Early stopping ---
    # If validation loss does not improve for `patience` consecutive epochs, stop training.
    # This prevents overfitting: the model starts memorizing training data without
    # generalizing to unseen data.
    patience: int = 15
    # Minimum improvement to count as "better" (prevents stopping on noise)
    min_delta: float = 1e-4

    # --- Checkpointing ---
    checkpoint_dir: str = "checkpoints/transformer"   # Where to save model snapshots
    save_every_n_epochs: int = 5                      # Save periodic checkpoints
    save_best_only: bool = True                       # Clean up old periodic checkpoints
    keep_n_checkpoints: int = 3                       # How many periodic checkpoints to keep

    # --- Logging ---
    log_every_n_steps: int = 200                      # Log training loss every N steps
    eval_every_n_steps: int = 1000                    # (Not currently used — validation is per-epoch)
    use_wandb: bool = False                           # Enable Weights & Biases experiment tracking
    wandb_project: str = "nba-transformer"
    wandb_run_name: Optional[str] = None
    experiment_name: str = "baseline"                 # Name for this experiment run

    # --- Loss weights ---
    # These control the relative importance of each prediction task in the combined loss.
    # See CombinedLoss in loss.py for details.
    spread_loss_weight: float = 1.0       # Primary objective: point spread
    score_loss_weight: float = 0.5        # Secondary: individual team scores
    win_prob_loss_weight: float = 0.1     # Auxiliary: win probability
    nll_weight: float = 1.0               # Weight for NLL component within each loss
    mse_weight: float = 0.1              # Weight for auxiliary MSE component

    # --- Gradient accumulation ---
    # Simulates larger batch sizes on limited GPU memory.
    # Effective batch size = batch_size * gradient_accumulation_steps.
    # E.g., batch_size=8 * accumulation=4 = effective batch of 32.
    gradient_accumulation_steps: int = 8

    # --- Reproducibility ---
    seed: int = 42  # Random seed for reproducible results

    # --- Mixed precision (AMP) ---
    # Uses float16 for speed and memory savings on CUDA GPUs.
    # float32 is still used for weight updates and numerically sensitive operations.
    use_amp: bool = True
    amp_dtype: str = "float16"  # "float16" (most GPUs) or "bfloat16" (newer GPUs, more stable)

    # --- Device ---
    device: str = "cuda"  # "cuda" (NVIDIA GPU), "cpu", or "mps" (Apple Silicon GPU)


@dataclass
class ExperimentConfig:
    """
    Complete experiment configuration — bundles all four config groups.

    This is the single object passed around during training. It can be serialized
    to YAML (for saving) and deserialized from YAML (for loading), making every
    experiment fully reproducible. Each checkpoint saves a copy of its config.
    """

    data: DataConfig = field(default_factory=DataConfig)            # What data to load
    model: ModelConfig = field(default_factory=ModelConfig)          # Network architecture
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)  # Optimization settings
    training: TrainingConfig = field(default_factory=TrainingConfig)      # Training loop behavior

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ExperimentConfig":
        """Create configuration from dictionary."""
        # Warn about unknown top-level sections (likely typos in YAML).
        # Field-level typos (e.g., hidden_dims) already crash with TypeError.
        expected_sections = {"data", "model", "optimizer", "training"}
        unknown_sections = set(config_dict.keys()) - expected_sections
        if unknown_sections:
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Unknown config sections ignored (possible typos): {sorted(unknown_sections)}. "
                f"Expected sections: {sorted(expected_sections)}"
            )

        data_config = DataConfig(**config_dict.get("data", {}))
        model_config = ModelConfig(**config_dict.get("model", {}))
        optimizer_config = OptimizerConfig(**config_dict.get("optimizer", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))

        return cls(
            data=data_config,
            model=model_config,
            optimizer=optimizer_config,
            training=training_config,
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
        """Pretty print configuration."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)


# ============================================================================
# Preset configurations for common experiment types.
# These provide sensible defaults for different model sizes, so you do not
# have to manually set dozens of hyperparameters every time.
# ============================================================================

def get_small_config() -> ExperimentConfig:
    """Small model for quick testing (~2M params). Trains fast, good for debugging."""
    return ExperimentConfig(
        model=ModelConfig(
            hidden_dim=128,
            event_encoder_layers=2,
            temporal_attention_layers=1,
        ),
        training=TrainingConfig(
            max_epochs=10,
            patience=5,
        ),
    )


def get_medium_config() -> ExperimentConfig:
    """Medium model (~5M params). Default baseline configuration."""
    return ExperimentConfig()


def get_large_config() -> ExperimentConfig:
    """Large model (~15M params). Best potential accuracy, but slow to train."""
    return ExperimentConfig(
        model=ModelConfig(
            hidden_dim=512,
            event_encoder_layers=6,
            event_encoder_heads=16,
            temporal_attention_layers=4,
            temporal_attention_heads=8,
            dropout=0.15,
        ),
        optimizer=OptimizerConfig(
            learning_rate=5e-5,
        ),
        training=TrainingConfig(
            max_epochs=100,
            patience=25,
        ),
    )


# ============================================================================
# Ablation configurations.
# An "ablation study" systematically varies one hyperparameter at a time while
# keeping everything else fixed. This reveals which components of the model
# actually contribute to performance (e.g., "does cross-attention fusion help?").
# ============================================================================

def get_ablation_configs() -> dict[str, ExperimentConfig]:
    """Get configurations for ablation studies (one hyperparameter varied at a time)."""
    configs = {}

    # Baseline: the standard medium model (everything else is compared to this)
    configs["baseline"] = get_medium_config()

    # Sequence length ablations: how many past games should the model look at?
    # More history = more context, but also more memory and compute.
    for n_games in [3, 5, 10, 20]:
        config = get_medium_config()
        config.data.n_history_games = n_games
        config.training.experiment_name = f"seq_len_{n_games}"
        configs[f"seq_len_{n_games}"] = config

    # Model size ablations: does a bigger model learn better, or just overfit?
    configs["model_small"] = get_small_config()
    configs["model_small"].training.experiment_name = "model_small"

    configs["model_medium"] = get_medium_config()
    configs["model_medium"].training.experiment_name = "model_medium"

    configs["model_large"] = get_large_config()
    configs["model_large"].training.experiment_name = "model_large"

    return configs


def test_config():
    """Test configuration system."""
    import tempfile
    import os

    print("Testing ExperimentConfig...")

    # Test default config
    config = ExperimentConfig()
    print(f"  Default config created")
    print(f"  Hidden dim: {config.model.hidden_dim}")
    print(f"  Batch size: {config.data.batch_size}")
    print(f"  Learning rate: {config.optimizer.learning_rate}")

    # Test from_dict
    config_dict = {
        "model": {"hidden_dim": 128},
        "training": {"max_epochs": 10},
    }
    config = ExperimentConfig.from_dict(config_dict)
    assert config.model.hidden_dim == 128
    assert config.training.max_epochs == 10
    print(f"  from_dict works")

    # Test YAML save/load
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = os.path.join(tmpdir, "config.yaml")
        config.to_yaml(yaml_path)

        loaded = ExperimentConfig.from_yaml(yaml_path)
        assert loaded.model.hidden_dim == 128
        print(f"  YAML save/load works")

        # Test JSON save
        json_path = os.path.join(tmpdir, "config.json")
        config.to_json(json_path)
        print(f"  JSON save works")

    # Test preset configs
    print("\nTesting preset configs...")
    small = get_small_config()
    medium = get_medium_config()
    large = get_large_config()
    print(f"  Small: hidden_dim={small.model.hidden_dim}")
    print(f"  Medium: hidden_dim={medium.model.hidden_dim}")
    print(f"  Large: hidden_dim={large.model.hidden_dim}")

    # Test ablation configs
    print("\nTesting ablation configs...")
    ablations = get_ablation_configs()
    print(f"  {len(ablations)} ablation configs created")
    for name in list(ablations.keys())[:5]:
        print(f"    - {name}")
    print(f"    ...")

    # Test string representation
    print("\nConfig string representation:")
    print(str(small)[:500] + "...")

    print("\nAll tests passed!")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(test_config())
