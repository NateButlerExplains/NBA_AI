"""
Training components for the transformer model.

Components:
    - loss: Gaussian NLL + MSE loss functions
    - metrics: MAE, Brier score, ECE, AUC
    - trainer: Training loop with early stopping
    - config: Hyperparameter management
"""

from src.transformer.training.loss import (
    gaussian_nll_loss,
    SpreadLoss,
    ScoreLoss,
    WinProbLoss,
    CombinedLoss,
)
from src.transformer.training.metrics import (
    MetricsCalculator,
    MetricResults,
    compute_auc,
    compute_calibration_error,
    compute_coverage,
    compute_crps,
)
from src.transformer.training.config import (
    DataConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
    ExperimentConfig,
    get_small_config,
    get_medium_config,
    get_large_config,
    get_ablation_configs,
)
from src.transformer.training.trainer import (
    Trainer,
    TrainingState,
)

__all__ = [
    # Loss functions
    "gaussian_nll_loss",
    "SpreadLoss",
    "ScoreLoss",
    "WinProbLoss",
    "CombinedLoss",
    # Metrics
    "MetricsCalculator",
    "MetricResults",
    "compute_auc",
    "compute_calibration_error",
    "compute_coverage",
    "compute_crps",
    # Config
    "DataConfig",
    "ModelConfig",
    "OptimizerConfig",
    "TrainingConfig",
    "ExperimentConfig",
    "get_small_config",
    "get_medium_config",
    "get_large_config",
    "get_ablation_configs",
    # Trainer
    "Trainer",
    "TrainingState",
]
