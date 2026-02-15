"""
Evaluation tools for the transformer model.

Components:
    - evaluate: Test set evaluation
    - ablation: Ablation study runner
    - visualize: Calibration plots, attention visualization
"""

from src.transformer.evaluation.evaluate import (
    Evaluator,
    EvaluationResults,
    load_and_evaluate,
)
from src.transformer.evaluation.ablation import (
    AblationRunner,
    AblationResult,
    run_quick_ablation,
)
from src.transformer.evaluation.visualize import (
    Visualizer,
    plot_training_curves,
)

__all__ = [
    # Evaluation
    "Evaluator",
    "EvaluationResults",
    "load_and_evaluate",
    # Ablation
    "AblationRunner",
    "AblationResult",
    "run_quick_ablation",
    # Visualization
    "Visualizer",
    "plot_training_curves",
]
