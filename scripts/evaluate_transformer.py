#!/usr/bin/env python
"""
Evaluation Script for NBA Transformer Model.

This script loads a trained model checkpoint and evaluates it on test data.
It can also run ablation studies (systematic experiments varying one
hyperparameter at a time to understand what helps performance).

Two main modes:
    1. Single evaluation: load one checkpoint, evaluate on test set, report metrics
    2. Ablation study: train and evaluate many model variants systematically

Usage:
    # Evaluate a trained checkpoint on test data
    python scripts/evaluate_transformer.py --checkpoint checkpoints/transformer/baseline/best.pt

    # Evaluate with visualization (calibration plots, error distributions, etc.)
    python scripts/evaluate_transformer.py --checkpoint best.pt --visualize --output-dir results/

    # Run ablation study (trains multiple variants, compares them)
    python scripts/evaluate_transformer.py --ablation --config configs/transformer/baseline.yaml

    # Quick test (minimal data, fast — for verifying the eval pipeline works)
    python scripts/evaluate_transformer.py --checkpoint best.pt --quick-test
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Add project root to Python's import search path so we can import src.transformer.*
sys.path.insert(0, str(Path(__file__).parent.parent))

from torch.utils.data import DataLoader

from src.transformer.tokenizer import PBPTokenizer
from src.transformer.dataset import NBAGameDataset, collate_games
from src.transformer.models import Phase1Model, ModelConfig
from src.transformer.training.config import (
    ExperimentConfig,
    get_small_config,
    get_medium_config,
)
from src.transformer.evaluation.evaluate import Evaluator, EvaluationResults
from src.transformer.evaluation.visualize import Visualizer


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_checkpoint(
    checkpoint_path: str,
    tokenizer: PBPTokenizer,
    device: str = "cpu",
) -> tuple:
    """
    Reconstruct a trained model from a saved checkpoint file.

    HOW CHECKPOINT LOADING WORKS:
        A checkpoint contains two key things:
        1. The model configuration (architecture details like hidden_dim, num_layers)
        2. The model state_dict (all the learned weight values)

        To reconstruct the model, we:
        1. Read the config to know what architecture to build
        2. Create a fresh (random) model with that architecture
        3. Load the saved weights into it, replacing the random ones

        The tokenizer is needed because vocab_sizes (how many unique tokens exist)
        is required to build the embedding layers, but it is not stored in the
        checkpoint (it comes from the tokenizer file).

    Args:
        checkpoint_path: Path to the .pt checkpoint file
        tokenizer: Tokenizer instance (provides vocab_sizes for embeddings)
        device: Device to load model onto ("cpu", "cuda", etc.)

    Returns:
        (model, model_config, experiment_name)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    # Load the checkpoint file. map_location ensures it loads to the right device
    # (e.g., if the checkpoint was saved on GPU but we are loading on CPU).
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle both old and new checkpoint formats (backward compatibility)
    config = checkpoint.get("config")
    if config is not None and isinstance(config, dict):
        # New format: full ExperimentConfig saved as a nested dict
        model_config_dict = config.get("model", {})
        training_config = config.get("training", {})
        experiment_name = training_config.get("experiment_name", "transformer")
    else:
        # Old format: model_config saved directly (legacy checkpoints)
        model_config_dict = checkpoint.get("model_config", {})
        experiment_name = checkpoint.get("experiment_name", "transformer")

    if not model_config_dict:
        raise ValueError("Checkpoint missing model configuration")

    # The tokenizer provides vocab_sizes, which the model needs for its embedding
    # layers. This is not saved in the checkpoint because it depends on the tokenizer.
    model_config_dict["vocab_sizes"] = tokenizer.vocab_sizes

    # Filter out any unexpected keys (e.g., from older checkpoint formats) to prevent
    # errors when constructing ModelConfig
    from dataclasses import fields as dc_fields
    valid_fields = {f.name for f in dc_fields(ModelConfig)}
    filtered_config = {k: v for k, v in model_config_dict.items() if k in valid_fields}

    # Build the model architecture from the config
    model_config = ModelConfig(**filtered_config)

    # Create a fresh model and load the trained weights into it
    model = Phase1Model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    return model, model_config, experiment_name


def create_test_loader(
    config: ExperimentConfig,
    tokenizer: PBPTokenizer,
) -> DataLoader:
    """Create test data loader."""
    logger = logging.getLogger(__name__)

    logger.info(f"Creating test dataset for seasons: {config.data.test_seasons}")
    test_dataset = NBAGameDataset(
        tokenizer=tokenizer,
        seasons=config.data.test_seasons,
        n_history_games=config.data.n_history_games,
        min_history_games=config.data.min_history_games,
        cache_size=config.data.cache_size,
    )
    logger.info(f"Test dataset: {len(test_dataset)} games")

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        collate_fn=collate_games,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory and torch.cuda.is_available(),
    )

    return test_loader


def evaluate_checkpoint(
    checkpoint_path: str,
    config: ExperimentConfig,
    device: str = "cuda",
    visualize: bool = False,
    output_dir: str = None,
) -> EvaluationResults:
    """
    Evaluate a trained checkpoint on test data and optionally generate plots.

    This is the main evaluation workflow:
        1. Load the tokenizer (needed to reconstruct the model)
        2. Load the model from the checkpoint (config + weights)
        3. Create a test data loader
        4. Run the model on all test data and compute metrics
        5. Optionally save results and generate visualization plots

    Args:
        checkpoint_path: Path to the .pt checkpoint file
        config: Experiment configuration (provides data/tokenizer paths)
        device: Device for evaluation ("cuda", "cpu", etc.)
        visualize: Whether to generate calibration plots, error distributions, etc.
        output_dir: Directory to save results and plots

    Returns:
        EvaluationResults with all computed metrics
    """
    logger = logging.getLogger(__name__)

    # Step 1: Load the tokenizer (must exist — cannot build during evaluation)
    tokenizer = PBPTokenizer()
    tokenizer_path = Path(config.data.tokenizer_path)
    if tokenizer_path.exists():
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        tokenizer.load(str(tokenizer_path))
    else:
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

    # Step 2: Reconstruct the model from the checkpoint
    # The tokenizer provides vocab_sizes needed to build the embedding layers
    model, model_config, experiment_name = load_checkpoint(
        checkpoint_path, tokenizer, device
    )

    # Step 3: Create test data loader
    test_loader = create_test_loader(config, tokenizer)

    # Step 4: Run evaluation — feeds all test data through the model and computes metrics
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        model_name=experiment_name,
        checkpoint_path=checkpoint_path,
    )

    results = evaluator.evaluate()
    evaluator.print_summary()

    # Step 5: Save results and optionally generate visualization plots
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save numeric results as JSON for later analysis
        results.save(str(output_path / "evaluation_results.json"))

        # Generate plots: calibration curves, error distributions, coverage plots, etc.
        if visualize:
            logger.info("Generating visualizations...")
            predictions_df = evaluator.get_predictions_df()

            viz = Visualizer(predictions_df, output_dir=str(output_path / "plots"))
            viz.plot_all()

    return results


def run_ablation(
    config_path: str,
    device: str = "cuda",
    output_dir: str = "experiments/ablation",
):
    """
    Run a full ablation study: systematically vary one hyperparameter at a time.

    ABLATION STUDY EXPLAINED:
        An ablation study answers questions like:
        - "Does cross-attention fusion actually help, or would simple concatenation work?"
        - "Do we need 5 history games, or would 3 be enough?"
        - "Is the large model better than the medium model, or does it overfit?"

        For each question, we train a model variant with ONLY that one thing
        changed, keeping everything else identical to the baseline. By comparing
        results, we can isolate the effect of each design decision.

    Args:
        config_path: Path to base config (the "baseline" all variants are compared to)
        device: Device for training ("cuda", "cpu")
        output_dir: Directory to save all experiment results
    """
    from src.transformer.evaluation.ablation import AblationRunner

    logger = logging.getLogger(__name__)
    logger.info(f"Starting ablation study with base config: {config_path}")

    # Load base config
    if config_path:
        config = ExperimentConfig.from_yaml(config_path)
    else:
        config = get_medium_config()

    # Run ablation
    runner = AblationRunner(
        base_config=config,
        output_dir=output_dir,
        device=device,
    )

    results = runner.run_all()
    report = runner.generate_report()

    print("\n" + "=" * 60)
    print("ABLATION STUDY COMPLETE")
    print("=" * 60)
    print(f"Total experiments: {len(results)}")
    print(f"Results saved to: {output_dir}")
    print(report)


def main():
    parser = argparse.ArgumentParser(description="Evaluate NBA Transformer Model")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, cpu, mps)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output files",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization plots",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run ablation study instead of single evaluation",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test with minimal data",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Determine device: prefer GPU (CUDA > MPS) for speed, fall back to CPU
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"  # NVIDIA GPU
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"   # Apple Silicon GPU
    else:
        device = "cpu"   # No GPU available

    logger.info(f"Using device: {device}")

    # Load configuration (same three modes as training script)
    if args.config:
        config = ExperimentConfig.from_yaml(args.config)
    elif args.quick_test:
        # Quick test: minimal data for verifying the eval pipeline works
        config = get_small_config()
        config.data.train_seasons = ["2023-2024"]
        config.data.val_seasons = ["2023-2024"]
        config.data.test_seasons = ["2023-2024"]
        config.data.tokenizer_path = "data/tokenized/test_tokenizer.json"
        config.data.batch_size = 8
        config.data.num_workers = 0
    else:
        config = get_medium_config()

    # Two main modes: ablation study or single checkpoint evaluation
    if args.ablation:
        # Ablation mode: train and evaluate many model variants
        output_dir = args.output_dir or "experiments/ablation"
        run_ablation(
            config_path=args.config,
            device=device,
            output_dir=output_dir,
        )
    elif args.checkpoint:
        # Single evaluation mode: load one checkpoint and evaluate on test data
        # Default output dir is a subfolder next to the checkpoint
        output_dir = args.output_dir or str(
            Path(args.checkpoint).parent / "evaluation"
        )
        evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            config=config,
            device=device,
            visualize=args.visualize,
            output_dir=output_dir,
        )
    else:
        parser.error("Either --checkpoint or --ablation must be specified")

    return 0


if __name__ == "__main__":
    sys.exit(main())
