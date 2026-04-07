"""
DataLoader utilities for NBA Transformer Model.

=============================================================================
WHAT IS A DATALOADER AND WHY DO WE NEED IT?
=============================================================================

In traditional data science, you might load an entire dataset into memory
(e.g., pd.read_csv()) and process it all at once. Deep learning changes this
paradigm for several reasons:

1. MEMORY CONSTRAINTS: Neural networks (especially transformers) require
   massive amounts of GPU memory. A single forward pass through a transformer
   might need 10-100x the memory of the raw data due to intermediate
   activations and attention matrices. We can't fit entire datasets in GPU
   memory, so we process data in small chunks called "batches."

2. STOCHASTIC GRADIENT DESCENT: Deep learning optimizes via gradient descent,
   updating model weights after seeing each batch. Processing smaller batches
   provides more frequent weight updates, which often leads to better
   generalization than processing the entire dataset at once.

3. PARALLELIZATION: DataLoaders can prefetch and preprocess data in background
   worker processes while the GPU is busy training on the current batch.
   This keeps the expensive GPU hardware fully utilized.

A DataLoader wraps around a Dataset object and handles:
- Batching: Grouping individual samples into batches
- Shuffling: Randomizing data order (when appropriate)
- Prefetching: Loading next batches while current batch trains
- Memory pinning: Optimizing CPU-to-GPU data transfer

=============================================================================
THE FACTORY PATTERN IN THIS MODULE
=============================================================================

This module uses the "factory pattern" - functions that create and return
configured objects. Instead of manually constructing DataLoaders everywhere
in your training code, you call create_dataloaders() which:

1. Creates Dataset objects with proper configuration
2. Wraps them in DataLoaders with appropriate settings
3. Returns a dictionary for easy access to train/val/test loaders

This pattern ensures consistency and reduces boilerplate code.

=============================================================================
USAGE EXAMPLE
=============================================================================

    from src.transformer.dataloader import create_dataloaders

    tokenizer = PBPTokenizer()
    tokenizer.load("data/tokenized/tokenizer.json")

    dataloaders = create_dataloaders(
        tokenizer=tokenizer,
        batch_size=32,
        n_history_games=5,
    )

    for batch in dataloaders["train"]:
        # Training loop
        pass
"""

import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from src.transformer.dataset import NBAGameDataset, collate_games
from src.transformer.tokenizer import PBPTokenizer


def create_dataloaders(
    tokenizer: PBPTokenizer,
    batch_size: int = 32,
    n_history_games: int = 5,
    min_history_games: int = 3,
    num_workers: int = 0,
    pin_memory: bool = True,
    train_seasons: Optional[list[str]] = None,
    val_seasons: Optional[list[str]] = None,
    test_seasons: Optional[list[str]] = None,
    db_path: Optional[str] = None,
    cache_size: int = 1000,
) -> dict[str, DataLoader]:
    """
    Create train, validation, and test DataLoaders using the factory pattern.

    This is the main entry point for creating data pipelines. It instantiates
    three Dataset objects (one each for train/val/test), wraps each in a
    DataLoader with appropriate settings, and returns them in a dictionary.

    WHY A FACTORY FUNCTION?
    -----------------------
    Creating DataLoaders requires many configuration decisions that should be
    consistent across your codebase:
    - Training loaders should drop incomplete final batches (drop_last=True)
    - Validation/test loaders should keep all samples (drop_last=False)
    - All loaders need the same collate function
    - Memory optimization settings should match your hardware

    By centralizing this logic in a factory function, we ensure consistency
    and make it easy to update settings project-wide.

    TRAIN VS. VAL VS. TEST SPLITS
    -----------------------------
    - TRAIN: Used to update model weights. Typically 70-80% of data.
    - VALIDATION (VAL): Used to tune hyperparameters and detect overfitting
      during training. Model never trains on this data. ~10-15% of data.
    - TEST: Held out completely until final evaluation. Gives unbiased
      estimate of model performance. ~10-15% of data.

    For temporal data like NBA games, we typically split by time:
    - Train: 2018-2022 seasons
    - Val: 2022-2023 season
    - Test: 2023-2024 season

    This prevents "future leakage" where the model learns patterns from
    games that haven't happened yet in the training timeline.

    Args:
        tokenizer: Initialized PBPTokenizer that converts play-by-play text
            into numerical token IDs the model can process.
        batch_size: Number of games per batch. Larger batches use more memory
            but provide more stable gradient estimates. Common values: 8-64.
            Start small (8-16) and increase until you hit memory limits.
        n_history_games: Number of previous games to include as context for
            each team. More history = more context but more memory/compute.
        min_history_games: Minimum historical games required. Games early in
            a season might be excluded if teams haven't played enough games.
        num_workers: Number of background processes for data loading.
            - 0: Load data in main process (simplest, good for debugging)
            - 1-4: Parallel loading (faster, recommended for training)
            - Too many: Diminishing returns, may cause memory issues
            See detailed explanation below in the DataLoader construction.
        pin_memory: Whether to use pinned (page-locked) memory for faster
            CPU-to-GPU transfer. See detailed explanation below.
        train_seasons: List of seasons for training, e.g., ["2021-2022"].
            If None, uses default split from dataset.
        val_seasons: List of seasons for validation. If None, uses default.
        test_seasons: List of seasons for testing. If None, uses default.
        db_path: Path to SQLite database with play-by-play data.
        cache_size: Number of processed games to cache in memory. Higher
            values speed up repeated access but use more RAM.

    Returns:
        Dictionary with keys "train", "val", "test", each mapping to a
        DataLoader. Missing splits (if seasons list is empty) are omitted.

    Example:
        >>> dataloaders = create_dataloaders(tokenizer, batch_size=32)
        >>> train_loader = dataloaders["train"]
        >>> for batch in train_loader:
        ...     loss = model(batch)
        ...     loss.backward()
    """
    # Dictionary to collect our DataLoaders. We'll add train, val, and test
    # as we create them.
    dataloaders = {}

    # =========================================================================
    # CREATE TRAINING DATALOADER
    # =========================================================================
    # The training loader has special settings optimized for gradient descent:
    # - shuffle=True: Breaks temporal correlations for better SGD gradients
    # - drop_last=True: Ensures all batches have the same size
    if train_seasons is not None and train_seasons != []:
        # First, create the Dataset - the object that knows how to fetch
        # and preprocess individual game samples.
        train_dataset = NBAGameDataset(
            tokenizer=tokenizer,
            seasons=train_seasons,
            # Explicit seasons provided, so don't use the built-in split
            split=None,
            n_history_games=n_history_games,
            min_history_games=min_history_games,
            db_path=db_path,
            cache_size=cache_size,
        )

        # Now wrap the Dataset in a DataLoader, which handles batching,
        # shuffling, parallel loading, and memory optimization.
        dataloaders["train"] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            # -----------------------------------------------------------------
            # SHUFFLE=TRUE: SAFE FOR THIS ARCHITECTURE
            # -----------------------------------------------------------------
            # Each sample is self-contained: __getitem__ fetches history via
            # SQL with `date < target_date`, so temporal integrity is baked
            # into each sample regardless of iteration order. Shuffling the
            # order of index access cannot cause data leakage.
            #
            # Future leakage is prevented by the train/val/test SEASON split,
            # not by iteration order within a split.
            #
            # Shuffling is beneficial because:
            # 1. Breaks correlations between adjacent chronological games
            #    (same teams, overlapping history windows) — improves gradient
            #    quality under SGD.
            # 2. Prevents ordering bias: without shuffling, early-season games
            #    always receive the largest gradients (before LR decay), biasing
            #    the model toward early-season patterns.
            # 3. Each epoch traverses the data in a different order, providing
            #    beneficial stochasticity that helps generalization.
            shuffle=True,
            # -----------------------------------------------------------------
            # COLLATE_FN: COMBINING SAMPLES INTO BATCHES
            # -----------------------------------------------------------------
            # When a DataLoader fetches multiple samples to form a batch,
            # it needs to combine them somehow. The collate_fn handles this.
            #
            # DEFAULT BEHAVIOR (without custom collate_fn):
            # PyTorch stacks tensors along a new batch dimension. This works
            # if all samples have identical shapes.
            #
            # WHY WE NEED A CUSTOM COLLATE:
            # NBA games have VARIABLE lengths - some games have 300 plays,
            # others have 600. You can't stack tensors of different sizes.
            #
            # Our collate_games function:
            # 1. Filters out failed samples (home_history is None)
            # 2. Finds the longest sequence in the batch
            # 3. PADS shorter sequences with zeros to match
            # 4. Preserves game_lengths so the model can build attention masks
            # 5. Returns a dictionary with properly batched tensors
            #
            # Note: collate_games does NOT create attention masks itself.
            # It preserves game_lengths in the batch, and the EventEncoder
            # uses those to build src_key_padding_mask during the forward pass.
            collate_fn=collate_games,
            # -----------------------------------------------------------------
            # NUM_WORKERS: PARALLEL DATA LOADING
            # -----------------------------------------------------------------
            # Data loading can be a bottleneck. While GPU trains on batch N,
            # CPU could be preparing batch N+1. num_workers controls this:
            #
            # num_workers=0:
            #   - Data loaded in main process
            #   - Simple, good for debugging
            #   - GPU may wait idle while data loads
            #
            # num_workers=1-4:
            #   - Spawn worker processes that prefetch data
            #   - While GPU trains, workers prepare next batches
            #   - Keeps GPU fed with data, maximizing utilization
            #   - Each worker uses additional RAM (copies of dataset)
            #
            # num_workers > 4:
            #   - Diminishing returns on most systems
            #   - Can cause memory pressure and context switching overhead
            #   - Rule of thumb: 2-4 workers, or num_cpus // num_gpus
            #
            # IMPORTANT: On Windows, num_workers > 0 requires specific
            # if __name__ == "__main__": guards due to multiprocessing quirks.
            num_workers=num_workers,
            # -----------------------------------------------------------------
            # PIN_MEMORY: FASTER CPU-TO-GPU TRANSFER
            # -----------------------------------------------------------------
            # GPU memory and CPU memory are separate. Moving data between them
            # takes time. Here's how pin_memory helps:
            #
            # NORMAL (pin_memory=False):
            #   CPU RAM (pageable) -> Staging area -> GPU RAM
            #   The OS might swap pageable memory to disk, causing delays.
            #
            # PINNED (pin_memory=True):
            #   CPU RAM (pinned/page-locked) -> GPU RAM (direct transfer)
            #   Pinned memory can't be swapped, enabling faster DMA transfer.
            #
            # WHEN TO USE:
            #   - pin_memory=True: When training on GPU (default)
            #   - pin_memory=False: When training on CPU (no GPU transfer)
            #
            # We check torch.cuda.is_available() to automatically disable
            # pinning when no GPU is present (pinning without GPU = waste).
            #
            # MEMORY COST: Pinned memory is slightly more expensive to
            # allocate and reduces available system RAM for other uses.
            pin_memory=pin_memory and torch.cuda.is_available(),
            # -----------------------------------------------------------------
            # DROP_LAST: HANDLING INCOMPLETE FINAL BATCHES
            # -----------------------------------------------------------------
            # If your dataset has 100 samples and batch_size=32:
            #   - Batches 1-3: 32 samples each (96 total)
            #   - Batch 4: Only 4 samples (100 - 96 = 4)
            #
            # drop_last=True discards that incomplete final batch.
            #
            # WHY FOR TRAINING:
            # 1. Batch normalization: Statistics are unreliable with tiny batches
            # 2. Consistent gradient scale: Prevents final batch from having
            #    outsized influence (4 samples vs 32)
            # 3. Distributed training: All processes need same batch sizes
            #
            # We only lose a few samples, which is negligible for large datasets.
            drop_last=True,
        )
        logging.info(f"Created train loader: {len(train_dataset)} games")

    # =========================================================================
    # CREATE VALIDATION DATALOADER
    # =========================================================================
    # Validation loader differs from training:
    # - drop_last=False: Evaluate on ALL samples, even incomplete final batch
    # - shuffle=False: Not needed since we're just evaluating, not training
    if val_seasons is not None and val_seasons != []:
        val_dataset = NBAGameDataset(
            tokenizer=tokenizer,
            seasons=val_seasons,
            # Explicit seasons provided, so don't use the built-in split
            split=None,
            n_history_games=n_history_games,
            min_history_games=min_history_games,
            db_path=db_path,
            cache_size=cache_size,
        )
        dataloaders["val"] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # No shuffling for validation - evaluate in order
            collate_fn=collate_games,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            # drop_last=False means we keep all samples for complete evaluation
            # We want validation metrics computed over the entire val set
            drop_last=False,
        )
        logging.info(f"Created val loader: {len(val_dataset)} games")

    # =========================================================================
    # CREATE TEST DATALOADER
    # =========================================================================
    # Test loader is configured identically to validation.
    # The key difference is WHEN it's used (only for final evaluation).
    if test_seasons is not None and test_seasons != []:
        test_dataset = NBAGameDataset(
            tokenizer=tokenizer,
            seasons=test_seasons,
            # Explicit seasons provided, so don't use the built-in split
            split=None,
            n_history_games=n_history_games,
            min_history_games=min_history_games,
            db_path=db_path,
            cache_size=cache_size,
        )
        dataloaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_games,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            drop_last=False,
        )
        logging.info(f"Created test loader: {len(test_dataset)} games")

    return dataloaders


def create_single_dataloader(
    tokenizer: PBPTokenizer,
    seasons: list[str],
    batch_size: int = 32,
    n_history_games: int = 5,
    min_history_games: int = 3,
    num_workers: int = 0,
    shuffle: bool = False,
    db_path: Optional[str] = None,
    cache_size: int = 1000,
) -> DataLoader:
    """
    Create a single DataLoader for specific seasons.

    This is a simpler alternative to create_dataloaders() when you only need
    one loader. Useful for:
    - Custom splits (e.g., training on only Western Conference teams)
    - Ablation studies (testing with different season combinations)
    - Inference on new data
    - Debugging with a subset of data

    WHY A SEPARATE FUNCTION?
    ------------------------
    The main create_dataloaders() function enforces certain conventions
    (train gets drop_last=True, etc.). This function gives you full control
    over all parameters for specialized use cases.

    Args:
        tokenizer: Initialized PBPTokenizer for converting text to tokens.
        seasons: List of seasons to include, e.g., ["2022-2023", "2023-2024"].
        batch_size: Number of games per batch.
        n_history_games: Number of previous games per team for context.
        min_history_games: Minimum historical games required.
        num_workers: Number of background data loading processes.
        shuffle: Whether to shuffle data order.
            WARNING: Set to False (default) for temporal data like NBA games.
            Only set True if you understand the implications for your use case
            (e.g., you're doing non-temporal analysis).
        db_path: Path to SQLite database with play-by-play data.
        cache_size: Number of processed games to cache in memory.

    Returns:
        A configured DataLoader ready for iteration.

    Example:
        >>> # Create loader for a single season
        >>> loader = create_single_dataloader(
        ...     tokenizer=tokenizer,
        ...     seasons=["2025-2026"],
        ...     batch_size=16,
        ... )
        >>> for batch in loader:
        ...     predictions = model(batch)
    """
    # Create the underlying dataset with the specified configuration
    dataset = NBAGameDataset(
        tokenizer=tokenizer,
        seasons=seasons,
        n_history_games=n_history_games,
        min_history_games=min_history_games,
        db_path=db_path,
        cache_size=cache_size,
    )

    # Wrap dataset in DataLoader with standard settings
    # Note: We don't set drop_last here, letting caller decide via batch_size
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_games,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def get_sample_batch(
    tokenizer: PBPTokenizer,
    seasons: list[str] = ["2023-2024"],
    batch_size: int = 4,
    n_history_games: int = 5,
    db_path: Optional[str] = None,
) -> dict:
    """
    Get a single sample batch for testing and debugging.

    This utility function is invaluable during development. It lets you
    quickly inspect the structure and content of batches without setting
    up a full training loop.

    COMMON USE CASES:
    -----------------
    1. SHAPE DEBUGGING: Check tensor dimensions match model expectations
       >>> batch = get_sample_batch(tokenizer)
       >>> print(batch["home_history"]["action_type_ids"].shape)
       torch.Size([4, 5, 500])  # [batch, history_games, max_plays]

    2. DATA INSPECTION: Verify tokenization and preprocessing are correct
       >>> batch = get_sample_batch(tokenizer)
       >>> print(batch["game_ids"])  # Check game IDs are sensible
       >>> print(batch["target_home_scores"])  # Verify labels exist

    3. MODEL TESTING: Run a forward pass to check model compiles
       >>> batch = get_sample_batch(tokenizer)
       >>> output = model(batch)  # Will error if shapes mismatch

    4. MEMORY PROFILING: Estimate GPU memory before full training
       >>> batch = get_sample_batch(tokenizer, batch_size=32)
       >>> mem = estimate_memory_per_batch(batch)

    Args:
        tokenizer: Initialized PBPTokenizer.
        seasons: Which seasons to sample from. Defaults to recent season.
        batch_size: Number of games in the sample batch. Keep small (4-8)
            for quick testing; increase to match training batch for
            accurate memory estimates.
        n_history_games: Number of historical games per team.
        db_path: Database path. Uses default if None.

    Returns:
        A single batch dictionary with the same structure as training batches.
        Keys typically include:
        - "game_ids": List of game identifiers
        - "home_history": Dict of tensors for home team's past games
        - "away_history": Dict of tensors for away team's past games
        - "target_home_scores": Home team final scores (labels)
        - "target_away_scores": Away team final scores (labels)
        - Additional metadata and attention masks

    Example:
        >>> batch = get_sample_batch(tokenizer, batch_size=8)
        >>> print(f"Batch has {len(batch['game_ids'])} games")
        >>> print(f"History tensor shape: {batch['home_history']['action_type_ids'].shape}")
    """
    # Create a minimal dataloader - we only need one batch
    # Small cache_size since we won't iterate through the whole dataset
    loader = create_single_dataloader(
        tokenizer=tokenizer,
        seasons=seasons,
        batch_size=batch_size,
        n_history_games=n_history_games,
        db_path=db_path,
        cache_size=batch_size * 2,  # Cache just enough for one batch + buffer
    )

    # next(iter(loader)) fetches the first batch from the iterator
    # This is a common Python pattern for getting one item from an iterable
    return next(iter(loader))


def estimate_memory_per_batch(
    batch: dict,
    include_gradients: bool = True,
) -> dict[str, float]:
    """
    Estimate GPU memory usage for a batch in megabytes (MB).

    GPU memory is the most common bottleneck in deep learning. This function
    helps you understand memory consumption BEFORE committing to a long
    training run that might crash with Out of Memory (OOM) errors.

    HOW NEURAL NETWORK MEMORY USAGE WORKS:
    ---------------------------------------
    When training a neural network, GPU memory is used for:

    1. MODEL PARAMETERS: The actual weights (relatively small)
       - A 100M parameter model = ~400 MB in float32

    2. ACTIVATIONS (Forward Pass): Intermediate results at each layer
       - These are HUGE for transformers due to attention matrices
       - Scales with sequence_length^2 for self-attention
       - This function estimates this via the input batch

    3. GRADIENTS (Backward Pass): Same size as activations
       - Needed to compute weight updates via backpropagation
       - Roughly DOUBLES memory during training
       - This is why include_gradients defaults to True

    4. OPTIMIZER STATES: Momentum, variance (for Adam)
       - Adam uses 2x the parameter memory for momentum + variance
       - Often a fixed cost, not measured here

    TYPICAL MEMORY BREAKDOWN:
    -------------------------
    For a transformer model:
    - Input batch tensors: 10-50 MB (what this function measures)
    - Attention matrices: 100 MB - 10 GB (depends on sequence length)
    - Gradients: Same as activations
    - Optimizer: 2-3x parameter count

    The input batch size this function measures is just the starting point.
    Actual GPU usage during training will be 10-100x higher due to
    activations, attention matrices, and gradients.

    USING THIS FUNCTION:
    --------------------
    1. Start with a small batch_size (4-8)
    2. Run estimate_memory_per_batch() to get baseline input memory
    3. Monitor actual GPU usage with nvidia-smi during training
    4. If OOM, reduce batch_size or sequence length

    Args:
        batch: A batch dictionary from the dataloader (from get_sample_batch
            or iterating through a DataLoader).
        include_gradients: If True, double the estimate to account for
            gradient storage during training. Set False for inference-only
            memory estimation.

    Returns:
        Dictionary with memory estimates in MB for each component:
        - Individual keys for each tensor in the batch
        - "total_forward": Total input tensor memory
        - "total_with_gradients": Estimated training memory (2x forward)

    Example:
        >>> batch = get_sample_batch(tokenizer, batch_size=32)
        >>> mem = estimate_memory_per_batch(batch)
        >>> print(f"Input memory: {mem['total_forward']:.1f} MB")
        >>> print(f"With gradients: {mem['total_with_gradients']:.1f} MB")

        # Rule of thumb: If input is 50 MB, expect 500+ MB actual GPU usage
        # during training due to activations and attention matrices.
    """

    def tensor_memory(t: torch.Tensor) -> float:
        """
        Calculate memory usage of a single tensor in MB.

        Memory = num_elements * bytes_per_element / (1024^2)

        For reference:
        - float32: 4 bytes per element
        - float16: 2 bytes per element
        - int64: 8 bytes per element
        - int32: 4 bytes per element
        """
        bytes_total = t.element_size() * t.nelement()
        megabytes = bytes_total / (1024 * 1024)
        return megabytes

    def dict_memory(d: dict) -> float:
        """
        Calculate total memory for a dictionary containing tensors.

        Recursively sums memory for all tensor values in the dict.
        Non-tensor values (strings, ints, etc.) are ignored.
        """
        total = 0.0
        for value in d.values():
            if isinstance(value, torch.Tensor):
                total += tensor_memory(value)
        return total

    # Calculate memory for each component in the batch
    estimates = {}
    total = 0.0

    for key, val in batch.items():
        if isinstance(val, dict):
            # Nested dict (e.g., home_history, away_history)
            mem = dict_memory(val)
        elif isinstance(val, torch.Tensor):
            # Direct tensor
            mem = tensor_memory(val)
        else:
            # Skip non-tensor items (strings, lists of game IDs, etc.)
            continue

        estimates[key] = mem
        total += mem

    # Total memory for forward pass (just input tensors)
    estimates["total_forward"] = total

    if include_gradients:
        # During training, PyTorch stores gradients for backpropagation.
        # For most tensors, this roughly doubles memory usage.
        #
        # NOTE: This is a simplified estimate. Actual training memory
        # also includes:
        # - Intermediate activations at each layer
        # - Attention score matrices (huge for transformers)
        # - Optimizer states (Adam uses 2x parameter memory)
        #
        # Real GPU usage during training is typically 5-20x the input size.
        estimates["total_with_gradients"] = total * 2

    return estimates


def test_dataloader():
    """
    Test dataloader utilities with real data.

    This function serves as both a test and a usage example. Run it to
    verify your data pipeline is working correctly before starting training.

    What it tests:
    1. Tokenizer loading/building
    2. Sample batch retrieval
    3. Memory estimation
    4. Multi-batch iteration

    Returns:
        0 on success (standard Unix convention for exit codes)

    Usage:
        python -m src.transformer.dataloader
    """
    # Set up logging to see progress messages
    logging.basicConfig(level=logging.INFO)

    # Load or build tokenizer
    # The tokenizer converts play-by-play text into numerical token IDs
    tokenizer_path = "data/tokenized/test_tokenizer.json"
    tokenizer = PBPTokenizer()

    try:
        tokenizer.load(tokenizer_path)
        print("Loaded existing tokenizer")
    except FileNotFoundError:
        # If no tokenizer exists, build one from the database
        # This scans all plays and creates a vocabulary of unique tokens
        print("Building tokenizer...")
        tokenizer.build_vocab_from_db(seasons=["2023-2024"], min_count=5)
        tokenizer.save(tokenizer_path)

    # =========================================================================
    # TEST 1: Get a sample batch
    # =========================================================================
    print("\n=== Testing Sample Batch ===")
    batch = get_sample_batch(
        tokenizer=tokenizer,
        seasons=["2023-2024"],
        batch_size=8,
        n_history_games=5,
    )

    # Inspect the batch structure
    print(f"Batch keys: {batch.keys()}")
    print(f"Game IDs: {batch['game_ids']}")
    print(f"Scores: {batch['target_home_scores']} - {batch['target_away_scores']}")

    # =========================================================================
    # TEST 2: Estimate memory usage
    # =========================================================================
    print("\n=== Memory Estimates ===")
    mem = estimate_memory_per_batch(batch)
    for key, val in mem.items():
        print(f"  {key}: {val:.2f} MB")

    # =========================================================================
    # TEST 3: Iterate through multiple batches
    # =========================================================================
    print("\n=== Testing Multiple Batches ===")
    loader = create_single_dataloader(
        tokenizer=tokenizer,
        seasons=["2023-2024"],
        batch_size=16,
        n_history_games=5,
    )

    # len(loader) gives total number of batches in the dataset
    print(f"Total batches: {len(loader)}")

    # Iterate through first few batches to verify everything works
    for i, batch in enumerate(loader):
        if i >= 3:
            break
        print(
            f"  Batch {i+1}: {len(batch['game_ids'])} games, "
            f"home_history shape {batch['home_history']['action_type_ids'].shape}"
        )

    print("\n=== Test Complete ===")
    return 0


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================
# When this file is run directly (python -m src.transformer.dataloader),
# execute the test function. When imported, this block is skipped.
if __name__ == "__main__":
    import sys

    # sys.exit() with the return value sets the process exit code
    # Exit code 0 = success, non-zero = error
    sys.exit(test_dataloader())
