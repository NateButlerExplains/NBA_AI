# Phase 1 Implementation Plan

## Status: Ready to Begin

**Goal**: Build supervised temporal baseline to validate sequence modeling approach

**Success Criteria**:
- Beat XGBoost baseline (MAE < 10.1 on spread prediction)
- Ablation shows sequences add 1+ point improvement over aggregated features
- Model trains successfully on full 27-season dataset

**Timeline**: 2-3 weeks (flexible based on complexity)

---

## Architecture Overview

### Minimal-Context Sequence Model

```
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 1 ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  STREAM 1: Historical PBP Sequences                          │
│  ┌────────────────────────────────────────────────────┐     │
│  │  For each team (last N games):                     │     │
│  │    Raw PBP events → Event Tokenizer                │     │
│  │                         ↓                          │     │
│  │    [action, player, clock, score_diff, timestamp]  │     │
│  │                         ↓                          │     │
│  │              Event Encoder (4-layer Transformer)   │     │
│  │                         ↓                          │     │
│  │              Game Embedding (pooling/[CLS])        │     │
│  │                         ↓                          │     │
│  │      Temporal Attention over game sequence         │     │
│  │                         ↓                          │     │
│  │              Team History Embedding                │     │
│  └────────────────────────────────────────────────────┘     │
│                         ↓                                    │
│  STREAM 2: Expected Roster (Minimal)                         │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Player IDs only: [p1, p2, ..., p13] per team     │     │
│  │                         ↓                          │     │
│  │        Player ID → Learned Embedding (64-d)        │     │
│  │                         ↓                          │     │
│  │        Set Transformer (handles variable size)     │     │
│  │                         ↓                          │     │
│  │              Roster Embedding per team             │     │
│  └────────────────────────────────────────────────────┘     │
│                         ↓                                    │
│  MINIMAL CONTEXT: Game Facts                                 │
│  ┌────────────────────────────────────────────────────┐     │
│  │  - Game timestamp                                  │     │
│  │  - Team locations (home city, away city)           │     │
│  │  - Home/away indicator                             │     │
│  │                         ↓                          │     │
│  │              Embedded as part of sequences         │     │
│  └────────────────────────────────────────────────────┘     │
│                         ↓                                    │
│  FUSION LAYER                                                │
│  ┌────────────────────────────────────────────────────┐     │
│  │  [Home History; Away History; Home Roster; Away    │     │
│  │   Roster; Context]                                 │     │
│  │                         ↓                          │     │
│  │        Cross-Attention Fusion (2-layer)            │     │
│  │                         ↓                          │     │
│  │              Matchup Representation                │     │
│  └────────────────────────────────────────────────────┘     │
│                         ↓                                    │
│  PROBABILISTIC PREDICTION HEADS                              │
│  ┌────────────────────────────────────────────────────┐     │
│  │  ┌──────────────┐  ┌──────────────┐                │     │
│  │  │ Spread Head  │  │ Score Heads  │                │     │
│  │  │ μ_spread, σ  │  │ μ_h, σ_h     │                │     │
│  │  │              │  │ μ_a, σ_a     │                │     │
│  │  └──────────────┘  └──────────────┘                │     │
│  │                                                     │     │
│  │  Loss: Negative Log-Likelihood (Gaussian)          │     │
│  │        + MSE for interpretability                  │     │
│  └────────────────────────────────────────────────────┘     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Week 1: Data Pipeline & Infrastructure

**Goal**: Get clean, tokenized sequences ready for model

#### 1.1 Event Tokenization (2-3 days)

**Input**: `PbP_Logs` table rows

**Output**: Tokenized event tensors

**Tasks**:
- [ ] Design event vocabulary
  - Action types (~50 unique)
  - Player IDs (5,118 players)
  - Clock encoding (sinusoidal)
  - Score differential
  - Period (1-4 + OT)
- [ ] Build tokenizer class
- [ ] Add minimal temporal context (timestamps, locations)
- [ ] Handle missing/malformed events
- [ ] Test on sample games

**Key Decisions**:
- Embedding dimensions per feature
- How to handle unknown players (new rookies)
- Max sequence length per game

#### 1.2 Game Sequence Construction (2 days)

**Input**: Tokenized events

**Output**: Historical game sequences per team

**Tasks**:
- [ ] Create sequence builder
  - Fetch last N games for each team
  - Concatenate/pad sequences
  - Handle variable game lengths
- [ ] Add game-level metadata
  - Timestamp
  - Final score
  - Home/away teams
- [ ] Implement efficient caching
- [ ] Build dataset class (PyTorch)

**Key Decisions**:
- How many historical games (N=5? N=10?)
- Padding strategy
- Memory management for 37K games

#### 1.3 Train/Val/Test Split (1 day)

**Critical**: Temporal split (no data leakage)

**Tasks**:
- [ ] Implement chronological splitting
  - Train: 2000-01 through 2022-23
  - Val: 2023-24 season
  - Test: 2024-25 + 2025-26 (current)
- [ ] Verify no overlap
- [ ] Document split statistics
- [ ] Create DataLoader with proper batching

**Key Decisions**:
- Exact cutoff dates
- How to handle team histories that span train/val boundary

#### 1.4 Roster & Context Features (1 day)

**Tasks**:
- [ ] Extract expected rosters
  - From InjuryReports for inference
  - From PBP (who played) for training
- [ ] Minimal context extraction
  - Game timestamps from `Games` table
  - Team locations (static mapping)
- [ ] Create roster padding/masking
- [ ] Test variable roster sizes

---

### Week 2: Model Implementation

**Goal**: Working transformer architecture

#### 2.1 Event Encoder (2 days)

**Tasks**:
- [ ] Implement embedding layer
  - Action type embedding
  - Player ID embedding
  - Clock encoding (sinusoidal)
  - Score context
  - Positional encoding
- [ ] Build Transformer encoder
  - 4 layers
  - Multi-head attention (8 heads?)
  - Feed-forward network
  - Layer norm + residual connections
- [ ] Game-level pooling
  - [CLS] token or average pooling?
- [ ] Test on sample sequences

**Key Decisions**:
- Hidden dimension (256? 512?)
- Number of attention heads
- Dropout rate
- Activation function

#### 2.2 Set Transformer for Rosters (1 day)

**Tasks**:
- [ ] Implement Set Transformer
  - Handles variable roster size
  - Permutation invariant
  - 2-layer architecture
- [ ] Player embedding lookup
- [ ] Test with different roster sizes (10-15 players)

**Key Decisions**:
- Number of induced points
- Embedding dimension for player IDs

#### 2.3 Temporal Attention & Fusion (2 days)

**Tasks**:
- [ ] Temporal attention over games
  - Cross-attention mechanism
  - Attend to last N games
- [ ] Fusion layer
  - Concatenate home/away histories
  - Concatenate rosters
  - Add minimal context
  - Cross-attention fusion (2-layer)
- [ ] Test end-to-end forward pass

#### 2.4 Prediction Heads (1 day)

**Tasks**:
- [ ] Implement probabilistic heads
  - Spread: (μ, log_σ²)
  - Home score: (μ, log_σ²)
  - Away score: (μ, log_σ²)
- [ ] Output layer with proper constraints
  - σ > 0 (via softplus)
- [ ] Test output shapes and distributions

---

### Week 3: Training & Evaluation

**Goal**: Trained model with baseline comparison

#### 3.1 Loss Function & Metrics (1 day)

**Tasks**:
- [ ] Implement Negative Log-Likelihood loss
  - Gaussian likelihood
  - Handles both μ and σ predictions
- [ ] Add auxiliary MSE loss (for interpretability)
- [ ] Implement evaluation metrics
  - MAE (spread, scores)
  - Calibration: Brier score, Log-loss, ECE
  - AUC (for win prediction)
- [ ] Test loss computation

**Key Decisions**:
- Loss weighting (NLL vs MSE)
- Whether to weight spread more than scores

#### 3.2 Training Loop (2 days)

**Tasks**:
- [ ] Setup training infrastructure
  - Optimizer (AdamW)
  - Learning rate scheduler (cosine? warmup?)
  - Gradient clipping
  - Mixed precision training (if needed)
- [ ] Implement training loop
  - Forward pass
  - Loss computation
  - Backprop
  - Logging (tensorboard/wandb)
- [ ] Checkpoint saving
- [ ] Early stopping (based on val MAE)
- [ ] Test on small subset

**Key Decisions**:
- Learning rate (1e-4? 3e-4?)
- Batch size (limited by GPU memory)
- Number of epochs
- Warmup steps

#### 3.3 Hyperparameter Tuning (2-3 days)

**Tasks**:
- [ ] Grid/random search over
  - Learning rate
  - Hidden dimension
  - Number of historical games
  - Dropout rate
  - Loss weights
- [ ] Track results systematically
- [ ] Select best model based on val MAE

#### 3.4 Evaluation & Analysis (2 days)

**Tasks**:
- [ ] Evaluate on test set
  - Spread MAE
  - Score MAE
  - Calibration metrics
- [ ] Compare to XGBoost baseline
- [ ] Error analysis
  - Where does model fail?
  - Error by team, season, game type
- [ ] Ablation studies
  - Remove sequences (use only aggregated features)
  - Remove roster context
  - Remove temporal attention
- [ ] Visualizations
  - Prediction distributions
  - Attention weights
  - Calibration plots

**Success Check**:
- MAE < 10.1 (beat XGBoost)
- Sequences add ≥1 point improvement

---

## Technical Specifications

### Model Architecture Sizes

| Component | Specification |
|-----------|---------------|
| Action embedding | 32-d |
| Player embedding | 64-d |
| Clock embedding | 16-d (sinusoidal) |
| Score context | 16-d |
| **Total event embedding** | **128-d** |
| Event encoder hidden | 256-d |
| Event encoder layers | 4 |
| Attention heads | 8 |
| Temporal attention layers | 2 |
| Set Transformer hidden | 64-d |
| Fusion hidden | 256-d |
| **Total parameters** | ~5-10M (estimate) |

### Training Configuration

| Setting | Value |
|---------|-------|
| Framework | PyTorch 2.x |
| Hardware | RTX 2070 SUPER (8GB) |
| Mixed precision | FP16 (if needed) |
| Batch size | 32-64 (tune based on memory) |
| Optimizer | AdamW |
| Learning rate | 1e-4 to 3e-4 (tune) |
| Scheduler | Cosine with warmup |
| Warmup steps | 500-1000 |
| Epochs | 50-100 (early stopping) |
| Gradient clipping | 1.0 |

### Data Specifications

| Dataset | Games | Samples | Usage |
|---------|-------|---------|-------|
| Train | ~30K | ~30K | 2000-2023 |
| Val | ~1,312 | ~1,312 | 2023-24 |
| Test | ~1,200 | ~1,200 | 2024-26 |
| **Full DB** | **37,366** | **37,366** | **All** |

---

## File Structure

```
src/
├── models/
│   ├── __init__.py
│   ├── event_encoder.py          # Transformer for PBP sequences
│   ├── set_transformer.py        # For roster encoding
│   ├── temporal_attention.py     # Cross-attention over games
│   ├── fusion.py                 # Combine all streams
│   ├── prediction_heads.py       # Probabilistic outputs
│   └── phase1_model.py           # Main model class
│
├── data/
│   ├── __init__.py
│   ├── tokenizer.py              # Event tokenization
│   ├── sequence_builder.py       # Historical game sequences
│   ├── dataset.py                # PyTorch Dataset
│   └── dataloader.py             # DataLoader with proper batching
│
├── training/
│   ├── __init__.py
│   ├── loss.py                   # NLL + MSE losses
│   ├── metrics.py                # MAE, Brier, ECE, etc.
│   ├── trainer.py                # Training loop
│   └── config.py                 # Hyperparameters
│
├── evaluation/
│   ├── __init__.py
│   ├── evaluate.py               # Test set evaluation
│   ├── ablation.py               # Ablation studies
│   └── visualize.py              # Plots and analysis
│
└── scripts/
    ├── train_phase1.py           # Main training script
    ├── evaluate_phase1.py        # Evaluation script
    └── compare_baselines.py      # Compare to XGBoost
```

---

## Risk Mitigation

### Technical Risks

| Risk | Mitigation |
|------|------------|
| **GPU memory overflow** | Start with small batch size, use gradient accumulation, mixed precision |
| **Sequence length too long** | Limit to top N events per game, use efficient attention |
| **Overfitting on small data** | Strong regularization (dropout 0.3-0.5), early stopping, cross-validation |
| **Model doesn't converge** | Careful initialization, learning rate warmup, gradient clipping |
| **Worse than XGBoost** | Not a failure - validates need for Phase 2 pre-training |

### Data Risks

| Risk | Mitigation |
|------|------------|
| **Data leakage** | Strict temporal split, careful validation |
| **Missing PBP data** | Handle gracefully, skip incomplete games |
| **Player ID mismatches** | Robust tokenizer with unknown token |
| **Era differences** | Model should learn implicitly, test on holdout eras |

---

## Success Metrics

### Primary Metrics

1. **Spread MAE < 10.0** (beat XGBoost baseline of 10.1)
2. **Ablation**: Sequences add ≥1 point improvement
3. **Calibration**: Well-calibrated uncertainty (ECE < 0.1)

### Secondary Metrics

4. **Score MAE**: Competitive with current system
5. **AUC**: Win prediction accuracy
6. **Computational efficiency**: Training time < 12 hours

### Qualitative

7. Model trains stably without divergence
8. Predictions are reasonable (no extreme outliers)
9. Uncertainty increases appropriately for uncertain games

---

## Next Steps After Phase 1

### If Successful (MAE < 10.0)
→ **Proceed to Phase 2**: Add pre-training on 18M plays (HGT)

### If Marginal (MAE 10.0-10.5)
→ **Iterate**: Try deeper model, more historical games, better tokenization
→ Then consider Phase 2

### If Unsuccessful (MAE > 10.5)
→ **Diagnose**:
- Is the model learning anything? (compare to random baseline)
- Is sequence information being used? (attention analysis)
- Data quality issues?

→ **Consider**:
- Simplify architecture (fewer layers, smaller hidden)
- Add more explicit features (hybrid approach)
- Pre-training might be necessary (skip to Phase 2)

---

## Open Questions (To Resolve During Implementation)

1. **How many historical games?** Start with N=5, ablate with N=3, N=10
2. **Pooling strategy?** [CLS] token vs average pooling
3. **Positional encoding?** Sinusoidal vs learned
4. **Attention mask?** Full attention vs causal mask (shouldn't matter for encoder)
5. **Player embeddings?** Random init vs pretrained (none available, skip for now)
6. **Loss weighting?** Equal weight vs prioritize spread
7. **Batch composition?** Random games vs chronological batches

---

## Timeline Summary

| Week | Focus | Deliverables |
|------|-------|--------------|
| **Week 1** | Data pipeline | Tokenized sequences, DataLoader ready |
| **Week 2** | Model implementation | Working forward pass, all components |
| **Week 3** | Training & eval | Trained model, comparison to baseline |

**Total**: ~3 weeks (flexible, could be 2-4 depending on issues)

---

## Getting Started

**Next immediate steps**:

1. [ ] Create file structure
2. [ ] Implement event tokenizer (first component)
3. [ ] Test on single game
4. [ ] Gradually build up from there

**First coding session**: Start with `src/data/tokenizer.py` - this is the foundation everything builds on.
