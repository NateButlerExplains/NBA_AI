# NBA Prediction Architecture

> **Status**: Phase 1 Implementation Ready
> **Last Updated**: February 2, 2026

---

## Overview

This document describes the neural architecture for NBA game prediction. The system uses a **minimal-context sequence modeling** approach that learns from raw play-by-play data without explicit feature engineering.

**Core Philosophy**: *"Provide facts, not interpretations. Let the model learn what matters."*

---

## Architecture Selection

### Chosen: Encoder-Only Transformer with Direct Prediction

**Why This Architecture**:
- Non-autoregressive (avoids error accumulation in numerical predictions)
- Learns from raw sequences without engineered features
- Produces calibrated uncertainty (μ, σ) for each prediction
- Scalable to pre-training (Phase 2: HGT, Phase 3: NBAFM)

**Alternatives Considered**:
- **PTIN** (Physics-Informed Transformer): Rejected - requires 43 engineered features, violates minimal-context philosophy
- **HGT** (Hierarchical Graph Transformer): Deferred to Phase 2 - adds pre-training on 18M plays
- **NBAFM** (Foundation Model): Deferred to Phase 3 - multi-task learning across prediction types

---

## Phase Progression

```
Phase 1: Supervised Baseline (Current)
├─ Goal: Validate sequence modeling approach
├─ Training: Supervised on game outcomes
├─ Timeline: 2-3 weeks
├─ Success: Beat XGBoost baseline (MAE < 10.1)
└─ Output: Trained model + ablation study results

Phase 2: Pre-training (HGT)
├─ Goal: Learn play dynamics from 18M plays
├─ Method: Self-supervised pre-training
├─ Timeline: 2-3 days per run
└─ Output: Pre-trained encoder for fine-tuning

Phase 3: Foundation Model (NBAFM)
├─ Goal: Multi-task learning (spreads, scores, player props)
├─ Method: Shared encoder, task-specific heads
├─ Timeline: 5-7 days per run
└─ Output: Production-ready foundation model
```

---

## Data Sources

### Primary: PBP_Logs + GameStates (Joined)

**Decision**: Use **both** PBP_Logs and GameStates together (1:1 on `game_id`, `play_id`)

**From PBP_Logs** (event details):
- Full action vocabulary (~50 types)
- Player involvement (who did it)
- Shot details (distance, result, coordinates)
- Clock information

**From GameStates** (state tracking):
- Clean parsed scores after each event
- Full roster state (all active players)
- `is_final_state` flag for supervision

**Rationale**:
- PBP provides event details, GameStates provides complete roster context
- Together they form complete (event, state) pairs at each timestep
- Natural supervision signal: events → state changes

### Context Data

**Minimal Context Only** (no feature engineering):
- **Game timestamp**: Captures era effects, season timing
- **Team locations**: Home city, away city (static mapping)
- **Home/away indicator**: Binary flag

**Explicitly Excluded**:
- Days rest (provide timestamps, let model learn)
- Back-to-back flags (provide timestamps, let model learn)
- Rolling averages (provide sequences, let model learn)
- Engineered schedule features (violates minimal-context philosophy)

### Roster Data

**Training**: Use actual players who played (from PBP)
**Inference**: Use expected available players (from InjuryReports)

**Fallback** (pre-2018 games without injury data):
- Use recent active players (last 10 games)
- Tested in ablation studies

---

## Phase 1 Architecture

### Three-Stream Model

```
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 1 ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  STREAM 1: Historical PBP Sequences                          │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Last N games per team:                            │     │
│  │    Raw PBP events → Event Tokenizer                │     │
│  │                         ↓                          │     │
│  │    [action, player, clock, score_diff, timestamp]  │     │
│  │                         ↓                          │     │
│  │              Event Encoder (4-layer Transformer)   │     │
│  │                         ↓                          │     │
│  │              Game Embedding (pooling)              │     │
│  │                         ↓                          │     │
│  │      Temporal Attention over game sequence         │     │
│  │                         ↓                          │     │
│  │              Team History Embedding                │     │
│  └────────────────────────────────────────────────────┘     │
│                         ↓                                    │
│  STREAM 2: Expected Roster                                   │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Player IDs: [p1, p2, ..., p13] per team          │     │
│  │                         ↓                          │     │
│  │        Player ID → Learned Embedding (64-d)        │     │
│  │                         ↓                          │     │
│  │        Set Transformer (permutation invariant)     │     │
│  │                         ↓                          │     │
│  │              Roster Embedding per team             │     │
│  └────────────────────────────────────────────────────┘     │
│                         ↓                                    │
│  STREAM 3: Minimal Context                                   │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Game timestamp, team locations, home/away         │     │
│  │              (embedded as part of fusion)          │     │
│  └────────────────────────────────────────────────────┘     │
│                         ↓                                    │
│  FUSION LAYER                                                │
│  ┌────────────────────────────────────────────────────┐     │
│  │  [Home History; Away History; Home Roster;         │     │
│  │   Away Roster; Context]                            │     │
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

### Model Specifications

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
| **Total parameters** | **~5-10M** |

---

## Key Design Decisions

### 1. Non-Autoregressive Prediction

**Decision**: Use direct prediction heads (not autoregressive)

**Rationale**:
- Autoregressive: Predict sequentially, each prediction depends on previous
- Direct: Predict all outputs in parallel
- For numerical predictions (spreads, scores), autoregressive causes error accumulation
- Direct prediction avoids compounding errors

### 2. Probabilistic Outputs

**Decision**: Predict (μ, σ) for each target, not point estimates

**Rationale**:
- Research shows calibration > accuracy for betting profitability
- Uncertainty quantification critical for risk management
- Enables Kelly criterion betting, confidence intervals
- Gaussian likelihood loss (NLL) trains well-calibrated models

### 3. Minimal Context Philosophy

**Decision**: No engineered features, only raw temporal data

**Rationale**:
- Avoid human bias in feature selection
- Let model learn what matters from sequences
- Provide facts (timestamps, locations), not interpretations (days_rest, rolling_avg)
- Simpler, more principled approach

### 4. Sequence Length (To Be Determined)

**Options**: N=3, 5, 10, 20 historical games

**Method**: Empirical testing in Phase 1 ablation studies

**Trade-off**: More context vs memory/compute

### 5. Roster Sources (To Be Determined)

**Options**:
- Oracle: Actual players who played (upper bound)
- Injury Reports: Expected available players (realistic)
- Fallback: Recent active players (pre-2018 historical)

**Method**: Empirical testing in Phase 1 ablation studies

---

## Training Strategy

### Data Split (Temporal)

- **Train**: 2000-01 through 2022-23 (~30K games)
- **Val**: 2023-24 season (~1,312 games)
- **Test**: 2024-25 + 2025-26 (~1,200 games)

**Critical**: No data leakage, strict chronological split

### Loss Function

**Primary**: Negative Log-Likelihood (Gaussian)
```
NLL = 0.5 * log(σ²) + 0.5 * ((y - μ) / σ)²
```

**Auxiliary**: MSE for interpretability
```
MSE = (y - μ)²
```

**Combined**: `Loss = λ_NLL * NLL + λ_MSE * MSE`

### Training Configuration

| Setting | Value |
|---------|-------|
| Framework | PyTorch 2.x |
| Hardware | RTX 2070 SUPER (8GB) |
| Batch size | 32-64 |
| Optimizer | AdamW |
| Learning rate | 1e-4 to 3e-4 (tune) |
| Scheduler | Cosine with warmup |
| Epochs | 50-100 (early stopping) |
| Gradient clipping | 1.0 |

---

## Success Criteria

### Primary Metrics

1. **Spread MAE < 10.0** (beat XGBoost baseline of 10.1)
2. **Ablation**: Sequences demonstrably useful vs aggregated features
3. **Calibration**: Well-calibrated uncertainty (ECE < 0.1)

### Secondary Metrics

4. **Score MAE**: Competitive with current system
5. **AUC**: Win prediction accuracy > 0.65
6. **Computational efficiency**: Training time < 12 hours per run

### Qualitative

7. Model trains stably without divergence
8. Predictions are reasonable (no extreme outliers)
9. Uncertainty increases appropriately for uncertain games

---

## Implementation Roadmap

### Week 1: Data Pipeline
- Event tokenization (PBP + GameStates joined)
- Sequence construction (last N games per team)
- Train/val/test split (chronological)
- Roster extraction (injury reports + fallback)

### Week 2: Model Implementation
- Event Encoder (Transformer)
- Set Transformer (rosters)
- Temporal Attention (game sequences)
- Fusion Layer + Prediction Heads

### Week 3: Experimentation (15-20 runs)
- **Ablation 1**: Roster sources (oracle/injury/fallback)
- **Ablation 2**: Sequence lengths (N=3/5/10/20)
- **Ablation 3**: Minimal context (none/temporal/full)
- **Ablation 4**: Model sizes (2M/5M/15M params)
- **Final**: Select best config, comprehensive evaluation

**Total Timeline**: 2-3 weeks

---

## Research Foundation

This architecture synthesizes insights from recent sports prediction research:

- **Sharp Sports Betting** (Millman 2021): Calibration > accuracy for profitability
- **PTIN** (Wang+ 2024): Transformer architectures work for NBA prediction
- **HGT** (Zhao+ 2024): Pre-training on play sequences improves downstream tasks
- **NBAFM** (Zhang+ 2024): Foundation models generalize across prediction types

**Key Insight**: Start simple (Phase 1 supervised), add pre-training (Phase 2), scale to foundation model (Phase 3).

---

## Next Steps

1. **Immediate**: Begin Week 1 implementation (data pipeline)
2. **First Component**: Event tokenizer (`src/data/tokenizer.py`)
3. **Test Early**: Validate on single game before scaling
4. **Track Progress**: Use TODO.md for task management

See [TODO.md](TODO.md) for detailed Sprint 18 implementation tasks.
