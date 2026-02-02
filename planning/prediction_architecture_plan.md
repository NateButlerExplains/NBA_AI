# Custom Model and Prediction Architecture Plan

## Status: Architecture Selected - Ready for Implementation

## Sprint Goal
Design the prediction and modeling architecture that transforms INPUTS into OUTPUTS.

---

## System Overview

### INPUTS

#### Core (Primary)
- **Play by Play Data** - Raw game event data (~492 plays/game, stored in `PbP_Logs`)
- **State by State Data** - Game states created 1:1 from play by play (`GameStates` table)

#### Secondary (Optional, Partially Set Up)
- **Injury Reports** - NBA Official PDFs, 26K+ records since Dec 2018
- **Betting Data** - Opening/Current/Closing lines from ESPN & Covers (3K games)
- **Schedule/Rest** - Days rest, game frequency, day of season

#### Tertiary (Future)
- Open to additional sources if high value
- Guiding principle: Focus on model/system over data collection

---

### OUTPUTS (Predictions)

#### Layer 1 - CORE METRIC
- Predicted final game spread (similar to Vegas open line)

#### Layer 2 - Game Totals
- Final Total
- Home Score
- Away Score

#### Layer 3 - Player Metrics
- Points, Rebounds, Assists, Steals, Blocks per player

#### Layer 4 - Point-in-Time Predictions
- All above metrics at any game state

#### Layer 5 - Confidence/Intervals
- Nice to have from core setup
- Explainability not a core goal

---

## Current System State

### What Exists Today

**Database (3-tier)**:
| DB | Size | Games | Purpose |
|----|------|-------|---------|
| current | 516 MB | 1,302 | Production (2025-26) |
| dev | 3.0 GB | 4,098 | Development (3 seasons) |
| full | 25 GB | 37,366 | Archive (27 seasons) |

**Core Data Available**:
- `PbP_Logs`: ~492 plays/game, raw JSON with actionType, clock, scores, player info
- `GameStates`: 1:1 with PBP, parsed snapshots (scores, period, clock, player points)
- `PlayerBox`: Full box scores (pts/reb/ast/stl/blk + shooting + minutes), 2023+
- `TeamBox`: Team aggregates, 2023+
- `Players`: 5,118 players with personId
- `InjuryReports`: 26K records since Dec 2018
- `Betting`: Opening/current/closing lines, 2007+

**Current Prediction System**:
- **Features**: 43 rolling averages from prior final game states
- **Models**: XGBoost (MAE 10.1), Ridge (MAE 11.2), MLP, Ensemble
- **Output**: Home/Away scores only
- **Live**: Pre-game prediction blended with actual score by time remaining

**Historical Data Availability**:
| Data | From | Notes |
|------|------|-------|
| PBP/GameStates | 2000-01 | ~18M plays in full DB |
| Betting | 2007-08 | ~21K games, ~93% coverage |
| InjuryReports | Dec 2018 | NBA Official PDFs |
| PlayerBox/TeamBox | 2023-24 | Backfill deferred |

### Key Gaps to Fill

1. **Player-level predictions** - Currently only team scores
2. **Spread as primary output** - Currently derived from score predictions
3. **Rich game state features** - Currently only uses final states from prior games
4. **Point-in-time predictions** - Currently just linear blending with actuals
5. **Sequence modeling** - PBP data stored but not used as sequences

---

## Architecture Design

### Current Considerations
*To be filled in during planning session*

### Key Questions
*See Clarifying Questions section below*

### Proposed Approach
*To be determined after clarification*

---

## Requirements (Confirmed)

### R1: Player Prediction Scope
**Answer**: 5 core stats only (pts/reb/ast/stl/blk)

### R2: Point-in-Time Granularity
**Answer**: Per-play / per-state granularity. Web app updates on-demand (no live streaming requirement).

### R3: Primary Use Case
**Answer**: All three (pre-game → in-game → historical backtest), with priority matching output hierarchy:
1. Pre-game predictions (primary focus)
2. In-game predictions (secondary)
3. Historical analysis

### R4: Model Architecture Philosophy
**Answer**: Open to exploration. Current system is baseline only - will design new architecture from scratch.

### R5: Spread vs Scores
**Answer**: No preference - optimize for prediction quality. Could go either way.

### R6: Training Data Strategy
**Answer**: Use as much of the 27 seasons as is useful. Open to data being used for training vs context/features in different ways.

---

### R7: Success Metric
**Answer**: Best possible system. Probably MAE, open to better metrics.

### R8: Player Prediction Constraints
**Answer**: No preference - whatever approach works best.

### R9: Roster/Lineup Information
**Answer**: Explore what exists in current data. Open to adding sources if absolutely necessary.

### R10: Cold Start Handling
**Answer**: Model should be inherently temporal to handle this naturally. Open to hybrid approaches.

### R11: Interpretability
**Answer**: Black-box is fine if it performs best.

### R12: Computational Constraints
**Answer**: Local dev/training on RTX 2070 SUPER. Cloud OK where necessary. Don't let compute drive planning - aim for optimal system.

### R13: Play-by-Play Sequence Usage
**Answer**: Open to explore, leaning toward full sequence modeling.

### R14: Betting Line Integration
**Answer**: Open to using Vegas open lines as features if advantageous, but system should not rely on them.

### R15: Player Identity
**Answer**: Per-player predictions tied to specific player IDs (not roster slots).

### R16: In-Game Update Trigger
**Answer**: On-demand is fine. Pre-game predictions of final values are most important. Every-play updates acceptable if beneficial.

### R17: Inference Latency
**Answer**: Not critical. Pre-game can be batched hours ahead. Web app updates should be seconds (not tens of seconds). Data is already delayed so predictions can be too.

### R18: Player Prediction Scope
**Answer**: Predict for all ~13 rostered players per team (not just starters/rotation).

### R19: Historical Era Handling
**Answer**: Model should handle era differences internally (e.g., era as feature, let model learn).

---

---

## Requirements Summary

| Aspect | Decision |
|--------|----------|
| **Player stats** | 5 core: pts/reb/ast/stl/blk |
| **Granularity** | Per-play/per-state |
| **Priority** | Pre-game → In-game → Backtest |
| **Architecture** | Open - design from scratch |
| **Spread target** | Optimize for quality (MAE likely) |
| **Training data** | Use all 27 seasons intelligently |
| **Player identity** | Per-player (NBA personId) |
| **Cold start** | Model should be inherently temporal |
| **Interpretability** | Black-box OK |
| **Compute** | RTX 2070 SUPER local, cloud available |
| **PBP sequences** | Explore full sequence modeling |
| **Betting lines** | Optional features, don't rely on |
| **Latency** | Seconds OK, not sub-100ms |
| **Lineups** | Explore what exists in data |

---

## Planning Session Log

### Session 1 - 2026-01-31
- Initial planning doc created
- Codebase exploration completed
- Current system: XGBoost predicting home/away scores, MAE 10.1
- Identified gaps: No player predictions, no native spread output, limited game-state features
- All requirements clarified (R1-R19)
- Player IDs confirmed: NBA personId (numeric strings like "201566")
- Ready to begin architecture research phase

### Session 2 - 2026-01-31
- Completed SOTA research (2025-2026 literature)
- Key findings: Transformers outperform LSTMs (AUC 0.8473), hybrids best (F1 92.1%)
- Designed 2 candidate architectures (HGT, NBAFM)
- User identified 2 key principles:
  1. Native multi-scale temporality (state→game→season→era)
  2. Hierarchical player representations (future focus)
- **Selected approach**: Temporal baseline → Pre-training
  - Phase 1: Supervised temporal model (validate sequences > features)
  - Phase 2: Add pre-training on 18M plays
  - Phase 3: Hierarchical player model (parked for more data)

### Session 3 - 2026-01-31
- Finalized Phase 1 architecture with three input streams
- **Key architectural decisions**:
  1. Encoder-only with direct prediction (not autoregressive) - avoids error accumulation
  2. Probabilistic heads (μ, σ outputs) - uncertainty without Monte Carlo simulation
  3. Three input streams: PBP sequences + roster context + schedule features
- **Data clarifications**:
  - Training: Uses actual rosters from PBP (who played)
  - Inference: Uses expected rosters from InjuryReports
  - This asymmetry is correct - model learns player impact from actual play
- **Scope decisions**:
  - Phase 1: Game outcomes only (spread, scores) - full 27 seasons available
  - Phase 2: Player stats - deferred (BoxScore data 2023+ only)
- **Schedule effects**: Confirmed critical (3-5 point impact) - explicit features required
- **Era/season context**: Implicit in temporal sequences (test explicit features via ablation later)
- Ready for Phase 1 implementation

### Session 4 - 2026-02-02

- **Refined philosophy**: "Provide facts, not interpretations. Let the model learn what matters."
- **Removed PTIN** (feature-based approach) - incompatible with philosophy
- **Confirmed non-autoregressive design**: Direct prediction avoids error accumulation
- **Validated against SOTA research**:
  - Calibration > accuracy for profitability (ROI +34.69% vs -35.17%)
  - Probabilistic heads are well-justified
  - Sequence modeling is underexplored (competitive advantage)
- **Clarified progression path**: Phase 1 → HGT (Phase 2) → NBAFM (optional)
- **Minimal context approach**: Timestamps, player IDs, locations (facts) - no engineered features
- **Transition to implementation**: Planning complete, ready to build Phase 1

---

## Next Steps

1. ✅ Explore current project structure
2. ✅ Understand existing data schemas
3. ✅ Clarify requirements
4. ✅ Research state-of-the-art approaches
5. ✅ Design candidate architectures with tradeoffs
6. ✅ Select approach and detail implementation plan
7. ⏳ **Phase 1**: Build temporal baseline (supervised) → See [Phase 1 Implementation Plan](phase1_implementation_plan.md)
8. ⏳ **Phase 2**: Add pre-training (if Phase 1 validates)
9. ⏳ **Phase 3**: Hierarchical player model (future)

---

## Implementation Status

**Current Phase**: Phase 1 Implementation (Week 0)

**See**: [Phase 1 Implementation Plan](phase1_implementation_plan.md) for detailed timeline and tasks

**Next Action**: Begin data pipeline implementation (event tokenizer)

---

## Architecture Research Phase

### Approach
- Research-first: Survey state-of-the-art before proposing
- User background: Intermediate-to-expert ML, willing to learn
- Constraints: None - complexity not a limiting factor
- Scope: Open to all approaches (transformers, GNNs, probabilistic, etc.)

---

## Research Findings (2025-2026 Literature)

### 1. Sequence Modeling Architectures

#### Transformers for Sports Events
- **Best results**: Transformer with BCE loss achieved AUC 0.8473 for NCAA basketball (outperforming LSTM)
- **Multi-scale approach**: Time-segment encoding + multi-level Transformer extracts short-term and long-term dependencies
- **Hybrid CNN-Transformer**: 1D CNN captures local patterns, Transformer models long-range dependencies
- **Axial Transformer**: Large-scale in-game forecasting for match, team, and player outcomes simultaneously

**Key paper**: [Forecasting NCAA Basketball Outcomes with Deep Learning](https://arxiv.org/html/2508.02725v1) - Transformer AUC 0.8473

#### LSTM/RNN Approaches (Still Competitive)
- **Long-sequence LSTM**: 8 seasons (9,840 games) achieves 72.35% accuracy, 76.13% AUC-ROC for NBA
- **Hybrid LSTM-Transformer**: Outperforms either alone (F1 92.1% vs 88.1% Transformer-only vs 85.9% LSTM-only)
- **Key insight**: Time dependencies are critical for NBA prediction - models without temporal architecture underperform

**Key paper**: [Long-Sequence LSTM Modeling for NBA Game Outcome Prediction](https://arxiv.org/abs/2512.08591)

### 2. Graph Neural Networks for Player Interactions

#### HIGFormer (KDD 2025) - Most Relevant
Three-component architecture:
1. **Player Interaction Network** - Historical player performance learning
2. **Team Interaction Network** - Historical team performance comprehension
3. **Match Comparison Transformer** - Match outcome prediction

**Key design**: Final prediction compares aggregated team representations (average of player embeddings)

**Key paper**: [Player-Team Heterogeneous Interaction Graph Transformer](https://arxiv.org/pdf/2507.10626)

#### Other GNN Approaches
- **GATv2-TCN**: Graph Attention + Temporal Convolution for NBA player performance
- **GCN for passing networks**: 71.54% accuracy modeling player interactions as graphs
- **Spatiotemporal Graph Transformer**: Ball trajectory + player interaction modeling

### 3. Player Embedding & Representation Learning

#### Approaches
- **Trainable embeddings**: Node type, edge type, and player identity embeddings (HIGFormer)
- **Column embeddings**: Categorical features → learnable parametric representations
- **play2vec**: Skip-gram model learning distributed representations of play segments
- **NETS (Neural Embeddings in Team Sports)**: Transformer + LSTM + team-wise pooling

#### For Variable-Length Rosters
- **Set Transformer**: Permutation-invariant, handles variable-size sets, captures pairwise interactions via attention
- **Deep Sets**: Simpler pooling (sum/average) - information loss but efficient
- **Set Transformer++**: Improved with SetNorm for deep architectures

**Key insight**: Set Transformer preserves permutation invariance while modeling player interactions

### 4. Pre-training Strategies

#### Self-Supervised Approaches
- **SoccerTransformer**: Self-supervised pre-training on event sequences → attack phase prediction (F1 0.814-0.862)
- **Masked Autoencoder**: Pre-training on multi-agent trajectories improves downstream classification
- **Seq2Event**: "Language of soccer" - treating actions as words with contextual meaning

**Relevance**: With 18M plays, similar pre-training could learn "language of basketball"

### 5. Multi-Task Learning

#### Joint Prediction Architectures
- **Hard Parameter Sharing (HPS)**: Shared backbone → task-specific heads (most common)
- **HIGFormer approach**: Player embeddings → team aggregation → multiple prediction heads
- **Axial Transformer**: Simultaneously predicts match, team, and player outcomes

**Key insight**: Multi-task learning improves efficiency and can share learned representations

### 6. Uncertainty Quantification

#### Probabilistic Approaches
- **Bayesian RNN with Monte Carlo dropout**: Calibrated sequential probabilities
- **TabPFN**: Probabilistic transformer for robust predictions with minimal tuning
- **Key finding**: Calibration > accuracy for betting applications (ROI +34.69% vs -35.17%)

**Key paper**: [Uncertainty-Aware Machine Learning for NBA Forecasting](https://www.mdpi.com/2078-2489/17/1/56)

### 7. Era Adaptation / Temporal Drift

#### Approaches
- **Raincoat**: Domain adaptation for time series under feature/label shifts
- **Continuous adaptation**: Fixed learning rate SGD for runtime/memory efficiency
- **Feature-based**: Era as explicit feature (confirmed viable approach)

**Key insight**: Temporal concept drift is a known challenge - explicit handling recommended

---

## Architecture Implications

Based on research, strong candidates for our system:

### Architecture Pattern A: Hierarchical Transformer
```
PBP Sequences → Event Encoder → Game-Level Transformer → Multi-Task Heads
                                      ↓
                            Player Embeddings (Set Transformer)
                                      ↓
                            Team Aggregation → Spread/Scores
                            Per-Player → Stats
```

### Architecture Pattern B: Graph + Sequence Hybrid
```
Historical Games → Player Graph (GNN) → Player Representations
                         ↓
Current Game PBP → Sequence Model (LSTM/Transformer) → Joint with Player Reps
                         ↓
                   Multi-Task Prediction
```

### Architecture Pattern C: Pre-trained Foundation Model
```
18M Plays → Self-Supervised Pre-training (Masked Event Prediction)
                         ↓
                   Fine-tune for:
                   - Game outcome (spread/scores)
                   - Player stats
                   - In-game updates
```

### Key Design Decisions to Make
1. **Event encoding**: How to represent individual PBP events
2. **Sequence architecture**: Transformer vs LSTM vs Hybrid
3. **Player handling**: Set Transformer vs GNN vs Both
4. **Multi-task structure**: Shared backbone or separate paths
5. **Pre-training**: Yes/No, and what objective
6. **Uncertainty**: Probabilistic heads or deterministic

---

## Detailed Candidate Architectures

### Option A: Hierarchical Game Transformer (HGT)

**Philosophy**: Treat basketball as a language - events are tokens, games are documents. Pre-train on event prediction, fine-tune for outcomes.

```
┌─────────────────────────────────────────────────────────────────┐
│                    HIERARCHICAL GAME TRANSFORMER                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐                                                │
│  │ PBP Event   │ actionType, clock, scores, playerIds           │
│  │ (raw JSON)  │                                                │
│  └──────┬──────┘                                                │
│         ▼                                                       │
│  ┌─────────────┐                                                │
│  │ Event       │ Embed: action_type + player_id + position +    │
│  │ Tokenizer   │ clock_encoding + score_diff + context          │
│  └──────┬──────┘                                                │
│         ▼                                                       │
│  ┌─────────────┐                                                │
│  │ Event       │ 6-layer Transformer encoder                    │
│  │ Encoder     │ Processes ~492 events per game                 │
│  └──────┬──────┘                                                │
│         ▼                                                       │
│  ┌─────────────┐                                                │
│  │ Game-Level  │ Cross-attention over historical games          │
│  │ Aggregator  │ (last N games per team/player)                 │
│  └──────┬──────┘                                                │
│         │                                                       │
│         ├──────────────┬──────────────┬────────────────┐        │
│         ▼              ▼              ▼                ▼        │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐    │
│  │ Spread    │  │ Score     │  │ Player    │  │ Uncertainty│    │
│  │ Head      │  │ Head      │  │ Stats Head│  │ Head       │    │
│  │ (1 value) │  │ (H/A/Tot) │  │ (26×5)    │  │ (σ per out)│    │
│  └───────────┘  └───────────┘  └───────────┘  └───────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Event Tokenization**:
| Component | Embedding Dim | Notes |
|-----------|---------------|-------|
| action_type | 32 | ~50 unique action types |
| player_id | 64 | 5,118 players → learned embedding |
| team | 16 | Home/Away indicator |
| period | 8 | 1-4 + OT |
| clock | 16 | Sinusoidal encoding |
| score_diff | 16 | Current margin |
| **Total** | **152** | Per-event vector |

**Training Strategy**:
1. **Phase 1 - Pre-training** (18M plays): Masked event prediction (like BERT)
   - Mask 15% of events, predict action_type + involved_player
   - ~2-3 days on cloud GPU
2. **Phase 2 - Fine-tuning** (37K games): Multi-task supervised learning
   - Loss = λ₁·spread_loss + λ₂·score_loss + λ₃·player_loss
   - ~4-6 hours local

**Inference**:
- Pre-game: Process last N games per team → predict
- In-game: Append current game events → re-predict
- Latency: ~200-500ms per prediction

**Compute**:
| Phase | Hardware | Time |
|-------|----------|------|
| Pre-training | Cloud (A100) | 2-3 days |
| Fine-tuning | RTX 2070 | 4-6 hours |
| Inference | RTX 2070 | <1 sec |

**Pros**:
- Leverages full 18M plays via pre-training
- Unified architecture for all outputs
- Strong sequence modeling (proven in research)
- Natural handling of in-game updates

**Cons**:
- Complex to implement
- Requires cloud for pre-training
- Player interactions implicit (not explicit graph)

**Expected Performance**: AUC ~0.82-0.85 (based on NCAA Transformer results)

---

### Option B: NBA Foundation Model (NBAFM)

**Philosophy**: Build a general-purpose basketball understanding model, then specialize for prediction tasks. Maximum use of data.

```
┌─────────────────────────────────────────────────────────────────┐
│                    NBA FOUNDATION MODEL                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              STAGE 1: PRE-TRAINING (18M plays)            │   │
│  │                                                          │   │
│  │  Task 1: Next Event Prediction (autoregressive)          │   │
│  │    "Given events 1-N, predict event N+1"                 │   │
│  │                                                          │   │
│  │  Task 2: Masked Event Modeling (bidirectional)           │   │
│  │    "Fill in masked events"                               │   │
│  │                                                          │   │
│  │  Task 3: Game Outcome Prediction (weak supervision)      │   │
│  │    "Predict final score from partial game"               │   │
│  │                                                          │   │
│  │  Architecture: 12-layer Transformer, 768-d hidden        │   │
│  │  Parameters: ~85M                                        │   │
│  │                                                          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              STAGE 2: FINE-TUNING (Task-Specific)         │   │
│  │                                                          │   │
│  │  ┌─────────────────┐  ┌─────────────────┐               │   │
│  │  │ Pre-Game Module │  │ In-Game Module  │               │   │
│  │  │                 │  │                 │               │   │
│  │  │ Input: Last 5   │  │ Input: Current  │               │   │
│  │  │ games per team  │  │ game events +   │               │   │
│  │  │ (encoded)       │  │ history         │               │   │
│  │  │                 │  │                 │               │   │
│  │  │ Output:         │  │ Output:         │               │   │
│  │  │ - Spread        │  │ - Updated preds │               │   │
│  │  │ - Scores        │  │ - Win prob      │               │   │
│  │  │ - Player stats  │  │ - Final scores  │               │   │
│  │  └─────────────────┘  └─────────────────┘               │   │
│  │                                                          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              STAGE 3: ADAPTATION (Optional)               │   │
│  │                                                          │   │
│  │  - Era-specific fine-tuning (e.g., 3-point era adapter)  │   │
│  │  - Team-specific adapters (LoRA-style)                   │   │
│  │  - Continuous learning on new seasons                    │   │
│  │                                                          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Model Specifications**:
| Component | Specification |
|-----------|---------------|
| Architecture | GPT-2 style decoder |
| Layers | 12 |
| Hidden dim | 768 |
| Attention heads | 12 |
| Parameters | ~85M |
| Context length | 1024 events (~2 games) |
| Vocabulary | ~5,200 (actions × players × positions) |

**Pre-training Objectives**:
1. **Next Event Prediction**: P(event_t | events_1:t-1)
2. **Masked Event Reconstruction**: P(event_mask | context)
3. **Score Prediction**: P(final_score | partial_game)

**Training Strategy**:
1. **Pre-training**: 18M plays, ~1 week on cloud
2. **Fine-tuning**: Task-specific heads, ~8-12 hours local
3. **Adaptation**: Era/team adapters as needed

**Compute**:
| Phase | Hardware | Time |
|-------|----------|------|
| Pre-training | Cloud (4×A100) | 5-7 days |
| Fine-tuning | RTX 2070 | 8-12 hours |
| Inference | RTX 2070 | ~500ms |

**Pros**:
- Maximum utilization of 18M plays
- Learns deep basketball semantics
- Flexible for multiple downstream tasks
- State-of-the-art potential

**Cons**:
- Highest complexity and compute cost
- Longest development time
- Risk of overfitting to pre-training objective
- May be overkill for the prediction task

**Expected Performance**: AUC ~0.84-0.88 (ceiling estimate based on scale)

---

## Architecture Comparison

| Dimension | A: HGT | B: NBAFM |
|-----------|--------|----------|
| **Complexity** | Medium | High |
| **Training time** | 2-3 days | 1 week |
| **Pre-training** | Yes (event) | Yes (multi-task) |
| **PBP sequence use** | Full | Full |
| **Player modeling** | Implicit | Implicit |
| **In-game updates** | Natural | Natural |
| **Cloud required** | Pre-train only | Yes |
| **Expected AUC** | 0.82-0.85 | 0.84-0.88 |
| **Dev effort** | Medium | High |
| **Interpretability** | Low | Low |

---

## Selected Approach

**Decision: Encoder-Only Transformer with Direct Prediction**

**Core Belief**: The NBA operates on 4 temporal layers (state→game→season→era), and the architecture must be natively temporal to capture this.

**Key Architecture Decisions**:
- **Encoder-only** (not autoregressive): Avoids error accumulation problems with numerical data
- **Direct prediction**: Outputs predictions directly rather than generating intermediate states
- **Probabilistic heads**: Output μ and σ for uncertainty quantification (no Monte Carlo simulation needed)
- **Three input streams**: Historical sequences + roster context + schedule features

**Future Vision** (parked for later):
- Hierarchical player representations (Base → Type → Individual)
- Player interaction networks
- Requires more player-level data

---

### Phase 1: Game Outcome Prediction (Supervised)

**Goal**: Validate sequence modeling for spread/score prediction with full context.

**Scope**: Game outcomes only (spread, home/away scores). Player stats deferred to Phase 2 (requires BoxScore data, 2023+ only).

**Architecture**: Three-Stream Encoder with Probabilistic Heads

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 1 ARCHITECTURE                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  STREAM 1: Historical PBP Sequences                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  For each team (last 5-10 games):                               │   │
│  │    PBP Events → Event Tokenizer → Event Encoder (4-layer)       │   │
│  │                        ↓                                        │   │
│  │              Game Embedding (via [CLS] or pooling)              │   │
│  │                        ↓                                        │   │
│  │              Temporal Attention over game sequence              │   │
│  │                        ↓                                        │   │
│  │                  Team History Embedding                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ↓                                         │
│  STREAM 2: Tonight's Expected Roster                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  For each team (~13 players):                                   │   │
│  │    Player ID → Learned Embedding (64-d)                         │   │
│  │    + Rolling Stats (from prior games)                           │   │
│  │    + Availability Status (Out/Questionable/Probable/Available)  │   │
│  │                        ↓                                        │   │
│  │              Set Transformer (permutation-invariant)            │   │
│  │                        ↓                                        │   │
│  │                  Roster Embedding (per team)                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ↓                                         │
│  STREAM 3: Schedule Context                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Features (derived from Games table):                           │   │
│  │    - Days rest (home/away)                                      │   │
│  │    - Back-to-back flag (home/away)                              │   │
│  │    - Games in last 7 days (home/away)                           │   │
│  │    - Travel distance (estimated from team locations)            │   │
│  │    - Day of season (normalized 0-1)                             │   │
│  │    - Home/Away indicator                                        │   │
│  │                        ↓                                        │   │
│  │                    MLP → Schedule Embedding                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ↓                                         │
│  FUSION LAYER                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  [Home History; Away History; Home Roster; Away Roster;         │   │
│  │   Schedule Features]                                            │   │
│  │                        ↓                                        │   │
│  │              Cross-Attention Fusion (2-layer)                   │   │
│  │                        ↓                                        │   │
│  │                  Matchup Representation                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ↓                                         │
│  PROBABILISTIC PREDICTION HEADS                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐       │   │
│  │  │ Spread Head   │  │ Home Score    │  │ Away Score    │       │   │
│  │  │ μ_spread, σ   │  │ μ_home, σ     │  │ μ_away, σ     │       │   │
│  │  └───────────────┘  └───────────────┘  └───────────────┘       │   │
│  │                                                                 │   │
│  │  Loss: NLL (Gaussian) or combined MSE + variance regularization │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Data Sources (Training vs Inference)**:

| Data | Training | Inference | Notes |
|------|----------|-----------|-------|
| PBP Sequences | ✅ Historical PBP | ✅ Historical PBP | Same - team's recent games |
| Roster | ✅ Actual players from PBP | ✅ Expected from InjuryReports | Training sees who actually played |
| Schedule | ✅ Derived from Games | ✅ Derived from Games | Same derivation |
| Labels | ✅ Final scores from Games | N/A | Ground truth |

**Key Insight**: Training uses actual rosters (who played, from PBP data). Inference uses expected rosters (injury reports + full roster). This is correct behavior - the model learns player impact from actual participation, then applies that to expected lineups.

**Schedule Feature Derivation** (from Games table):
```python
# Pseudocode for schedule features
for team in [home, away]:
    team_games = get_team_games(team_id, before=game_date)
    days_rest = (game_date - team_games[-1].date).days
    is_b2b = days_rest == 1
    games_last_7 = len([g for g in team_games if g.date > game_date - 7])

schedule_features = [
    home_days_rest, away_days_rest,
    home_b2b, away_b2b,
    home_games_7d, away_games_7d,
    day_of_season,  # normalized 0-1
    1.0  # home indicator (always 1 for home team perspective)
]
```

**Specifications**:
| Component | Specification |
|-----------|---------------|
| Event embedding | 128-d (action + player + clock + context) |
| Event encoder | 4-layer Transformer, 256-d hidden |
| Temporal attention | 2-layer cross-attention over game embeddings |
| Set Transformer | 2-layer, handles variable roster size |
| Schedule MLP | 2-layer, 64-d hidden |
| Fusion | 2-layer cross-attention |
| Training data | Full DB (37K games, 27 seasons) |
| Training time | 4-8 hours local (RTX 2070) |

**Success Criteria**:
- Spread MAE < 10.0 (beat current XGBoost baseline of 10.1)
- Ablation shows sequences > aggregated features
- Ablation shows roster context adds value

**Deliverables**:
1. Event tokenization pipeline
2. Roster embedding with Set Transformer
3. Schedule feature derivation
4. PyTorch model implementation
5. Training loop with validation
6. Comparison vs XGBoost baseline

**In-Game Prediction Mode**:

The same architecture handles in-game predictions by appending current game events:

```
PRE-GAME:
  Stream 1: [Team A last 5 games] + [Team B last 5 games]
  Stream 2: [Expected rosters from injury reports]
  Stream 3: [Schedule features]
  → Predict final spread/scores

IN-GAME (e.g., halftime):
  Stream 1: [Team A last 5 games] + [Team B last 5 games] + [CURRENT GAME so far]
  Stream 2: [Actual players on court from PBP]
  Stream 3: [Same schedule features]
  → Predict final spread/scores (conditioned on current state)
```

The encoder naturally handles variable-length sequences. Current game events provide strong signal that overwhelms pre-game priors as the game progresses.

---

### Phase 2: Add Pre-training (Cloud)

**Goal**: Leverage 18M plays via self-supervised learning.

**Trigger**: Phase 1 shows sequences add value over features.

**Pre-training Objective**: Masked Event Prediction
- Mask 15% of events in a game
- Predict: action_type + involved_player
- Learn basketball "grammar" without labels

**Timeline**: 2-3 days cloud (A100)

**Expected Lift**: +2-4 points AUC based on SoccerTransformer results

---

### Phase 3: Hierarchical Player Model (Future)

**Goal**: Add explicit player representations with type hierarchy.

**Trigger**: More player-level data available, Phase 2 complete.

**Architecture**:
- Base Player Encoder (all players)
- Type Adapters (archetypes)
- Individual Fine-tuning (per-player)
- Player Interaction Networks

**Parked Until**: Sufficient player data for type-level training

---

### Experimental: SportVU Tracking Integration

**Status**: Alternative approach, not core. Explore if Phase 1 succeeds.

**Data Available**: 2015-16 SportVU tracking data (~632 games, publicly leaked)
- XY coordinates at 25fps for all 10 players + ball
- Source: [github.com/linouk23/NBA-Player-Movements](https://github.com/linouk23/NBA-Player-Movements)

**Why This Matters**: Tracking data captures spatial information (spacing, defensive pressure, player movement patterns) that PBP events don't encode. No one has used tracking for outcome prediction - only play classification.

#### Option A: Direct Prediction (Limited)

Train only on 2015-16 games with tracking + game state → outcome prediction.

```
Tracking frames (25fps) → Tracking Encoder → Tracking Embedding
GameState features → State Encoder → State Embedding
                          ↓
                   Concatenate + Fusion
                          ↓
                   Prediction Heads (spread, scores)
```

**Limitation**: Only ~632 games. Too small for robust training. Useful as proof-of-concept only.

#### Option B: Cross-Modal Distillation (More Interesting)

Use 2015-16 as a "Rosetta Stone" to teach the PBP encoder what tracking captures.

```
┌─────────────────────────────────────────────────────────────────────────┐
│            CROSS-MODAL ALIGNMENT (2015-16 only)                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  TRACKING PATH:                                                         │
│    Tracking frames → Tracking Encoder → z_tracking (256-d)              │
│                                                                         │
│  PBP PATH:                                                              │
│    PBP events → PBP Encoder → z_pbp (256-d)                             │
│                                                                         │
│  ALIGNMENT LOSS:                                                        │
│    L_align = ||z_tracking - z_pbp||² (same game, same window)           │
│                                                                         │
│  PREDICTION LOSS:                                                       │
│    L_pred = MSE(prediction, actual_outcome)                             │
│                                                                         │
│  TOTAL: L = L_pred + λ·L_align                                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

TRAINING:
  Phase A: Train both encoders jointly on 2015-16 with alignment + prediction loss
  Phase B: Freeze tracking encoder, fine-tune PBP encoder on full 27 seasons

INFERENCE:
  Only use PBP encoder (tracking knowledge is distilled)
```

**Key Insight**: The PBP encoder learns to approximate tracking-derived representations. At inference, we don't need tracking data - the PBP encoder has internalized what "good spacing" or "defensive pressure" looks like from the event patterns.

**Research Questions**:
1. Is 632 games enough to learn meaningful alignment?
2. Does the distilled PBP encoder actually improve predictions on non-tracking games?
3. What tracking features transfer best? (spacing, speed, ball movement?)

**Prerequisites**:
- Phase 1 architecture working
- SportVU data downloaded and processed
- Timestamp alignment between tracking and PBP

**Effort**: High. Novel research direction, not established path.

---

---

## Separate System: Hierarchical Compositional Framework

**Status**: Independent system concept. Not part of Phase 1-3 roadmap.

**Philosophy**: Build prediction bottom-up through explicit layers, each modeling a specific abstraction level. Fundamentally different from the encoder-based approach above - this is a separate system that could be built independently or in parallel.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 HIERARCHICAL COMPOSITIONAL FRAMEWORK                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  LAYER 4: GAME PREDICTION                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Inputs: Team predictions + game context                        │   │
│  │    - Schedule effects (B2B, rest, travel)                       │   │
│  │    - Home/away advantage                                        │   │
│  │    - Referee tendencies (optional)                              │   │
│  │    - Betting line context (optional)                            │   │
│  │  Output: Final spread, scores, player stats                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ↑                                         │
│  LAYER 3: TEAM PREDICTION                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Inputs: Lineup interaction outputs + team-level features       │   │
│  │    - Team rolling stats (pace, efficiency, etc.)                │   │
│  │    - Coaching style embeddings                                  │   │
│  │    - Team form/momentum                                         │   │
│  │  Output: Team strength estimate (neutral setting)               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ↑                                         │
│  LAYER 2: PLAYER INTERACTION / SYNERGY                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Architecture: Graph Neural Network                             │   │
│  │    - Nodes: Player embeddings from Layer 1                      │   │
│  │    - Edges: Teammate synergy, opponent matchups                 │   │
│  │    - Message passing: How players affect each other             │   │
│  │  Output: Lineup-aware player representations                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ↑                                         │
│  LAYER 1: PLAYER PREDICTION                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Goal: Model each player's individual quality/performance       │   │
│  │  Architecture: Per-player model or shared encoder + player ID   │   │
│  │    - Rolling stats (pts, reb, ast, efficiency)                  │   │
│  │    - Player archetype/role embedding                            │   │
│  │    - Career trajectory features                                 │   │
│  │  Output: Player quality vector (before lineup context)          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Layer Details**:

| Layer | Input | Model | Output |
|-------|-------|-------|--------|
| 1. Player | Player history, stats | MLP or per-player model | Player embedding (64-d) |
| 2. Interaction | Player embeddings × lineup | GNN (GAT/GCN) | Context-aware player emb |
| 3. Team | Interaction outputs + team stats | Aggregation + MLP | Team strength score |
| 4. Game | Team scores + context | MLP or attention | Final predictions |

**Training Approaches**:

1. **End-to-end**: Train all layers jointly with final prediction loss
2. **Layer-wise**: Pre-train each layer, then fine-tune together
3. **Hybrid**: Train L1-L2 on player/lineup data, L3-L4 on game outcomes

**Comparison: Two Different Systems**:

| Aspect | Compositional System | Encoder System (Phase 1-3) |
|--------|---------------------|-------------------|
| Philosophy | Bottom-up, explicit layers | End-to-end learning |
| Interpretability | High - each layer explainable | Low - black box |
| Modularity | High - layers independent | Low - monolithic |
| Player modeling | Explicit - per-player models | Implicit - learned embeddings |
| Error propagation | Risk of compounding | End-to-end optimized |
| Data efficiency | Better for small data | Needs more data |
| Emergent patterns | May miss cross-layer patterns | Can capture unexpected patterns |
| Implementation | More complex architecture | Simpler single system |
| Player predictions | Natural fit (Layer 1 output) | Requires additional head |
| Primary data | Player stats, lineup combinations | PBP sequences |

**When to Build This System**:

- If interpretability and modularity are priorities
- If player-level predictions are the primary goal
- If you want to validate each abstraction layer independently
- If you want explicit control over how player interactions are modeled

**When to Build Encoder System Instead**:

- If you want to validate sequence modeling first
- If simplicity of implementation matters
- If you trust end-to-end learning to find patterns
- If PBP sequences are the primary signal

**These Are Independent**:

These two systems could be built in parallel or sequentially. They use some of the same underlying data but take fundamentally different approaches. Results from one could inform the other, but they don't share architecture.

**Effort**: Medium-High. More architectural complexity but well-established components (GNNs, per-player models).

---

### Implementation Roadmap

```
Week 1: Data Pipeline
  ├─ Event tokenization from PBP_Logs
  ├─ Historical game sequence construction
  ├─ Train/val/test split (temporal)
  └─ DataLoader with variable-length handling

Week 2: Model Implementation
  ├─ Event Encoder (Transformer)
  ├─ Game-level pooling
  ├─ Temporal Attention module
  └─ Multi-task prediction heads

Week 3: Training & Evaluation
  ├─ Training loop with logging
  ├─ Hyperparameter tuning
  ├─ Ablation studies (sequence vs features)
  └─ Comparison vs XGBoost baseline

Week 4: Analysis & Decision
  ├─ Error analysis
  ├─ Feature importance (if interpretable)
  ├─ Decision: proceed to pre-training?
  └─ Document findings

Month 2+: Pre-training (if Phase 1 succeeds)
  ├─ Cloud setup
  ├─ Pre-training run
  ├─ Fine-tuning
  └─ Final evaluation
```

---

## Research Sources

### Primary Papers
- [NCAA Basketball Deep Learning](https://arxiv.org/html/2508.02725v1) - Transformer vs LSTM comparison
- [Long-Sequence LSTM for NBA](https://arxiv.org/abs/2512.08591) - 8-season temporal modeling
- [HIGFormer (KDD 2025)](https://arxiv.org/pdf/2507.10626) - Player-Team graph transformer
- [Uncertainty-Aware NBA Forecasting](https://www.mdpi.com/2078-2489/17/1/56) - Bayesian RNN approach
- [Stacked Ensemble NBA Prediction](https://www.nature.com/articles/s41598-025-13657-1) - SHAP interpretability
- [Set Transformer](https://www.researchgate.net/publication/333918639_Set_Transformer_A_Framework_for_Attention-based_Permutation-Invariant_Neural_Networks) - Permutation-invariant sets

### Related Work
- [GameSense Basketball Tracking](https://www.nature.com/articles/s41598-025-29586-y) - Hierarchical transformer
- [Seq2Event Soccer](https://eprints.soton.ac.uk/458099/1/KDD22_paper_CReady_v20220606.pdf) - Event language modeling
- [SoccerTransformer Pre-training](https://dtai.cs.kuleuven.be/events/MLSA24/papers/2.pdf) - Self-supervised approach
- [Sports ML Systematic Review](https://arxiv.org/html/2410.21484v1) - Betting context

---

## Context Summary (for new conversations)

**Project**: NBA game/player prediction system using play-by-play data as primary input.

**Data Available**:
- 18M play-by-play events (2000-2026, 37K games)
- ~492 plays/game with: actionType, clock, scores, playerIds, teamTricode
- PlayerBox stats (pts/reb/ast/stl/blk + shooting) for 2023+ seasons
- 5,118 players with stable personId identifiers
- Betting lines (2007+), injury reports (2018+)

**Current System**: XGBoost on 43 hand-crafted features → home/away scores (MAE 10.1)

**Target Outputs** (priority order):
1. Game spread prediction (primary metric)
2. Final scores (home/away/total)
3. Player stats (pts/reb/ast/stl/blk) for all ~26 players per game
4. Point-in-time predictions during games
5. Confidence intervals (nice-to-have)

**Key Requirements**:
- Pre-game predictions most important
- Full sequence modeling of PBP data preferred
- Per-player predictions (not roster slots)
- Model should handle 27 seasons including era changes
- Black-box OK, RTX 2070 SUPER local + cloud available
- Inference latency: seconds acceptable

**Key Files**:
- `DATA_MODEL.md` - Complete database schema reference
- `src/predictions/features.py` - Current 43-feature engineering
- `src/database_updater/game_states.py` - PBP → GameState parsing
- `src/database_updater/boxscores.py` - PlayerBox schema

**Research Questions**:
1. Best architecture for sequence modeling PBP data?
2. How to jointly predict game outcomes + player stats?
3. How to handle variable-length player rosters?
4. Pre-training strategies with 18M plays?
