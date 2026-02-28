# NBA Prediction Architecture

> **Status**: Phase 1 Complete (15 experiments) | Phase 2 Complete (7 experiments) | Phase 3 In Progress
> **Last Updated**: February 27, 2026

---

## Overview

This project predicts NBA game outcomes (point spreads, scores, win probabilities) using a **sequence modeling** approach. A transformer processes each team's full season of historical games — scores, opponents, rosters, per-player contributions, and recent in-game score trajectories — to produce probabilistic predictions with calibrated uncertainty.

Phase 1 explored PBP (play-by-play) history, roster encoding, and schedule embeddings across 15 experiments, establishing a Spread MAE ceiling of ~12.2. Phase 2 redesigned the architecture around richer per-game representations and full-season context, breaking that ceiling (MAE 11.61) and substantially improving win prediction (AUC 0.592 → 0.687). Phase 2 maximized standard transformer approaches for direct score prediction; future phases explore alternative architectures.

---

## Phase Progression

```text
Phase 1a: PBP History Only (Complete — 8 experiments)
├─ Two-stream architecture: home history + away history
├─ Enriched tokenization: 10-component tokens (added shot features)
├─ GameStates ablation: proved PBP adds no signal beyond score trajectories
├─ Finding: Spread MAE ceiling at ~12.2 regardless of PBP enrichment
└─ Output: Baseline models, ablation studies, GameStates comparison

Phase 1b: + Roster Stream (Complete — 2 experiments)
├─ RosterEncoder: shared player_emb (64-d) → self-attention → attention pooling
├─ Finding: Roster improves win prediction (AUC 0.583) but not spread
└─ Output: Best val_loss (37.25), deepest training (epoch 31)

Phase 1c: + Schedule Stream (Complete — 2 experiments)
├─ Temporal embeddings: days_before_target + season_game_number
├─ Finding: No improvement — positional encoding already captures ordering
└─ Output: Confirmed schedule embeddings are too indirect

Combined Phase 1: All Features (Complete — 3 experiments)
├─ PBP + Roster + Schedule + Rest Days + per-team combine + SimpleFusion
├─ Finding: Best overall — Win AUC 0.592, Spread MAE 12.20
└─ Output: Phase 1 best model, transition insights for Phase 2

Phase 2: Maximize Standard Transformer (Complete — 7 experiments)
├─ Goal: maximize standard transformer approaches for direct score prediction
├─ Exp 1: Baseline — full-season context, per-player points, 512-d
│         Broke 12.2 ceiling: MAE 11.70, AUC 0.674
├─ Exp 2: N=10 recent games — marginal gain, not worth 70% more compute
├─ Exp 3: +Player form encoder, +team-relative scores, 8 pool queries
│         MAE 11.67, AUC 0.682
├─ Exp 4: Loss fix (MSE 1.0, sigma cap), gated dynamics, rest
│         simplification, fusion dropout 0.1 — regressed vs Exp 3
├─ Exp 5a: Fusion residual + revert harmful loss changes
│         MAE 11.61, AUC 0.687, 90% Coverage 72.7% — Phase 2 best
├─ Exp 6: Derived spread + model reduction — regressed (MAE 11.84)
├─ Exp 7: PLE + cross-attention fusion + Huber — regressed (MAE 11.73)
└─ Outcome: ~11.6 MAE plateau; standard transformer approaches
           exhausted → move to Phase 3

Phase 3: Alternative Architectures (In Progress — 6 experiments planned)
├─ Goal: break ~11.6 MAE plateau with different model architectures
├─ Exp 1: Time-aware bidirectional GRU (replace temporal attention)
├─ Exp 2: Self-supervised pre-training (masked game prediction on 33K games)
├─ Exp 3: Multi-stat player contributions (10 stats + position from PlayerBox)
├─ Exp 4: Player interaction graph (self-attention between players in games)
├─ Exp 5: Full heterogeneous graph (HIGFormer-inspired multi-pass)
└─ Exp 6: Best-of-everything (combine winners from Exps 1-5)
```

---

## Data Sources

All data comes from existing database tables — no external APIs required.

**GameStates** (primary):
- Score trajectories for recent N games (full ~500 states per game, regulation only)
- Final state for ALL prior season games: scores, per-player points from `players_data` JSON
- Team-relative encoding: scores[0] is always the team's score, margin from team's perspective
- GameStates buckets swapped/inverted when team was away (consistent perspective)

**Games** (scheduling):
- Game dates → days-before-target positional encoding (calendar distance)
- Rest days per team (Embedding(30, 64))
- Season identification for train/val/test splits

**Player Embeddings** (128-d, trained from scratch):
- Shared between per-game encoder and roster encoder
- Random initialization — no Phase 1 transfer
- Frequent players get 250-350+ gradient updates across 5 training seasons

**Data Split** — strict chronological, no leakage:
- Train: 2018-19 through 2022-23 (5 seasons, ~5,800 games)
- Val: 2023-24 (~1,230 games)
- Test: 2024-25 + 2025-26 (~1,800 games)
- Minimum 3 prior same-season games required per team

---

## Phase 1 Summary

Phase 1 explored four data streams across 15 experiments using a two-stream PBP transformer (5.7-6.2M params). Each team's last 5 games of play-by-play data were encoded through a 4-layer transformer, processed by temporal attention, and fused for prediction. The architecture tested PBP-only, GameStates-only, roster, schedule, and combined configurations.

### Phase 1 Results

| # | Experiment | Spread MAE | Avg Score MAE | Win Acc | Win AUC | Params |
| - | ---------- | ---------- | ------------- | ------- | ------- | ------ |
| 1 | PBP Baseline (5 seasons) | 12.20 | — | — | — | 5.7M |
| 6 | Shot Features (5 seasons) | 12.29 | 10.15 | 52.7% | 0.567 | 5.85M |
| 8 | GameStates Only (5 seasons) | 12.36 | 10.14 | 52.0% | 0.509 | 5.59M |
| 12 | Roster (5 seasons) | 12.33 | 10.12 | 56.9% | 0.583 | 6.1M |
| **13** | **Combined v1 (all features)** | **12.20** | **10.18** | **57.6%** | **0.592** | **6.07M** |
| 15 | Roster Only (no PBP) | 12.59 | 10.32 | 52.4% | 0.523 | 911K |

### Key Phase 1 Findings

1. **Spread MAE ceiling at ~12.2**: All configurations converged to the same spread accuracy regardless of input enrichment, history length, or training data volume.

2. **PBP adds no signal beyond score trajectories**: GameStates-only (5 features per row) matched full PBP (10 features per play). Detailed play-by-play contributes nothing beyond score evolution.

3. **Roster identity helps win prediction, not spread**: RosterEncoder achieved best Win AUC (0.583) and deepest training (epoch 31), but Spread MAE stayed at ~12.3.

4. **~2-point gap vs XGBoost**: XGBoost (MAE ~10.1) uses 43 engineered features — the transformer needed richer data utilization, not more model capacity.

**Insights driving Phase 2**: The bottleneck was data utilization. With only 5 recent games as context, most of the season was thrown away. The model knew WHO was playing but not HOW WELL. GameStates `players_data` contained per-player points that were discarded. Every prior game needed to contribute.

---

## Phase 2 Architecture (Current)

Phase 2 maximizes standard transformer approaches for direct score prediction. It replaces PBP with richer per-game representations over the full season. Each historical game is encoded as a holistic embedding containing team-relative scores, opponent identity, location, and per-player point contributions. Temporal attention processes all prior same-season games (3-82) with calendar-distance positional encoding.

### Architecture Diagram

```text
═══════════════════════════════════════════════════════════
                 PHASE 2 ARCHITECTURE (Exp 4)
═══════════════════════════════════════════════════════════

Per Team (home and away independently):

┌──── Per-Game Encoder (ALL prior season games) ─────────┐
│                                                          │
│  For EACH of up to 82 prior games:                      │
│                                                          │
│    Scores (team-relative, normalized):                   │
│      [team_score, opp_score, margin, total]              │
│      → Linear(4, 128) → LN → GELU                      │
│                                                          │
│    Opponent: team_id → Embedding(30, 64)                 │
│    Location: was_home → Embedding(2, 32)                 │
│                                                          │
│    Player Contributions:                                 │
│      For each player in game:                            │
│        Concat([player_emb(128-d), norm_pts]) → 129-d     │
│        → Linear(129, 256) → LN → GELU                   │
│      Attention Pool (1 query, 4 heads) → 256-d           │
│                                                          │
│    Concat([score(128), opp(64), loc(32), player(256)])   │
│    → Linear(480, 512) → LN → game_context (512-d)       │
│                                                          │
│  Gated Dynamics Merge (recent N games only):             │
│    GameStates (~500 states) → GS Encoder (4L, 8H)       │
│    → Attention Pool → dynamics (512-d)                   │
│    gate = σ(Linear(512, 512))                            │
│    game_repr = context + gate · proj(dynamics)            │
│                                                          │
│  Older games: game_repr = game_context (no dynamics)     │
│                                                          │
└──────────────────────────────────────────────────────────┘
                        ↓
          Sequence of 3-82 game_reprs per team

┌──── Temporal Attention ────────────────────────────────┐
│                                                          │
│  days_before_target → Embedding(180, 512)                │
│  3-layer pre-norm Transformer (8 heads, FF=2048)         │
│  8-query attention pool → Concat(8×512)                  │
│  → Linear(4096, 512) → LN → season_repr (512-d)         │
│                                                          │
└──────────────────────────────────────────────────────────┘

┌──── Target Game Context ───────────────────────────────┐
│                                                          │
│  Player Form Encoder (per roster player):                │
│    Recent appearances: (points, days_ago) tuples          │
│    → 1-layer Transformer (d=64, 4 heads)                 │
│    → Attention Pool → form_vector (64-d)                 │
│                                                          │
│  Roster Encoder:                                         │
│    Concat([player_emb(128-d), form(64-d)]) per player    │
│    → Linear(192, 512) → LN → GELU                       │
│    → 2-layer self-attention (8 heads, FF=2048)           │
│    → Attention Pool → roster_repr (512-d)                │
│                                                          │
│  Rest: rest_days → Embedding(30, 64) → rest (64-d)      │
│                                                          │
└──────────────────────────────────────────────────────────┘

┌──── Per-Team Combine ──────────────────────────────────┐
│                                                          │
│  Concat([season(512), roster(512), rest(64)])             │
│  → Linear(1088, 512) → LN → team_repr (512-d)           │
│                                                          │
└──────────────────────────────────────────────────────────┘

     ↓ (home_team_repr)          ↓ (away_team_repr)

┌──── Fusion ────────────────────────────────────────────┐
│                                                          │
│  Concat([home, away, home−away, home⊙away]) → 2048-d    │
│  → Linear(2048, 1024) → LN → GELU → Dropout(0.1)       │
│  → Linear(1024, 512) → LN → GELU → Dropout(0.1)        │
│  → matchup_repr (512-d)                                  │
│                                                          │
└──────────────────────────────────────────────────────────┘
                        ↓

┌──── Prediction Heads ──────────────────────────────────┐
│                                                          │
│  Spread: Linear(512→256) → LN → GELU → Linear(256→1)   │
│          σ via softplus, clamped to [1.0, 8.0]           │
│                                                          │
│  Scores: Linear(512→256→256→1) (deeper, per target)     │
│          σ via softplus + min_std 5.0                    │
│          Bias init: home=110, away=108                   │
│                                                          │
│  Win: P(home > away) via Gaussian CDF on score dists    │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### Model Specifications

| Component | Specification |
| --------- | ------------- |
| **Hidden dimension** | 512 |
| **Player embeddings** | 128-d, ~4,000 players |
| **Total parameters** | ~39M (Exp 4 config) |
| Per-game encoder | Concat([score(128), opp(64), loc(32), player(256)]) → Linear(480, 512) |
| GameStates encoder | 4-layer Transformer, 8 heads, FF=2048, attention pool (1 query) |
| Gated dynamics | gate = σ(Linear(512)) · Linear(512) + context; additive for recent N |
| Temporal attention | 3-layer Transformer, 8 heads, FF=2048, 8-query attention pool |
| Player form encoder | 1-layer Transformer (d=64, 4 heads), attention pool per player |
| Roster encoder | Linear(192, 512) per player → 2-layer self-attention (8 heads) → attention pool |
| Rest embedding | Embedding(30, 64) — no projection to 512 |
| Team combine | Linear(1088, 512) — concat season(512) + roster(512) + rest(64) |
| Fusion | Concat 4 interaction features (2048-d) → MLP 2048→1024→512, dropout 0.1 |
| Prediction heads | Spread: 512→256→1; Scores: 512→256→256→1 (deeper); σ clamped |
| Context window | All prior same-season games (3-82 per team) |
| Recent dynamics | Last 5 games with full GameStates (~500 states each) |

### GameStates Encoder Detail

The GameStates encoder processes in-game score trajectories for recent games:

| Feature | Embedding | Dimension |
| ------- | --------- | --------- |
| Period | Embedding(5, 16) | 16 |
| Clock bucket | Embedding(721, 16) | 16 |
| Home score bucket | Embedding(51, 16) | 16 |
| Away score bucket | Embedding(51, 16) | 16 |
| Margin bucket | Embedding(121, 16) | 16 |
| **Total** | Concat → Linear(80, 512) | **512** |

Score buckets are team-relative: when the team was away, home/away buckets are swapped and margin is inverted. This ensures the encoder always sees scores from the team's perspective.

---

## Phase 2 Experiment Results

> **Hardware**: RTX 2070 SUPER (8GB VRAM) | ~6.5 min/epoch
> **Test Set**: 2024-2025 + 2025-2026 seasons

| # | Experiment | Config | Spread MAE | Win AUC | Win Acc | Brier | ECE | Best Epoch | Params |
| - | ---------- | ------ | ---------- | ------- | ------- | ----- | --- | ---------- | ------ |
| 1 | Baseline (N=5, full architecture) | `phase2_baseline` | 11.70 | 0.674 | 62.9% | 0.2305 | 0.046 | 8 | 38.2M |
| 2 | N=10 recent games | `phase2_recent10` | 11.68 | 0.678 | 63.1% | 0.2277 | 0.021 | 11 | 38.2M |
| 3 | +Player form, +team-relative, 8 queries | `phase2_enhanced` | **11.67** | **0.682** | — | — | — | — | ~38.3M |
| 4 | Loss fix, gated dynamics, rest simplify | `phase2_exp4_fixes` | 11.69 | 0.673 | 61.9% | 0.2312 | 0.043 | 8 | 39.1M |
| **5a** | **Fusion residual + revert loss** | `phase2_exp5a_revert_loss` | **11.61** | **0.687** | **63.0%** | **0.2273** | — | 8 (ES 23) | ~39M |
| 6 | Derived spread + model reduction | `phase2_exp6_simplify` | 11.84 | 0.668 | 61.4% | 0.2338 | — | 7 (ES 17) | ~33M |
| 7 | PLE + cross-attention fusion + Huber | `phase2_exp7_ple_crossattn` | 11.73 | 0.669 | 62.2% | 0.2306 | — | 7 (ES 22) | ~39M |

**Exp 2 findings**: N=10 is marginally better than N=5 across all metrics (Spread MAE -0.02, AUC +0.004) but at 70% more compute per epoch (~11 min vs ~6.5 min). Calibration improved notably (ECE 0.046→0.021). Diminishing returns — N=15/20 not worth running.

**Exp 3 findings**: Player form encoder, team-relative scores, and 8 temporal pool queries collectively improved spread (11.70→11.67) and win prediction (AUC 0.674→0.682). Best results so far.

**Exp 4 findings**: The four fixes regressed vs Exp 3 (MAE 11.69 vs 11.67, AUC 0.673 vs 0.682). The sigma cap (max_std=8.0) crushed 90% coverage from ~78% to 64.4% — the model can no longer express appropriate uncertainty. Gated dynamics, MSE weight boost, and reduced fusion dropout had no net positive effect. The spread collapse diagnosis was valid (Exp 3 predicted near-zero spreads), but these particular fixes overcorrected on uncertainty while not improving point predictions.

**Exp 5a findings**: Fusion residual connection (output = interaction(combined) + diff) ensures raw team-difference signal always reaches prediction heads. Removed redundant final LN/GELU/Dropout in fusion. Reverted sigma cap (removed spread_max_std) and set MSE weight to 0.3 (balanced between 0.1 trap and 1.0 domination). New Phase 2 best: Spread MAE 11.61, Win AUC 0.687, 90% Coverage 72.7%. Best epoch 8, early stopped at 23.

**Exp 6 findings**: Derived spread from scores (spread = home - away, no separate head) plus model reduction (2 temporal layers, 4 pool queries, reduced FF dims). The derivation constraint hurt — the model benefits from an independent spread head that can learn spread-specific patterns beyond raw score difference. Model reduction also cost performance. Regressed to MAE 11.84.

**Exp 7 findings**: Three changes tested simultaneously: (1) PLE (Piecewise Linear Encoding) for score features, (2) cross-attention fusion replacing concat+MLP, (3) Huber loss replacing MSE component. All three changes together regressed (MAE 11.73, AUC 0.669 vs Exp 5a's 11.61, 0.687). Cross-attention fusion may have been too aggressive a change — it replaces a proven interaction-feature approach with attention that must learn which features matter. PLE adds complexity without clear gain at these feature dimensions. Phase 2 exploration exhausted.

### Phase 2 vs Phase 1 vs XGBoost

| Metric | Phase 1 Best | Phase 2 Best | Delta | XGBoost |
| ------ | ------------ | ------------ | ----- | ------- |
| Spread MAE | 12.20 | **11.61** | **-0.59** | ~10.1 |
| Win Accuracy | 57.6% | **63.0%** | **+5.4pp** | — |
| Win AUC | 0.592 | **0.687** | **+0.095** | — |
| Brier Score | 0.2471 | **0.2273** | **-0.020** | — |

---

## Loss Function

### Current Configuration (Exp 4)

```text
Total = 1.0 × SpreadLoss + 0.5 × ScoreLoss + 0.3 × WinProbLoss + 0.1 × ConsistencyLoss

SpreadLoss = 1.0 × NLL(target, μ_spread, σ_spread) + 1.0 × MSE(target, μ_spread)
ScoreLoss  = 1.0 × NLL(targets, μ_scores, σ_scores) + 1.0 × MSE(targets, μ_scores)
WinProbLoss = BCE(home_win_prob, actual_outcome)
ConsistencyLoss = MSE(μ_spread, μ_home − μ_away)
```

Where `NLL = 0.5 × log(2π) + log(σ) + 0.5 × ((y − μ) / σ)²`

Win probability is derived analytically: `P(home wins) = Φ((μ_home − μ_away) / √(σ²_home + σ²_away))` — not predicted by a separate head.

### The Sigma Inflation Trap

In Exps 1-3, MSE weight was 0.1 and spread sigma was uncapped. The model learned to inflate σ_spread (~10.7), which reduced the NLL gradient on μ by ~70× compared to pure MSE. The equilibrium: predict μ ≈ 0 with high uncertainty (mean |predicted spread| was 3.68 vs actual 12.45).

Exp 4 fixes: MSE weight raised to 1.0 (provides direct, σ-independent gradient on μ) and spread sigma capped at 8.0 (prevents the inflation trap). Score sigma remains uncapped with min_std=5.0.

### Consistency Loss

Encourages coherent predictions: `MSE(μ_spread, μ_home − μ_away)`. The spread and score heads predict from the same fusion representation but have independent parameters. Without this term, they can diverge.

---

## Training Configuration

| Setting | Value |
| ------- | ----- |
| Optimizer | AdamW (lr=1e-4, weight_decay=0.1, betas=(0.9, 0.98)) |
| LR schedule | Cosine annealing (warmup 5% of steps, min_lr=0.1×initial) |
| Max epochs | 100 (early stopping does the work) |
| Early stopping | val_spread_mae, patience=15, 3-epoch smoothing |
| Effective batch | 32 (micro-batch 4 × gradient accumulation 8) |
| Gradient clipping | max_norm=1.0 |
| Mixed precision | AMP float16 |
| EMA | decay=0.999 (used for evaluation) |
| Augmentation | Home/away swap with negated spread (doubles effective data) |
| Gradient checkpointing | Applied to GameStates encoder (saves VRAM) |
| Per-epoch time | ~6.5 min (N=5) on RTX 2070 SUPER |
| Typical convergence | Best epoch 8-11, early stop ~23-26 |

### Per-Component Dropout

| Component | Dropout |
| --------- | ------- |
| GameStates encoder | 0.1 |
| Temporal attention | 0.1 |
| Player contribution pool | 0.2 |
| Roster self-attention | 0.2 |
| Fusion MLP | 0.1 (was 0.3 pre-Exp 4) |
| Prediction head MLPs | 0.3 |

---

## Key Design Decisions

### 1. Attention Pooling Everywhere

Phase 1 mean-pooled rich per-event embeddings into a single vector — collapsing most learned representation. Phase 2 uses attention pooling with learned queries at every aggregation point: player contributions within games, GameStates dynamics, temporal season summary, and roster encoding. The model learns WHAT to focus on.

### 2. Full Season Context

Every prior same-season game contributes (3-82 tokens). At O(82²) ≈ 6,700 operations, temporal attention is trivially cheap compared to Phase 1's O(750²) per-game PBP. Early-season predictions with sparse context (3-5 games) naturally produce higher uncertainty via padding masks.

### 3. Gated Dynamics Merge

Only the last N games have full GameStates trajectories. The original approach zero-filled the dynamics dimension for older games (94% of samples), forcing all games through a Linear(1024, 512) bottleneck where half the input was zeros. The gated approach: older games pass through unchanged as pure context; recent games additively contribute gated dynamics: `game_repr = context + gate · proj(dynamics)`.

### 4. Concatenation Over Addition

Different information types (scores, opponent identity, player contributions, game dynamics) should not interfere in the same embedding dimensions. Concatenation + linear projection lets the model learn how to combine without information loss. Used throughout: per-game features, per-team combine, fusion interaction features.

### 5. Explicit Interaction Features in Fusion

Rather than requiring the MLP to learn subtraction and multiplication from raw concatenated team vectors, fusion provides explicit `home - away` (strength gap) and `home * away` (feature interactions) alongside the raw vectors.

### 6. Team-Relative Score Encoding

Scores are encoded from each team's perspective: scores[0] is always the team's score, scores[2] is always the team's margin. GameStates buckets are swapped when the team was away. This eliminates the need for the model to learn home/away perspective switching.

### 7. Probabilistic Outputs with Sigma Cap

Predict (μ, σ) for each target via Gaussian NLL. Enables calibrated uncertainty and Kelly criterion betting. Spread σ is clamped to [1.0, 8.0] to prevent the sigma inflation trap. Score σ uses softplus + min_std=5.0 without upper cap.

### 8. Consistency Loss

Lightweight penalty `MSE(μ_spread, μ_home − μ_away)` encourages coherent predictions across the independent spread and score heads without tightly coupling them.

### 9. Player Form Encoder

Learns per-player scoring patterns from recent appearances rather than relying on static embeddings. Each roster player's recent (points, days_ago) history is processed through a small transformer (d=64) and concatenated with the player's ID embedding before roster self-attention. Players with no recent history use a learned fallback embedding.

---

## File Structure

```text
src/transformer/
├── __init__.py
├── tokenizer.py                # PBP → 10-component discrete tokens (Phase 1)
├── sequence_builder.py         # Historical game sequence construction (Phase 1)
├── dataset.py                  # Phase 1 PyTorch Dataset
├── dataloader.py               # DataLoader factory functions
├── models/
│   ├── __init__.py
│   ├── event_encoder.py        # EventEmbedding + Transformer encoder (Phase 1)
│   ├── temporal_attention.py   # Attention over game history (Phase 1)
│   ├── fusion.py               # SimpleFusion + CrossAttentionFusion (Phase 1)
│   ├── roster_encoder.py       # Self-attention roster encoder (Phase 1)
│   ├── prediction_heads.py     # Gaussian (μ, σ) output heads (shared)
│   └── phase1_model.py         # Phase1Model
├── training/
│   ├── __init__.py
│   ├── loss.py                 # Phase 1 NLL + MSE + BCE combined loss
│   ├── metrics.py              # MAE, AUC, ECE, coverage, CRPS
│   ├── trainer.py              # Phase 1 training loop
│   └── config.py               # Phase 1 ModelConfig + ExperimentConfig
├── evaluation/
│   ├── __init__.py
│   ├── evaluate.py             # Test set evaluation
│   ├── ablation.py             # Ablation study runner
│   └── visualize.py            # Calibration plots, attention viz
├── gamestates/
│   ├── __init__.py
│   ├── tokenizer.py            # Score trajectory tokenizer (Phase 1)
│   ├── sequence_builder.py     # GameStates sequence construction (Phase 1)
│   ├── dataset.py              # GameStates dataset (Phase 1)
│   └── model.py                # GameStates-only model (Phase 1)
└── phase2/
    ├── __init__.py
    ├── config.py               # Phase2Config dataclass
    ├── dataset.py              # Phase2Dataset with full-season context
    ├── sequence_builder.py     # Season context + GameStates builder
    ├── cache_builder.py        # Pre-cache per-game features to disk
    ├── models/
    │   ├── __init__.py
    │   ├── phase2_model.py     # Phase2Model (main model)
    │   ├── per_game_encoder.py # Scores + opponent + location + players
    │   ├── gamestates_encoder.py # 4-layer Transformer for score trajectories
    │   ├── temporal_attention.py # 3-layer Transformer + 8-query pool
    │   ├── player_form_encoder.py # Per-player scoring history
    │   ├── roster_encoder.py   # Project-up-first + self-attention + pool
    │   ├── fusion.py           # Interaction features + MLP
    │   ├── piecewise_linear.py # PLE for score features (Exp 7)
    │   └── temporal_gru.py    # Time-aware bidirectional GRU (Phase 3 Exp 1)
    └── training/
        ├── __init__.py
        ├── loss.py             # Phase2CombinedLoss + ConsistencyLoss
        ├── trainer.py          # Training loop with EMA, augmentation
        └── ema.py              # Exponential moving average

scripts/
├── train_transformer.py        # Phase 1 training entry point
├── train_gamestates.py         # Phase 1 GameStates training
├── train_phase2.py             # Phase 2 training entry point
├── evaluate_transformer.py     # Evaluation entry point
└── generate_performance_chart.py

configs/transformer/
├── phase2_baseline.yaml        # Exp 1: full Phase 2 architecture
├── phase2_recent10.yaml        # Exp 2: N=10 recent games
├── phase2_enhanced.yaml        # Exp 3: +player form, +team-relative, 8 queries
├── phase2_exp4_fixes.yaml      # Exp 4: loss fix, gated dynamics, rest simplify
├── phase2_exp5a_revert_loss.yaml # Exp 5a: fusion residual + revert harmful loss changes
├── phase2_exp7_ple_crossattn.yaml # Exp 7: PLE + cross-attention fusion + Huber
├── phase3_exp1_gru.yaml        # Phase 3 Exp 1: time-aware bidirectional GRU
├── phase2_ordinal_pos.yaml     # Ablation: ordinal vs days-before positional encoding
├── combined_v1.yaml            # Phase 1 best: all features + SimpleFusion
├── combined_v2.yaml            # Phase 1: CrossAttentionFusion variant
├── roster_only.yaml            # Phase 1 ablation: roster + rest only
├── full_baseline.yaml          # Phase 1a: original PBP baseline
└── ...                         # Additional Phase 1 configs (ablations, variants)
```

---

## Project Roadmap

### Phase 3 — Alternative Architectures, Direct Score Prediction

Same prediction target (final game scores), same data pipeline. Diversify architecture choices to break the ~11.6 MAE plateau. Six experiments planned across four lines:

**Line A — GRU Temporal**:
- **Exp 1**: Replace temporal attention (3-layer transformer + pool) with a time-aware bidirectional GRU. GRU gates provide exponential decay natively — the inductive bias the transformer must learn from positional encoding. Calendar distance embedded as 64-d input feature. Same MultiQueryAttentionPool over GRU hidden states.

**Line B — Pre-Training**:
- **Exp 2**: Self-supervised pre-training via masked game prediction on all 33K games (6x more than supervised training). Pre-train per-game encoder + temporal module, then fine-tune with prediction heads. Expand training seasons from 5 to 8+.

**Line C — Richer Player Data**:
- **Exp 3**: Expand PlayerContributionEncoder from 1 stat (points) to 10 stats (pts, reb, ast, stl, blk, tov, min, fgm, fg3m, plus_minus) + position embedding. Blocked on box score data backfill for historical seasons.

**Line D — Heterogeneous Graph (HIGFormer-inspired)**:
- **Exp 4**: Add player-player self-attention within games before pooling. Currently players are pooled via attention to a learned query with no interaction. Self-attention lets players exchange information (e.g., "LeBron + AD together" produces a different representation).
- **Exp 5**: Full multi-pass heterogeneous architecture: (1) player interaction within games, (2) per-player trajectory tracking across games, (3) temporal aggregation enriched by player context, (4) team fusion.
- **Exp 6**: Combine winners from all lines into the best possible direct prediction model.

### Future Avenues (Phase 4+)

1. **Player Props**: Predict individual player stats (pts/reb/ast/blk/stl) alongside game scores. Builds on whichever Phase 3 architecture wins. The per-player encoding pipeline is already in place; extend prediction heads to player-level outputs.

2. **Live Prediction**: Game states as additional input for in-progress games. The GameStates encoder already processes score trajectories — extend to real-time updates for live win probability and final score projections.

3. **Generative / Next-State Prediction**: Predict the next game state (like LLMs do next-token) if direct prediction approaches plateau. A generative model could simulate game trajectories and produce distributional predictions through sampling.

### Cross-Cutting Concerns

These are not tied to a specific phase — they represent fundamental improvements applicable across all architectures:

1. **Data Utilization**: 20+ NBA seasons available in the database, but only using 5 for training. More historical data could help the model learn longer-term patterns, though older seasons may have different dynamics (pace, three-point revolution, etc.).

2. **Compute Utilization**: Under 50% VRAM on an RTX 2070 SUPER (8GB), training runs complete in hours, and we have not needed AWS. There is substantial headroom to scale model size, batch size, or data volume before hitting compute limits.

3. **Temporal Freshness**: The strict chronological train/test split means the model cannot learn from recent team quality changes (trades, injuries, hot streaks). Current setup: train on 2018-2023, test on 2024-2026. The model has never seen any 2024+ data during training. Continuous learning or rolling updates could ensure every game serves as training data, context, or test data — no wasted information.

4. **Player/Team Signal Preservation**: The encoding pipeline may have bottlenecks where player and team identity gets washed out. Players go through attention pooling (variable players) to 256-d, then into 512-d game context, then pooled again across games, then combined. Individual identity is compressed through multiple bottlenecks. The model needs to know LeBron is LeBron, not just "a player who scored 27." Teams and players vary season-to-season and within-season — the representation must preserve this granularity rather than averaging a star player into the whole team signal.

---

## Research Foundation

- **Sharp Sports Betting** (Millman 2021): Calibration > accuracy for profitability
- **PTIN** (Wang+ 2024): Transformer architectures work for NBA prediction
- **HGT** (Zhao+ 2024): Pre-training on play sequences improves downstream tasks
- **NBAFM** (Zhang+ 2024): Foundation models generalize across prediction types
