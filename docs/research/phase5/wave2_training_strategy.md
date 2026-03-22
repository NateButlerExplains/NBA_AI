# Phase 5 Training Strategy: 4-Level Hierarchical Model

**Date**: 2026-03-16
**Status**: Design document -- pre-implementation
**Target**: Beat Phase 3 Exp 9 ensemble (Spread MAE 10.66, Win AUC 0.718)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Recommended Training Pipeline](#2-recommended-training-pipeline)
3. [Loss Functions Per Level](#3-loss-functions-per-level)
4. [Data Usage Plan](#4-data-usage-plan)
5. [Validation Strategy](#5-validation-strategy)
6. [Ablation Plan](#6-ablation-plan)
7. [Risk Analysis](#7-risk-analysis)
8. [Research Basis](#8-research-basis)

---

## 1. Executive Summary

### The Core Problem

Training 4 interconnected levels that must each be independently meaningful while producing
a coherent final prediction is a multi-task, multi-timescale optimization problem. The
literature on hierarchical models (recommendation systems, NLP, computer vision) converges
on a clear answer: **staged pre-training followed by end-to-end fine-tuning with
discriminative learning rates**.

### Recommended Approach: Three-Phase Training

| Phase | What | Duration | Key Idea |
|-------|------|----------|----------|
| **Phase A** | Bottom-up pre-training (L1, then L2) | ~60% of compute | Each level learns meaningful representations against level-specific targets |
| **Phase B** | Top-down assembly (L3 + L4 on frozen L1-L2) | ~15% of compute | Upper levels learn to use lower-level representations |
| **Phase C** | End-to-end fine-tuning (all levels, discriminative LR) | ~25% of compute | Joint optimization with 10x-100x lower LR on pre-trained levels |

This is the same pattern used in:
- **NLP**: Pre-train language model, then fine-tune on downstream task (BERT/GPT paradigm)
- **Recommendation systems**: Pre-train user/item embeddings, then fine-tune in the ranking model
- **Computer vision**: Pre-train backbone on ImageNet, freeze early layers, fine-tune later layers
- **Hierarchical RL**: Train sub-policies, then meta-controller coordinates them

### Why Not Pure End-to-End?

1. **Gradient starvation**: L1 is 4 layers removed from the final loss. Gradients attenuate
   through L2, L3, L4, making L1 updates noisy and slow.
2. **Team contamination**: End-to-end training incentivizes L1 to encode team-level patterns
   (e.g., "this player is on the Warriors") because that directly helps spread prediction,
   but destroys player-level transferability.
3. **Validation impossibility**: If L1 is only trained through L4, we cannot validate whether
   player representations are meaningful independent of the prediction pipeline.
4. **Sample efficiency**: L1 can leverage 2001-2017 data that the main model never sees for
   prediction, dramatically increasing its training signal.

### Why Not Pure Bottom-Up?

1. **Representation mismatch**: L1 optimized purely for next-game stat prediction may learn
   features that are redundant for spread prediction (e.g., minutes played is easy to predict
   but nearly useless for spread after controlling for other stats).
2. **Interface rigidity**: Frozen representations create a hard information bottleneck. If L1
   is missing something L4 needs, there is no way to recover it.
3. **Lost synergy**: The best features for spread prediction may only emerge when all levels
   are trained together.

The hybrid approach gets the best of both worlds: meaningful per-level representations from
pre-training, plus joint optimization to fix interface mismatches.

---

## 2. Recommended Training Pipeline

### Phase A: Bottom-Up Pre-Training

#### Step A1: Level 1 -- Player Ability Vectors

**Goal**: Learn a 24-32 dimensional ability vector for each player that captures their
intrinsic basketball ability, separated from team context.

**Training objective**: Multi-task learning with three heads:

```
L_total = w1 * L_stats + w2 * L_impact + w3 * L_contrastive

Where:
  L_stats       = Next-game box score prediction (MSE on normalized stats)
  L_impact      = RAPM-approximation prediction (Huber loss on estimated impact)
  L_contrastive = Player similarity regularization (triplet loss)
```

**Why multi-task?** Each task provides a different training signal:
- **L_stats** (weight 0.5): Forces the model to encode what a player *does* -- scoring,
  rebounding, playmaking, etc. This is the most data-rich signal (every game provides a
  supervision signal for every player).
- **L_impact** (weight 0.3): Forces the model to encode how much a player *matters* for
  winning. We approximate RAPM from box-score-derived BPM (available for every
  player-season). This bridges the gap between production and impact.
- **L_contrastive** (weight 0.2): Prevents representation collapse. Without this, the model
  could map all players to similar vectors and rely on the stats head to differentiate.
  Triplet loss ensures similar players (by archetype) are close in embedding space and
  dissimilar players are far apart.

**Architecture**:
```
Input:  [last N games box scores, position, age, experience, physical profile]
        N = 20 games (sliding window)

Encoder: Temporal model (GRU or Transformer) over game sequence
         Per-game: 16 box stats + position embedding + age/experience features
         Output: 24-32 dim ability vector with uncertainty estimate

Heads:
  stats_head:       Linear(32, 16) -> next-game box score predictions
  impact_head:      MLP(32, 64, 1) -> BPM-approximation
  contrastive_head: L2-normalized embedding -> triplet loss with hard negatives
```

**Data**: 2001-2017 seasons (pre-training data pool). This is intentionally disjoint from
the main model's train/val/test splits (2018-2026). L1 sees more historical data to learn
general player dynamics.

**Training details**:
- Batch: 256 player-game samples
- Optimizer: AdamW, LR 3e-4 with cosine schedule
- Epochs: ~50 (early stopping on held-out 2017 data)
- Temporal ordering: Strictly causal -- never use future games to predict past
- Team-agnostic: No team identity features allowed. This forces the model to learn
  intrinsic player ability.

**Expected outcome**: The ability vector should:
- Cluster by playing style/archetype (not by team)
- Show smooth temporal evolution (no discontinuities at team changes)
- Predict next-game stats with correlation > 0.5 for major categories
- Rank players reasonably by the impact head (top players should have highest values)

#### Step A2: Level 2 -- Player Synergy Graph

**Goal**: Learn pairwise and higher-order player interaction effects that modify the
sum-of-parts prediction.

**Training objective**: Lineup net rating prediction.

```
L_synergy = Huber(predicted_lineup_net_rtg - observed_lineup_net_rtg)
```

**Why lineup net rating?** This is the most direct observable of player interactions.
The synergy model takes L1 ability vectors for 5 players and predicts the lineup's
net rating. The difference between predicted and sum-of-individual-ratings IS the
synergy signal.

**Architecture**:
```
Input:  5 player ability vectors (from frozen L1), each 24-32 dim

Stage 1 - Pairwise FM:
  For each pair (i,j): synergy_ij = <v_i, v_j> (FM inner product, 10 pairs)
  Also: archetype compatibility features (learned)

Stage 2 - GATv2 Aggregation:
  Nodes: 5 players with L1 ability vectors as features
  Edges: 10 pairwise connections with FM synergy scores as edge features
  2-layer GATv2 with 4 attention heads
  Readout: Mean pool + attention-weighted pool

Output: Lineup synergy adjustment (scalar, centered at 0)
```

**Data**: 2007-2017 lineup data from NBA.com (requires play-by-play parsing to reconstruct
stint-level lineups). Minimum 50 possessions per lineup for inclusion.

**Training details**:
- Batch: 128 lineup samples
- L1 vectors: Frozen (from Step A1)
- Optimizer: AdamW, LR 1e-3
- Epochs: ~30
- Regularization: L2 on FM interaction weights, dropout 0.1 on GATv2

**The lineup sparsity problem**: Teams use ~600 lineups per season averaging only 25-30
possessions each. Mitigation strategies:
1. Pool across seasons (same 5 players in different seasons treated as related samples)
2. Use 2-man and 3-man combinations as auxiliary targets (much more data)
3. Hierarchical shrinkage: rare lineups shrink toward the sum-of-pairs prediction

**Expected outcome**: Synergy model should:
- Predict lineup net rating better than sum-of-individual-abilities
- Identify known positive synergies (e.g., elite PnR ball-handler + rim-running center)
- Identify known negative synergies (e.g., two ball-dominant players)

### Phase B: Top-Down Assembly

#### Step B1: Level 3 -- Team Residuals (on frozen L1+L2)

**Goal**: Learn what team-level factors (coaching, system, organization) add beyond the
player-level prediction.

**Training objective**: Predict the residual between sum(L1 + L2) team ratings and actual
team performance.

```
team_pred_from_players = sum(L1 ability vectors * projected_minutes) + L2 synergy
residual = actual_team_net_rtg - team_pred_from_players

L_team = MSE(L3_output, residual)
```

**Architecture**:
```
Input:
  Slow features (updated seasonally):
    - Coach embedding (learned, 16-dim per coach)
    - Coach tenure (games with current team, nonlinear transform)
    - Roster continuity percentage
    - 3-year team performance trend (organizational quality proxy)

  Medium features (rolling 15-game window):
    - Four Factors differentials (eFG%, TOV%, ORB%, FTR) -- 8 features
    - Net rating (rolling)
    - Pace, pace variability
    - Assist rate, 3PA rate (system proxies)
    - Opponent 3PA rate, opponent paint points (defensive scheme proxies)

  Continuity gate:
    - Modulates weight between team-history features and player-composition features
    - Follows FiveThirtyEight insight: ~35% team / ~65% player on average

Encoder: MLP(input_dim, 128, 64, 1) with residual connections
Output: Team residual adjustment (scalar, centered at 0)
```

**Data**: 2018-2023 (same as main model training split). L3 trains on the residual from
frozen L1+L2, so it only needs to capture what those levels missed.

**Training details**:
- Batch: 64 team-game samples
- L1, L2: Frozen
- Optimizer: AdamW, LR 5e-4
- Epochs: ~40
- Key insight: L3 capacity should be small. Berry & Fowler estimate coaching at ~15-25% of
  win variance after accounting for talent. If L3 is too large, it will overfit to noise.

#### Step B2: Level 4 -- Game Context (on frozen L1+L2+L3)

**Goal**: Combine team-level predictions with game-specific context to produce the final
spread prediction.

**Training objective**: Spread prediction (primary) with auxiliary win probability.

```
L_game = w1 * Huber(predicted_spread - actual_spread, delta=8.0)
       + w2 * BCE(predicted_win_prob, actual_win_outcome)
       + w3 * NLL_gaussian(mu, sigma, actual_spread)  [calibration]

Where w1=0.5, w2=0.3, w3=0.2
```

**Architecture**:
```
Input:
  From lower levels (frozen):
    - Home team rating: sum(L1) + L2 + L3 for home
    - Away team rating: sum(L1) + L2 + L3 for away
    - Team rating differential

  Game-specific context:
    - Home/away indicator
    - Rest days (home, away) -- bucketed {0, 1, 2, 3+}
    - Back-to-back status (home, away)
    - Travel distance (home, away) -- log-scaled
    - Altitude differential
    - Timezone crossings with direction
    - Games in last 7 days (home, away)
    - Season phase (early/mid/late/playoffs)

Encoder:
  Context MLP: MLP(context_dim, 64, 32) -> context embedding
  Fusion: Concat(team_differential, home_rating, away_rating, context_embedding)
  Output MLP: MLP(fusion_dim, 128, 64, 3) -> (spread_mu, spread_sigma, win_logit)
```

**Data**: 2018-2023 (training), 2023-24 (validation).

**Training details**:
- Batch: 128 games
- L1, L2, L3: Frozen
- Optimizer: AdamW, LR 3e-4
- Epochs: ~60
- The Gaussian NLL loss forces calibrated uncertainty estimates

### Phase C: End-to-End Fine-Tuning

**Goal**: Joint optimization of all four levels against the final spread prediction target,
with discriminative learning rates to preserve pre-trained representations.

**Training objective**: Same as Step B2 (spread + win probability + calibration).

**Learning rate schedule** (discriminative):
```
L1 (player ability):     1e-5  (100x lower than pre-training)
L2 (synergy):            3e-5  (30x lower than pre-training)
L3 (team residuals):     1e-4  (5x lower than phase B training)
L4 (game context):       3e-4  (same as phase B training)
```

**Rationale for discriminative LR**:
- L1 has the most data and the most to lose from overfitting to spread signal
- L2 should change slowly to preserve learned synergy patterns
- L3 can adapt more because it was trained on residuals that may shift during fine-tuning
- L4 adapts fastest because it is closest to the output and most task-specific

This follows the ULMFiT discriminative fine-tuning paradigm (Howard & Ruder, 2018) and
the hierarchical observation that early/deep layers need less adaptation than task-specific
layers.

**Training details**:
- Batch: 128 games
- All levels unfrozen with per-level LR (use param_groups in PyTorch)
- Optimizer: AdamW with per-group weight decay
- Epochs: ~20 (aggressive early stopping -- this is fine-tuning, not training)
- Gradient clipping: 1.0 (prevents L1 gradient explosion from deep backprop)
- Monitor: L1 embedding drift (cosine similarity between Phase A and Phase C embeddings).
  If drift > 0.3, reduce L1 LR further.

**Warmup schedule for Phase C**:
```
Epoch 1-3:   Only L4 unfrozen (warmup)
Epoch 4-6:   L3 + L4 unfrozen
Epoch 7-10:  L2 + L3 + L4 unfrozen
Epoch 11+:   All levels unfrozen (full fine-tuning)
```

This "gradual unfreezing" prevents catastrophic forgetting in lower levels. It is the same
technique used in ULMFiT and has been validated for hierarchical models in vision (layer-wise
unfreezing in YOLO architectures) and NLP (progressive fine-tuning in BERT variants).

---

## 3. Loss Functions Per Level

### Summary Table

| Level | Phase A Loss | Phase B Loss | Phase C Loss | Target |
|-------|-------------|-------------|-------------|--------|
| **L1** | 0.5*MSE_stats + 0.3*Huber_impact + 0.2*Triplet_contrastive | Frozen | 0.05*MSE_stats (auxiliary) + backprop from L4 | Next-game stats, BPM approx, player similarity |
| **L2** | Huber_lineup_net_rtg + 0.1*MSE_2man + 0.1*MSE_3man | Frozen | Backprop from L4 | Lineup net rating, 2/3-man net ratings |
| **L3** | N/A | MSE_team_residual | Backprop from L4 | Team performance residual after L1+L2 |
| **L4** | N/A | 0.5*Huber_spread + 0.3*BCE_win + 0.2*NLL_gauss | Same | Spread, win prob, calibrated uncertainty |

### Detailed Loss Specifications

#### L1: Multi-Task Player Loss

```python
def l1_loss(pred_stats, true_stats, pred_impact, true_bpm, embeddings, triplets):
    # Stats prediction: MSE on z-scored box-score stats
    # Weight stats by stabilization rate (fast-stabilizing stats get more weight)
    stat_weights = torch.tensor([
        0.8,   # minutes (stabilizes ~10 games)
        1.0,   # points
        0.6,   # oreb
        0.8,   # dreb
        0.9,   # assists
        0.7,   # steals
        0.7,   # blocks
        0.8,   # turnovers
        0.5,   # personal fouls
        0.9,   # FGA
        0.9,   # FGM
        0.7,   # 3PA
        0.5,   # 3PM (stabilizes ~240 games -- downweight)
        0.8,   # FTA
        0.8,   # FTM
        0.3,   # plus_minus (extremely noisy -- heavily downweight)
    ])
    L_stats = weighted_mse(pred_stats, true_stats, stat_weights)

    # Impact prediction: Huber loss on BPM approximation
    # Huber delta=2.0 because BPM has heavy tails (superstars >> average)
    L_impact = F.huber_loss(pred_impact, true_bpm, delta=2.0)

    # Contrastive: Triplet loss with hard negative mining
    # Positive: Same player, different games (temporal consistency)
    # Negative: Different player, same position (force discrimination within position)
    anchor, positive, negative = embeddings[triplets]
    L_contrastive = F.triplet_margin_loss(anchor, positive, negative, margin=0.5)

    return 0.5 * L_stats + 0.3 * L_impact + 0.2 * L_contrastive
```

**Stat weighting rationale**: Medvedovsky's stabilization research shows that different stats
stabilize at wildly different rates (minutes ~10 games, 3PT% ~240 games, plus-minus ~1000+
possessions). Stats that stabilize slowly have high observation noise, so we downweight them
in the loss. This is equivalent to inverse-variance weighting under a heteroscedastic noise
model.

#### L2: Lineup Net Rating Loss

```python
def l2_loss(pred_lineup_rtg, true_lineup_rtg, pred_2man, true_2man, pred_3man, true_3man):
    # Primary: 5-man lineup net rating
    # Huber loss because lineup ratings have heavy tails and outliers
    L_5man = F.huber_loss(pred_lineup_rtg, true_lineup_rtg, delta=10.0)

    # Auxiliary: 2-man and 3-man combination ratings (much more data)
    L_2man = F.huber_loss(pred_2man, true_2man, delta=10.0)
    L_3man = F.huber_loss(pred_3man, true_3man, delta=10.0)

    return L_5man + 0.1 * L_2man + 0.1 * L_3man
```

**Why auxiliary 2/3-man losses?** The lineup sparsity problem is severe: ~600 lineups per
team-season averaging 25-30 possessions each. 2-man combinations have 10x more data and
3-man combinations have 5x more. The auxiliary losses provide gradient signal even when
5-man data is sparse, following the multi-task auxiliary learning paradigm (Navon et al.,
2020).

#### L3: Team Residual Loss

```python
def l3_loss(pred_residual, true_residual):
    # Simple MSE on the residual
    # L2 regularization built into AdamW weight decay
    return F.mse_loss(pred_residual, true_residual)
```

L3's loss is intentionally simple. The hard work is in defining the target (the residual
after L1+L2). If L3's capacity is right-sized (~5K-20K params), MSE with weight decay
provides sufficient regularization.

#### L4: Game-Level Composite Loss

```python
def l4_loss(pred_mu, pred_sigma, pred_win_logit, true_spread, true_win):
    # Spread prediction: Huber loss (robust to blowouts)
    L_spread = F.huber_loss(pred_mu, true_spread, delta=8.0)

    # Win classification: Binary cross-entropy
    L_win = F.binary_cross_entropy_with_logits(pred_win_logit, true_win)

    # Calibration: Gaussian NLL (forces sigma to be meaningful)
    L_calib = F.gaussian_nll_loss(pred_mu, true_spread, pred_sigma.exp())

    return 0.5 * L_spread + 0.3 * L_win + 0.2 * L_calib
```

**Why three sub-losses?** Spread MAE is the primary metric, but win probability and
calibration provide complementary gradient signals:
- L_win prevents the model from optimizing spread at the expense of direction (a prediction
  of +0.1 is nearly useless for win prediction but "close" in MAE)
- L_calib forces the uncertainty estimate to be meaningful, enabling downstream applications
  (betting edge detection, confidence-weighted predictions)

### Loss Weighting Strategy

We use **fixed weights** rather than learned/adaptive weights (GradNorm, uncertainty
weighting). Rationale:

1. Our losses operate at different levels, not different tasks on the same input. GradNorm
   and uncertainty weighting were designed for multi-task learning on shared representations,
   not hierarchical staged training.
2. We have strong domain priors about relative importance (spread > win > calibration).
3. Fewer hyperparameters to tune. We can always revisit this if training dynamics show
   gradient imbalance.

---

## 4. Data Usage Plan

### Overview

```
          2001                    2017  2018         2023  2023-24  2024-2026
          |-- L1 pre-training ---|     |--- Main train --|  Val       Test
          |-- L2 pre-training ---|
                                       |--- L3/L4 train -|
                                       |--- Phase C FT --|
```

### Season Allocation

| Data Range | Seasons | Games | Purpose |
|-----------|---------|-------|---------|
| 2001-2016 | 16 | ~19,700 | L1 pre-training (primary) |
| 2016-17 | 1 | ~1,230 | L1 pre-training validation |
| 2007-2017 | 11 | ~13,500 | L2 pre-training (lineup data available from 2007) |
| 2018-2023 | 6 | ~7,400 | L3 training, L4 training, Phase C fine-tuning |
| 2023-24 | 1 | ~1,230 | Validation (all levels) |
| 2024-2026 | 2 | ~2,000 | Test (final evaluation) |

### L1-Specific Data Considerations

**Why use 2001-2017 for L1?** Two reasons:
1. **More data**: 16 seasons of player-game observations provide much richer training signal
   for learning player dynamics than the 6-season main training window.
2. **No leakage**: L1 pre-training data is strictly before the main model's training period.
   Players who span both eras (e.g., LeBron James, 2003-present) are fine because L1 only
   uses their 2001-2017 games during pre-training and their 2018+ games during main training.

**Temporal ordering**: Within the L1 training window, all data is processed chronologically.
For each player-game sample, the model only sees the player's previous N games. No future
information leaks into the input features.

**Career-spanning players**: Players whose careers span multiple data splits (pre-training
and main training) are handled naturally:
- During L1 pre-training: model learns from their 2001-2017 games
- During Phase C fine-tuning: model processes their 2018-2023 games
- The L1 model does NOT retain per-player state between phases. Instead, it re-processes
  the most recent N games at inference time to produce the current ability vector.

### L2-Specific Data Considerations

**Lineup data availability**: NBA.com lineup endpoints go back to 2007-08. Play-by-play
data for lineup reconstruction goes back to 1996-97 via pbpstats, but quality improves
significantly from 2000 onward. We use 2007-2017 for L2 pre-training because:
- Official lineup statistics are more reliable than reconstructed lineups
- 11 seasons provides ~150,000 5-man lineup observations with 50+ possessions each
- Pre-dates main training window (no leakage)

**Minimum possession thresholds**:
- 5-man lineups: 50+ possessions (primary target)
- 3-man combinations: 100+ possessions (auxiliary target)
- 2-man combinations: 200+ possessions (auxiliary target)

### L3 and L4 Data

L3 and L4 train on the same 2018-2023 window as the final model. This is because:
- L3 needs to learn the residual AFTER L1+L2, which requires running L1+L2 on the same era
- L4 game-context features (rest, travel, altitude) are era-specific (home court advantage
  has declined from ~3.5 to ~2.0 points since 2014)
- 6 seasons (~7,400 games) is sufficient for the small parameter counts of L3 and L4

### Handling Players Who Span Splits

This is the trickiest data engineering challenge. Consider a player active from 2015-2025:

| Phase | Data used | Player's contribution |
|-------|-----------|----------------------|
| A1 (L1 pre-train) | 2015-2017 games | 3 seasons of training signal |
| B1 (L3 train) | 2018-2023 games | L1 processes last 20 games to produce ability vector |
| B2 (L4 train) | 2018-2023 games | Ability vector as input feature |
| C (fine-tune) | 2018-2023 games | All levels active |
| Inference | 2024-2026 games | L1 processes last 20 games from 2024+ |

The key principle: **L1 never stores per-player hidden state across phases**. It is a
function from (recent game history) -> (ability vector), recomputed fresh at each inference
point. This means:
- No information leaks between pre-training and main training eras
- The model handles player progression naturally (a player who improves between 2017 and
  2018 will have a different ability vector because different games are in the window)
- New players (rookies in 2024) get ability vectors from their actual games, not from
  pre-training era data

---

## 5. Validation Strategy

### Per-Level Validation

Each level has independent validation metrics that can be checked without the full pipeline.

#### L1 Validation: Player Ability Quality

| Metric | How to compute | Target |
|--------|---------------|--------|
| **Stats prediction R^2** | Correlation between predicted and actual next-game stats | > 0.30 for major stats (pts, reb, ast) |
| **Impact ranking Spearman rho** | Rank players by impact head vs rank by BPM | > 0.65 |
| **Embedding cluster purity** | K-means (K=9) on embeddings; measure purity vs known archetypes | > 0.50 |
| **Temporal smoothness** | Mean cosine sim between ability vectors 5 games apart for same player | > 0.85 |
| **Team-change stability** | Cosine sim of ability vector before/after player trades | > 0.70 (ability should not change at trade) |
| **Rookie convergence** | Games until ability vector is within 0.1 cosine of 30-game vector | < 15 games |

The **team-change stability** metric is critical. If a player's ability vector changes
dramatically when they change teams, L1 is encoding team-level features (contamination).
The vector should remain relatively stable across team changes, with only gradual drift from
new role/usage patterns.

#### L2 Validation: Synergy Quality

| Metric | How to compute | Target |
|--------|---------------|--------|
| **Lineup net rating MAE** | Predict lineup ratings on held-out lineups | Better than sum-of-individuals baseline |
| **Synergy sign accuracy** | For known positive/negative synergies, check sign of predicted synergy | > 60% |
| **Complementarity test** | PnR guard + rim-runner should score higher than guard + guard | Directionally correct |
| **Cross-validation across teams** | Train on 25 teams, validate on 5 | Similar performance |

The critical baseline is **sum-of-individual-abilities**. If L2 does not beat this,
it is adding nothing. L-RAPM research (arXiv 2601.15000) demonstrates that lineup-level
ratings DO have predictive power beyond individual ratings, so this bar should be achievable.

#### L3 Validation: Team Residual Quality

| Metric | How to compute | Target |
|--------|---------------|--------|
| **Residual prediction R^2** | How well L3 explains the L1+L2 residual | > 0.05 (small but real) |
| **Coach effect detection** | After coaching change, does L3 prediction shift? | Detectable direction change |
| **Continuity correlation** | Higher roster continuity -> smaller L3 residual? | Negative correlation |
| **Team ranking** | Rank teams by L3 output vs rank by coaching quality proxies | Spearman > 0.40 |

Note the low R^2 target for L3. The team residual after accounting for player quality and
synergy is small. Berry & Fowler estimate coaching at ~30% of win variance, but much of
that overlaps with roster construction (which is captured by L1+L2). The pure coaching/system
residual may explain only 5-10% of remaining variance.

#### L4 Validation: End-to-End Prediction Quality

| Metric | How to compute | Target (beat Phase 3 Exp 9) |
|--------|---------------|----------------------------|
| **Spread MAE** | Mean absolute error on test set | < 10.66 |
| **Win AUC** | Area under ROC curve for win prediction | > 0.718 |
| **Win accuracy** | Percentage of correct winner predictions | > 66.5% |
| **Calibration (ECE)** | Expected calibration error | < 0.03 |
| **90% coverage** | Fraction of games within 90% prediction interval | > 0.85 |

### Cross-Level Validation

Beyond per-level metrics, we need cross-level checks:

| Check | What it catches | How to test |
|-------|----------------|-------------|
| **L1 ablation** | Is L1 actually used? | Replace L1 with random vectors, measure L4 degradation |
| **L2 ablation** | Is L2 adding value? | Set L2 synergy output to 0, measure L4 degradation |
| **L3 ablation** | Is L3 necessary? | Set L3 residual to 0, measure L4 degradation |
| **Interface gradient flow** | Are gradients flowing in Phase C? | Log gradient norms at each level boundary per epoch |
| **Representation drift** | Is Phase C destroying pre-trained features? | Track cosine similarity of L1/L2 weights vs Phase A checkpoints |

### Validation Schedule

| After step | What to validate | Pass criteria |
|-----------|-----------------|---------------|
| A1 complete | L1 per-level metrics | All targets met |
| A2 complete | L2 per-level metrics + L1 stability | L2 beats sum-of-individuals |
| B1 complete | L3 per-level metrics | R^2 > 0.05 |
| B2 complete | L4 per-level metrics (with frozen L1-L3) | MAE < 11.5 (pre-fine-tuning) |
| C epoch 5 | Gradient flow + representation drift | No collapse or extreme drift |
| C complete | All metrics | Beat Phase 3 Exp 9 targets |

---

## 6. Ablation Plan

### A. Level Contribution Ablation (After Phase C)

The most important ablation: systematically remove each level to measure its marginal
contribution.

| Experiment | Configuration | What it measures |
|-----------|--------------|-----------------|
| **Full model** | L1 + L2 + L3 + L4 | Baseline (best result) |
| **No L1** | Random embeddings + L2 + L3 + L4 | L1's contribution to final prediction |
| **No L2** | L1 + zero synergy + L3 + L4 | L2's contribution (synergy value) |
| **No L3** | L1 + L2 + zero residual + L4 | L3's contribution (coaching/system value) |
| **No L4 context** | L1 + L2 + L3 + team diff only | Context features' contribution |
| **L1 only** | L1 + zero synergy + zero residual + basic context | Pure player-ability prediction |
| **L1 + L2 only** | L1 + L2 + zero residual + basic context | Player + synergy, no team effects |

**Expected contribution ordering** (based on variance budget from literature):
1. L1 (largest): Player talent explains ~50-60% of variance
2. L4 context (second): Home court + rest + travel explain ~5-10%
3. L3 (third): Coaching/system adds ~5-10%
4. L2 (smallest but real): Synergy adds ~3-5%

If L2 contributes less than L3, this is acceptable -- synergy effects are subtle and
only matter in specific matchups. If L1 contributes less than expected, it indicates
the ability vectors are not sufficiently team-agnostic.

### B. Training Strategy Ablation

| Experiment | Description | What it tests |
|-----------|------------|--------------|
| **No pre-training** | Skip Phase A entirely, train all levels end-to-end from scratch | Value of staged pre-training |
| **No fine-tuning** | Skip Phase C entirely, freeze L1+L2 forever | Value of end-to-end fine-tuning |
| **Uniform LR** | Phase C with same LR for all levels | Value of discriminative LR |
| **No gradual unfreezing** | Phase C unfreezes all levels simultaneously | Value of gradual unfreezing |
| **Different L1 targets** | Stats-only, Impact-only, Contrastive-only | Relative value of each L1 training signal |

### C. Architecture Ablation

| Experiment | Description | What it tests |
|-----------|------------|--------------|
| **L1 dim sweep** | 8, 16, 24, 32, 48 dim ability vectors | Optimal embedding dimensionality |
| **L2 FM only** | Remove GATv2, keep only FM interactions | Whether graph attention adds value |
| **L2 GATv2 only** | Remove FM, keep only GATv2 | Whether explicit pairwise interaction adds value |
| **L3 capacity sweep** | 1K, 5K, 20K, 100K params | Optimal team residual capacity |
| **L1 window sweep** | 5, 10, 20, 40 game windows | Optimal temporal context length |

### D. Data Ablation

| Experiment | Description | What it tests |
|-----------|------------|--------------|
| **L1 recent-only** | Pre-train L1 on 2012-2017 only (6 seasons instead of 16) | Value of historical depth |
| **L2 no auxiliary** | Train L2 on 5-man lineups only (no 2/3-man aux losses) | Value of auxiliary lineup targets |
| **No pre-training data separation** | Use 2018-2023 for both L1 pre-training and main training | Value of data separation |

### Ablation Execution Order

Priority order (highest information value first):

1. **Level contribution ablation** (A) -- most critical: does the hierarchy help at all?
2. **No pre-training vs full pipeline** (B.1) -- second most critical: is staged training
   worth the complexity?
3. **L1 dimension sweep** (C.1) -- determines a key architectural hyperparameter
4. **L1 target ablation** (B.5) -- which training signal matters most for L1?
5. Remaining ablations as compute budget allows

---

## 7. Risk Analysis

### Risk 1: L1 Team Contamination

**What**: During Phase C fine-tuning, L1 learns to encode team identity because it helps
spread prediction (e.g., "this player is on the Warriors" is predictive even without
modeling the player's actual ability).

**How to detect**: Monitor the team-change stability metric. If cosine similarity of ability
vectors before/after trades drops below 0.60 during Phase C, contamination is occurring.

**How to fix**:
1. Reduce L1 learning rate further (1e-6 instead of 1e-5)
2. Add an adversarial head during Phase C: a team-classifier that L1 must FOOL (gradient
   reversal layer). If L1 encodes team identity, the adversarial head will succeed; the
   gradient reversal forces L1 to remove team information.
3. Freeze L1 entirely during Phase C if contamination persists (sacrifice some end-to-end
   performance for clean representations)

**Severity**: High. Team-contaminated L1 vectors will fail at inference time for traded
players and new rosters, leading to poor early-season predictions.

### Risk 2: L2 Overfitting on Sparse Lineup Data

**What**: With only 25-30 possessions per lineup, L2 memorizes specific lineups rather than
learning generalizable synergy patterns.

**How to detect**: Large gap between training lineup rating prediction accuracy and
cross-validation accuracy (especially on unseen player combinations).

**How to fix**:
1. Increase minimum possession threshold from 50 to 100 (fewer but more reliable samples)
2. Weight samples by sqrt(possessions) to give more reliable lineups more influence
3. Increase reliance on 2/3-man auxiliary losses (more data, less noise)
4. Add dropout (0.2-0.3) to GATv2 layers
5. Reduce L2 capacity (fewer GATv2 layers, fewer attention heads)

**Severity**: Medium. L2 is expected to have the smallest contribution. If it overfits,
setting synergy to 0 loses ~3-5% of the total signal.

### Risk 3: L3 Capturing Player Signal

**What**: L3 learns player-level patterns from the rolling team stats (e.g., a team's Four
Factors are really just reflections of their star player's shooting).

**How to detect**: L3 predictions correlate highly with L1 aggregated team ratings (should
be uncorrelated by construction since L3 trains on the residual).

**How to fix**:
1. Verify the residual computation is correct (L3 target = actual - L1_pred - L2_pred)
2. Add L1-decorrelation regularizer: penalize correlation between L3 output and mean L1
   ability for the team
3. Use only features that are genuinely team-level (coach identity, continuity) and remove
   features that are player-level in disguise (team eFG% is heavily influenced by individual
   shooters)

**Severity**: Medium. If L3 is just relearning player information, it wastes capacity but
does not actively harm predictions. However, it will make ablation results misleading
(L3 ablation would appear to remove important signal that actually belongs to L1).

### Risk 4: Catastrophic Forgetting During Phase C

**What**: End-to-end fine-tuning destroys the useful representations learned during
pre-training, especially in L1 where the fine-tuning gradient signal is weakest.

**How to detect**: Monitor cosine similarity between Phase A and Phase C embeddings.
Track L1 validation metrics (stats prediction R^2, impact ranking) during Phase C.

**How to fix**:
1. Gradual unfreezing (already in the plan) -- the most effective mitigation
2. Elastic weight consolidation (EWC): add a penalty for deviating from pre-trained weights,
   weighted by Fisher information (how important each weight is for the pre-training task)
3. If L1 metrics degrade more than 10%, freeze L1 entirely and accept the Phase B result
4. Use a separate "anchor" loss that keeps L1 embeddings close to their pre-trained values

**Severity**: High. This is the most common failure mode of fine-tuning hierarchical models.
The gradual unfreezing schedule is specifically designed to mitigate this.

### Risk 5: Insufficient Signal in L1 Pre-Training Target

**What**: Box-score stats + BPM are noisy proxies for true player ability. The resulting
L1 embeddings may not capture what matters for game outcomes.

**How to detect**: L1 ability vectors have low correlation with known good player metrics
(EPM, RAPM). The full model (after Phase C) does not outperform a simpler model that just
uses raw rolling stats.

**How to fix**:
1. Add play-by-play-derived features as L1 targets if available (usage rate, on/off
   differential, etc.)
2. Compute actual RAPM from PBP data and use it as the impact target (more work but
   better signal)
3. Use a different L1 architecture that does not depend on explicit targets -- e.g.,
   a masked autoencoder that reconstructs randomly masked box-score features

**Severity**: Medium. Even noisy pre-training helps (BERT works despite noisy masked LM
targets). Phase C fine-tuning will compensate to some degree.

### Risk 6: Leakage Through Overlapping Player Careers

**What**: A player active from 2010-2025 appears in both L1 pre-training data (2010-2017)
and main training data (2018-2023). The model might learn patterns specific to this player's
2017 ability when predicting 2018 games.

**How to detect**: Compare model accuracy on (a) games with many career-spanning players
vs (b) games with mostly single-era players. If (a) is much better, there may be leakage.

**How to fix**: This is actually NOT leakage if handled correctly. L1 does not store
per-player state across phases. At inference time for a 2018 game, L1 processes only the
player's 2018 games (sliding window). The pre-training phase teaches L1 HOW to process
game sequences, not WHAT specific players are like. This is analogous to how BERT learns
language patterns from a pre-training corpus but processes new text at inference time.

**Severity**: Low. This is a conceptual concern more than a practical one, as long as the
sliding window architecture is correctly implemented.

### Risk 7: Phase C Underfitting (Interface Mismatch)

**What**: Pre-trained L1+L2 representations are not well-suited for the L3+L4 downstream
task. Phase C fine-tuning cannot bridge the gap in ~20 epochs.

**How to detect**: Phase C shows steady improvement for all 20 epochs without plateauing
(suggesting more epochs would help but we are stopping too early). Alternatively, the
"no pre-training" ablation performs similarly to the full pipeline.

**How to fix**:
1. Increase Phase C epochs (up to 50)
2. Increase L1/L2 learning rates during Phase C (accept more drift for better task
   alignment)
3. Add "adapter" layers between levels -- small MLPs that transform representations without
   modifying the original weights (parameter-efficient fine-tuning approach from NLP)
4. Re-examine whether L1's pre-training objectives produce features that are useful for
   spread prediction. If not, redesign L1 targets.

**Severity**: Medium. The mitigation (more epochs, higher LR) is straightforward.

### Risk Summary Matrix

| Risk | Likelihood | Impact | Mitigation Difficulty | Priority |
|------|-----------|--------|----------------------|----------|
| L1 team contamination | Medium | High | Medium (gradient reversal) | 1 |
| Catastrophic forgetting | Medium | High | Low (gradual unfreezing) | 2 |
| L2 lineup overfitting | High | Low | Low (dropout, thresholds) | 3 |
| Insufficient L1 signal | Low | Medium | Medium (need RAPM data) | 4 |
| L3 capturing player signal | Medium | Low | Low (decorrelation reg) | 5 |
| Career overlap leakage | Low | Low | Already mitigated | 6 |
| Phase C underfitting | Low | Medium | Low (more epochs) | 7 |

---

## 8. Research Basis

### Core Paradigms Informing This Design

**1. Pre-train then fine-tune (BERT/GPT paradigm)**
The most successful training paradigm in deep learning. Pre-train on a large dataset with
self-supervised or multi-task objectives, then fine-tune on the downstream task. Our L1
pre-training on 2001-2017 data followed by Phase C fine-tuning on 2018-2023 directly
mirrors this pattern.

- Howard, J. & Ruder, S. (2018). "Universal Language Model Fine-tuning for Text
  Classification." ACL. Introduced discriminative learning rates and gradual unfreezing.
- Devlin, J. et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers." NAACL.

**2. Hierarchical Bayesian modeling (James-Stein, partial pooling)**
Our L1 population -> archetype -> individual hierarchy is a direct neural implementation
of Bayesian partial pooling. Players with sparse data are "shrunk" toward their archetype
prior. This is mathematically optimal (James & Stein, 1961) and practically validated
for NBA player modeling (Miyakawa's BPR, EPAA by Elmore et al.).

- James, W. & Stein, C. (1961). "Estimation with Quadratic Loss."
- Elmore, R. et al. (2024). "Expected Points Above Average." arXiv:2405.10453.

**3. Multi-task learning with auxiliary objectives**
L1's multi-task loss (stats + impact + contrastive) follows the rich literature showing
that auxiliary tasks improve main task performance by providing complementary gradient
signals and preventing overfitting to a single objective.

- Ruder, S. (2017). "An Overview of Multi-Task Learning in Deep Neural Networks."
  arXiv:1706.05098.
- Navon, A. et al. (2020). "Auxiliary Learning by Implicit Differentiation." ICLR 2021.

**4. Two-tower models (recommendation systems)**
Our player-level architecture (L1+L2 producing per-team ratings) mirrors the two-tower
paradigm in recommendation systems, where user and item embeddings are learned separately
and combined for prediction. The key innovation from rec systems is that embeddings should
be task-useful (predict interactions) not just descriptive (reconstruct features).

- Yi, X. et al. (2019). "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item
  Recommendations." RecSys.

**5. GradNorm and loss balancing**
While we use fixed loss weights, the GradNorm literature informs our weight choices. The
principle is that gradient magnitudes across tasks should be similar to prevent any single
task from dominating optimization.

- Chen, Z. et al. (2018). "GradNorm: Gradient Normalization for Adaptive Loss Balancing."
  ICML.

**6. Graph Factorization Machines (GraphFM)**
Our L2 architecture combines FM pairwise interactions with GATv2 aggregation, directly
inspired by GraphFM which integrates the FM interaction function into GNN feature
aggregation.

- Li, H. et al. (2021). "GraphFM: Graph Factorization Machines for Feature Interaction
  Modeling." arXiv:2105.11866.

**7. Curriculum learning**
The Phase A -> B -> C progression is a form of curriculum learning: L1 learns the "easy"
task (individual player stats), L2 learns the harder task (player interactions), L3/L4
learn the hardest task (game outcomes). Each level benefits from the representations
learned by simpler levels below it.

- Bengio, Y. et al. (2009). "Curriculum Learning." ICML.

### NBA-Specific Research Underpinning Design Decisions

| Design Decision | Research Basis |
|----------------|---------------|
| L1 24-32 dim vectors | NBA2Vec succeeds with 8 dim; PCA shows 4-5 components capture 70-80% variance; archetype studies find 9-12 clusters. 24-32 dim provides headroom. |
| L1 stat-specific loss weighting | Medvedovsky (2020): stabilization rates range from ~10 games (minutes) to ~1000+ possessions (plus-minus). Noisy stats should be downweighted. |
| L1 BPM as impact target | Myers (BBRef): BPM is trained against 14-year RAPM and captures most predictive box-score information. Available for every player-season. |
| L2 lineup rating target | L-RAPM (arXiv 2601.15000): lineup ratings have predictive power beyond sum-of-individual ratings. This is the synergy signal. |
| L2 50+ possession minimum | Medvedovsky: 5-man lineup offensive rating needs ~550 possessions to stabilize. 50 is aggressively low but necessary for data volume. |
| L3 ~15-25% variance budget | Berry & Fowler: coaching explains ~30% of win variance; minus overlap with talent leaves ~15-25% for pure system effects. |
| L3 continuity-gated attention | FiveThirtyEight: optimal team-vs-player weighting is ~35/65 on average, modulated by roster continuity. |
| L4 ~2.0 HCA baseline | Post-2014 HCA decline from ~3.5 to ~2.0 points. Entine & Small (2008) decomposed HCA into rest, crowd, and familiarity components. |
| L4 directional travel asymmetry | Song et al. (2022): eastward jet lag costs ~1.3 points; westward has no significant effect. |
| L4 delta=8.0 for Huber loss | Vegas spread MAE is ~8-9 points (the practical floor). Errors beyond 8 points are largely from irreducible randomness (blowouts, garbage time). |

### Sources

**Hierarchical Training**
- [Hierarchy-Aware Fine-Tuning of Vision-Language Models](https://arxiv.org/html/2512.21529)
- [Fine-tuning LLMs for Domain Adaptation: Training Strategies and Scaling](https://www.nature.com/articles/s41524-025-01564-y)
- [HiPreNets: Progressive Training for High-Precision Neural Networks](https://arxiv.org/html/2506.15064)
- [An adaptive and stability-promoting layerwise training approach](https://www.sciencedirect.com/science/article/abs/pii/S0045782525002105)
- [Greedy Layer-Wise Pretraining Tutorial](https://machinelearningmastery.com/greedy-layer-wise-pretraining-tutorial/)

**Multi-Task Learning**
- [Towards Consistent Multi-Task Learning (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Qin_Towards_Consistent_Multi-Task_Learning_Unlocking_the_Potential_of_Task-Specific_Parameters_CVPR_2025_paper.pdf)
- [Rep-MTL: Representation-level Task Saliency](https://arxiv.org/abs/2507.21049)
- [Multitask Learning 1997-2024: Fundamentals (HDSR 2025)](https://hdsr.mitpress.mit.edu/pub/7fcc3jhv)
- [Deep Multi-Task Learning: A Review](https://link.springer.com/article/10.1007/s41060-025-00892-y)
- [End-To-End Multi-Task Learning With Attention](https://www.semanticscholar.org/paper/End-To-End-Multi-Task-Learning-With-Attention-Liu-Johns/619cf9d39abb93fe1ab17921c163fc5734ac1e70)
- [An Overview of Multi-Task Learning in Deep Neural Networks](https://arxiv.org/pdf/1706.05098)

**Loss Balancing**
- [GradNorm: Gradient Normalization for Adaptive Loss Balancing (ICML 2018)](https://arxiv.org/pdf/1711.02257)
- [Uncertainty Weighted Gradients for Model Calibration (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Lin_Uncertainty_Weighted_Gradients_for_Model_Calibration_CVPR_2025_paper.pdf)
- [Strategies for Balancing Multiple Loss Functions in Deep Learning](https://medium.com/@baicenxiao/strategies-for-balancing-multiple-loss-functions-in-deep-learning-e1a641e0bcc0)

**Curriculum Learning**
- [Curriculum Reinforcement Learning: Easy to Hard](https://arxiv.org/pdf/2506.06632)
- [Curriculum Learning for Graph Neural Networks](https://openreview.net/forum?id=fTyGT5fulj)
- [On The Power of Curriculum Learning in Training Deep Networks](https://arxiv.org/abs/1904.03626)

**Player Models and Sports Prediction**
- [NBA2Vec: Dense Feature Representations of NBA Players](https://arxiv.org/pdf/2302.13386)
- [Player-Team Heterogeneous Interaction Graph for Soccer Prediction](https://arxiv.org/html/2507.10626v1)
- [GraphFM: Graph Factorization Machines](https://arxiv.org/html/2105.11866v5)
- [Sports Match Outcome Prediction with Graph Representation Learning](https://summit.sfu.ca/_flysystem/fedora/2022-08/input_data/22492/etd21919.pdf)
- [From Players to Champions: Machine Learning for Match Outcome Prediction](https://arxiv.org/html/2505.01902v1)
- [Lineup Regularized Adjusted Plus-Minus (L-RAPM)](https://arxiv.org/abs/2601.15000)
- [RAPM: Regularized Adjusted Plus-Minus](https://www.nbastuffer.com/analytics101/regularized-adjusted-plus-minus-rapm/)

**Recommendation Systems**
- [Two-Tower Model for Recommendation Systems: Deep Dive](https://www.shaped.ai/blog/the-two-tower-model-for-recommendation-systems-a-deep-dive)
- [Embedding-Based Retrieval with Two-Tower Models (Snap)](https://eng.snap.com/embedding-based-retrieval)
- [Two-Tower Recommendation at Uber](https://www.uber.com/blog/innovative-recommendation-applications-using-two-tower-embeddings/)

**Layer Freezing and Fine-Tuning**
- [Why Warmup the Learning Rate? Underlying Mechanisms](https://arxiv.org/html/2406.09405v1)
- [Freezing Layers in Deep Learning and Transfer Learning](https://www.exxactcorp.com/blog/deep-learning/guide-to-freezing-layers-in-ai-models)
- [Optimal Transfer Protocol by Incremental Layer Defrosting](https://arxiv.org/pdf/2303.01429)

**Ablation Studies**
- [Ablation Studies in Artificial Neural Networks](https://arxiv.org/abs/1901.08644)
- [AutoAblation: Automated Parallel Ablation Studies](https://dcatkth.github.io/papers/autoablation.pdf)

**NBA-Specific Research** (from Level 1-4 research docs)
- Berry & Fowler: coaching explains ~30% of win variance
- Medvedovsky (2020): stat-specific stabilization rates
- Entine & Small (2008): HCA decomposition and rest effects
- Song et al. (2022): directional travel asymmetry
- FiveThirtyEight: 35/65 Elo-RAPTOR weighting with continuity
- Snarr (EPM): skills-based per-stat optimization
- Miyakawa (BPR): hierarchical Bayesian with cold-start handling
