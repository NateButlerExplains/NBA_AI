# Wave 2: Integration Architecture -- Full Forward Pass Design

**Date**: 2026-03-16
**Purpose**: Define the complete data flow from Level 1 (Player) through Level 4 (Game Context) to produce a final spread prediction. Resolves all interface questions, specifies tensor shapes, aggregation mechanisms, gradient flow, and training strategy.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Level 1: Player Ability Vectors](#2-level-1-player-ability-vectors)
3. [L1 to L2 Interface](#3-l1-to-l2-interface)
4. [Level 2: Player Synergy Network](#4-level-2-player-synergy-network)
5. [L2 to L3 Interface](#5-l2-to-l3-interface)
6. [Level 3: Team Model](#6-level-3-team-model)
7. [L3 to L4 Interface](#7-l3-to-l4-interface)
8. [Level 4: Game Context and Prediction](#8-level-4-game-context-and-prediction)
9. [Prediction Heads](#9-prediction-heads)
10. [Full Forward Pass: ASCII Diagram](#10-full-forward-pass-ascii-diagram)
11. [Training Strategy](#11-training-strategy)
12. [Batch Structure and Data Loading](#12-batch-structure-and-data-loading)
13. [Roster Entry Point](#13-roster-entry-point)
14. [Parameter Budget](#14-parameter-budget)
15. [Design Decisions and Rationale](#15-design-decisions-and-rationale)
16. [Open Questions](#16-open-questions)

---

## 1. Architecture Overview

The system is a 4-level hierarchy where each level produces representations consumed by the next. The guiding principle is **additive residual composition**: each level adds information that the previous levels cannot capture.

```
Level 1 (Player)   -->  "What can this player do?"
Level 2 (Synergy)  -->  "How do these players interact?"
Level 3 (Team)     -->  "What does the team do beyond player talent?"
Level 4 (Context)  -->  "What does today's game context do to the matchup?"
```

**Key architectural decisions** (justified in Section 15):

| Decision | Choice | Alternative Considered |
|----------|--------|----------------------|
| Training regime | Bottom-up pre-train, then end-to-end fine-tune | Pure end-to-end |
| L2 player modification | Dual output: context-adjusted vectors + interaction features | Modify-only or features-only |
| L2-to-L3 aggregation | Gated attention pooling (permutation-invariant) | Mean pooling, DeepSets |
| L3 architecture | Residual MLP over team vector + explicit team features | Separate team embedding |
| L3-to-L4 comparison | [concat, difference, hadamard product] | Concat-only |
| Prediction heads | Multi-task: spread (mu, sigma) + win probability + total | Spread-only |
| Gradient flow | L1 frozen at inference; L2-L4 end-to-end in fine-tune | All frozen except L4 |

---

## 2. Level 1: Player Ability Vectors

Level 1 runs independently per player. It consumes a player's historical game log, physical profile, and career metadata to produce a dense ability vector.

### 2.1 Input

For player `i` at game time `t`:

```
game_log_i:     (G, F_box)     # G recent games, F_box=16 box-score features per game
                                # [min, pts, oreb, dreb, ast, stl, blk, tov, pf,
                                #  fga, fgm, fg3a, fg3m, fta, ftm, plus_minus]
physical_i:     (F_phys,)      # height_inches, weight_lbs, wingspan_inches, age_days
draft_info_i:   (F_draft,)     # draft_round, draft_pick, years_experience
position_i:     (1,)           # categorical: 0-4 or soft multi-hot
```

### 2.2 Architecture (from Wave 1 research)

```
Hierarchical Encoder:
  1. Generic NBA player prior         -->  prior_vec (d,)         [shared across all players]
  2. Position/archetype specialization -->  archetype_vec (d,)     [K=10 soft archetypes]
  3. Individual temporal encoder       -->  individual_vec (d,)    [LSTM/Transformer over game_log]

Combination:
  ability_i = prior_vec + archetype_vec + individual_vec    # residual composition
  uncertainty_i = sigma_network(game_log_i)                 # (d,) per-dimension uncertainty
```

### 2.3 Output

```python
player_ability:    (d_player,)    # d_player = 32 (core ability vector)
player_uncertainty: (d_player,)   # per-dimension confidence (used by L2 for weighting)
player_archetype:  (K,)          # K=10 soft archetype membership probabilities
```

**d_player = 32** is chosen based on literature:
- PCA captures ~80% of box-score variance in 5 components
- NBA2Vec succeeds with 8 dimensions for play outcomes
- Player archetype studies find 9-12 natural clusters
- We need headroom for defensive, playmaking, and efficiency dimensions not captured by raw box-score PCA

### 2.4 Update Cadence

Level 1 vectors are **recomputed after every game**. At inference time, the latest ability vector is a fixed input to Level 2. During bottom-up pre-training, Level 1 is trained on its own loss (predicting next-game box scores or RAPM-derived targets). During end-to-end fine-tuning, Level 1 gradients are either frozen or receive a reduced learning rate (10x lower than L2-L4).

---

## 3. L1 to L2 Interface

### 3.1 The Interface Question

**Q: Does L2 modify the player vectors, or produce separate interaction features alongside them?**

**A: Both.** L2 produces two outputs per player:

1. **Context-adjusted player vector** `h_i` in R^d_player: the player's ability after accounting for teammate and opponent context. This is the player's ability _in this specific lineup_, not in a vacuum.
2. **Pairwise interaction features** that are aggregated into a team-level synergy signal.

The motivation is that player ability is genuinely context-dependent (a ball-dominant guard is less effective next to another ball-dominant guard) while also producing emergent team-level properties (total floor spacing, defensive versatility) that are not properties of any individual player.

### 3.2 What Feeds Into L2

For a single team's side of one game:

```python
# Expected roster: up to P players (P=15 max NBA roster, padded)
# Active roster: up to A players expected to play (A <= P, typically 8-13)

player_abilities:    (A, d_player)    # from Level 1, for each active player
player_uncertainties: (A, d_player)   # from Level 1
player_archetypes:   (A, K)          # soft archetype memberships
roster_mask:         (A,)            # 1.0 for active, 0.0 for padding

# Opponent context (for matchup-aware interactions)
opp_abilities:       (A_opp, d_player)
opp_archetypes:      (A_opp, K)
opp_mask:            (A_opp,)
```

### 3.3 Roster Entry Point

**Q: Where do pre-game rosters enter?**

Pre-game rosters are the **gating mechanism** between Level 1 and Level 2. The pipeline:

1. Before a game, determine the expected active roster for each team (injury reports, rest decisions).
2. Look up each active player's current Level 1 vector (computed from their most recent game).
3. Feed only active players' vectors into Level 2.
4. Players not in the active roster simply do not contribute to Level 2.

This is the cleanest way to handle injuries and load management: a missing player's ability vector is never fed into the synergy network, so their contribution is naturally absent. No special "injury adjustment" is needed.

---

## 4. Level 2: Player Synergy Network

### 4.1 Architecture

Level 2 is a graph network operating on the roster graph. Nodes are active players. Edges connect all players on the same team (fully connected within-team) and optionally cross-team (for matchup interactions).

The architecture has four stages, following the synthesis from Wave 1 Level 2 research:

```
Stage 1: Archetype Interaction (dense prior, ~100 params)
Stage 2: FM Pairwise Residual  (player-specific, ~16K params)
Stage 3: GATv2 Message Passing  (contextualizes each player, ~200K params)
Stage 4: Lineup Aggregation     (team-level representation, ~50K params)
```

### 4.2 Stage 1: Archetype Interaction Matrix

A learned K x K symmetric matrix M captures interaction effects at the archetype level. This provides a dense, data-efficient prior for all player pairs.

```python
# archetype_i: (K,) soft membership from L1
# M: (K, K) learned symmetric interaction matrix

archetype_synergy_ij = archetype_i @ M @ archetype_j.T    # scalar
# For all pairs in a team of A players:
# archetype_synergies: (A, A) matrix of pairwise archetype interactions
```

This captures general patterns like "two ball-dominant playmakers have negative synergy" or "a rim-running center + a floor-spacing power forward have positive synergy." With K=10, M has only 55 unique parameters (upper triangle + diagonal), making it learnable even with limited data.

### 4.3 Stage 2: Factorization Machine Pairwise Residual

Player-specific synergy vectors capture individual deviations from the archetype-level prediction.

```python
# v_i: (d_syn,) per-player synergy vector, d_syn=32
# Learned per player (stored alongside L1 ability vector)

fm_synergy_ij = dot(v_i, v_j)    # scalar player-specific residual

total_pairwise_ij = archetype_synergy_ij + fm_synergy_ij
```

With Bayesian shrinkage toward zero:
- Players with extensive co-occurrence data: FM residual is data-driven
- Players who have never played together: pure archetype prediction
- New teammates post-trade: archetype prior dominates initially

### 4.4 Stage 3: GATv2 Message Passing

GATv2 (Brody et al., 2022) contextualizes each player's representation based on their teammates (and optionally opponents).

```python
# Input: player_abilities (A, d_player) + pairwise synergy scores
# 1 layer, 4 heads, d_player=32

# For each player i, attending over teammates j:
#   edge_features_ij = [archetype_synergy_ij, fm_synergy_ij,
#                       shared_minutes_ij, years_together_ij]  # (4,)
#
#   alpha_ij = GATv2_attention(h_i, h_j, edge_features_ij)     # attention weight
#   message_ij = V * h_j                                        # value projection
#
#   h_i_new = h_i + sum_j(alpha_ij * message_ij)               # residual update

# Output per player: h_i_context in R^d_player
# This is the "context-adjusted player vector"
```

The GATv2 layer performs within-team message passing first (how does each player's ability change given their specific teammates), then optionally a cross-team pass (how does facing a specific opponent affect each player). The cross-team pass uses separate attention parameters.

```
Within-team GATv2:  (A, d_player) --> (A, d_player)    # teammate context
Cross-team GATv2:   (A, d_player) x (A_opp, d_player) --> (A, d_player)  # opponent context
```

### 4.5 Stage 4: Lineup Aggregation

Aggregate player-level representations into a team-level vector. We use **gated attention pooling** rather than simple mean pooling, because not all players contribute equally to team identity.

```python
# Gated attention pooling (Ilse et al., 2018 - Attention-based MIL)
#
# gate_i = sigmoid(W_gate @ h_i + b_gate)              # (1,) importance gate per player
# attn_i = softmax_over_players(W_attn @ h_i + b_attn) # (1,) normalized attention weight
# a_i = gate_i * attn_i                                 # combined weight
#
# team_player_agg = sum_i(a_i * h_i)                   # (d_player,) weighted sum

# Additionally, aggregate pairwise synergies into a team synergy vector:
# All C(A,2) pairwise scores -> (d_syn_team,) via a small MLP
team_synergy = MLP([sum of pairwise features])           # (d_syn_team,) where d_syn_team=64
```

### 4.6 Level 2 Output

```python
# Per team:
team_player_agg:     (d_player,)      # = (32,)  gated attention pool of context-adjusted players
team_synergy:        (d_syn_team,)    # = (64,)  aggregated pairwise interaction features
player_vectors:      (A, d_player)    # = (A, 32) context-adjusted individual vectors (for props)

# Combined L2 team output:
l2_team = concat(team_player_agg, team_synergy)  # (96,)
```

---

## 5. L2 to L3 Interface

### 5.1 The Interface Question

**Q: How do we aggregate player-level + interaction-level information into a team representation? Does L3 operate on individual player vectors or on the aggregated team vector?**

**A: L3 operates on the aggregated team vector from L2, augmented with explicit team-level features.** L3 does NOT see individual player vectors. This is a deliberate information bottleneck: L3 should capture what is NOT explained by player talent and interactions, i.e., the coaching/system/organizational residual.

### 5.2 What Feeds Into L3

```python
l2_team:            (d_l2,)          # = (96,) aggregated player + synergy representation
team_features:      (d_team_feat,)   # explicit team-level features (see below)
roster_continuity:  (1,)             # fraction of minutes returning from last season
```

### 5.3 Explicit Team Features

These are pre-computed rolling statistics that capture team-level behavior beyond player composition:

```python
# Four Factors (rolling 15-game window), offensive and defensive = 8 features
four_factors:        (8,)     # [eFG%, TOV%, ORB%, FTR] x [offense, defense]

# Efficiency metrics (rolling 15-game window) = 4 features
efficiency:          (4,)     # [ORtg, DRtg, NetRtg, Pace]

# Coaching features = 4 features
coaching:            (4,)     # [coach_tenure_games, coach_career_winpct,
                              #  is_new_coach_flag, games_since_coach_change]

# Defensive scheme proxies (rolling 15-game window) = 4 features
defense_scheme:      (4,)     # [opp_3PA_rate, opp_paint_pts_rate, steal_rate, block_rate]

# Organization/continuity = 2 features
org:                 (2,)     # [roster_continuity, multi_year_trend_3yr]

# Total: d_team_feat = 22
```

### 5.4 Continuity-Gated Weighting

Following FiveThirtyEight's insight (optimal split: ~35% team-level, ~65% player-level), L3 uses roster continuity to gate how much it trusts team-level history vs. the player-derived representation from L2.

```python
# gate = sigmoid(w_cont * roster_continuity + b_cont)    # scalar in [0, 1]
# l3_input = gate * team_history_repr + (1 - gate) * l2_team
```

High continuity (>0.7): trust team-level trends strongly.
Low continuity (<0.3): rely mostly on player composition from L2.

---

## 6. Level 3: Team Model

### 6.1 Architecture

Level 3 is a residual MLP that takes the L2 aggregation plus team features and produces a team representation. It is explicitly designed to model what L1+L2 cannot capture.

```python
# Inputs
l2_team:         (96,)        # from Level 2
team_feat:       (22,)        # explicit team features
# --> concat + project:
l3_input = Linear(118, d_team)(concat(l2_team, team_feat))    # (d_team,)
                                                                # d_team = 128

# Residual MLP (2 blocks)
# Block: LayerNorm -> Linear -> GELU -> Dropout -> Linear -> residual add
l3_repr = ResidualMLP(l3_input, layers=2, hidden=256)          # (128,)

# Output
team_repr:       (d_team,)    # = (128,)
```

### 6.2 What L3 Captures

Based on the literature (Berry & Fowler: ~30% coaching variance; Maymin: ~6 wins synergy; FiveThirtyEight: 35% Elo weight), Level 3 captures:

- **Coaching system execution quality**: How well the team executes relative to talent level (captured via Four Factors trends and coach identity features)
- **Organizational culture**: Persistent team-level effects beyond current roster (captured via multi-year trend and continuity)
- **Tactical adaptation**: Recent changes in play style or defensive scheme (captured via rolling defensive scheme proxies)

The residual formulation is key: L3 does not predict the spread directly. It enriches the team representation with information that player talent alone does not explain.

---

## 7. L3 to L4 Interface

### 7.1 The Interface Question

**Q: Two teams' representations need to be compared. How does the matchup aspect enter?**

**A: Triple representation -- concatenation + difference + Hadamard product.** This is the standard NLI (Natural Language Inference) approach adapted for team matchups, which gives the downstream network the most information about the relationship between the two teams.

### 7.2 Matchup Construction

```python
# team_home: (d_team,)    # = (128,) from Level 3
# team_away: (d_team,)    # = (128,) from Level 3

matchup_concat = concat(team_home, team_away)              # (256,)
matchup_diff   = team_home - team_away                      # (128,)
matchup_hadamard = team_home * team_away                    # (128,)

matchup = concat(matchup_concat, matchup_diff, matchup_hadamard)  # (512,)
```

**Why all three?**
- **Concatenation** preserves full information about each team independently. Needed because the model must know absolute team strength, not just relative.
- **Difference** directly encodes the relative advantage. A Linear layer over the difference is equivalent to computing `w_home @ team_home - w_away @ team_away`, which is the natural form for a spread prediction (spread = home_strength - away_strength).
- **Hadamard product** captures interaction effects: dimensions where both teams are strong (or both weak) have different matchup implications than dimensions where one is strong and the other weak. This is the "style matchup" signal -- e.g., a fast-paced team (high pace dimension) playing another fast-paced team produces a high Hadamard value on that dimension, indicating a pace-up game.

---

## 8. Level 4: Game Context and Prediction

### 8.1 Context Features

Level 4 adds all game-specific features that are not intrinsic to team identity:

```python
# Home court
home_flag:          (1,)      # 1.0 for home, -1.0 for away perspective
arena_altitude:     (1,)      # normalized altitude in feet (Denver=1.0)

# Rest and schedule (per team, so x2 for home and away)
rest_days_home:     (1,)      # days since last game (capped at 5)
rest_days_away:     (1,)
is_b2b_home:        (1,)      # binary: second game in consecutive days
is_b2b_away:        (1,)
games_7d_home:      (1,)      # games played in last 7 days
games_7d_away:      (1,)

# Travel
travel_dist_home:   (1,)      # miles from previous game venue (log-scaled)
travel_dist_away:   (1,)
tz_crossings_home:  (1,)      # timezone crossings (signed: east=positive)
tz_crossings_away:  (1,)

# Season context
season_progress:    (1,)      # fraction of season completed [0, 1]
is_playoffs:        (1,)      # binary

# Total: d_context = 14
```

### 8.2 Architecture

```python
# matchup:    (512,)    from L3->L4 interface
# context:    (14,)     game-specific features

# Project context to match hidden dim
context_proj = Linear(14, 64)(context)                      # (64,)

# Combine matchup with context
l4_input = concat(matchup, context_proj)                    # (576,)

# Prediction MLP (3 layers with residual connections)
l4_hidden = Linear(576, 256)(l4_input)                      # (256,)
l4_hidden = ResidualBlock(256, 512)(l4_hidden)              # (256,)
l4_hidden = ResidualBlock(256, 512)(l4_hidden)              # (256,)

# --> prediction heads (see Section 9)
```

---

## 9. Prediction Heads

### 9.1 Multi-Task Outputs

The model produces three prediction types from the shared L4 representation:

```python
l4_repr: (256,)

# 1. SPREAD (primary task)
#    Gaussian: mu (point estimate) and sigma (uncertainty)
spread_mu    = Linear(256, 1)(l4_repr)                    # (1,) predicted margin
spread_sigma = Softplus(Linear(256, 1)(l4_repr))          # (1,) > 0, predicted std

# 2. WIN PROBABILITY (auxiliary task, strongly correlated with spread)
#    Derived from spread Gaussian: P(home_win) = Phi(mu / sigma)
#    But also predict independently for calibration:
win_logit    = Linear(256, 1)(l4_repr)                    # (1,)
win_prob     = sigmoid(win_logit)                          # (1,)

# 3. TOTAL POINTS (auxiliary task, partially orthogonal to spread)
total_mu     = Linear(256, 1)(l4_repr)                    # (1,)
total_sigma  = Softplus(Linear(256, 1)(l4_repr))          # (1,) > 0
```

### 9.2 Loss Functions

```python
# Spread: Gaussian NLL (primary loss)
loss_spread = -log N(actual_margin | spread_mu, spread_sigma^2)

# Win probability: Binary cross-entropy
loss_win = BCE(win_prob, actual_home_win)

# Total: Gaussian NLL
loss_total = -log N(actual_total | total_mu, total_sigma^2)

# Combined (multi-task)
loss = 1.0 * loss_spread + 0.3 * loss_win + 0.3 * loss_total
```

The spread loss dominates because it is the primary prediction target. Win probability and total are auxiliary tasks that provide complementary gradient signal:
- Win probability forces the model to be well-calibrated on the binary outcome.
- Total points is partially orthogonal to spread (a blowout and a close game can have the same total) and helps learn pace/efficiency effects that benefit Level 4 context features.

### 9.3 Consistency Regularization

The Gaussian-derived win probability should agree with the direct win probability head:

```python
# Phi = standard normal CDF
gaussian_win_prob = Phi(spread_mu / spread_sigma)
loss_consistency = MSE(win_prob, gaussian_win_prob.detach())  # stop gradient on Gaussian side
```

This is a soft constraint (weight 0.1) that encourages the two win probability estimates to agree without forcing them to be identical.

### 9.4 Optional: Player Props Head

For future use, individual player vectors from L2 can be fed to prop prediction heads:

```python
# player_i_vector: (d_player,) from L2 output
# game_context: (d_context,)
#
# player_pts_mu = Linear(d_player + d_context, 1)(concat(player_i, game_context))
# player_reb_mu = ...
# player_ast_mu = ...
```

This is not part of the initial implementation but the architecture supports it because L2 preserves per-player vectors.

---

## 10. Full Forward Pass: ASCII Diagram

```
                         PRE-GAME ROSTER (injury reports, lineup decisions)
                              |                    |
                    Active Home Roster       Active Away Roster
                     (8-13 players)          (8-13 players)
                              |                    |
                              v                    v
     +-----------------------------------------------------------------+
     |                    LEVEL 1: PLAYER MODELS                       |
     |   (runs independently per player, pre-computed before game)     |
     |                                                                 |
     |   For each active player i:                                     |
     |     game_log_i (G, 16)  ----+                                   |
     |     physical_i (4,)  -------+--> Hierarchical Encoder           |
     |     draft_info_i (3,)  -----+         |                         |
     |                                       v                         |
     |                              ability_i    (32,)                 |
     |                              uncertainty_i (32,)                |
     |                              archetype_i   (10,)                |
     +-----------------------------------------------------------------+
                    |                                |
         home_abilities (A_h, 32)          away_abilities (A_a, 32)
         home_archetypes (A_h, 10)         away_archetypes (A_a, 10)
                    |                                |
                    v                                v
     +-----------------------------------------------------------------+
     |                 LEVEL 2: PLAYER SYNERGY                         |
     |         (per team, with optional cross-team attention)          |
     |                                                                 |
     |  Stage 1: Archetype Interaction                                 |
     |    S_arch_ij = arch_i @ M @ arch_j    (A, A) pairwise scores   |
     |                                                                 |
     |  Stage 2: FM Pairwise Residual                                  |
     |    S_fm_ij = <v_i, v_j>               (A, A) player-specific   |
     |                                                                 |
     |  Stage 3: GATv2 Message Passing                                 |
     |    h_i = GATv2(ability_i, teammates, edge_features)             |
     |    [optional cross-team attention with opponent]                 |
     |    h_i: (32,) context-adjusted player vector                    |
     |                                                                 |
     |  Stage 4: Lineup Aggregation                                    |
     |    team_player_agg = GatedAttnPool(h_1..h_A)     (32,)        |
     |    team_synergy = MLP(aggregate pairwise features) (64,)        |
     |    l2_team = concat(team_player_agg, team_synergy) (96,)        |
     +-----------------------------------------------------------------+
                    |                                |
           l2_home (96,)                     l2_away (96,)
                    |                                |
                    v                                v
     +-----------------------------------------------------------------+
     |                    LEVEL 3: TEAM MODEL                          |
     |          (per team, adds coaching/system/org signal)            |
     |                                                                 |
     |  team_features: (22,)  [Four Factors, efficiency, coaching,     |
     |                         defense scheme, continuity]             |
     |                                                                 |
     |  Continuity gate:                                               |
     |    gate = sigmoid(w * continuity + b)                           |
     |    l3_input = gate * team_history + (1-gate) * l2_team          |
     |                                                                 |
     |  Input projection:                                              |
     |    x = Linear(118, 128)(concat(l2_team, team_features))         |
     |                                                                 |
     |  Residual MLP: 2 blocks, hidden=256                             |
     |    team_repr: (128,)                                            |
     +-----------------------------------------------------------------+
                    |                                |
          home_repr (128,)                  away_repr (128,)
                    |                                |
                    +----------- MATCHUP -----------+
                    |                                |
                    v                                v
     +-----------------------------------------------------------------+
     |               LEVEL 4: GAME CONTEXT + PREDICTION                |
     |                                                                 |
     |  Matchup construction:                                          |
     |    concat(home, away)          (256,)                           |
     |    home - away                 (128,)                           |
     |    home * away                 (128,)                           |
     |    matchup = concat(all)       (512,)                           |
     |                                                                 |
     |  Context features:             (14,)                            |
     |    [home_flag, altitude, rest x2, b2b x2, games_7d x2,         |
     |     travel x2, tz x2, season_progress, is_playoffs]            |
     |                                                                 |
     |  context_proj = Linear(14, 64)  (64,)                           |
     |  l4_input = concat(matchup, context_proj)  (576,)               |
     |                                                                 |
     |  Prediction MLP:                                                |
     |    Linear(576, 256) + 2x ResidualBlock(256, 512) -> (256,)      |
     |                                                                 |
     |  Heads:                                                         |
     |    spread_mu, spread_sigma   (1,), (1,)   [Gaussian NLL]        |
     |    win_prob                  (1,)          [BCE]                 |
     |    total_mu, total_sigma     (1,), (1,)   [Gaussian NLL]        |
     +-----------------------------------------------------------------+
                                |
                                v
                    FINAL PREDICTION:
                      spread = mu +/- sigma
                      P(home_win) = win_prob
                      total = total_mu +/- total_sigma
```

---

## 11. Training Strategy

### 11.1 Three-Phase Training

The system is trained in three phases, motivated by the principle that lower levels need stable representations before higher levels can learn effectively.

#### Phase A: Level 1 Pre-Training (Bottom-Up)

**Duration**: Until convergence (~50-100 epochs)
**What trains**: Level 1 only
**Loss**: Next-game box score prediction + archetype clustering loss
**Data**: All player-game records (2001-2026, ~750K player-games)

```
L1 Loss = MSE(predicted_box_scores, actual_box_scores)
        + KL(archetype_posterior || archetype_prior)
        + contrastive_loss(similar_players_close, different_players_far)
```

**Goal**: Learn stable player ability vectors that capture intrinsic ability separated from team context. The contrastive loss encourages the embedding space to be well-structured (similar players nearby, different players far apart).

#### Phase B: Level 2-3 Pre-Training (Middle-Out)

**Duration**: Until convergence (~30-50 epochs)
**What trains**: Levels 2 and 3 (Level 1 frozen, producing fixed ability vectors)
**Loss**: Team-level net rating prediction
**Data**: All team-game records (2001-2026, ~63K team-games)

```
L2-3 Loss = MSE(predicted_team_net_rating, actual_team_net_rating)
          + lambda_arch * archetype_regularization
```

**Goal**: Learn synergy effects and team-level representations with stable L1 inputs. The archetype regularization penalizes FM synergy vectors that are too large, biasing toward the archetype-level prior (L-RAPM-inspired: default to additive, deviate only with evidence).

#### Phase C: End-to-End Fine-Tuning

**Duration**: ~20-30 epochs with early stopping
**What trains**: All levels (L1 at 0.1x learning rate, L2-L4 at full rate)
**Loss**: Full multi-task loss (spread + win + total)
**Data**: All games with final scores (2001-2026, ~31K games)

```
Full Loss = 1.0 * loss_spread + 0.3 * loss_win + 0.3 * loss_total
          + 0.1 * loss_consistency
          + 0.01 * L2_weight_decay
```

**Goal**: Allow all levels to co-adapt for the downstream prediction task. The reduced L1 learning rate prevents catastrophic forgetting of the pre-trained player representations. The full loss provides gradient signal from the final prediction back through all levels.

### 11.2 Gradient Flow Diagram

```
Phase A (L1 pre-train):
  [L1] <-- grad --  L1 Loss
  [L2] [L3] [L4]   (not instantiated)

Phase B (L2-3 pre-train):
  [L1] ---- frozen, no grad ----
     |
     v
  [L2] <-- grad --+
     |             |
     v             +-- L2-3 Loss
  [L3] <-- grad --+
  [L4]   (not instantiated)

Phase C (end-to-end fine-tune):
  [L1] <-- grad (0.1x lr) --+
     |                        |
     v                        |
  [L2] <-- grad (1.0x lr) --+
     |                        +-- Full Multi-Task Loss
     v                        |
  [L3] <-- grad (1.0x lr) --+
     |                        |
     v                        |
  [L4] <-- grad (1.0x lr) --+
```

### 11.3 Learning Rate Schedule

```python
# Phase C learning rates (AdamW optimizer):
lr_base = 3e-4

param_groups = [
    {"params": level1.parameters(), "lr": lr_base * 0.1},   # 3e-5
    {"params": level2.parameters(), "lr": lr_base},          # 3e-4
    {"params": level3.parameters(), "lr": lr_base},          # 3e-4
    {"params": level4.parameters(), "lr": lr_base},          # 3e-4
]

# Cosine annealing with warm restarts, T_0=5 epochs
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
```

---

## 12. Batch Structure and Data Loading

### 12.1 What Is a Sample?

**A sample is one game.** Each game produces:

```python
sample = {
    # Level 1 inputs (pre-computed, loaded from cache)
    "home_abilities":      (A_h, 32),    # home active players' ability vectors
    "home_uncertainties":  (A_h, 32),
    "home_archetypes":     (A_h, 10),
    "home_mask":           (A_h,),
    "away_abilities":      (A_a, 32),
    "away_uncertainties":  (A_a, 32),
    "away_archetypes":     (A_a, 10),
    "away_mask":           (A_a,),

    # Level 2 auxiliary (pre-computed)
    "home_fm_vectors":     (A_h, 32),    # per-player synergy vectors
    "away_fm_vectors":     (A_a, 32),

    # Level 3 inputs
    "home_team_features":  (22,),
    "away_team_features":  (22,),

    # Level 4 inputs
    "context_features":    (14,),

    # Targets
    "home_score":          (1,),
    "away_score":          (1,),
    "spread":              (1,),         # home_score - away_score
    "total":               (1,),         # home_score + away_score
    "home_win":            (1,),         # binary
}
```

### 12.2 Batching

Games are batched with padding to the maximum roster size in the batch:

```python
batch_size = 64
max_roster = max(A_h, A_a across batch)    # typically 13

# Padded tensors:
home_abilities:    (B, max_roster, 32)
home_mask:         (B, max_roster)          # 1.0 for real players, 0.0 for padding
```

### 12.3 Cache Strategy

Level 1 ability vectors are the most expensive to compute (requiring each player's full game history). The cache strategy:

1. **Build once**: After Phase A training, run inference on all players at every date in the dataset. Store the resulting ability vectors in a date-indexed cache.
2. **Load at game time**: For each game in the training set, look up each active player's ability vector from the cache at the game's date.
3. **Phase C fine-tuning**: When L1 weights change during fine-tuning, we do NOT rebuild the cache. Instead, L1 runs live in the forward pass with the fine-tuned weights. This is more expensive but necessary for end-to-end gradients.

For inference, the live forward pass through L1 uses the player's most recent game data.

---

## 13. Roster Entry Point

### 13.1 Training Time

During training, we know the actual rosters (which players played). We use the actual active players (those with >0 minutes in the box score) as the roster for that game. This gives the model the "perfect information" roster.

### 13.2 Inference Time

At inference time, we need pre-game roster predictions. The pipeline:

```
1. Scrape injury reports (NBA official injury report, mandatory 5:00 PM ET)
2. Cross-reference with team depth charts
3. Determine probable active roster (8-13 players per team)
4. Look up each player's latest L1 ability vector
5. Feed into L2 forward pass
```

Players listed as "Out" are excluded. Players listed as "Questionable" or "Probable" can be handled probabilistically (run the model twice, weight by estimated availability probability) or with a simple threshold (include if >50% chance of playing).

### 13.3 Missing Player Handling

If a player is unexpectedly unavailable at game time (last-minute scratch):

- Remove their ability vector from the L2 input
- L2 naturally computes a weaker team representation (fewer contributing players)
- The model does NOT need a special "injury adjustment" feature because the player's absence is directly reflected in the L2 input set

This is a key advantage of the player-level architecture over team-level approaches: the model automatically adjusts for any roster change without needing to learn a separate injury-impact model.

---

## 14. Parameter Budget

| Component | Parameters | Notes |
|-----------|-----------|-------|
| **Level 1** | | |
| Generic prior | 32 | Learned mean vector |
| Archetype embeddings (K=10) | 10 x 32 = 320 | Per-archetype mean |
| Position encoder | 5 x 16 = 80 | Position embedding |
| Physical profile MLP | 4->64->32 = 2,336 | Project physical features |
| Temporal encoder (2-layer Transformer) | ~265K | d=128, 4 heads, FF=512 |
| Output projection | 128->32 = 4,128 | To ability vector dim |
| Uncertainty head | 128->32 = 4,128 | Per-dimension sigma |
| **L1 subtotal** | **~276K** | |
| | | |
| **Level 2** | | |
| Archetype interaction matrix M | 55 | K=10, symmetric |
| FM synergy vectors | ~500 x 32 = 16K | Per active player |
| GATv2 within-team (1 layer, 4 heads) | ~33K | d=32, 4 heads |
| GATv2 cross-team (1 layer, 4 heads) | ~33K | Separate params |
| Edge feature projection | 4->32 = 160 | For GATv2 edge features |
| Gated attention pooling | 32->1 x 2 = 128 | Gate + attention networks |
| Synergy aggregation MLP | varies->64 = ~8K | Pairwise to team |
| Lineup MLP | 64->128->64 = ~16K | Higher-order implicit |
| **L2 subtotal** | **~106K** (+ 16K player-specific) | |
| | | |
| **Level 3** | | |
| Input projection | 118->128 = 15,232 | Concat L2 + features |
| Residual MLP block 1 | 128->256->128 = 65,920 | With bias |
| Residual MLP block 2 | 128->256->128 = 65,920 | With bias |
| LayerNorms (x2) | 256 each = 512 | Per block |
| Continuity gate | 3 | w, b for gate + bias |
| **L3 subtotal** | **~147K** | |
| | | |
| **Level 4** | | |
| Context projection | 14->64 = 960 | Project context features |
| Input projection | 576->256 = 147,712 | Matchup + context |
| Residual block 1 | 256->512->256 = 262,656 | With bias |
| Residual block 2 | 256->512->256 = 262,656 | With bias |
| LayerNorms (x2) | 512 each = 1,024 | Per block |
| Spread head (mu + sigma) | 256->1 x 2 = 514 | Two linear layers |
| Win prob head | 256->1 = 257 | One linear layer |
| Total head (mu + sigma) | 256->1 x 2 = 514 | Two linear layers |
| **L4 subtotal** | **~676K** | |
| | | |
| **TOTAL** | **~1.2M** (+ 16K per-player synergy vectors) | |

This is deliberately compact. The current Phase 3 model has ~4.4M parameters. The reduction reflects the shift from a monolithic architecture to a hierarchical one where each level has a clear, bounded role.

---

## 15. Design Decisions and Rationale

### 15.1 Why Dual Output from L2 (Context-Adjusted Vectors + Interaction Features)?

**Alternative A: Modify-only.** L2 just updates player vectors. Team representation is the pool of modified vectors.
- Problem: Loses explicit synergy information. The GATv2 attention weights encode synergy implicitly, but there is no mechanism to capture team-level emergent properties (e.g., total floor spacing, defensive switching versatility) that are properties of the group, not any individual.

**Alternative B: Features-only.** L2 produces interaction features but does not modify player vectors.
- Problem: Downstream levels would see the same player vectors regardless of context. A point guard's representation would be identical whether paired with a shooting-heavy lineup or an iso-heavy one. This contradicts the "Who You Play Affects How You Play" finding (Luo & Krishnamurthy, 2023).

**Our choice (A+B):** Context-adjusted vectors capture how each player's effective ability changes with teammates. Separate interaction features capture emergent team properties. Both signals are needed.

### 15.2 Why Gated Attention Pooling (Not Mean or DeepSets)?

**Mean pooling** treats all players equally. But a team's identity is more defined by its stars than its bench. Mean pooling washes out star contributions in a 13-player roster.

**DeepSets** (sum-pool after per-element MLP) is permutation-invariant and universal, but the sum operation makes the representation scale-dependent on roster size. A 13-player roster would have a larger norm than an 8-player roster, creating a confound.

**Gated attention pooling** (Ilse et al., 2018) learns which players to weight heavily. It naturally handles variable roster sizes, is permutation-invariant, and produces representations whose magnitude reflects team quality rather than roster size. The gating mechanism also allows the network to learn that the 9th and 10th players contribute negligibly, which is empirically true in the NBA.

### 15.3 Why [Concat, Diff, Hadamard] for Matchup?

This pattern comes from the NLI literature (Mou et al., 2016; Chen et al., 2017) where it is standard for comparing two sentence representations. Empirically, difference alone misses absolute scale (is -5 a close game between bad teams or a blowout between good teams?), concatenation alone requires the network to learn subtraction (wasteful), and Hadamard product alone loses sign information. The combination provides complementary views with minimal parameter overhead.

### 15.4 Why Bottom-Up Pre-Training Before End-to-End?

**Pure end-to-end** training of a 4-level hierarchy is prone to:
1. **Gradient vanishing**: Gradients from the spread loss must flow back through L4, L3, L2 matchup construction, L2 GATv2, and into L1. This is 8+ nonlinear layers.
2. **Representation collapse**: Without pre-training, L1 might learn degenerate representations that are easy for L4 to predict from but carry no meaningful player information.
3. **Conflicting gradients**: L1 receives gradients from every game, but its representation should capture general player ability, not be tuned for any specific opponent.

Bottom-up pre-training ensures:
- L1 learns meaningful, transferable player representations (grounded in box-score prediction).
- L2-L3 learn to use stable L1 outputs before the whole stack is tuned together.
- End-to-end fine-tuning then makes small, coordinated adjustments across all levels.

This is analogous to how BERT is pre-trained on masked language modeling before being fine-tuned on downstream tasks. The hierarchical structure benefits from hierarchical training.

### 15.5 Why Multi-Task Heads?

A spread-only model misses complementary learning signals:

- **Win probability** provides a calibration anchor. A model predicting Home -7 with sigma=3 should imply ~99% home win probability. If the win head disagrees, the model has an internal inconsistency that the consistency loss will correct.
- **Total points** is partially orthogonal to spread. Two games can have the same spread (+5) but very different totals (185 vs 225). Learning to predict totals teaches the model about pace and efficiency effects that indirectly help spread prediction by better modeling game dynamics.
- **Shared representation** between heads acts as regularization, preventing the model from overfitting to spread-specific patterns.

### 15.6 Why d_player=32 and d_team=128?

**d_player=32**: The literature converges on 16-32 dimensions being sufficient for player representation. NBA2Vec uses 8 dimensions successfully; PCA on box scores needs ~5 for 80% variance; archetype studies find 9-12 clusters. We use 32 to have headroom for defensive and efficiency dimensions that are poorly captured by standard box-score PCA.

**d_team=128**: A team has 5 starters and 8-13 active players, each represented by 32 dimensions. After aggregation, 128 dimensions can capture the relevant information without excessive compression. The ratio 128/32 = 4x is a common expansion factor when aggregating set elements into a set representation. Level 3 adds 22 explicit features (projected to 128), so the team representation needs enough capacity to encode both player-derived and team-level information.

### 15.7 Why Roster Continuity as a Gate?

FiveThirtyEight discovered that the optimal blend of team-level (Elo) vs. player-level (RAPTOR) information varies by roster stability. Their optimal weights:
- Average team: 35% Elo / 65% RAPTOR
- High-continuity team (returning core): up to 55% Elo
- Low-continuity team (new roster): down to 0% Elo

Our continuity gate implements this principle: when continuity is high, team-level historical features (captured by L3's team features) are more predictive because the team's identity is stable. When continuity is low, the model should rely more on the player-level composition from L2 because team-level history is no longer representative.

---

## 16. Open Questions

### 16.1 Cross-Team Attention in L2

Should L2's GATv2 include cross-team message passing? Arguments:

**For**: Matchup effects are real. A rim-protector center's value increases against a paint-heavy opponent. Cross-team attention could capture positional matchup advantages directly at the player level.

**Against**: Cross-team attention means the team representation is no longer team-intrinsic -- it changes with every opponent. This complicates caching and means L2 must run at game time rather than being pre-computable per team. It also conflates L2's role (teammate synergy) with L4's role (game-specific context).

**Recommendation**: Start without cross-team attention. Let L4's matchup construction (concat + diff + Hadamard) handle cross-team effects. If the model hits a performance ceiling, add cross-team attention as an ablation.

### 16.2 FM Synergy Vector Storage

FM synergy vectors (d_syn=32 per player) are player-specific parameters. Options:

1. **Learned parameters**: Store in the model as an embedding table, trained end-to-end. Simple but does not transfer to new players.
2. **Predicted from L1**: A small MLP predicts the synergy vector from the ability vector: `v_i = MLP(ability_i)`. Generalizes to new players but may not capture player-specific chemistry.
3. **Hybrid**: Initialize from MLP prediction, then allow gradient updates for players with sufficient data.

**Recommendation**: Option 3 (hybrid). The MLP provides a reasonable initialization for all players, and gradient updates refine it for well-observed players.

### 16.3 L3 Team History Encoder

The current design uses hand-crafted rolling features for L3. An alternative is a learned team history encoder:

```python
# Feed last N team-game results through a small Transformer or GRU
team_game_sequence: (N, d_game_summary)    # last 15-20 games
team_history_repr = TemporalEncoder(team_game_sequence)    # (d_team_hist,)
```

This would allow L3 to learn what aspects of team history matter, rather than relying on hand-crafted Four Factors and efficiency metrics.

**Recommendation**: Start with hand-crafted features (simpler, interpretable, fewer parameters). If L3 consistently underfits, add a learned encoder.

### 16.4 Playoff Mode

The regular season and playoffs may require different model behavior:
- Playoff series feature repeated matchups (7-game series), allowing coaches to make adjustments
- Effort and intensity increase in playoffs
- Rotation shortens (8-player instead of 10-12)

**Recommendation**: Include `is_playoffs` as a context feature in L4. If systematic differences emerge, consider a playoff-specific fine-tuning phase or a learned mode switch.

### 16.5 Real-Time Roster Uncertainty

For pre-game prediction when rosters are uncertain (questionable players), we could:

1. Run the model multiple times with different roster scenarios and weight by probability
2. Add a "player availability probability" feature to L2's input
3. Always use the most likely roster

**Recommendation**: Option 1 is most principled and allows computing prediction intervals that account for roster uncertainty. It requires 2-4 forward passes per game (one per likely roster configuration) but the model is small enough that this is feasible.

---

## Summary of Tensor Shapes Through the Pipeline

| Interface | Tensor | Shape | Description |
|-----------|--------|-------|-------------|
| L1 output | player_ability | (d_player,) = (32,) | Per-player ability vector |
| L1 output | player_uncertainty | (d_player,) = (32,) | Per-dimension confidence |
| L1 output | player_archetype | (K,) = (10,) | Soft archetype membership |
| L1->L2 | home_abilities | (A_h, 32) | Active home roster abilities |
| L1->L2 | away_abilities | (A_a, 32) | Active away roster abilities |
| L2 internal | pairwise_synergy | (A, A) | Archetype + FM scores |
| L2 internal | h_context | (A, 32) | GATv2-adjusted player vectors |
| L2 output | l2_team | (96,) | team_player_agg(32) + team_synergy(64) |
| L2->L3 | l2_team + team_feat | (96,) + (22,) = (118,) | Concatenated input to L3 |
| L3 output | team_repr | (128,) | Team representation |
| L3->L4 | matchup | (512,) | concat(256) + diff(128) + hadamard(128) |
| L4 input | matchup + context | (512,) + (64,) = (576,) | Full prediction input |
| L4 output | spread_mu | (1,) | Predicted margin |
| L4 output | spread_sigma | (1,) | Prediction uncertainty |
| L4 output | win_prob | (1,) | Home win probability |
| L4 output | total_mu | (1,) | Predicted total points |
| L4 output | total_sigma | (1,) | Total uncertainty |

**Batched shapes** (B = batch size, P = max padded roster):

| Tensor | Batched Shape |
|--------|--------------|
| home_abilities | (B, P, 32) |
| home_mask | (B, P) |
| pairwise_synergy | (B, P, P) |
| l2_team | (B, 96) |
| team_repr | (B, 128) |
| matchup | (B, 512) |
| spread_mu | (B, 1) |

---

## References

### Directly Informing This Design

- Luo & Krishnamurthy (2023). "Who You Play Affects How You Play." [arXiv:2303.16741](https://arxiv.org/abs/2303.16741) -- GATv2 player interaction graphs
- Brody, Alon & Yahav (2022). "How Attentive are Graph Attention Networks?" ICLR. [arXiv:2105.14491](https://arxiv.org/abs/2105.14491) -- GATv2 attention mechanism
- Rendle (2010). "Factorization Machines." IEEE ICDM. -- Pairwise interaction modeling
- Ilse, Tomczak & Welling (2018). "Attention-based Deep Multiple Instance Learning." ICML. [arXiv:1802.04712](https://arxiv.org/abs/1802.04712) -- Gated attention pooling
- Zaheer et al. (2017). "Deep Sets." NeurIPS. -- Permutation-invariant set functions
- HIGFormer (2025). "Player-Team Heterogeneous Interaction Graph Transformer." KDD. [arXiv:2507.10626](https://arxiv.org/abs/2507.10626) -- Hierarchical player-team graph
- arXiv:2601.15000 (2026). "Lineup Regularized Adjusted Plus-Minus." -- L-RAPM informed priors
- Maymin, Maymin & Shen (2013). "NBA Chemistry." IJCSS. -- Synergy quantification (~6 wins)
- Berry & Fowler (2019). "How Much Do Coaches Matter?" Sloan. -- 30% coaching variance
- FiveThirtyEight NBA Methodology. -- Continuity-gated Elo/RAPTOR weighting
- Mou et al. (2016). "Natural Language Inference by Tree-Based Convolution." ACL. -- [concat, diff, product] comparison
- Liu et al. (2020). "Learning Agent Representations for Ice Hockey." NeurIPS. -- VaRLAE hierarchical prior
- Song et al. (2022). "Eastward Jet Lag in the NBA." Frontiers in Physiology. -- Travel effects
- Entine & Small (2008). "Rest in the NBA Home-Court Advantage." JQAS. -- Rest/HCA quantification
- Medvedovsky (2020). "NBA Stabilization Rates." -- Stat-specific stabilization
- Snarr. "EPM." Dunks & Threes. -- Skills concept for per-stat optimization
- Erhan et al. (2010). "Why Does Unsupervised Pre-training Help Deep Learning?" JMLR. -- Bottom-up pre-training motivation

### Sports Prediction Architecture Surveys
- [Interactive sequential generative models for team sports](https://link.springer.com/article/10.1007/s10994-024-06648-2)
- [Predicting sport event outcomes using deep learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC12453701/)
- [A Systematic Review of Machine Learning in Sports Betting](https://arxiv.org/html/2410.21484v1)
- [Graph Neural Networks to Predict Sports Outcomes](https://arxiv.org/abs/2207.14124)

### Recommendation System Analogies
- [Embedding in Recommender Systems: A Survey](https://arxiv.org/html/2310.18608v2)
- [KHGCN: Knowledge-Enhanced Recommendation with Hierarchical Graph Capsule Network](https://pmc.ncbi.nlm.nih.gov/articles/PMC10137578/)
