# Wave 2: Level 1 Architecture Deep-Dive -- Three Paradigms Compared

**Date**: 2026-03-16
**Purpose**: Head-to-head comparison of VaRLAE, Kalman Filter (DARKO-style), and EPM two-stage approaches for building per-player ability vectors. Synthesizes Wave 1 research into a concrete hybrid architecture proposal.

---

## Table of Contents

1. [Paradigm Summaries](#1-paradigm-summaries)
2. [Head-to-Head Comparison Table](#2-head-to-head-comparison-table)
3. [Hybrid Recommendation](#3-hybrid-recommendation)
4. [Concrete Architecture Proposal](#4-concrete-architecture-proposal)
5. [Training Procedure](#5-training-procedure)
6. [Cold-Start Behavior](#6-cold-start-behavior)
7. [Team-Agnostic Mechanism](#7-team-agnostic-mechanism)

---

## 1. Paradigm Summaries

### 1.1 Paradigm A: VaRLAE (Variational Recurrent Ladder Agent Encoder)

**Source**: Liu, Schulte, Poupart, Rudd, Javan. "Learning Agent Representations for Ice Hockey." NeurIPS 2020.

VaRLAE is a Conditional Variational Autoencoder (CVAE) extended with (a) a Ladder structure for hierarchical latent variables and (b) recurrent (RNN) components for modeling game history. It was originally designed for ice hockey event-level prediction with 4.5M+ events and 1,000+ players.

**Core idea**: Learn a generative model p(player_action | game_context, player_identity) where player identity is captured by a latent variable z. The model learns a *context-specific shared prior* p(z | context) that represents what a "generic player" would do in this context. The *posterior* q(z | context, player_identity, observation) represents what THIS specific player does. For data-rich players, the posterior diverges from the prior (the model has learned their idiosyncrasies). For data-poor players, the posterior stays close to the prior (Bayesian shrinkage).

**Mathematical framework**:

The generative model defines a hierarchy of L latent variable groups z = {z_1, ..., z_L}:

```
Generative model (top-down):
  p(x, z | c) = p(x | z_1, c) * prod_{l=1}^{L-1} p(z_l | z_{l+1}, c) * p(z_L | c)

Inference model (bottom-up then top-down merge):
  q(z | x, c) = prod_{l=1}^{L} q(z_l | z_{l+1}, x, c)

ELBO:
  log p(x|c) >= E_q[log p(x|z,c)] - sum_{l=1}^{L} KL[q(z_l|...) || p(z_l|z_{l+1},c)]
```

The Ladder structure uses **precision-weighted merging** between the bottom-up encoder output (data-dependent approximate likelihood) and the top-down generative prior:

```
Bottom-up pass: compute d_l = encoder_l(x)  -> produces mu_bu, sigma_bu for each level
Top-down pass:  compute p_l = prior_l(z_{l+1}, c) -> produces mu_td, sigma_td for each level

Merge (precision-weighted):
  precision_merged = 1/sigma_bu^2 + 1/sigma_td^2
  mu_merged = (mu_bu/sigma_bu^2 + mu_td/sigma_td^2) / precision_merged
  sigma_merged = 1/sqrt(precision_merged)
```

This is the mechanism that produces shrinkage: when the bottom-up data is sparse (high sigma_bu), the merged posterior collapses toward the top-down prior.

**Recurrent component**: An RNN (GRU/LSTM) processes the sequence of game events to build a context vector c_t at each timestep. This context feeds into both the prior and posterior networks.

**Key strengths**:
- Native hierarchical latent structure (maps to population -> archetype -> individual)
- Built-in Bayesian shrinkage for cold-start
- Recurrent component handles sequential game data
- Learns the prior from data (not hand-specified)

**Key weaknesses**:
- Complex training (KL annealing, posterior collapse risk)
- No explicit aging model -- temporal evolution is implicit in the RNN
- Originally designed for event-level prediction, not stat-level player modeling
- Computationally expensive (full variational inference per player per game)

---

### 1.2 Paradigm B: Kalman Filter (DARKO-style)

**Source**: Medvedovsky, "DARKO" (Daily Adjusted and Regressed Kalman Optimized). Publicly documented at darko.app, kmedved.com, and nbastuffer.com.

DARKO combines three components: (1) exponential decay weighting of historical game stats, (2) a modified Kalman filter for state estimation under noise, and (3) gradient boosted decision trees to combine the two projections nonlinearly. It updates projections for every NBA player, every box-score stat, every day.

**Kalman filter formulation for player ability**:

```
State equation:     x_t = F * x_{t-1} + w_t,     w_t ~ N(0, Q)
Observation eq:     y_t = H * x_t + v_t,          v_t ~ N(0, R)

Where:
  x_t = player ability state vector (d dimensions) at time t
  F   = state transition matrix (encodes aging/drift)
  Q   = process noise covariance (how much ability changes game-to-game)
  H   = observation matrix (maps latent ability to expected box-score stats)
  R   = observation noise covariance (game-outcome variance, very large in basketball)
  y_t = observed box-score stats from game t

Prediction step:
  x_{t|t-1} = F * x_{t-1|t-1}
  P_{t|t-1} = F * P_{t-1|t-1} * F' + Q

Update step:
  K_t = P_{t|t-1} * H' * (H * P_{t|t-1} * H' + R)^{-1}    [Kalman gain]
  x_{t|t} = x_{t|t-1} + K_t * (y_t - H * x_{t|t-1})         [state update]
  P_{t|t} = (I - K_t * H) * P_{t|t-1}                        [covariance update]
```

**Exponential decay component**: Each past game gets weight beta^t, where beta in (0,1) and t = days since the game. With beta = 0.99, a game from one year ago has weight 0.025, making it roughly 40x less important than yesterday's game. This is applied independently per stat, with stat-specific decay rates optimized via differential evolution.

**Stabilization rates (from Medvedovsky 2020)**:

| Stat | Padding Games (lower = faster stabilization) |
|------|----------------------------------------------|
| Minutes | ~10 |
| Usage Rate | ~41 |
| Assist % | ~80 |
| Turnover % | ~81 |
| True Shooting % | ~135 |
| 3-Point % | ~242 |
| Effective FG % | ~247 |
| Plus/Minus | ~1,000+ |

**Padding formula (Bayesian shrinkage equivalent)**:

```
Expected Stat = (Actual * n + League_Avg * X) / (n + X)
```

where X = stat-specific padding constant, n = sample size. This is mathematically equivalent to a Beta(alpha, beta) conjugate prior for proportions where the pseudo-counts equal X.

**Gradient boosted trees**: A GBDT combines the exponential decay projection and the Kalman filter projection into a final prediction. This allows learning nonlinear interactions between the two signals that neither can capture alone. Context features (opponent strength, rest days, home/away, pace, altitude, travel distance) are also fed to the GBDT.

**Rookie handling ("padding method")**: New players are initialized with ~240 league-average pseudo-observations. Their projection is dominated by this prior until sufficient real games accumulate. The decay of prior influence is stat-specific: minutes projections stabilize after ~10 games, but shooting percentages require 100+ games.

**Key strengths**:
- Computationally trivial (matrix multiply per game, no gradient computation)
- Principled uncertainty tracking (covariance matrix P)
- Stat-specific stabilization rates match basketball reality
- Daily updates are trivial
- Proven predictive accuracy (#1 RMSE for NBA game prediction)

**Key weaknesses**:
- Linear state transition model cannot capture nonlinear aging/development curves
- No shared structure across players (each player tracked independently)
- Cannot learn latent ability dimensions -- operates on observed stat space
- Relies on GBDT for nonlinear corrections (two separate models)
- No hierarchical prior (league -> position -> individual)
- Not team-agnostic by design (observed stats confound player ability with team system)

---

### 1.3 Paradigm C: EPM Two-Stage (Box-Score SPM -> RAPM Adjustment)

**Source**: Snarr, "Estimated Plus-Minus (EPM)," dunksandthrees.com. Also RPM (Engelmann & Ilardi), BPR (Miyakawa).

EPM is a two-stage pipeline that first builds a Statistical Plus-Minus (SPM) model from box-score stats, then uses those SPM predictions as a Bayesian prior for Regularized Adjusted Plus-Minus (RAPM) computation.

**Stage 1: Estimated Skills -> SPM Model**

Step 1a -- Estimate "true skills" per stat:
- For each stat category (3PT%, 2PT%, assist rate, turnover rate, rebound rate, etc.), learn the optimal exponential decay factor via differential evolution
- Goal: predict next-game stat values that minimize RMSE over 20+ years of data (2001-present)
- Factors modeled per-stat: age sensitivity, team/opponent strength, seasonality, days rest, sample size
- Output: "estimated skills" = noise-reduced predictions of each stat at each point in time

Step 1b -- SPM regression:
- Train a regression model mapping estimated skills -> RAPM
- Training target: multi-year RAPM computed from 5+ million possessions (2001-2019)
- Two separate models: one for box-score stats only (available back to 2001), one incorporating tracking data (available from 2013-14)
- Features include: scoring rates, shooting percentages, assist/turnover rates, rebound rates, plus tracking data (speed, distance, touch time, etc.) when available
- Features are adjusted relative to league average at each point in time to account for era effects

Step 1c -- SPM output:
- Each player at each point in time receives an SPM estimate: their predicted impact per 100 possessions based on their box-score/tracking stats

**Stage 2: RAPM with SPM Bayesian Prior**

The SPM values serve as the *prior mean* in a prior-informed ridge regression:

```
RAPM formulation:
  y = X * beta + epsilon

  X: design matrix (rows = possessions, columns = 2 * N_players)
     X_ij = +1 if player j on offense, -1 if on defense, 0 otherwise
  y: points scored per possession
  beta: player impact coefficients (what we solve for)

Prior-informed ridge regression:
  minimize: ||y - X*beta||^2 + lambda * ||beta - mu_0||^2

  Solution: beta_hat = (X'X + lambda*I)^{-1} (X'y + lambda * mu_0)

  where mu_0 = SPM estimates (the Bayesian prior)
```

Players with lots of on-off data get beta_hat close to their raw RAPM (data overwhelms prior). Players with little playing time get beta_hat close to their SPM estimate (prior dominates).

**Exponential decay in RAPM**: EPM applies exponential decay to all possessions since 2002, weighting recent possessions more heavily. This makes the metric responsive to current form rather than career averages.

**Key strengths**:
- RAPM target is inherently team-agnostic (controls for every teammate and opponent on the court)
- SPM prior provides excellent cold-start behavior (box-score stats available immediately)
- Proven best-in-class predictive performance (EPM wins metric comparisons)
- Stat-specific stabilization via estimated skills
- Daily updating via rolling exponential decay

**Key weaknesses**:
- Requires possession-level lineup data (not just box scores)
- RAPM computation is expensive (5+ million possession ridge regression)
- Produces a scalar per player (O-EPM, D-EPM), not a rich ability vector
- Season-level RAPM is inherently slow to stabilize (1000+ possessions needed)
- The SPM -> RAPM pipeline is two disconnected models, not end-to-end
- No learned latent structure -- the ability "vector" is just the estimated skills
- No explicit aging model

---

## 2. Head-to-Head Comparison Table

| Requirement | A: VaRLAE | B: Kalman Filter | C: EPM Two-Stage |
|-------------|-----------|-------------------|-------------------|
| **Team-agnostic** | Partially. Can learn to disentangle if trained with appropriate loss, but nothing in the architecture enforces it. Observed stats are still team-context-contaminated. | No. Operates on raw box-score stats which are confounded by team system, pace, usage distribution, and teammate quality. | Yes, by design. RAPM controls for every teammate and opponent. The gold standard for team-agnostic evaluation. |
| **Hierarchical pre-training** | Native fit. The ladder hierarchy of latent variables maps directly to population -> archetype -> individual. Shared prior = population level. Intermediate latent levels = archetype. Posterior = individual. | No hierarchy. Each player is tracked independently with no sharing of information across players, positions, or archetypes. | Partial. The SPM prior provides a form of shrinkage toward "what a player with these stats is typically worth," but there is no explicit position/archetype hierarchy. |
| **Aging/career curves** | Implicit only. The RNN can learn temporal patterns from sequences of games, but there is no explicit aging model. The recurrent hidden state would need to encode age-related trends. | Can be encoded in the state transition matrix F. In principle, F could be age-dependent: F(age) applies different drift rates at different career stages. However, DARKO does not do this -- it uses a fixed decay rate. | Not modeled. EPM treats each time point independently. CARMELO/PREDATOR add age-based adjustments as separate features, but EPM itself has no aging model. |
| **Game-by-game updates** | Yes. The RNN hidden state updates with each new game event. However, the full variational inference is expensive -- requires forward pass through encoder and decoder per update. | Excellent. The Kalman update is a single matrix multiply per game. This is the strongest paradigm for efficient sequential updates. O(d^2) per update where d = state dimension. | Partially. The estimated skills update daily via exponential decay (fast). But the RAPM component requires re-solving the full ridge regression to incorporate new possessions (slow). In practice, EPM uses a rolling window. |
| **Cold-start (rookies)** | Strong. The shared prior provides a meaningful initialization for any new player. As games accumulate, the posterior diverges from the prior at a rate controlled by observation noise. The ladder hierarchy means rookies start at population level and gradually specialize. | Moderate. The padding method initializes with X league-average pseudo-observations. Simple and effective but not position-aware -- all rookies start at the same league-average point regardless of position or draft pedigree. | Moderate. The SPM prior provides initial estimates from box-score stats (available from game 1). But the SPM model was trained on established players, so its predictions for rookies with 5 games are unreliable. RAPM is essentially useless for players with <100 possessions. |
| **Output: 24-32d ability vector** | Native. The latent variable z is an explicit vector of configurable dimension. The encoder produces this vector; the decoder reconstructs observations from it. We choose the dimension. | Not native. The Kalman state tracks observed stats (points, rebounds, etc.), not a learned latent space. Could be extended by adding a learned observation matrix H that maps a latent ability vector to expected stats, but this is an extension, not the core design. | Not native. EPM outputs a scalar (or O/D split). The intermediate "estimated skills" are a vector of stat projections (~15-20 stats), but these are in observed stat space, not a learned latent space. Could use the estimated skills vector as a proxy ability vector. |
| **Feed into Level 2 (graph)** | Excellent. The latent z vector is differentiable and can be used as node features in a GNN. If the VaRLAE and the graph network are trained end-to-end, gradients flow from graph-level loss back to the player encoder, jointly optimizing the ability representation for team-interaction prediction. | Moderate. The Kalman state is a fixed vector that can be used as GNN node features, but there are no gradients -- the Kalman filter is not a neural network module. The graph network cannot influence what the player model learns. | Poor for end-to-end. The estimated skills or RAPM values can be used as GNN node features, but the EPM pipeline is not differentiable. No gradient flow from the graph back to the player model. |

### Summary Scores

| Requirement | A: VaRLAE | B: Kalman | C: EPM |
|-------------|-----------|-----------|--------|
| Team-agnostic | 2/5 | 1/5 | 5/5 |
| Hierarchical pre-training | 5/5 | 1/5 | 2/5 |
| Aging/career curves | 2/5 | 3/5 | 1/5 |
| Game-by-game updates | 3/5 | 5/5 | 3/5 |
| Cold-start (rookies) | 5/5 | 3/5 | 3/5 |
| Output: ability vector | 5/5 | 2/5 | 2/5 |
| Feed into Level 2 (graph) | 5/5 | 3/5 | 2/5 |
| **Total** | **27/35** | **18/35** | **18/35** |

VaRLAE wins on architecture fit, but its critical weakness is team-agnostic evaluation. Kalman wins on computational efficiency and updates. EPM wins on team-agnostic evaluation. The hybrid must combine all three strengths.

---

## 3. Hybrid Recommendation

### 3.1 Core Thesis

The optimal architecture uses **VaRLAE's hierarchical variational structure** as the backbone, with **EPM-inspired team-agnostic training targets** as the supervision signal, and **Kalman-inspired update mechanics** for efficient game-by-game inference.

### 3.2 What We Take from Each Paradigm

**From VaRLAE (architecture)**:
- Hierarchical latent variable structure with L=3 levels
- Precision-weighted merging between prior and data-dependent posterior
- Learned shared prior that provides cold-start initialization
- Differentiable encoder-decoder framework that enables end-to-end training with Level 2

**From Kalman Filter (update mechanism)**:
- Explicit state + uncertainty tracking (mean and covariance)
- Stat-specific stabilization rates encoded in diagonal process noise Q
- Efficient O(d^2) per-game update rule
- Aging curve encoded in state transition F(age)

**From EPM (training target and team-agnostic signal)**:
- Estimated skills concept: stat-specific noise reduction via exponential decay
- SPM -> RAPM pipeline as training target (not as runtime architecture)
- RAPM values as the ground truth for what a team-agnostic player impact looks like
- Stat-level predictions as an auxiliary reconstruction objective

### 3.3 The Hybrid Design: VaKE (Variational Kalman Encoder)

We propose **VaKE**: a Variational Autoencoder with Kalman-style updates, trained on EPM-style team-agnostic targets. The name reflects the three-paradigm synthesis.

Key innovations:
1. **Train the VAE to reconstruct RAPM-adjusted stats, not raw stats** -- this makes the latent space team-agnostic by construction
2. **Replace the RNN-based sequential inference with an amortized Kalman update** -- this gives us O(d^2) updates instead of full encoder forward passes
3. **Use the Ladder hierarchy with three explicit levels** mapping to population, archetype, and individual

---

## 4. Concrete Architecture Proposal

### 4.1 Overview

```
Input: Game-level box-score stats for player i at game t
  y_t^i in R^16  (min, pts, oreb, dreb, ast, stl, blk, tov, pf, fga, fgm, fg3a, fg3m, fta, ftm, plus_minus)

Plus context features:
  c_t in R^12   (age, rest_days, home_flag, opponent_drtg, team_pace, usage_share,
                  minutes_share, season_progress, career_games_played, days_since_last_game,
                  opponent_pace, altitude)

Output: Player ability vector
  z_i in R^32   (the latent ability representation)
  Sigma_i in R^32x32  (uncertainty covariance, stored as diagonal for efficiency)
```

### 4.2 Layer-by-Layer Architecture

#### Level 0: Population Prior Network

Learns the distribution of a "generic NBA player" conditioned on context.

```
Input:  c_t in R^12 (context features, NOT player-specific stats)
Layers: Linear(12, 64) -> LayerNorm -> GELU
        Linear(64, 64) -> LayerNorm -> GELU

Output: mu_pop in R^32, log_sigma_pop in R^32
        via Linear(64, 32) for mu, Linear(64, 32) for log_sigma

Population prior: p_pop(z | c) = N(mu_pop, diag(sigma_pop^2))
```

This represents: "given this game context (opponent strength, home/away, pace, etc.), what would a league-average player's ability vector look like?"

Parameter count: 12*64 + 64 + 64*64 + 64 + 64*32 + 32 + 64*32 + 32 = ~7K

#### Level 1: Archetype Prior Network

Learns K=10 archetype distributions. Each archetype specializes the population prior.

```
Input:  z_pop sample from Level 0 (R^32) + physical_profile in R^4 (height, weight, wingspan, draft_position_normalized)
Layers: Linear(36, 128) -> LayerNorm -> GELU
        Linear(128, 128) -> LayerNorm -> GELU

Archetype assignment (soft):
  logits_k = Linear(128, K=10)     [K=10 archetype logits]
  pi_k = softmax(logits_k)          [archetype membership probabilities]

Per-archetype parameters:
  mu_arch_k = Linear(128, 32) for each k (or: shared Linear(128, 32*K) reshaped)
  log_sigma_arch_k = Linear(128, 32) for each k

Mixture prior: p_arch(z | z_pop, physical) = sum_k pi_k * N(mu_arch_k, diag(sigma_arch_k^2))

Precision-weighted merge with population:
  For the selected archetype k (or mixture):
  prec_pop = 1 / sigma_pop^2
  prec_arch = 1 / sigma_arch_k^2
  prec_merged = prec_pop + prec_arch
  mu_L1 = (mu_pop * prec_pop + mu_arch_k * prec_arch) / prec_merged
  sigma_L1 = 1 / sqrt(prec_merged)
```

This represents: "this player has physical profile suggesting 60% wing archetype, 30% guard archetype, 10% big archetype. Given that mix, the prior ability vector is..."

Parameter count: ~36*128 + 128 + 128*128 + 128 + 128*10 + 128*32*10 + 128*32*10 = ~100K

#### Level 2: Individual Posterior Network (Encoder)

Takes the player's actual game data and updates the archetype prior to produce an individual posterior.

```
Input:  y_t in R^16 (box-score stats for this game)
        c_t in R^12 (context features)
        z_prior in R^32 (from Level 1 merged prior)
        n_games in R^1 (games played so far -- controls shrinkage weight)

Concatenated input: R^(16 + 12 + 32 + 1) = R^61

Layers: Linear(61, 256) -> LayerNorm -> GELU -> Dropout(0.1)
        Linear(256, 256) -> LayerNorm -> GELU -> Dropout(0.1)
        Linear(256, 128) -> LayerNorm -> GELU

Output: mu_ind in R^32, log_sigma_ind in R^32
        via Linear(128, 32) for mu, Linear(128, 32) for log_sigma

Individual posterior: q(z | y, c, prior) = N(mu_ind, diag(sigma_ind^2))

Precision-weighted merge with Level 1 prior:
  prec_L1 = 1 / sigma_L1^2
  prec_ind = 1 / sigma_ind^2
  prec_final = prec_L1 + prec_ind
  mu_final = (mu_L1 * prec_L1 + mu_ind * prec_ind) / prec_final
  sigma_final = 1 / sqrt(prec_final)

Note: For a single game, sigma_ind will be large (high uncertainty from one game),
so the merged result stays close to the Level 1 prior. After many games, the
accumulated individual evidence drives sigma_ind down and the posterior diverges.
```

Parameter count: 61*256 + 256 + 256*256 + 256 + 256*128 + 128 + 128*32 + 32 + 128*32 + 32 = ~120K

#### Multi-Game Aggregation: Kalman-Style Sequential Update

Rather than re-running the encoder on all historical games, we maintain a running state.

```
State at time t-1:  mu_{t-1}, P_{t-1} = diag(sigma_{t-1}^2)

Prediction step (drift):
  F(age) = I + diag(f_1(age), ..., f_32(age))
  where f_j(age) is a learned aging drift per dimension:
    f_j(age) = MLP_aging(age, j)  -- small network producing per-dimension drift
    For age < 27: f_j ~ small positive (improvement)
    For age > 30: f_j ~ small negative (decline)
    Magnitude depends on dimension (physical dims decline faster than skill dims)

  mu_{t|t-1} = F(age) * mu_{t-1}
  P_{t|t-1} = F(age) * P_{t-1} * F(age)' + Q

  where Q = diag(q_1, ..., q_32) is the process noise
  q_j values are learned and stat-specific:
    - Physical/volume dims (scoring rate, minutes): low q (stable)
    - Efficiency dims (3PT%, TS%): medium q (moderate variance)
    - Defensive dims: high q (noisier signal)

Update step (new game observation):
  Run encoder on new game data -> get mu_obs, sigma_obs from Level 2 network
  This is the "observation" in Kalman terminology.

  K_t = P_{t|t-1} / (P_{t|t-1} + diag(sigma_obs^2))    [Kalman gain, element-wise for diagonal case]
  mu_t = mu_{t|t-1} + K_t * (mu_obs - mu_{t|t-1})
  P_t = (1 - K_t) * P_{t|t-1}                            [element-wise]

  This is O(d) per game (since we use diagonal covariance), extremely efficient.
```

**Aging MLP** (shared across all players):
```
Input:  age in R^1, dimension_index (one-hot or embedding) in R^8
Layers: Linear(9, 32) -> GELU -> Linear(32, 1) -> tanh * 0.01

Output: drift value f_j(age) in (-0.01, 0.01) per game step
```

Parameter count: ~1K

#### Decoder Network

Reconstructs team-agnostic statistics from the ability vector. This is the "D" in VAE.

```
Input:  z in R^32 (sampled or mean from posterior)
        c_t in R^12 (context features)
        Concatenated: R^44

Layers: Linear(44, 256) -> LayerNorm -> GELU
        Linear(256, 256) -> LayerNorm -> GELU

Output heads:
  1. Stat reconstruction head:
     Linear(256, 16)  -> reconstructed box-score stats y_hat
     (opponent/team-adjusted, not raw)

  2. RAPM prediction head:
     Linear(256, 2)   -> predicted O-RAPM, D-RAPM

  3. Next-game stat prediction head:
     Linear(256, 16)  -> predicted next-game box-score stats

  4. Archetype classification head (auxiliary):
     Linear(256, 10)  -> softmax -> archetype probabilities
```

Parameter count: 44*256 + 256 + 256*256 + 256 + 256*16 + 256*2 + 256*16 + 256*10 = ~90K

#### Total Model Size

```
Level 0 (Population prior):    ~7K params
Level 1 (Archetype prior):     ~100K params
Level 2 (Individual encoder):  ~120K params
Kalman components:             ~1K params
Decoder:                       ~90K params
Stat-specific Q, R matrices:   ~64 params (32 process noise + 32 observation noise)
Player identity embeddings:    ~2000 players * 32d = ~64K params (optional)
----------------------------------------------
Total:                         ~380K params (lightweight by design)
```

This is intentionally small. The player model should be a compact, efficient module, not a large transformer. The heavy lifting happens in Level 2 (graph network) and Level 3 (game simulator).

### 4.3 Architecture Diagram

```
                    TRAINING FLOW
                    =============

  Raw Box Score (y_t)          Context (c_t)         Physical Profile
        |                         |                       |
        v                         v                       v
  [Stat Normalizer]        [Context Encoder]       [Profile Encoder]
        |                    /    |    \                   |
        |                   /     |     \                  |
        |                  v      v      v                 |
        |          [Level 0: Population Prior]              |
        |                  |                               |
        |                  v                               |
        |          [Level 1: Archetype Prior] <------------|
        |                  |
        |                  v (precision-weighted merge)
        |                  |
        +-------> [Level 2: Individual Encoder]
                           |
                           v (precision-weighted merge)
                           |
                      z ~ q(z|y,c)    <---- [reparameterization trick]
                           |
                    +------+------+------+
                    |      |      |      |
                    v      v      v      v
                [Stat   [RAPM  [Next  [Arch
                Recon]  Pred]  Game]  Class]
                    |      |      |      |
                    v      v      v      v
                  L_rec  L_rapm L_pred L_arch   +   KL terms
                    \      |      /      /            |
                     \     |     /      /             |
                      v    v    v      v              v
                          Total Loss


                    INFERENCE FLOW (per game update)
                    ================================

  Game t box score (y_t) + context (c_t)
        |
        v
  [Level 2 Encoder] -> mu_obs, sigma_obs  (single-game posterior)
        |
        v
  [Kalman Update]:
    mu_t = mu_{t-1} + K_t * (mu_obs - mu_{t-1})
    where K_t = P_{t-1} / (P_{t-1} + sigma_obs^2)
        |
        v
  z_t = mu_t   (ability vector, 32d)
  P_t = updated uncertainty
        |
        v
  [Feed to Level 2 Graph Network as node features]
```

---

## 5. Training Procedure

### 5.1 Data Requirements

| Data Source | What We Use It For | Availability |
|-----------|-------------------|--------------|
| Box scores (PlayerBox) | Primary observations y_t | 2001-2026, all games (31,743 games in our DB) |
| Multi-year RAPM | Training target for team-agnostic supervision | Must compute ourselves from play-by-play, or use published EPM/DPM values |
| Player metadata | Physical profile, draft position, age | Basketball-Reference, our Players table |
| Play-by-play data | Computing RAPM if needed | 2001-2026 via pbpstats or NBA API |

**RAPM computation note**: Computing our own RAPM requires possession-level lineup data (which player was on court for each possession). This is available from pbpstats.com or can be derived from play-by-play data. We need ~5 million possessions for a reliable 20-year RAPM. Alternative: use published EPM/DPM values as proxy targets for a smaller initial training set.

**Pragmatic shortcut**: For initial development, we can use DPM (from DARKO) or EPM (from dunksandthrees.com) as proxy RAPM targets. Both are publicly available and daily-updated. We should validate that our model's team-agnostic ability vectors correlate highly with these reference metrics.

### 5.2 Loss Function

```
L_total = L_reconstruction + alpha * L_rapm + beta * L_next_game + gamma * L_archetype + delta * L_KL

Where:

L_reconstruction = MSE(y_hat, y_adjusted)
  y_adjusted = box-score stats adjusted for opponent and team context
  (subtract team-system effects, normalize for pace and opponent quality)
  Weight: 1.0

L_rapm = MSE(rapm_pred, rapm_target)
  rapm_target = multi-year RAPM or EPM/DPM proxy
  Split into O-RAPM and D-RAPM
  Weight: alpha = 2.0 (this is the most important loss -- drives team-agnostic learning)

L_next_game = MSE(next_game_pred, y_{t+1})
  Predicting next game's stats from current ability vector
  Tests whether the latent space captures predictive information
  Weight: beta = 0.5

L_archetype = CrossEntropy(arch_pred, arch_target)
  arch_target = cluster labels from pre-computed player archetype clustering
  Auxiliary loss to encourage archetype-structured latent space
  Weight: gamma = 0.1

L_KL = sum_{l=0}^{L} KL[q(z_l|...) || p(z_l|z_{l+1}, c)]
  Sum of KL divergences at each ladder level
  For Gaussians: KL[N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2)]
    = log(sigma_p/sigma_q) + (sigma_q^2 + (mu_q - mu_p)^2) / (2*sigma_p^2) - 0.5
  Weight: delta = 0.01 initially, warmed up to 1.0 over first 20 epochs (KL annealing)
```

### 5.3 Training Loop

**Phase 1: Pre-training the hierarchy (20 epochs)**

Train on ALL player-game observations to learn the population prior and archetype structure.

```
for epoch in 1..20:
  for batch in shuffle(all_player_games):
    y, c, rapm_target, physical_profile = batch

    # Forward pass through full hierarchy
    mu_pop, sig_pop = population_prior(c)
    pi_k, mu_arch, sig_arch = archetype_prior(sample(mu_pop, sig_pop), physical_profile)
    mu_ind, sig_ind = individual_encoder(y, c, merge(mu_pop, sig_pop, mu_arch, sig_arch))

    z = reparameterize(merge_all(mu_pop, sig_pop, mu_arch, sig_arch, mu_ind, sig_ind))

    y_hat, rapm_hat, next_hat, arch_hat = decoder(z, c)

    loss = compute_total_loss(...)
    loss.backward()
    optimizer.step()

  # KL annealing: delta = min(1.0, epoch / 20)
```

**Phase 2: Sequential training (30 epochs)**

Train on player career sequences to learn the Kalman update dynamics.

```
for epoch in 1..30:
  for player in shuffle(all_players):
    games = sorted_games_for_player(player)  # chronological

    # Initialize state from Level 0+1 prior
    mu, P = initialize_from_hierarchy(player.physical_profile)

    total_loss = 0
    for t, game in enumerate(games):
      y_t, c_t, rapm_t = game

      # Prediction step
      mu_pred = F(player.age_at_game(t)) @ mu
      P_pred = F @ P @ F.T + Q

      # Observation from encoder
      mu_obs, sig_obs = individual_encoder(y_t, c_t, mu_pred)

      # Kalman update
      K = P_pred / (P_pred + sig_obs**2)
      mu = mu_pred + K * (mu_obs - mu_pred)
      P = (1 - K) * P_pred

      # Decode and compute loss
      y_hat, rapm_hat, next_hat, arch_hat = decoder(mu, c_t)
      total_loss += compute_loss(y_hat, rapm_hat, next_hat, arch_hat, y_t, rapm_t)

    # Backprop through entire sequence (truncated BPTT every 20 games for memory)
    total_loss.backward()
    optimizer.step()
```

**Phase 3: Fine-tuning with Level 2 (joint training)**

Once Level 2 (graph network) exists, fine-tune the player encoder jointly.

```
for epoch in 1..10:
  for game in shuffle(all_games):
    home_players = get_player_vectors(game.home_roster)  # each is R^32
    away_players = get_player_vectors(game.away_roster)   # each is R^32

    # Level 2: Graph network processes player interactions
    game_prediction = graph_network(home_players, away_players)

    # Loss includes game outcome prediction
    loss = game_loss(game_prediction, game.actual_result) + player_model_losses
    loss.backward()  # gradients flow back through player vectors to encoder
```

### 5.4 Optimizer and Hyperparameters

```
Optimizer:        AdamW
Learning rate:    1e-3 (Phase 1), 3e-4 (Phase 2), 1e-4 (Phase 3)
Weight decay:     1e-4
Batch size:       256 player-games (Phase 1), 32 player careers (Phase 2)
Gradient clipping: max_norm = 1.0
Scheduler:        CosineAnnealingWarmRestarts (T_0=10, T_mult=2)
KL annealing:     Linear warmup from 0.01 to 1.0 over 20 epochs
Dropout:          0.1 in encoder
```

---

## 6. Cold-Start Behavior

### 6.1 Walk-Through: Rookie's First 100 Games

**Draft Day (0 games played)**:

```
Available information:
  - Draft position (e.g., #3 overall)
  - Physical profile (6'5", 200 lbs, 6'9" wingspan)
  - Age (19.5 years)
  - College stats (if applicable, else international/G-League)

Initialization:
  1. Population prior: mu_pop = population_prior(context)
     -> Generic NBA player baseline vector (32d)
     -> sigma_pop is large everywhere (high uncertainty)

  2. Archetype prior: soft assignment based on physical profile
     -> 6'5" with long wingspan -> 55% wing, 30% guard, 15% combo
     -> mu_arch = weighted mixture of archetype means
     -> sigma_arch still fairly large but smaller than sigma_pop

  3. Draft position adjustment:
     -> #3 pick -> scale mu_arch by 1.2x on overall ability dimensions
     -> (Draft position provides strong signal about expected ability level)
     -> Encoded as a feature in the physical profile input

  4. Initial ability vector:
     z_0 = precision_weighted_merge(mu_pop, sigma_pop, mu_arch, sigma_arch)
     P_0 = merged covariance (still large -- high uncertainty)

  Result: A 32d vector close to the archetype-weighted population mean,
  with high uncertainty in all dimensions. This player "looks like" a
  typical wing-type lottery pick.
```

**Games 1-5 (Summer League / Preseason)**:

```
Each game provides one observation (y_t).
Kalman gain K is small because P_{t|t-1} is large relative to sigma_obs.

After 5 games:
  - Minutes dimension: partially updated (stabilizes fast, ~10 games)
  - Usage dimension: starting to diverge from prior
  - Shooting dimensions: still dominated by prior (need 100+ games)
  - Overall ability: slight update from prior, mostly still archetype-driven

  Uncertainty P has decreased slightly but remains large.
  The player vector is ~80% archetype prior, ~20% observed data.
```

**Games 10-20 (First month of regular season)**:

```
After 15 games:
  - Minutes and usage dimensions: ~50% data-driven, 50% prior
  - Scoring volume: emerging pattern (is this a 15 PPG or 8 PPG player?)
  - Assist/turnover patterns: starting to stabilize
  - Shooting: still heavily prior-dependent (need more sample)

  The player vector has noticeably diverged from the generic archetype.
  We can now distinguish "high-usage wing" from "3-and-D wing" with
  moderate confidence. P has decreased by ~40% in fast-stabilizing dims.

  The archetype soft-assignment may have shifted:
  initially 55% wing, now 45% wing / 35% guard / 20% combo
  as the data reveals playmaking tendencies.
```

**Games 40-60 (Mid-season)**:

```
After 50 games:
  - Most dimensions are now data-driven (70-80% data, 20-30% prior)
  - Shooting percentages starting to stabilize
  - Defensive dimensions still noisy but trending
  - The ability vector looks like a real, individual player, not an archetype

  P has decreased substantially. The model is fairly confident in the
  player's overall profile. The vector could now meaningfully contribute
  to Level 2 game predictions without heavy regression to the mean.
```

**Games 80-100 (End of first season)**:

```
After 82+ games:
  - All dimensions except the noisiest (defensive impact, 3PT%) are
    largely data-driven
  - The player vector is a mature, individualized representation
  - P is small in most dimensions

  This player now looks like a veteran in terms of representation quality.
  The remaining uncertainty is in dimensions that are inherently hard to
  measure (defensive impact, spacing gravity, etc.).

  The aging model F(age=20) applies small positive drift to most dimensions,
  reflecting expected sophomore-year improvement.
```

### 6.2 Uncertainty Reduction Rate

The key insight from the Kalman framework is that uncertainty reduction is **stat-specific and automatic**. The per-dimension Kalman gain K_j controls how fast dimension j of the ability vector incorporates new data:

```
K_j = P_j / (P_j + R_j)

Where:
  P_j = current uncertainty in dimension j of ability vector
  R_j = observation noise for dimension j (learned, stat-specific)

High R_j (noisy stat like 3PT%):
  K_j is small -> slow updates, prior dominates longer
  After N games: P_j ~ R_j / N (roughly)
  Stabilizes in ~242 games (matching empirical 3PT% stabilization)

Low R_j (stable stat like minutes):
  K_j is large -> fast updates, data dominates quickly
  After N games: P_j ~ R_j / N
  Stabilizes in ~10 games (matching empirical minutes stabilization)
```

This naturally reproduces the empirical stabilization rates from Medvedovsky (2020) without hardcoding them -- the model learns R_j during training.

### 6.3 Comparison: Paradigm-Specific Cold-Start

| Scenario | VaKE (Hybrid) | Pure VaRLAE | Pure Kalman | Pure EPM |
|----------|---------------|-------------|-------------|----------|
| Draft day | Archetype-weighted prior from physical profile + draft position | Shared prior only (no physical profile input in original) | League-average padding (position-agnostic) | No estimate possible |
| 5 games | 80% prior, 20% data; uncertainty high | Similar but no Kalman structure for efficient sequential updates | 98% prior, 2% data (padding dominates) | SPM estimate from 5 games (very noisy) |
| 20 games | 50% prior, 50% data; archetype assignment updating | Similar | 90% prior for slow stats, 50% for fast stats | SPM from 20 games (moderate noise) |
| 50 games | 20-30% prior, 70-80% data; individual profile clear | Similar | 40-80% prior depending on stat | SPM reliable; RAPM still useless |
| 82 games | 10-20% prior, 80-90% data; mature vector | Similar | 10-50% prior depending on stat | SPM good; RAPM marginally useful |

---

## 7. Team-Agnostic Mechanism

### 7.1 The Problem

Raw box-score stats are severely confounded by team context:

- A player on a fast-paced team (e.g., 2018 Rockets) gets more possessions per game, inflating counting stats
- A player on a pass-heavy team (e.g., 2014 Spurs) may have higher assist rates but lower usage
- A player sharing the court with great shooters benefits from defensive attention being spread
- A player on a bad team may have inflated stats from garbage time

If we train the encoder on raw stats, the ability vector will encode team-system artifacts, not intrinsic player ability.

### 7.2 Multi-Layered Team-Agnostic Approach

We use four complementary mechanisms:

#### Mechanism 1: Input Normalization (Pre-Encoder)

Before feeding stats to the encoder, adjust for known team-context confounds:

```
y_adjusted[i] = y_raw[i] for i in stat_indices

Adjustments:
  1. Pace adjustment: divide counting stats by team_possessions / league_avg_possessions
     (removes pace inflation/deflation)

  2. Minutes normalization: convert counting stats to per-36-minute rates
     (removes playing time effects)

  3. Opponent adjustment: multiply by league_avg_opp_stat / actual_opp_stat
     (e.g., if facing top-5 defense, adjust scoring up)

  4. Usage context: include teammates' usage rates as context features
     (lets the model learn that 20 PPG with Steph Curry is different from 20 PPG as the sole option)
```

#### Mechanism 2: RAPM Training Target (Primary)

The most powerful mechanism: train the RAPM prediction head with heavy weight (alpha = 2.0).

```
RAPM is inherently team-agnostic because it solves:
  y = X * beta + epsilon
where X encodes ALL 10 players on the court simultaneously.

By training our encoder to predict RAPM from box-score inputs, we force the latent space to capture
the team-agnostic component of the box scores. The encoder must learn to "undo" the team-context
confounds in order to accurately predict RAPM.

This is the same principle behind EPM's success: SPM models trained to predict RAPM
learn to extract the team-agnostic signal from box scores.
```

#### Mechanism 3: Transfer/Trade Consistency Loss (Auxiliary)

Players who change teams provide natural experiments for team-agnostic learning.

```
L_transfer = ||z_before_trade - z_after_trade||^2 * trade_weight

For each player who changed teams:
  z_before = ability vector from last 20 games before trade
  z_after  = ability vector from first 20 games after trade
  trade_weight = 1.0 (or higher for mid-season trades)

Rationale: A player's intrinsic ability doesn't change when they change teams.
Their box-score stats will change (different system, pace, teammates), but the
ability vector should remain similar. This loss directly penalizes the encoder
for learning team-system artifacts.

We have ~200-300 trades per season across ~25 seasons = ~5,000-7,500 natural experiments.
```

#### Mechanism 4: Teammate Context Conditioning

Provide teammate information as context features, not as part of the ability vector.

```
Context features c_t include:
  - Team offensive rating (season-to-date)
  - Team pace
  - Minutes share (player's minutes / team total)
  - Usage share (player's usage / team total)
  - Teammate average quality (mean EPM/DPM of teammates)

By making these context features (inputs to the prior, not the ability vector),
the encoder can condition on team context when interpreting stats but the
ability vector z itself is context-independent.

The decoder then reconstructs stats CONDITIONED on context, meaning:
  z captures intrinsic ability
  c captures the environment
  decoder(z, c) produces environment-specific stat predictions
```

### 7.3 Validating Team-Agnostic Quality

We validate that the learned ability vectors are truly team-agnostic using:

1. **Trade test**: For players who changed teams, compute the cosine similarity of their ability vectors before and after the trade. Target: >0.85 similarity.

2. **RAPM correlation**: Compute Pearson correlation between the RAPM prediction head output and actual multi-year RAPM. Target: r > 0.70.

3. **Cross-team prediction**: Train a simple model to predict which team a player is on from their ability vector. If the ability vector is truly team-agnostic, this should be near chance (1/30 = 3.3%).

4. **Pace independence**: Regress ability vector dimensions on team pace. No dimension should have |r| > 0.15 with pace after training.

---

## Appendix A: Implementation Sequence

```
Step 1: Data preparation
  - Extract all PlayerBox data from our SQLite DB (31,743 games, 2001-2026)
  - Compute per-game player stat vectors (16 stats)
  - Compute context features (12 dims)
  - Acquire or compute RAPM targets
    Option A: Compute our own from play-by-play (most principled, most work)
    Option B: Use published EPM/DPM as proxy (fastest path)
  - Pre-compute player archetypes via k-means on career stat profiles (K=10)

Step 2: Build VaKE model
  - Implement population prior network
  - Implement archetype prior network
  - Implement individual encoder
  - Implement decoder with 4 heads
  - Implement precision-weighted merging
  - Implement KL computation for ladder hierarchy
  - Test on synthetic data

Step 3: Phase 1 training (hierarchy pre-training)
  - Train on all player-game observations
  - Validate: do archetype clusters make basketball sense?
  - Validate: does the population prior produce reasonable "average player" vectors?

Step 4: Implement Kalman update module
  - Implement state transition F(age) with learned aging MLP
  - Implement diagonal process noise Q (learned)
  - Implement sequential update loop
  - Test: feed a player's career game-by-game, check that vector evolves sensibly

Step 5: Phase 2 training (sequential)
  - Train on player career sequences
  - Validate: do career trajectories show expected patterns?
  - Validate: does uncertainty decrease with more games?
  - Validate: does rookie behavior match Section 6 description?

Step 6: Integration tests
  - Trade test: vectors stable across team changes?
  - RAPM correlation: r > 0.70?
  - Pace independence: no dimension correlated with pace?
  - Archetype recovery: do learned archetypes match basketball intuition?

Step 7: Connect to Level 2
  - Use ability vectors as GNN node features
  - Fine-tune jointly (Phase 3 training)
```

## Appendix B: Key Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Posterior collapse (KL term drives all z's to prior) | High | KL annealing, free bits (minimum KL per dimension), ladder structure (proven to help) |
| RAPM targets are noisy for low-minute players | Medium | Weight L_rapm by player minutes; use multi-year RAPM for target smoothing |
| Model memorizes player identity from stat patterns rather than learning intrinsic ability | Medium | Transfer consistency loss; dropout; validate with trade test |
| Aging MLP overfits to career-end data (survivorship bias -- only good players have long careers) | Medium | Include all players (not just long careers); regularize aging MLP heavily |
| Diagonal covariance is too restrictive (ability dimensions are correlated) | Low | Start diagonal for efficiency; upgrade to low-rank (e.g., rank-4) covariance if diagonal proves insufficient |
| 32 dimensions is too many/few | Low | Run ablations at 16, 24, 32, 48; evaluate downstream Level 2 performance |

## Appendix C: Comparison with Existing Phase 3/4 Architecture

Our current Phase 3 system (best: Exp 9 ensemble, AUC 0.718) uses a completely different player representation: per-game box-score stats are encoded via a fixed `PlayerContributionEncoder` with attention pooling. There is no persistent player state across games -- each game is encoded independently.

VaKE would replace this with a persistent, learned ability vector that:
- Carries information from all past games (not just recent window)
- Has uncertainty quantification (we know what we don't know)
- Is team-agnostic (current system uses raw stats)
- Has cold-start handling (current system fails for new players with short history)
- Is differentiable end-to-end (enables joint optimization with downstream models)

The expected benefit is primarily in the early-season period (first 20 games) and for traded players, where the current fixed-window approach has the least data.

---

## References

### VaRLAE and Hierarchical VAEs
- Liu, G., Schulte, O., Poupart, P., Rudd, M., Javan, M. (2020). "Learning Agent Representations for Ice Hockey." NeurIPS 2020. [Paper](https://proceedings.neurips.cc/paper/2020/hash/d90e5b6628b4291225cba0bdc643c295-Abstract.html)
- Sonderby, C.K. et al. (2016). "Ladder Variational Autoencoders." NeurIPS 2016. [arXiv](https://arxiv.org/abs/1602.02282)
- Zhao, S. et al. (2017). "Learning Hierarchical Features from Generative Models." ICML 2017. [Blog](https://ermongroup.github.io/blog/hierarchy/)
- Havtorn, J.D. et al. (2021). "Hierarchical VAEs Know What They Don't Know." ICML 2021. [Paper](http://proceedings.mlr.press/v139/havtorn21a/havtorn21a.pdf)

### Kalman Filters and DARKO
- Medvedovsky, K. (2020). "NBA Stabilization Rates and the Padding Approach." [Blog](https://kmedved.com/2020/08/06/nba-stabilization-rates-and-the-padding-approach/)
- DARKO DPM documentation. [NBAstuffer](https://www.nbastuffer.com/analytics101/darko-daily-plus-minus/)
- DraftKings Engineering. "Kalman Filters for NBA Player Ratings." [Blog](https://careers.draftkings.com/life-at-draftkings/engineering/how-we-use-kalman-filters-for-nba-player-ratings/)
- Glickman, M.E. (1999). "Parameter estimation in large dynamic paired comparison experiments." *Applied Statistics*.
- Kain, K. & Logan, T. (2014). "A State-Space Perspective on Modelling and Inference for Online Skill Rating." [arXiv](https://arxiv.org/abs/2308.02414)

### EPM and RAPM
- Snarr, T. "About Estimated Plus-Minus (EPM)." [Dunks & Threes](https://dunksandthrees.com/about/epm)
- Engelmann, J. "NBA Adjusted Plus-Minus: How to Build It." [Substack](https://jeremiasengelmann.substack.com/p/nba-adjusted-plus-minus-how-to-build)
- Sill, J. (2010). "Improved NBA Adjusted +/- Using Regularization and Out-of-Sample Testing." MIT Sloan.
- Miyakawa, E. "Bayesian Performance Rating." [EvanMiya](https://blog.evanmiya.com/p/bayesian-performance-rating)

### Bayesian Hierarchical Models for Basketball
- Vaci, N. et al. (2019). "Large data and Bayesian modeling -- aging curves of NBA players." *Behavior Research Methods*. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6690864/)
- Foo, K. (2020). "Bayesian Hierarchical Modelling of NBA 3 Point Shooting." [Medium](https://medium.com/analytics-vidhya/bayesian-hierarchical-modelling-of-nba-3-point-shooting-in-2018-19-season-f96ba77bcaab)
- Elmore et al. (2024). "Expected Points Above Average: A Novel NBA Player Metric Based on Bayesian Hierarchical Modeling." [arXiv](https://arxiv.org/html/2405.10453v1)
- Efron, B. & Morris, C. (1975). "Data Analysis Using Stein's Estimator and Its Generalizations." *JASA*.

### Player Embeddings and Representations
- Alcorn, M.A. & Nguyen, A. (2023). "NBA2Vec: Dense Feature Representations of NBA Players." [arXiv](https://arxiv.org/abs/2302.13386)
- Luo, T. & Krishnamurthy, A. (2023). "Who You Play Affects How You Play." [arXiv](https://arxiv.org/abs/2303.16741)

### State-Space Models for Rating
- Poropudas, J. (2011). "Kalman Filter Algorithm for Rating and Prediction in Basketball." [hamahakkimies](https://www.hamahakkimies.com/project/kalman-rating)
- Kain & Logan (2023). "A State-Space Perspective on Modelling and Inference for Online Skill Rating." *JRSS-C*. [Paper](https://academic.oup.com/jrsssc/article/73/5/1262/7734616)
