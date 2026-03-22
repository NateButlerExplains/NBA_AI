# Level 1: Individual Player Models -- Academic Literature Review

**Purpose**: Inform the design of per-player "ability vector" models for the NBA_AI prediction system.
**Date**: 2026-03-16

---

## Table of Contents

1. [Regularized Adjusted Plus-Minus (RAPM)](#1-regularized-adjusted-plus-minus-rapm)
2. [Modern Player Metrics (EPM, RAPTOR, LEBRON, DARKO)](#2-modern-player-metrics)
3. [Player Trajectory / Aging Curve Modeling](#3-player-trajectory--aging-curve-modeling)
4. [Latent Player Representations](#4-latent-player-representations)
5. [Bayesian Hierarchical Models for Players](#5-bayesian-hierarchical-models-for-players)
6. [Transfer Learning / Few-Shot Learning for Player Modeling](#6-transfer-learning--few-shot-learning-for-player-modeling)
7. [Synthesis: Design Implications for Our System](#7-synthesis-design-implications-for-our-system)

---

## 1. Regularized Adjusted Plus-Minus (RAPM)

RAPM is the gold standard for team-agnostic player evaluation. It measures each player's per-possession impact on point differential while controlling for every other player on the court.

### 1.1 Historical Development

| Year | Contribution | Author(s) |
|------|-------------|-----------|
| 2004 | Introduced Adjusted Plus-Minus (APM) to basketball | Dan Rosenbaum |
| 2010 | Regularized APM via ridge regression (Sloan best paper) | Joe Sill |
| 2011+ | Prior-informed xRAPM (basis for ESPN RPM) | Jeremias Engelmann |
| 2014 | ESPN Real Plus-Minus (RPM) | Engelmann & Steve Ilardi |
| 2026 | Lineup-RAPM (L-RAPM) with informed priors per lineup | arXiv:2601.15000 |

### 1.2 Mathematical Formulation

**Data structure**: Each row is one possession (or stint). There are ~230,000 possessions per NBA season, ~5 million over 20 years.

**Design matrix** X:
- Rows = possessions
- Columns = 2 x (number of players): one offensive variable and one defensive variable per player
- X_ij = 1 if player j is on offense in possession i, 1 if on defense, 0 otherwise

**Target vector** y:
- y_i = points scored on possession i (typically 0, 1, 2, or 3)

**OLS formulation (APM)**:

```
y = X*beta + epsilon
beta_hat = (X'X)^{-1} X'y
```

Problem: X'X is nearly singular due to multicollinearity (players who always play together).

**Ridge regression formulation (RAPM)**:

```
minimize: ||y - X*beta||^2 + lambda * ||beta||^2

Solution: beta_hat = (X'X + lambda*I)^{-1} X'y
```

**Bayesian interpretation**: Ridge regression is equivalent to placing a zero-mean Gaussian prior on each coefficient:

```
beta_j ~ N(0, sigma^2 / lambda)
```

This shrinks all player ratings toward zero (league average), with the strength controlled by lambda.

**Prior-informed RAPM (xRAPM)**: Instead of shrinking toward zero, shrink toward a box-score-derived prior mu_0:

```
beta_hat = (X'X + lambda*I)^{-1} (X'y + lambda * mu_0)
```

This is the key innovation behind RPM, EPM, and other modern metrics.

### 1.3 Lambda Selection

Sill (2010) found optimal lambda in the range 2,000-3,000 via K-fold cross-validation. Engelmann confirmed similar values. The lambda differs for offensive vs. defensive coefficients -- offense benefits from lower regularization because players exert more individual control over offensive outcomes.

### 1.4 Practical Implementation Notes (from Engelmann)

- Use sparse matrices (only ~10 of ~500 players on court per possession)
- Do NOT normalize the X matrix (no z-scoring)
- Include home-court advantage and "rubber-band effect" (score differential) as controls
- Separate lambda for offense and defense improves accuracy
- Standard errors: `variance = SSE * (X'X + lambda*I)^{-1} / (n - p - 1)`

### 1.5 Stabilization

RAPM is notoriously noisy for single seasons:

- **Single-season RAPM**: High noise, year-to-year MSE ~1.4, MAE ~0.85
- **3-year RAPM**: Much more stable
- **5-year RAPM**: Often cited as yielding reasonable results
- **Plus/minus data**: Should be regressed by ~1,000+ league-average possessions
- Prior-informed variants (xRAPM, EPM) converge much faster because box-score priors are available immediately

### 1.6 Key Takeaway for Our System

RAPM provides the mathematical foundation for separating player ability from team context. The ridge regression framework with informative priors maps directly to our hierarchical pre-training concept: the prior is the "generic player" model, and the RAPM update is the individualization. However, RAPM produces a scalar (or 2-scalar O/D split), not the multi-dimensional ability vector we need. We should use RAPM-like thinking as a training signal, not as the representation itself.

---

## 2. Modern Player Metrics

### 2.1 EPM (Estimated Plus-Minus) -- Taylor Snarr, Dunks & Threes

**Architecture**: Two-stage model combining Statistical Plus-Minus (SPM) with RAPM.

**Stage 1 -- Skills**: Individually optimized per-stat estimates of true talent. Each stat has its own:
- Exponential decay factor (beta^t, where t = days ago)
- Stabilization rate (optimized via differential evolution)
- Adjustments for age, league trends, home-court, back-to-back, opponent strength

**Stage 2 -- Prior-informed RAPM**: SPM predictions serve as Bayesian priors in exponentially-decayed RAPM across 4,700+ cumulative RAPM models from 2002 to present. Each model uses all possessions since 2002, weighting recent data more heavily.

**Data**: Box score, play-by-play, and player tracking (from 2013-14 onward). Uses entire career history, not single seasons.

**Prediction accuracy** (game-level margin RMSE):
- Traditional SRS-style: 12.25
- EPM weighted by predicted minutes: 12.14
- EPM + inactivity: 12.10

**Metric comparison** (team-level retrodiction RMSE, 2013-2019):
| Metric | RMSE |
|--------|------|
| **EPM** | **2.48** |
| RPM | 2.60 |
| RAPTOR | 2.63 |
| BPM | 2.71 |
| RAPM | 2.80 |
| PER | 3.20 |

EPM and RPM (both using RAPM + Bayesian prior) consistently outperform all others. EPM leads overall.

**Key insight**: The "skills" concept -- individually optimized per-stat projections with stat-specific decay rates -- is directly relevant to our ability vector design. Different stats stabilize at very different rates.

### 2.2 RAPTOR -- FiveThirtyEight (Nate Silver et al., 2019)

**Architecture**: Blend of two components:
1. **Box component**: Individual stats including player tracking and play-by-play derivatives
2. **On-off component**: Team performance with player on/off court (RAPM-like)

**Key features**:
- Uses player tracking data (available from 2013-14)
- Descriptive RAPTOR: Pure on-court performance
- PREDATOR: Predictive variant incorporating age, height, draft position, All-NBA status
- Per-100-possession scale

**Limitations acknowledged by creators**:
- Assumes player performance is "largely linear and additive"
- Does not account for coaching, systems, or synergies between teammates

**Prediction context**: In team-level retrodiction, RAPTOR achieved 2.63 RMSE. FiveThirtyEight blends RAPTOR (65%) with Elo (35%) for team-level game predictions.

### 2.3 LEBRON -- Basketball Index

**Full name**: Luck-adjusted player Estimate using a Box prior Regularized ON-OFF

**Architecture**: Combines luck-adjusted RAPM with box-score stabilization:
- Box prior uses boxPIPM weightings stabilized via offensive archetypes (player role clustering)
- On-off component uses actual RAPM calculations (not estimates)
- Luck adjustment removes 3PT% variance, FT% variance from on-off data

**Scale**: Per 100 possessions, 0 = average, -2.7 = replacement level

**Distinguishing feature**: Only public impact metric using the full combination of (a) actual RAPM (not estimated), (b) role-adjusted stabilization, and (c) luck adjustment.

### 2.4 DARKO (DPM) -- Kostya Medvedovsky

**Full name**: Daily Adjusted and Regressed Kalman Optimized projections

**Architecture**: Composite predictive metric using:
1. Box score stats with exponential decay: weight = beta^t (beta in [0,1], t = games ago)
2. Plus-minus stats blended proportionally to total possessions played
3. Kalman filter for noise reduction and sequential updating

**Key differentiators**:
- Built from the ground up as a **projection system** (forward-looking, not retrospective)
- Updates daily for every player, every box-score stat
- Attempts to remove noise and luck from observations
- Most trusted all-in-one metric among NBA executives

**Prediction performance**: DPM beats all other public metrics in RMSE for game-level prediction. Rankings: DPM > EPM > LEBRON.

### 2.5 ESPN Net Points (2025) -- Dean Oliver & Jeremias Engelmann

**Architecture**: Play-by-play credit assignment system.
- Divides credit/blame for every shot, rebound, turnover, foul across all 10 players
- Accounts for shot creation (self-created vs. assisted)
- Weights by difficulty of each action
- Single-game granularity (unlike RAPM which needs large samples)

**Key insight for us**: Net Points demonstrates that play-by-play data can assign credit at the individual possession level, not just aggregated over seasons. This per-event credit assignment aligns with our goal of updating ability estimates after every game.

### 2.6 BPM (Box Plus-Minus) -- Daniel Myers, Basketball Reference

**Architecture**: Pure box-score regression trained against 14-year RAPM from Engelmann.
- Uses position-weighted box-score stats (a block by a guard is worth more than by a center)
- Features: USG%, TS%, ORB%, DRB%, STL%, BLK%, AST%, plus interaction terms
- No play-by-play or tracking data required

**Relevance**: BPM serves as the standard box-score prior for RAPM-based metrics. Its coefficients tell us which box-score stats are most predictive of true impact -- useful for designing our ability vector features.

### 2.7 Bayesian Box APM (Intraocular)

**Architecture**: A fully Bayesian model that learns box-score-to-impact weights simultaneously with RAPM:

```
Likelihood:   Y ~ N(X*beta, sigma)
Prior:        beta ~ N(sum_j(alpha_j * Z_j), tau)
Hyperprior:   alpha_j ~ N(0, xi)
```

Two levels of regularization:
- **tau**: Weight of box-score prior vs. plus-minus likelihood
- **xi**: Regularization of box-score coefficient weights themselves

Offensive and defensive coefficients learned separately, with stronger regularization on defense (defensive box stats are weaker predictors, R^2 ~59% vs offense ~66%).

**Key insight**: This hierarchical Bayesian approach to learning the prior weights is exactly the kind of framework we should consider for our ability vector model.

---

## 3. Player Trajectory / Aging Curve Modeling

### 3.1 Bayesian Latent Cognitive Variable Model (B-Ianus)

**Paper**: Vaci et al. (2019), "Large data and Bayesian modeling -- aging curves of NBA players," *Behavior Research Methods*.

**Data**: 50 years, 2,845 players, 400 analyzed in detail.

**Methodology**:
- Bayesian latent variable model with three hierarchical levels
- Separates pre-peak development and post-peak decline into distinct latent factors (phi_1 and phi_2)
- Tests linear, exponential, logistic, and power-law functions for each phase

**Mathematical formulation**:
```
Pre-peak (before ~27):  Performance_i = alpha * exp(beta_p * age_i)
Post-peak (after ~27):  Performance_i = alpha * age_i^{beta_p}
```

**Key findings**:
- **Peak age**: ~27 years across WS, VORP, and PER
- **Post-peak decline**: Follows a **power-law** function (rapid initial decline that slows with age)
- **Pre-peak growth**: Exponential
- **Critical correlation**: Players with steeper development curves show **shallower** declines (knowledge accumulation preserves performance despite physical decline)
- **Minutes effect**: More minutes correlate with both better development and slower decline (but causality ambiguous)
- **Individual variability**: Enormous -- some players show minimal decline

### 3.2 LSTM + Autoencoder Career Trend Prediction

**Paper**: arXiv:2509.25858 (2025), "Aging Decline in Basketball Career Trend Prediction Based on Machine Learning and LSTM Model"

**Methodology**:
1. Structure career data as 7-year developmental sequences (ages 22-28) with 48 features
2. Autoencoder compresses sequences to latent embeddings
3. K-means clustering on embeddings identifies career trajectory types
4. LSTM predicts 3-year target sequences (ages 29-31) conditioned on trajectory type

**Results**: 22.83% reduction in MAE and 189.47% improvement in R^2 vs. standard LSTM by incorporating cluster assignments. Reveals distinct trajectory patterns between star players and regular players.

**Key insight**: Career trajectory typing via clustering before prediction is powerful. Our hierarchical model should learn these trajectory archetypes.

### 3.3 Kolmogorov-Arnold Networks (KAN) for Nonlinear Age Effects

**Paper**: Frontiers in Sports and Active Living (2025), "Nonlinear age effects on basketball player performance: insights from Kolmogorov-Arnold Networks"

**Data**: 2,786 NBA player-season samples (2019-2024)

**Approach**: Interpretable ML framework with age-group-specific feature analysis, using KAN architecture for nonlinear modeling of age effects on performance.

### 3.4 CARMELO Projection System (FiveThirtyEight)

**Full name**: Career-Arc Regression Model Estimator with Local Optimization

**Methodology**: Similarity-based k-nearest-neighbors approach:
- Compares current players to historical players with similar statistical profiles at the same age
- Key features: age (most important), height, weight, draft position, RAPTOR ratings
- Produces probabilistic forecasts (80% confidence intervals)
- Explicitly models that players improve through ~27 then decline

**Key insight for cold start**: CARMELO is essentially a few-shot / transfer learning system -- it projects new/young players by finding similar historical players. This is the same principle we want for rookies.

### 3.5 Stat-Specific Stabilization Rates (Medvedovsky, 2020)

Using the "padding approach" with differential evolution over 750,000+ player-game observations (2001-present):

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

**Projection formula**:
```
Expected Metric = (Actual * n + League_Avg * X) / (n + X)
```
where X = padding number, n = sample size (games).

**Key insight**: Different ability dimensions stabilize at radically different rates. Our model must represent this -- some dimensions of the ability vector should update quickly (usage patterns), others slowly (shooting efficiency), and the model should know which is which.

---

## 4. Latent Player Representations

### 4.1 NBA2Vec (2023)

**Paper**: arXiv:2302.13386, "NBA2Vec: Dense Feature Representations of NBA Players"

**Architecture**:
- Inspired by Word2Vec (analogy: players are "words", possessions are "sentences")
- Input: 10 players on court (5 offense, 5 defense) for each play
- **8-dimensional shared player embedding** per player
- 5 offensive embeddings averaged, 5 defensive embeddings averaged, concatenated
- Hidden layer (128 units, ReLU)
- Output: Distribution over 23 play outcome types

**Training**: 3.5 million plays, 1,551 distinct players. K-L divergence of 0.3 vs. empirical play distribution.

**Key findings**:
- Embedding space naturally separates by position (centers vs. guards)
- Forwards scatter across both clusters (reflects positional flexibility)
- Successfully predicts 2017 NBA Playoffs series outcomes

**Key insight**: Only 8 dimensions suffice to capture meaningful player differentiation when learned from play outcomes. This suggests our ability vector may not need to be very high-dimensional.

### 4.2 Player2Vec (Perez, 2021, Santa Clara University)

**Methodology**: PCA and autoencoder dimensionality reduction on player statistics, with embeddings used to predict net rating of 5-player lineups.

**Relevance**: Demonstrates that lineup-level prediction from individual player embeddings is feasible.

### 4.3 VaRLAE -- Learning Agent Representations for Ice Hockey (NeurIPS 2020)

**Paper**: Liu, Schulte, Poupart, Rudd, Javan. NeurIPS 2020.

**Architecture**: Variational Recurrent Ladder Agent Encoder (VaRLAE)
- **Variational encoder** embeds player information with hierarchy of latent variables
- **Context-specific shared prior** induces shrinkage for sparse players (directly addresses cold-start)
- Hierarchy of latent variables prevents posterior collapse

**Data**: 4.5M+ ice hockey events, 1,000+ players (including many bench players with sparse data)

**Key contributions**:
1. Shared prior across all players (population-level) that gets specialized per individual
2. Hierarchy of latent variables at different abstraction levels
3. Handles data-sparse players via shrinkage toward the shared prior

**Results**: State-of-the-art on (a) player identification, (b) expected goals estimation, (c) score prediction.

**Key insight for our system**: VaRLAE is the closest existing work to our proposed architecture. The "context-specific shared prior with shrinkage" is essentially our "generic NBA player -> position -> individual" hierarchy. The Bayesian shrinkage naturally handles cold-start.

### 4.4 PCA on NBA Box Scores

Multiple analyses converge on similar findings:
- **PC1** (~39% variance): Offensive volume/production
- **PC2** (~28% variance): Interior vs. perimeter orientation
- **4 components**: Capture ~68% of variance (with tracking data)
- **5 components**: Capture ~80% of variance

Typical interpretable dimensions:
1. Scoring volume
2. Interior dominance (rebounds, blocks, FG%)
3. Perimeter skills (3PT, assists)
4. Defensive activity (steals, blocks, deflections)
5. Efficiency (TS%, TOV%)

### 4.5 Player Clustering / Archetype Studies

Data-driven clustering consistently finds that traditional 5-position labels are outdated. Typical findings:

- **5 clusters**: Rough position analogs but data-driven boundaries
- **9 clusters**: Captures archetypes like "Aggressive Scorer", "Floor General", "High-Efficiency Scorer", "Low-Efficiency Defender"
- **13 clusters**: Fine-grained role differentiation

**Paper**: Duman, Sennaroglu, Tuzkaya (2024), "A cluster analysis of basketball players for each of the five traditionally defined positions," *Journal of Sports Engineering and Technology*.

**Key insight**: 8-12 archetype clusters appear to be the natural dimensionality of player "types" in the NBA. This informs our embedding dimensionality choices.

### 4.6 GATv2-TCN: Player Performance as Graph Problem

**Paper**: Luo & Krishnamurthy (2023), "Who You Play Affects How You Play," arXiv:2303.16741

**Architecture**: Graph Attention Network v2 + Temporal Convolution Network
- Dynamic player interaction graph where edges represent co-play relationships
- GATv2 captures attention weights between players (who influences whom)
- TCN handles temporal sequences of per-game statistics

**Key insight**: Player performance is not independent -- it depends on who they play with and against. Our "ability vector" must represent intrinsic ability that is then modulated by team context, not the raw observed performance.

---

## 5. Bayesian Hierarchical Models for Players

### 5.1 Theoretical Foundation: Partial Pooling and James-Stein

The mathematical foundation for our hierarchical approach comes from James & Stein (1961) and the subsequent Bayesian hierarchical modeling literature:

**Key result** (James-Stein): Shrinking individual estimates toward a group mean always reduces total estimation error (in 3+ dimensions). This is the mathematical justification for hierarchical player models.

**Partial pooling** in hierarchical Bayesian models:
```
Population level:  mu ~ prior
Group level:       theta_g ~ N(mu, sigma_group)
Individual level:  y_ij ~ N(theta_g[j], sigma_obs)
```

Each player's estimate is "pulled" toward the group mean in proportion to their data scarcity. Players with lots of data get estimates close to their raw performance; players with little data get estimates close to the group mean.

**Classic sports example**: Efron & Morris (1975) showed this with batting averages -- predictions from 45 at-bats were dramatically improved by shrinking toward the grand mean.

### 5.2 EPAA: Expected Points Above Average (2024)

**Paper**: Elmore, Williams, Schliep et al. (2024), arXiv:2405.10453, published in *Annals of Applied Statistics* (2025).

**Hierarchical structure**:
```
Shot selection:  N_i^1,...,N_i^K | w_i  ~ Multinomial(N_i, p_{w_i})
Shot accuracy:   M_i^k | z_i           ~ Binomial(N_i^k, q_{z_i}^k)
Selection prior: p_w ~ Dirichlet(alpha,...,alpha)
Accuracy prior:  q_z^k ~ Beta(1, 1)
Cluster membership: pi ~ Dirichlet(beta,...,beta)
```

- K=7 court regions, L shot-selection clusters, J accuracy clusters
- Players assigned to latent clusters via membership variables
- **Position-agnostic**: No traditional position labels used
- Gibbs sampling with 10,000 iterations (3,000 burn-in)

**Key finding**: EPAA correlates only 0.24 with PER and BPM, suggesting it captures genuinely different aspects of player value.

**Relevance**: Demonstrates that Bayesian clustering can identify meaningful player types without relying on position labels. The latent cluster assignments serve as a kind of learned "position" embedding.

### 5.3 Bayesian Performance Rating (Evan Miyakawa, EvanMiya.com)

**Architecture**: Three-component Bayesian model for college basketball:

1. **Multi-year RAPM**: 4 consecutive seasons of play-by-play data
2. **Box BPR**: College-specific box-score regression (distinct from NBA BPM)
3. **Preseason projection**: Recruiting ratings + historical performance

**Hierarchical prior structure**:
```
E[Points/poss] = intercept + sum(B_off) - sum(B_def)
B_off_j ~ N(Box_BPR_off_j, tau_off)
B_def_j ~ N(Box_BPR_def_j, tau_def)
```

**Cold-start handling**:
- **Established players**: Prior from Box BPR (offense R^2 ~66%, defense ~59%)
- **Transfers**: Emphasizes box scores over on-off (on-off less reliable across team changes)
- **Freshmen**: Incorporates recruiting star ratings as prior
- Prior weight decays from ~85% early-season to ~15% end-of-season

**Key insight**: The differentiated prior handling for transfers vs. established players vs. freshmen is directly relevant. Our system needs analogous mechanisms for:
- Veterans (strong individual prior)
- Players who changed teams (discount team-context-dependent features)
- Rookies (rely on draft position, college stats, physical profile)

### 5.4 Bayesian Hierarchical Team Modeling

**Paper**: Scitepress (2023), "Bayesian Hierarchical Modelling of Basketball Team Performance"

Estimates offensive and defensive ability per scoring method (2PT, 3PT, FT) for each team. Uses hierarchical structure to share information across teams and scoring methods.

### 5.5 Direct Mapping to Our Hierarchy

Our proposed hierarchy maps cleanly to the Bayesian partial pooling framework:

| Our Level | Bayesian Analog | What It Provides |
|-----------|----------------|------------------|
| Generic NBA player | Population prior mu | Baseline for all players |
| Position/archetype | Group-level theta_g | Position-specific adjustments |
| Individual player | Individual theta_i | Player-specific ability |

The Bayesian framework provides:
- **Automatic regularization**: Sparse players shrink toward their group
- **Uncertainty quantification**: Know how confident we are in each estimate
- **Natural cold-start**: New players start at group prior and update
- **Principled combination**: Box-score signal + on-off signal + prior knowledge

---

## 6. Transfer Learning / Few-Shot Learning for Player Modeling

### 6.1 Similarity-Based Player Projection (CARMELO / DARKO)

Both CARMELO and DARKO handle new/sparse players by leveraging historical comparables:

**CARMELO approach**:
- Find historically similar players (same age, size, draft position, statistical profile)
- Weight projections by similarity score (0-100 scale)
- Produce probabilistic career trajectory forecasts

**DARKO approach**:
- New players start with league-average priors
- Exponential decay weighting ensures recent data dominates quickly
- Kalman filter provides principled uncertainty reduction

### 6.2 Prototype Networks and Meta-Learning

**Prototype networks** (Snell et al., 2017) learn a metric space where classification is performed by computing distances to class prototypes (mean embeddings).

**Application to NBA cold-start**:
- "Classes" = player archetypes (e.g., 3-and-D wing, rim-running center)
- "Prototype" = mean embedding of all players in that archetype
- "Few-shot" = a rookie's first 10-20 games
- Classification = finding the closest archetype prototype

This is mathematically equivalent to Bayesian shrinkage toward the nearest cluster centroid, connecting few-shot learning to the hierarchical Bayesian framework.

### 6.3 VaRLAE Shared Prior for Sparse Players (NeurIPS 2020)

As discussed in Section 4.3, the VaRLAE architecture explicitly addresses the sparse-player problem:
- Shared prior learned across all players
- Shrinkage effect means sparse players get representations close to the prior
- As data accumulates, individual representations diverge from prior

This is the most directly relevant existing work for our cold-start problem.

### 6.4 Draft Projection Models

**ESPN draft model** and academic work (e.g., SMU Data Science Review, Sloan Conference) show:
- Scout rankings are the single most important predictor of NBA success
- Adding college stats reduces uncertainty by ~10%
- Age at draft is critical (younger = more upside)
- Physical measurements (height, wingspan) provide additional signal

**Implication**: For rookies, our system should initialize the ability vector using:
1. Draft position -> strong prior on overall ability level
2. College stats (adjusted for competition level) -> initial ability vector shape
3. Physical profile (height, weight, wingspan) -> position archetype prior
4. Then rapidly update with actual NBA performance data

### 6.5 Cross-League Transfer

**Paper**: Frontiers in Sports and Active Living (2025), "A deep learning-based study of player styles and cross-league performance adaptation mechanisms: a case study of the NBA and CBA."

Demonstrates that player style embeddings can transfer across leagues, suggesting learned representations capture intrinsic properties, not just league-specific patterns.

---

## 7. Synthesis: Design Implications for Our System

### 7.1 Ability Vector Design

Based on the literature, we recommend:

**Dimensionality**: 16-32 dimensions for the core ability vector.
- PCA shows ~4-5 components capture 70-80% of box-score variance
- NBA2Vec succeeds with 8 dimensions for play-outcome prediction
- Player archetype studies find 9-12 natural clusters
- We need additional dimensions for aspects not captured by box scores (defense, playmaking gravity, etc.)
- Recommendation: 24-dimensional base ability vector with 8 additional context-dependent dimensions (32 total)

**Interpretable components** (informed by PCA and clustering literature):
1. Scoring volume / usage
2. Scoring efficiency (TS%, shot selection quality)
3. Interior play (rebounding, rim protection, post scoring)
4. Perimeter play (3PT shooting, ball handling, spacing)
5. Playmaking (assist creation, decision-making)
6. Defensive impact (steals, blocks, contest rates)
7. Physical profile / athleticism proxy
8. Durability / minutes capacity

### 7.2 Hierarchical Pre-Training Architecture

```
Level 0 (Population):     Generic NBA player prior ~ N(mu_0, Sigma_0)
                          Learned from all players, all eras

Level 1 (Archetype):      Archetype prior ~ N(mu_k, Sigma_k)
                          K = 9-12 learned archetypes (not traditional positions)
                          Each archetype has own mean and covariance

Level 2 (Individual):     Player posterior ~ N(mu_i, Sigma_i)
                          Updated after every game via Bayesian update

Cold-start flow:
  Rookie drafted -> initialize at population prior
  Physical profile + draft position -> soft-assign to archetype(s)
  College stats (if available) -> initial offset from archetype mean
  First 10 NBA games -> begin individual updates
  After ~40 games -> individual data dominates prior
```

### 7.3 Update Mechanism

The literature strongly supports a **Kalman filter** approach for game-by-game updates:

```
State:       x_t = player ability vector at time t  (dimension d)
Transition:  x_t = F * x_{t-1} + w_t,  w_t ~ N(0, Q)
Observation: y_t = H * x_t + v_t,       v_t ~ N(0, R)

Kalman update:
  K_t = P_{t|t-1} H' (H P_{t|t-1} H' + R)^{-1}
  x_{t|t} = x_{t|t-1} + K_t (y_t - H x_{t|t-1})
  P_{t|t} = (I - K_t H) P_{t|t-1}
```

Where:
- F = state transition (encodes aging curves + career development)
- Q = process noise (how much ability can change game-to-game)
- R = observation noise (game outcome variance -- very high in basketball)
- H = observation matrix (maps ability to expected box-score stats)

**Key design choices**:
- Q should be **stat-specific** (usage stabilizes in ~10 games, 3PT% in ~240)
- F should encode position-specific aging curves (peak at ~27, power-law decline)
- R should be very large (single-game basketball stats are extremely noisy)
- The Kalman gain K automatically balances prior vs. new evidence

### 7.4 Team Context Separation

The RAPM literature shows how to separate player ability from team context:

1. **Training signal**: Use RAPM-derived values as supervision (already team-agnostic)
2. **Architecture**: Predict RAPM from box-score sequences (like BPM/EPM's SPM models)
3. **Opponent adjustment**: Normalize observations by opponent defensive quality
4. **Teammate adjustment**: Regress out teammate quality effects (usage redistribution, spacing)
5. **System adjustment**: Control for coach/system effects (pace, shot distribution)

### 7.5 Training Data Strategy

| Signal Type | Source | What It Captures | Stabilization |
|-------------|--------|------------------|---------------|
| Box scores | Every game | Observed production | Fast (10-80 games) |
| On-off | Stint-level | True impact | Slow (1000+ possessions) |
| Tracking | NBA.com (2013+) | Movement, effort | Medium (50-100 games) |
| Play-by-play | Every game | Event credit | Medium (100+ games) |
| Multi-year RAPM | Career | Career ability | Very slow (3-5 seasons) |

### 7.6 Key Papers to Revisit During Implementation

| Priority | Paper | Why |
|----------|-------|-----|
| 1 | Liu et al., NeurIPS 2020 (VaRLAE) | Closest architecture to our goal; shared prior + hierarchy |
| 2 | Vaci et al., 2019 (Aging curves) | Mathematical aging model we should embed in transition matrix F |
| 3 | Engelmann Substack (RAPM tutorial) | Practical RAPM implementation with code |
| 4 | Snarr, Dunks & Threes (EPM) | Best public metric; "skills" concept for per-stat optimization |
| 5 | Elmore et al., 2024 (EPAA) | Bayesian hierarchical model with spatial shooting |
| 6 | Miyakawa, EvanMiya (BPR) | Practical hierarchical Bayesian with cold-start handling |
| 7 | Medvedovsky, 2020 (Stabilization) | Stat-specific stabilization rates for ability vector design |
| 8 | arXiv:2601.15000, 2026 (L-RAPM) | Lineup-level priors from individual ratings |
| 9 | arXiv:2509.25858, 2025 (LSTM aging) | Autoencoder + LSTM for career trajectory prediction |
| 10 | Luo & Krishnamurthy, 2023 (GATv2-TCN) | Graph attention for player interaction effects |

---

## References

### RAPM and Plus-Minus Foundations
- Rosenbaum, D. (2004). "Measuring How NBA Players Help Their Teams Win." 82games.com.
- Sill, J. (2010). "Improved NBA Adjusted +/- Using Regularization and Out-of-Sample Testing." MIT Sloan Sports Analytics Conference. [PDF](https://supermariogiacomazzo.github.io/STOR538_WEBSITE/Articles/Basketball/Basketball_Sill.pdf)
- Engelmann, J. (2011+). xRAPM methodology. [stats-for-the-nba.appspot.com](https://stats-for-the-nba.appspot.com)
- Engelmann, J. "NBA Adjusted Plus-Minus: How to Build It." [Substack](https://jeremiasengelmann.substack.com/p/nba-adjusted-plus-minus-how-to-build)
- Jacobs, J. (2017-2018). "Deep Dive on Regularized Adjusted Plus-Minus." [Squared Statistics](https://squared2020.com/2017/09/18/deep-dive-on-regularized-adjusted-plus-minus-i-introductory-example/)
- arXiv:2601.15000 (2026). "Lineup Regularized Adjusted Plus-Minus (L-RAPM)." [arXiv](https://arxiv.org/abs/2601.15000)

### Modern Metrics
- Snarr, T. "About Estimated Plus-Minus (EPM)." [Dunks & Threes](https://dunksandthrees.com/about/epm)
- Snarr, T. "Metric Comparison." [Dunks & Threes](https://dunksandthrees.com/blog/metric-comparison)
- Silver, N. et al. (2019). "How Our RAPTOR Metric Works." [FiveThirtyEight](https://fivethirtyeight.com/features/how-our-raptor-metric-works/)
- Basketball Index. "LEBRON Introduction." [BBall Index](https://www.bball-index.com/lebron-introduction/)
- Medvedovsky, K. "DARKO." [DARKO app](https://apanalytics.shinyapps.io/DARKO/)
- Ilardi, S. & Engelmann, J. (2014). "Introducing Real Plus-Minus." [ESPN](https://www.espn.com/nba/story/_/id/10740818/introducing-real-plus-minus)
- Oliver, D. & Engelmann, J. (2025). "Introducing Net Points." [ESPN](https://www.espn.com/nba/story/_/id/44093220/introducing-net-points-latest-nba-metric-amazing-early-findings)
- Myers, D. "About Box Plus/Minus (BPM)." [Basketball-Reference](https://www.basketball-reference.com/about/bpm2.html)
- "Bayesian Box Adjusted Plus-Minus." [Intraocular](https://www.intraocular.net/posts/bayesian-box-plus-minus)

### Aging Curves and Career Trajectories
- Vaci, N. et al. (2019). "Large data and Bayesian modeling -- aging curves of NBA players." *Behavior Research Methods*. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6690864/)
- arXiv:2509.25858 (2025). "Aging Decline in Basketball Career Trend Prediction Based on Machine Learning and LSTM Model." [arXiv](https://arxiv.org/abs/2509.25858)
- Frontiers (2025). "Nonlinear age effects on basketball player performance: insights from KAN." [Frontiers](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2025.1693433/full)
- Medvedovsky, K. (2020). "NBA Stabilization Rates and the Padding Approach." [Blog](https://kmedved.com/2020/08/06/nba-stabilization-rates-and-the-padding-approach/)
- FiveThirtyEight. "How We're Predicting The Career Of Every NBA Player (CARMELO)." [FiveThirtyEight](https://fivethirtyeight.com/features/how-were-predicting-nba-player-career/)

### Latent Player Representations
- arXiv:2302.13386 (2023). "NBA2Vec: Dense Feature Representations of NBA Players." [arXiv](https://arxiv.org/abs/2302.13386)
- Perez, A. (2021). "Player2Vec: Representation Learning of NBA Players." Santa Clara University. [Thesis](https://scholarcommons.scu.edu/cseng_mstr/23/)
- Liu, G. et al. (2020). "Learning Agent Representations for Ice Hockey." NeurIPS 2020. [NeurIPS](https://proceedings.neurips.cc/paper/2020/hash/d90e5b6628b4291225cba0bdc643c295-Abstract.html)
- Wang et al. (2024). "player2vec: A Language Modeling Approach to Understand Player Behavior in Games." [arXiv](https://arxiv.org/abs/2404.04234)
- arXiv:1511.04351 (2015). "A Scalable Framework for NBA Player and Team Comparisons Using Player Tracking Data." [arXiv](https://arxiv.org/abs/1511.04351)

### Bayesian Hierarchical Models
- Elmore, R. et al. (2024). "Expected Points Above Average: A Novel NBA Player Metric Based on Bayesian Hierarchical Modeling." *Annals of Applied Statistics* (2025). [arXiv](https://arxiv.org/abs/2405.10453)
- Miyakawa, E. "Bayesian Performance Rating." [EvanMiya Blog](https://blog.evanmiya.com/p/bayesian-performance-rating)
- Efron, B. & Morris, C. (1975). "Data Analysis Using Stein's Estimator and Its Generalizations." *JASA*.
- James, W. & Stein, C. (1961). "Estimation with Quadratic Loss." *Proc. Fourth Berkeley Symposium*.

### Graph Neural Networks and Player Interaction
- Luo, R. & Krishnamurthy, V. (2023). "Who You Play Affects How You Play." [arXiv](https://arxiv.org/abs/2303.16741)
- PMC (2023). "Enhancing Basketball Game Outcome Prediction through Fused GCN and Random Forest." [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10217531/)

### Player Clustering and Archetypes
- Duman, E.A. et al. (2024). "A cluster analysis of basketball players for each of the five traditionally defined positions." *Journal of Sports Engineering and Technology*.
- Frontiers (2025). "Player archetypes within basketball: optimizing roster composition." [Frontiers](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2025.1639431/full)

### Rating Systems and Kalman Filters
- Herbrich, R. et al. (2006). "TrueSkill: A Bayesian Skill Rating System." NeurIPS.
- Minka, T. et al. (2018). "TrueSkill 2." [Microsoft Research](https://www.microsoft.com/en-us/research/wp-content/uploads/2018/03/trueskill2.pdf)
- Poropudas, J. (2011). "Kalman Filter Algorithm for Rating and Prediction in Basketball." [Link](https://www.hamahakkimies.com/project/kalman-rating)
- arXiv:2308.02414 (2023). "A State-Space Perspective on Modelling and Inference for Online Skill Rating." [JRSS-C](https://academic.oup.com/jrsssc/article/73/5/1262/7734616)

### Few-Shot and Transfer Learning
- Stanford CS231n (2024). "Scoring with Few Shots: Applying Few-Shot Learning to Basketball Analytics." [PDF](https://cs231n.stanford.edu/2024/papers/scoring-with-few-shots-applying-few-shot-learning-to-basketball-.pdf)
- Frontiers (2025). "A deep learning-based study of player styles and cross-league performance adaptation." [Frontiers](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2025.1639972/full)
- Snell, J. et al. (2017). "Prototypical Networks for Few-Shot Learning." NeurIPS.
