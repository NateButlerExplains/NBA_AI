# Level 2: Player Synergy and Interaction Modeling -- Academic Literature Review

Research compiled 2026-03-16. Covers academic approaches to modeling how NBA players affect each other's performance when playing together. Level 2 sits on top of Level 1 (individual player ability vectors) and captures chemistry/synergy effects through graph-based and factorized interaction models.

---

## Table of Contents

1. [Graph Neural Networks in Basketball/Sports](#1-graph-neural-networks-in-basketballsports)
2. [Skills Plus Minus (SPM) and NBA Chemistry](#2-skills-plus-minus-spm-and-nba-chemistry)
3. [Factorization Machines for Player Interactions](#3-factorization-machines-for-player-interactions)
4. [Bayesian Pairwise Interaction Models](#4-bayesian-pairwise-interaction-models)
5. [Hypergraph Networks and Higher-Order Interactions](#5-hypergraph-networks-and-higher-order-interactions)
6. [Dynamic/Temporal Graph Networks](#6-dynamictemporal-graph-networks)
7. [Player Archetype and Clustering Approaches](#7-player-archetype-and-clustering-approaches)
8. [Synthesis: Practical Architecture for Level 2](#8-synthesis-practical-architecture-for-level-2)

---

## 1. Graph Neural Networks in Basketball/Sports

### 1.1 "Who You Play Affects How You Play" (Luo & Krishnamurthy, 2023)

**Paper:** [arXiv:2303.16741](https://arxiv.org/abs/2303.16741)

This is the most directly relevant paper for our Level 2 design. It presents GATv2-TCN, a model that predicts individual player performance conditioned on who else is on the court.

**Architecture:**

1. **Dynamic Player Interaction Graph**: Players are nodes. Edges connect players who share the court (teammates and opponents). Edge weights reflect co-occurrence frequency and recency.

2. **GATv2 Attention Layer**: Uses GATv2 (Brody, Alon & Yahav, 2022) rather than the original GAT. The critical difference is that GATv2 computes *dynamic* attention -- the attention score ranking is conditioned on the query node, making it strictly more expressive than GAT's static attention. The attention mechanism learns how much each player "pays attention to" (is affected by) each neighbor:

   ```
   e_ij = a^T LeakyReLU(W [h_i || h_j])      # GATv2 attention
   alpha_ij = softmax_j(e_ij)                   # normalized attention weights
   h_i' = sigma(sum_j alpha_ij W h_j)           # aggregated representation
   ```

   In GATv2, the weight matrix W is applied *after* concatenation (not before), enabling the attention function to be a universal approximator over node pairs.

3. **Temporal Convolution Network (TCN)**: Handles the multivariate time series of player statistics. Provides temporal predictive power independent of the graph structure. Dilated causal convolutions capture patterns across multiple time scales.

4. **Team and Position Embeddings**: Auxiliary embeddings help the model account for a player's team membership and positional role when computing attention to neighbors.

**Results (2022-23 NBA season, 691 games, 582 players):**
- Superior RMSE, MAE, and correlation (CORR) vs. baselines
- 35/59 correct predictions on held-out game day
- Key finding: Player performance predictions improve significantly when graph context (who else is playing) is included vs. individual-only models

**Takeaway for our design:** This validates the core premise -- graph attention over player interactions improves prediction. Their GATv2 + TCN architecture maps well to our Level 2 plan. However, they model game-level performance prediction, not the latent synergy vectors we need. We should adopt the GATv2 attention mechanism but apply it to learn persistent synergy embeddings rather than per-game predictions.

### 1.2 HIGFormer: Heterogeneous Interaction Graph Transformer (2025)

**Paper:** [KDD 2025](https://arxiv.org/abs/2507.10626)

HIGFormer introduces a three-component architecture for match outcome prediction (applied to soccer, but architecture-transferable):

1. **Player Interaction Network**: Encodes individual player performance through heterogeneous interaction graphs. Combines local graph convolutions (for fine-grained neighbor interactions) with a global graph-augmented transformer (for long-range dependencies).

2. **Team Interaction Network**: Constructs team-to-team interaction graphs from historical match relationships. Models how teams perform against each other over time.

3. **Match Comparison Transformer**: Jointly analyzes both player-level and team-level representations to predict outcomes.

**Key architectural insight:** The hybrid local-GCN + global-transformer design captures both fine-grained player interactions (who specifically plays with whom) and coarse-grained team patterns. This two-scale approach is relevant to our archetype-level vs. player-level synergy design.

### 1.3 Sports Analytics with GNNs Survey (2024)

**Paper:** [Preprints.org 202410.0046](https://www.preprints.org/manuscript/202410.0046/v1)

Recent survey covering GNN and GCN applications across sports analytics. Key findings relevant to us:

- **Graph construction matters**: How you define nodes (players, teams, game events) and edges (co-occurrence, passing, matchups) dramatically affects what the GNN can learn.
- **Heterogeneous data integration**: Best results come from combining multiple data sources (box scores, event data, video features) in graph-based representations.
- **Player embeddings**: Match outcomes can be predicted by comparing aggregated team representations computed as the average of player embeddings. This validates our planned approach of aggregating Level 1 + Level 2 player vectors into team representations.

### 1.4 Zhao et al. (2023) -- Fused GCN + Random Forest

**Paper:** [Entropy 2023, 25(5), 765](https://www.mdpi.com/1099-4300/25/5/765)

Applied Graph Convolutional Networks to basketball game outcome prediction by transforming structured match data into graphs representing passing interactions among players. Achieved 71.54% prediction accuracy with fused GCN + Random Forest, outperforming non-graph baselines. Demonstrates that explicitly modeling spatial/relational structure between players outperforms flat vector approaches.

### 1.5 TacticExpert (2025)

**Paper:** [arXiv:2503.10722](https://arxiv.org/abs/2503.10722)

A vertically integrated large model for basketball that uses a Spatial-Temporal Propagation Symmetry-Aware Graph Transformer. Notable features:

- **Mixture of Tactics Experts**: Uses contrastive learning to differentiate offensive tactical patterns. Relevant to our archetype idea -- different tactical contexts activate different interaction patterns.
- **Geometric deep learning**: Models symmetries of the basketball court (identity, X-rotation, Y-rotation, Z-rotation) in the spatio-dependent coding layer.
- **LLM grounding**: Uses lightweight graph grounding to enable zero-shot generalization to novel teams/players.
- **2.4x efficiency improvement** through dense training with sparse inference.

**Relevance:** The mixture-of-experts approach for different tactical contexts is directly applicable. Player synergies may differ in half-court offense vs. transition vs. post-up situations.

### 1.6 GATv2: How Attentive Are Graph Attention Networks? (Brody, Alon & Yahav, ICLR 2022)

**Paper:** [arXiv:2105.14491](https://arxiv.org/abs/2105.14491)

The foundational paper for the GATv2 architecture used in multiple basketball/sports GNN papers. Key results:

- **Static vs. dynamic attention**: Original GAT computes attention where the ranking of attention scores is *unconditioned* on the query node. This means for any query node, the neighbors are always ranked in the same order -- a severe expressiveness limitation.
- **GATv2 fix**: By changing the order of operations (apply W after concatenation, not before), GATv2 becomes a universal approximator over attention functions.
- **Empirical**: Outperforms GAT across 12 benchmarks. Much more robust to edge noise -- dynamic attention can learn to decay noisy/false edges, while GAT cannot.
- **Available in**: PyTorch Geometric, Deep Graph Library, TensorFlow GNN.

**Takeaway:** GATv2 is the correct choice for our player interaction graph attention. The dynamic attention property is critical because player A's attention to player B should depend on player A's own characteristics (a ball-dominant scorer attends differently to a shooter vs. another ball-handler).

---

## 2. Skills Plus Minus (SPM) and NBA Chemistry

### 2.1 Maymin, Maymin & Shen: NBA Chemistry (IJCSS, 2013)

**Paper:** [SSRN:1935972](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1935972) | [PDF](https://philipmaymin.com/papers/Maymin%20Maymin%20and%20Shen%20-%20NBA%20Chemistry%20-%20IJCSS.pdf)

The seminal work on quantifying player synergy in basketball.

**Framework:**

1. **Skill Decomposition**: Each player's offense and defense is evaluated across three skill categories: *scoring*, *rebounding*, and *ball-handling*. This produces a 6-dimensional skill vector per player.

2. **Game Simulation**: Games are simulated possession-by-possession using the skill ratings of the 10 players on the court. The simulation engine models how skills interact -- e.g., a great ball-handler creates open shots for teammates, but two great ball-handlers may not combine as well as one great ball-handler plus one great shooter.

3. **Synergy Calculation**: For a 5-player lineup, synergy = (simulated team effectiveness) - (sum of individual player values). Positive synergy means the lineup is greater than the sum of its parts; negative synergy means the opposite.

**Key Findings:**

- **Synergies can explain up to 6 wins** per season. This is large -- the difference between a playoff team and a lottery team.
- **Skill-specific synergy effects are asymmetric:**
  - *Offensive ball-handling has negative synergy with itself*: A lineup with one great ball-handler doesn't need another. The second ball-handler's value is diminished.
  - *Defensive ball-handling has positive synergy with itself*: Defenders who create turnovers feed off each other -- one defender pressuring a pass creates opportunities for another to intercept.
  - *Scoring has diminishing returns*: Multiple elite scorers compete for shots.
  - *Rebounding has positive synergy*: Multiple good rebounders create better box-out positioning for each other.
- **Context-dependent player value**: A player's value depends on the other 9 players on the court. The same player can be worth +3 wins on one team and +1 win on another.
- **Trade identification**: The framework identified 200+ mutually beneficial trades where both teams improve because the traded players' skills better complement their new teammates. Traditional rating systems cannot find these because they assign fixed values to players.

**Mathematical Framework (reconstructed):**

Let player i have skill vector `s_i = [s_i^{off_score}, s_i^{off_rebound}, s_i^{off_handle}, s_i^{def_score}, s_i^{def_rebound}, s_i^{def_handle}]`.

For a 5-player lineup L = {p1, ..., p5}:
```
V_individual(L) = sum_{i in L} value(s_i)          # sum of individual values
V_lineup(L) = simulate(s_p1, ..., s_p5)            # simulated lineup value
Synergy(L) = V_lineup(L) - V_individual(L)         # synergy residual
```

The simulation captures non-linear interactions that the linear sum misses. The genius is that the skill decomposition reduces the space from player-identity (500+ NBA players) to skill-type (6 dimensions), making the simulation tractable.

**Takeaway for our design:** This confirms that (a) synergy effects are real and significant (up to 6 wins), (b) specific skill-type combinations have predictable positive/negative interactions, and (c) reducing the problem from player-identity space to skill-type space is critical for tractability. Our archetype-based regularization approach directly follows this insight.

### 2.2 RAPM and the Multicollinearity Problem

**Sources:** [NBAStuffer](https://www.nbastuffer.com/analytics101/regularized-adjusted-plus-minus-rapm/) | [Squared Statistics](https://squared2020.com/2017/09/18/deep-dive-on-regularized-adjusted-plus-minus-i-introductory-example/)

Regularized Adjusted Plus-Minus (RAPM) provides essential background for understanding why interaction modeling is hard:

**The Basic APM Model:**

For each possession p with outcome y_p (points scored), the model assigns a coefficient beta_i to each player:
```
y_p = sum_{i in home_lineup} beta_i - sum_{i in away_lineup} beta_i + epsilon_p
```

RAPM adds L2 regularization (ridge regression) to handle multicollinearity:
```
minimize: sum_p (y_p - y_hat_p)^2 + lambda * sum_i beta_i^2
```

**The Sparsity Problem for Interactions:**

Extending RAPM to pairwise interactions means adding beta_{ij} terms for every pair:
```
y_p = sum_i beta_i + sum_{i<j in same_lineup} beta_{ij} + epsilon_p
```

With ~500 active NBA players, this creates ~125,000 pairwise interaction parameters. Most player pairs rarely or never share the court:
- It takes ~550 possessions for a five-man lineup's offensive rating to "stabilize" (Medvedovsky)
- ~850 possessions for defensive rating stabilization
- **Only ~25 five-man lineups per season reach 550 possessions** (less than 1 per team)

This extreme data sparsity is *the* central challenge for Level 2. Any approach must handle the fact that most player pairs have very limited shared court time, and most five-man lineups have essentially no statistically reliable data.

### 2.3 Wei et al. (2025) -- Synergy Through Social Network Analysis

**Paper:** [Acta Psychologica, 2025](https://www.sciencedirect.com/science/article/pii/S0001691825010601)

Recent study applying Social Network Analysis to professional basketball lineup synergy:

- Identified **10 key player roles** correlated with lineup excellence
- Found specific role combinations that enhance or diminish performance:
  - **C6 (Core Ball Handler) + C1 (Off-Ball Shooter)**: Significantly improved lineup net ratings (positive synergy)
  - **C6 (Core Ball Handler) + C2 (Versatile Big)**: Enhanced performance of most teammates
  - **C3 (Ball Dominant Scorer)**: Had *negative* impact on performance of other roles (negative synergy)
- Network centrality metrics identify players with superior tactical influence

**Takeaway:** This validates Maymin's findings with modern data. Ball-dominant scorers create negative synergy, while complementary role combinations (ball-handler + shooter, ball-handler + versatile big) create positive synergy. Our model should learn these patterns automatically through the interaction mechanism.

---

## 3. Factorization Machines for Player Interactions

### 3.1 Core Formulation (Rendle, 2010)

**Paper:** [IEEE ICDM 2010](https://www.ismll.uni-hildesheim.de/pub/pdfs/Rendle2010FM.pdf)

Factorization Machines provide the mathematical foundation for efficient pairwise interaction modeling. The FM prediction equation is:

```
y_hat(x) = w_0 + sum_{i=1}^{n} w_i x_i + sum_{i=1}^{n} sum_{j=i+1}^{n} <v_i, v_j> x_i x_j
```

Where:
- `w_0` is the global bias
- `w_i` are linear feature weights (analogous to our Level 1 individual effects)
- `v_i in R^k` is a k-dimensional latent vector for feature i
- `<v_i, v_j> = sum_{f=1}^{k} v_{i,f} * v_{j,f}` is the dot-product interaction between features i and j

**The key insight for player interactions:**

If each player i is represented by a latent vector `v_i in R^k`, then the pairwise interaction between players i and j is modeled as `<v_i, v_j>`. This has three critical properties:

1. **Parameter efficiency**: Instead of learning O(n^2) explicit interaction parameters, we learn O(nk) latent vectors. With ~500 NBA players and k=32, this is 16,000 parameters instead of 125,000.

2. **Generalization to unseen pairs**: Even if players i and j have never shared the court, their interaction can be estimated from `<v_i, v_j>` because each vector is learned from *all* of that player's interactions. Player i's vector is shaped by his interactions with players a, b, c, ..., and player j's vector is shaped by his interactions with players x, y, z, .... The dot product generalizes.

3. **Linear-time computation** via the kernel trick:
   ```
   sum_{i<j} <v_i, v_j> x_i x_j = 1/2 * sum_{f=1}^{k} [(sum_i v_{i,f} x_i)^2 - sum_i v_{i,f}^2 x_i^2]
   ```
   This reduces computation from O(kn^2) to O(kn).

### 3.2 Application to Player Interaction Modeling

**Proposed formulation for our Level 2:**

Let each player i have:
- Level 1 ability vector: `a_i in R^d` (from Phase 5 Level 1)
- Synergy vector: `v_i in R^k` (learned at Level 2)

For a lineup L = {p1, p2, p3, p4, p5}, the synergy contribution is:
```
Synergy(L) = sum_{i<j in L} <v_{p_i}, v_{p_j}>
           = 1/2 * sum_{f=1}^{k} [(sum_{i in L} v_{p_i,f})^2 - sum_{i in L} v_{p_i,f}^2]
```

This captures all C(5,2) = 10 pairwise interactions in a 5-player lineup with O(5k) computation.

**Enrichment: Typed interactions.**

Rather than a single dot product, we can decompose interactions by type:
```
Interaction(i, j) = <v_i^{off}, v_j^{off}> + <v_i^{def}, v_j^{def}> + <v_i^{space}, v_j^{space}>
```

Where `v^{off}`, `v^{def}`, `v^{space}` are separate latent vectors capturing offensive chemistry, defensive chemistry, and spacing chemistry respectively. This mirrors SPM's decomposition into scoring/rebounding/ball-handling synergies.

### 3.3 DeepFM: Deep Factorization Machines (Guo et al., IJCAI 2017)

**Paper:** [arXiv:1703.04247](https://arxiv.org/abs/1703.04247)

DeepFM extends FM by combining the FM component (for explicit low-order interactions) with a deep neural network (for implicit high-order interactions):

```
y_hat = sigmoid(y_FM + y_DNN)
```

Where:
- `y_FM = w_0 + sum w_i x_i + sum_{i<j} <v_i, v_j> x_i x_j` (standard FM)
- `y_DNN = MLP(concat(v_1, v_2, ..., v_n))` (deep component)

**Critical design choice:** The FM and DNN components share the same embedding vectors `v_i`. This means:
- No separate feature engineering is needed
- The embeddings are trained by both the explicit interaction signal (FM) and the implicit pattern signal (DNN)
- The shared embedding acts as a regularizer

**Relevance to our design:** The DeepFM architecture maps naturally to our player synergy model. The FM component captures explicit pairwise synergies (player A + player B = +X points), while the DNN component captures higher-order patterns that emerge from the full lineup composition. The shared embedding ensures the synergy vectors are well-regularized.

### 3.4 GraphFM: Graph Factorization Machines (2021)

**Paper:** [arXiv:2105.11866](https://arxiv.org/abs/2105.11866)

GraphFM combines FM with GNN by representing feature interactions as a graph and applying graph neural network operations. Relevant because it bridges the gap between the FM-style interaction modeling (Section 3) and the GNN-based approaches (Section 1). The key idea is that not all pairwise interactions are equally important -- the graph structure encodes *which* interactions to attend to.

---

## 4. Bayesian Pairwise Interaction Models

### 4.1 The Kernel Interaction Trick (Agrawal et al., ICML 2019)

**Paper:** [Proceedings of ICML 2019](https://proceedings.mlr.press/v97/agrawal19a.html) | [arXiv:1905.06501](https://arxiv.org/abs/1905.06501)

This paper addresses the fundamental computational challenge of Bayesian inference over pairwise interactions in high dimensions.

**Problem:** With p features (players), there are O(p^2) pairwise interactions. Maintaining a full posterior over all interactions is computationally intractable for large p.

**Solution:** Many hierarchical interaction models admit a Gaussian process representation where, instead of maintaining a posterior over all O(p^2) interactions, you only need to maintain O(p) kernel hyperparameters. This implicit representation allows:
- MCMC in time and memory *linear in p* per iteration
- Orders of magnitude runtime reduction over naive MCMC
- Lower Type I and Type II error than LASSO-based approaches

**Mathematical insight:**

The interaction model `y = sum_i beta_i x_i + sum_{i<j} beta_{ij} x_i x_j + epsilon` can be reformulated using a GP with kernel:
```
k(x, x') = sigma_main^2 * sum_i x_i x_i' + sigma_int^2 * sum_{i<j} x_i x_j x_i' x_j'
```

The second term factorizes as:
```
sum_{i<j} x_i x_j x_i' x_j' = 1/2 * [(sum_i x_i x_i')^2 - sum_i x_i^2 x_i'^2]
```

This is the same kernel trick as in factorization machines. The Bayesian version adds proper uncertainty quantification and sparsity-inducing priors.

**Takeaway for our design:** If we want principled uncertainty estimates on synergy effects (which we do, for sparse player pairs), this Bayesian FM framework is the right foundation. It provides the same computational efficiency as standard FM but with calibrated uncertainty.

### 4.2 Bayesian Factor Analysis for Inference on Interactions (Ferrari & Dunson, JASA 2021)

**Paper:** [JASA 2021, Vol 116, 1521-1532](https://arxiv.org/abs/1904.11603)

The FIN (Factor analysis for INteractions) framework addresses inference on interactions among correlated predictors -- directly analogous to our correlated player statistics problem.

**Core approach:**

1. **Shared latent factors**: Both predictors (player stats) and response (team performance) are modeled through shared latent variables. This induces dimension reduction in characterizing both main effects and interactions.

2. **Quadratic regression in latent space**: By including quadratic terms in the latent variables within the response component, the model captures interactions without explicitly enumerating all pairwise terms.

3. **Automatic shrinkage**: The prior structure resembles a mixture of two normals with different variances. Higher-order interactions receive stronger shrinkage toward zero:
   - The mixture places higher weight on the zero-centered component as interaction order increases
   - Products of latent elements are naturally affected by the shrinkage prior
   - This encodes the prior belief that higher-order interactions are typically smaller than lower-order ones

**Relevance:** This is directly applicable to handling sparse player pairs. The hierarchical shrinkage means:
- Player pairs with lots of shared court time: the data speaks, interaction estimates are data-driven
- Player pairs with little shared court time: estimates are shrunk toward the archetype-level interaction
- Player pairs with no shared court time: interaction estimate comes entirely from the latent factor structure (analogous to FM generalization)

### 4.3 Expected Points Above Average (EPAA) -- Bayesian Hierarchical NBA Metric

**Paper:** [arXiv:2405.10453](https://arxiv.org/html/2405.10453v1)

A 2024 paper applying Bayesian hierarchical modeling to NBA player evaluation. Key methodological features:

- **Hierarchical structure**: Players are grouped by team, teams by conference. Each level has its own variance component. This naturally handles the fact that teammates are correlated.
- **Full posterior inference**: Provides uncertainty intervals on all player estimates, not just point estimates.
- **Shrinkage toward group means**: Low-minute players are shrunk toward team averages; small-sample teams are shrunk toward conference averages.

**Takeaway:** The hierarchical Bayesian framework with shrinkage is ideal for our interaction model. We can structure it as: player-pair interactions are shrunk toward archetype-pair interactions, which are shrunk toward the global mean. This handles sparsity gracefully.

---

## 5. Hypergraph Networks and Higher-Order Interactions

### 5.1 Why Hypergraphs?

Standard pairwise graphs model dyadic relationships (player A affects player B). But basketball lineups exhibit *higher-order* interactions:
- A three-player combination (pick-and-roll ball-handler + roll man + corner shooter) has synergy that cannot be decomposed into three pairwise interactions
- Five-man lineup effects often exceed the sum of all 10 pairwise effects

A hypergraph generalizes a graph by allowing *hyperedges* that connect arbitrary subsets of nodes. A 5-player lineup is naturally a hyperedge connecting 5 player nodes.

### 5.2 HGNN: Hypergraph Neural Networks (Feng et al., AAAI 2019)

**Paper:** [arXiv:1809.09401](https://arxiv.org/abs/1809.09401)

The foundational hypergraph neural network. Key formulation:

**Hypergraph definition:** G = (V, E, W) with:
- V: vertex set (players)
- E: hyperedge set (lineups, or sub-lineup groups)
- W: diagonal matrix of hyperedge weights
- H: |V| x |E| incidence matrix where H_{i,j} = 1 if vertex i belongs to hyperedge j

**Spectral convolution on hypergraphs:**
```
X^{(l+1)} = sigma(D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2} X^{(l)} Theta^{(l)})
```

Where:
- D_v: diagonal vertex degree matrix
- D_e: diagonal hyperedge degree matrix
- Theta: learnable filter parameters
- The product `H W D_e^{-1} H^T` propagates signals along vertex -> hyperedge -> vertex paths

**Message passing interpretation:**
1. **Vertex-to-hyperedge**: Aggregate node features for each hyperedge (e.g., average all player features in a lineup)
2. **Hyperedge-to-vertex**: Propagate hyperedge features back to constituent nodes (e.g., each player receives information from all lineups they participate in)

### 5.3 Tensorized Hypergraph Neural Networks (Wang et al., SDM 2024)

**Paper:** [arXiv:2306.02560](https://arxiv.org/abs/2306.02560) | [SIAM SDM](https://epubs.siam.org/doi/pdf/10.1137/1.9781611978032.15)

The first HGNN based on adjacency *tensors* rather than incidence matrices, enabling true higher-order message passing:

**Problem with standard HGNN:** The incidence-based approach (H W D_e^{-1} H^T) reduces to first-order approximations. It aggregates node features within each hyperedge independently, losing information about *which specific nodes* are together.

**Tensorized solution:**
- Represent the hypergraph with an adjacency tensor A of order matching the hyperedge cardinality
- For 5-player lineups: A is a 5th-order tensor A_{i,j,k,l,m}
- Direct message passing on the tensor captures true 5-way interactions

**Computational tractability:** Uses partially symmetric CP decomposition to reduce complexity from exponential to linear:
```
A ≈ sum_{r=1}^{R} lambda_r * v_r^{(1)} otimes v_r^{(2)} otimes ... otimes v_r^{(K)}
```

Where R is the CP rank. This factorizes the tensor into a sum of rank-1 tensors, each defined by K vectors.

**Handling non-uniform hypergraphs** (lineups of different sizes -- relevant for pre-substitution partial lineups):
- Global node approach: Add a virtual node connected to all hyperedges
- Multi-uniform processing: Process hyperedges of each size separately, then combine

### 5.4 Higher-Order GNN via Hypergraph Encodings (ICLR 2025)

**Paper:** [OpenReview](https://openreview.net/forum?id=oeMK0Js4lq)

Proposes encoding higher-order information into standard GNN frameworks via hypergraph encodings, avoiding the need for specialized hypergraph architectures. This is practically attractive because it lets us use well-optimized GATv2 implementations while still capturing some higher-order effects.

### 5.5 Practical Considerations for Basketball

**The combinatorial challenge:**

| Interaction Order | Count for 500 Players | Example |
|---|---|---|
| Pairwise (2-way) | 124,750 | Player A + Player B |
| 3-way | 20,708,500 | Pick-and-roll trio |
| 4-way | 2,573,031,125 | Four-man unit |
| 5-way | 255,244,687,600 | Full lineup |

Direct enumeration beyond pairwise is intractable. The factorized approaches (CP decomposition, FM-style latent vectors) are necessary.

**Recommended approach for our model:**

1. **Explicit pairwise**: Use FM-style factorized interactions `<v_i, v_j>` for all teammate pairs
2. **Implicit higher-order**: Use a small MLP or attention layer over the full 5-player lineup embedding (sum/mean of player vectors) to capture residual higher-order effects
3. **Do NOT attempt explicit 3-way+ enumeration**: The data is far too sparse. Let the neural network learn implicit higher-order patterns from the aggregated lineup representation.

---

## 6. Dynamic/Temporal Graph Networks

### 6.1 Why Temporal Dynamics Matter

Player interaction graphs in the NBA are inherently dynamic:
- **Mid-season trades**: A player's interaction vectors must update when teammates change
- **Free agency**: New teammate combinations appear every summer
- **Player development**: A young player's synergy profile evolves as their skills mature
- **Injury effects**: Extended absences change a player's rhythm with teammates
- **Recency weighting**: Recent shared court time should matter more than games from 3 seasons ago

### 6.2 Temporal Graph Networks (Rossi et al., ICML 2020 Workshop)

**Paper:** [arXiv:2006.10637](https://arxiv.org/abs/2006.10637) | [GitHub](https://github.com/twitter-research/tgn)

TGN is the foundational framework for deep learning on continuous-time dynamic graphs. Key components:

1. **Per-node temporal memory**: Each node maintains a memory state that is updated with each interaction. For our case, each player has a memory vector that evolves with each game played.

2. **Memory update**: When an event occurs (e.g., player i and j share the court), their memories are updated:
   ```
   m_i(t) = MSG(m_i(t-), m_j(t-), delta_t, e_{ij})
   m_j(t) = MSG(m_j(t-), m_i(t-), delta_t, e_{ij})
   ```
   Where MSG is a learnable message function and delta_t is the time since last interaction.

3. **Time-aware graph attention**: Aggregates neighbor information with attention weights that depend on both features *and* timing:
   ```
   h_i(t) = sum_j alpha(h_i, h_j, delta_t) * V * h_j
   ```
   More recent interactions get higher attention weights.

4. **Efficiency**: Because the memory module captures historical context, a single-layer graph attention suffices, yielding up to 30x speedup over multi-layer models.

**Relevance to our design:** TGN's per-node memory is directly applicable. Each player's synergy vector should be a *living* representation that updates with each game. When a player is traded, the memory naturally adapts as new interaction events flow in with new teammates.

### 6.3 DURENDAL: Temporal Heterogeneous Graph Framework (2023)

**Paper:** [arXiv:2310.00336](https://arxiv.org/abs/2310.00336)

A framework for repurposing any heterogeneous graph learning model to evolving networks:

- **Snapshot-based approach**: The network is divided into time snapshots (e.g., monthly periods). Each snapshot is a static heterogeneous graph.
- **Cross-snapshot connections**: Edges connect the same node across consecutive snapshots, encoding temporal evolution.
- **Any base model**: Can use GATv2, GraphSAGE, HGT, etc. as the base heterogeneous GNN within each snapshot.

**Relevance:** This is pragmatically useful. Rather than implementing complex continuous-time dynamics, we can use a snapshot approach:
- Each month (or each set of ~15 games) is a snapshot
- Within a snapshot, the player interaction graph is static
- Between snapshots, exponential decay weights connect the same player's representations

### 6.4 SE-HTGNN: Simple and Efficient Heterogeneous Temporal GNN (2025)

**Paper:** [arXiv:2510.18467](https://arxiv.org/abs/2510.18467)

Achieves up to 10x speedup over state-of-the-art temporal heterogeneous GNNs while maintaining best forecasting accuracy:

- **Core innovation**: Integrates temporal modeling *into* spatial learning via a dynamic attention mechanism that retains attention information from historical graph snapshots to guide subsequent attention computation.
- **No separate temporal module**: Previous methods used decoupled temporal and spatial learning (e.g., GNN + RNN), weakening spatio-temporal interactions. SE-HTGNN unifies them.
- **LLM-prompted type understanding**: Uses LLMs to capture implicit properties of node types as prior knowledge.

**Takeaway:** The unified spatial-temporal attention is elegant and efficient. Rather than separately computing graph attention (spatial) and temporal decay (temporal), a single mechanism does both. This is the architecture we should aspire to for our Level 2 temporal updates.

### 6.5 DyHAN: Dynamic Heterogeneous Attention Networks (2020)

**Paper:** [PMC:7148053](https://pmc.ncbi.nlm.nih.gov/articles/PMC7148053/)

Hierarchical attention for dynamic heterogeneous graphs:
1. **Node-level attention**: How much should node i attend to neighbor j? (Within a single snapshot)
2. **Semantic-level attention**: How important is each edge type? (teammate vs. opponent vs. same-position)
3. **Temporal-level attention**: How important is each historical snapshot? (recent games vs. older games)

The three levels of attention are composed hierarchically. This maps well to our problem where we need to weight: which players matter most (node), which relationship types matter most (semantic), and which time periods matter most (temporal).

### 6.6 Practical Temporal Strategy for Our Model

Given the NBA's temporal structure, we recommend a **hybrid approach**:

1. **Career-level synergy vectors**: Persistent latent vectors for each player, updated with exponential decay. These capture long-term chemistry patterns. Weight: `w(t) = beta^(days_since_game)` with `beta ~ 0.997` (half-life ~230 days, roughly one season).

2. **Season-level snapshots**: Reset/re-weight when a player changes teams. The pre-trade synergy vectors with former teammates are frozen and decay naturally. New teammate synergies begin accumulating.

3. **Recency boost**: Within a season, recent games (last ~20) receive additional weight for the interaction update, capturing short-term rhythm effects.

---

## 7. Player Archetype and Clustering Approaches

### 7.1 Why Archetypes?

The fundamental challenge of Level 2 is data sparsity: most player pairs have insufficient shared court time for reliable interaction estimates. Archetypes solve this by pooling information:

- Instead of learning 124,750 pairwise interactions (500 players), learn interactions between ~10 archetypes: only 45 pairs
- Each archetype represents a *functional role* (e.g., "ball-dominant creator," "3&D wing," "rim-running big")
- Player-level interactions can then be modeled as *deviations from archetype-level interactions*, with the archetype providing a strong prior

This is analogous to Bayesian hierarchical modeling: archetype interactions are the group-level parameters, and player-specific interactions are shrunk toward them.

### 7.2 K-Means Clustering of NBA Players

**Sources:** Multiple recent studies (2022-2025)

The standard approach clusters players using per-minute or per-possession box-score statistics:

**Methodology:**
1. Normalize stats to per-36-minutes or per-100-possessions
2. Apply PCA for dimensionality reduction (typically 90% variance retained at ~16 PCs)
3. Apply K-Means with k chosen by silhouette score or elbow method

**Typical findings with k=8-10 clusters:**

| Cluster | Archetype Name | Key Characteristics |
|---|---|---|
| 1 | Elite Scorer | High PPG, high usage, moderate efficiency |
| 2 | 3-and-D Wing | High 3P%, high steals, low usage |
| 3 | Floor General | High assists, high usage, moderate scoring |
| 4 | Rim Protector | High blocks, high rebounds, low scoring |
| 5 | Stretch Big | High rebounds, moderate 3P%, interior defense |
| 6 | Role Player | Average stats, moderate minutes |
| 7 | Energy Big | High ORB%, high FG% (dunks/layups), low minutes |
| 8 | Two-Way Wing | Balanced offense/defense, versatile |
| 9 | Bench Scorer | Moderate PPG, low minutes, variable efficiency |
| 10 | Specialist | Extreme in 1-2 stats, minimal in others |

**PCA variance explained (typical):**
- PC1 (~39%): Overall offensive production and efficiency
- PC2 (~28%): Perimeter-oriented vs. interior-oriented play
- PC3 (~10%): Playmaking vs. scoring
- PC4 (~7%): Defensive activity (steals + blocks)

### 7.3 Ke, Bian & Chandra (2024) -- Unified ML Framework for Roster Construction

**Paper:** [Applied Soft Computing, Vol 153](https://www.sciencedirect.com/science/article/pii/S1568494624000723)

A two-phase framework combining unsupervised clustering with supervised team optimization:

**Phase 1 (Unsupervised):**
- PCA reduces player features
- K-Means identifies **10 player clusters** (for NBA) and **4 elite clusters**
- Each cluster represents a distinct functional role

**Phase 2 (Supervised):**
- A neural network learns the optimal *combination* of cluster types for winning teams
- Optimal NBA roster: 2 players from each of 2 elite clusters + best available from 2 non-elite clusters
- The model learns that certain cluster combinations are synergistic while others are redundant

**Key insight:** The optimal team isn't the one with the most elite players -- it's the one with the best *combination* of archetypes. This directly supports our interaction modeling approach.

### 7.4 Penner (2025) -- Optimizing Championship Roster Composition

**Paper:** [Frontiers in Sports and Active Living, 2025](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2025.1639431/full)

Large-scale study (22,500 players across 110 leagues) analyzing player archetypes and championship team composition:

- **9 distinct archetypes** identified via K-Means on 13 per-48-min stats
- Championship teams consistently feature balanced archetype distributions
- Top-performing teams overrepresent three specific archetypes:
  - "Aggressive Scorer" (high volume, moderate efficiency)
  - "High Efficiency Scorer" (elite efficiency, moderate volume)
  - "Floor General" (elite assist rate, moderate scoring)
- Multiple linear regression within each archetype predicts per-minute scoring contribution

**Takeaway:** Archetype balance matters more than raw talent concentration. Our interaction model should capture this -- having two "Floor Generals" should show diminishing returns, while one "Floor General" + one "High Efficiency Scorer" should show positive synergy.

### 7.5 How to Use Archetypes in Level 2

**Proposed two-tier interaction architecture:**

**Tier 1: Archetype-level interactions (dense data, strong signal)**
```
For archetype clusters C_a, C_b:
    Synergy_archetype(C_a, C_b) = W_arch[a, b]    # learned K x K interaction matrix
```
This is a small, fully learnable matrix with ~45 parameters (for K=10). It captures general patterns like "point guard + shooting guard = positive synergy."

**Tier 2: Player-level residual interactions (sparse data, refined signal)**
```
For players i (archetype a) and j (archetype b):
    Synergy_player(i, j) = Synergy_archetype(a, b) + <v_i^{residual}, v_j^{residual}>
```

The residual FM interaction captures player-specific deviations from the archetype mean. With Bayesian shrinkage:
- Players with lots of shared data: residual is data-driven
- Players with little shared data: residual is shrunk toward zero (archetype prediction dominates)
- Players who have never played together: pure archetype-level prediction

This gives us reliable predictions even for novel player combinations (e.g., post-trade) while allowing the model to capture specific chemistry effects when data supports it.

---

## 8. Synthesis: Practical Architecture for Level 2

### 8.1 Design Principles (from literature)

| Principle | Source | Implementation |
|---|---|---|
| Dynamic attention over player graphs | Luo & Krishnamurthy 2023, GATv2 | GATv2 attention for teammate interactions |
| FM-style factorized interactions | Rendle 2010, DeepFM 2017 | Latent synergy vectors with dot-product interaction |
| Bayesian shrinkage for sparse pairs | Ferrari & Dunson 2021, RAPM literature | Hierarchical archetype -> player regularization |
| Archetype-based regularization | Maymin 2013, Ke 2024, Penner 2025 | K-Means archetypes as group-level priors |
| Temporal memory per node | TGN (Rossi 2020) | Exponentially decayed synergy vectors |
| Higher-order via aggregation, not enumeration | Tensorized HGNN, DeepFM DNN component | MLP over aggregated lineup embedding |

### 8.2 Recommended Architecture

```
Level 1 Output: player_ability_i in R^d (individual skill vector)
                                |
                                v
        +------------------------------------------+
        |          Level 2: Synergy Module          |
        |                                           |
        |  1. Archetype Assignment:                 |
        |     a_i = softmax(W_arch * player_i)      |
        |     (soft cluster assignment, K=10)       |
        |                                           |
        |  2. Archetype Interaction:                |
        |     S_arch = a_i^T * M * a_j              |
        |     (K x K learned interaction matrix)    |
        |                                           |
        |  3. Player-Specific Residual (FM):        |
        |     S_player = <v_i, v_j>                 |
        |     (k-dim synergy vectors, k=32)         |
        |                                           |
        |  4. GATv2 Message Passing:                |
        |     Attend over teammate graph            |
        |     Aggregate neighbor synergy signals    |
        |                                           |
        |  5. Lineup Aggregation:                   |
        |     h_team = Pool(h_1, ..., h_5)          |
        |     h_higher = MLP(h_team)                |
        |     (implicit higher-order interactions)  |
        +------------------------------------------+
                                |
                                v
              synergy_adjusted_team_embedding in R^d
```

### 8.3 Key Mathematical Formulation

For a game with home lineup H = {h1, ..., h5} and away lineup A = {a1, ..., a5}:

```
# Pairwise teammate synergy (FM-style)
Syn_team(H) = sum_{i<j in H} [a_hi^T M a_hj + <v_hi, v_hj>]

# GATv2 contextualization
for each player i in H:
    h_i = GATv2(player_i, neighbors={H \ {i}})

# Lineup-level higher-order
h_H = MeanPool(h_1, ..., h_5)
h_higher = MLP(h_H)

# Opponent matchup adjustment (secondary)
Matchup(H, A) = sum_{i in H, j in A} sigma(<u_i, u_j>)  # separate opponent vectors u

# Final team embedding
team_embedding(H, A) = base_team(H) + Syn_team(H) + h_higher + Matchup(H, A)
```

### 8.4 Handling the Sparsity Challenge

The literature consistently identifies sparsity as the central challenge. Our multi-level approach addresses it at each layer:

| Sparsity Level | Solution | Data Requirement |
|---|---|---|
| Novel pair (never played together) | Archetype interaction matrix M | Only need archetype assignments |
| Sparse pair (<50 min together) | FM interaction shrunk to archetype | ~10 games of each player's data with any teammates |
| Moderate pair (50-500 min) | FM interaction with partial data | Direct co-occurrence data starts contributing |
| Dense pair (>500 min) | Full FM interaction + GATv2 attention | Enough data for reliable pair-specific estimates |

### 8.5 Temporal Update Strategy

Based on TGN (Section 6.2) and the snapshot approach (Section 6.3):

```
# After each game g at time t:
for each player pair (i, j) who shared the court:
    # Update synergy vectors with exponential decay
    v_i = decay(t) * v_i_prev + (1 - decay(t)) * gradient_update(v_i, loss_g)

    # decay(t) = beta^(t - t_prev), beta ~ 0.997
    # Half-life ~230 days (one season)

# On trade event (player i moves from team A to team B):
    # Former teammates: synergy vectors freeze and decay naturally
    # New teammates: interaction begins from archetype-level prior
    # Player's own synergy vector is preserved (it encodes their interaction style)
```

### 8.6 Parameter Budget Estimate

| Component | Parameters | Notes |
|---|---|---|
| Archetype assignment matrix | d x K = 256 x 10 = 2,560 | Soft clustering |
| Archetype interaction matrix M | K x K = 100 | Symmetric, only 55 unique |
| FM synergy vectors (per player) | k = 32 per player | ~500 active players = 16,000 |
| GATv2 (1 layer, 4 heads) | ~4 x (3 x d x d/4) = 3 x 256^2 = 196,608 | Standard GATv2 |
| Lineup MLP (higher-order) | 256 -> 128 -> 64 = 41,024 | Small overhead |
| Opponent matchup vectors | k_opp = 16 per player = 8,000 | Separate from teammate synergy |
| **Total** | **~264,000** | Modest addition to Level 1 |

---

## Key References

### Primary Papers (Most Relevant)

1. Luo, R. & Krishnamurthy, V. (2023). "Who You Play Affects How You Play: Predicting Sports Performance Using Graph Attention Networks With Temporal Convolution." [arXiv:2303.16741](https://arxiv.org/abs/2303.16741)

2. Maymin, A., Maymin, P. & Shen, E. (2013). "NBA Chemistry: Positive and Negative Synergies in Basketball." International Journal of Computer Science in Sport. [SSRN:1935972](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1935972)

3. Rendle, S. (2010). "Factorization Machines." IEEE International Conference on Data Mining. [PDF](https://www.ismll.uni-hildesheim.de/pub/pdfs/Rendle2010FM.pdf)

4. Guo, H. et al. (2017). "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction." IJCAI 2017. [arXiv:1703.04247](https://arxiv.org/abs/1703.04247)

5. Brody, S., Alon, U. & Yahav, E. (2022). "How Attentive are Graph Attention Networks?" ICLR 2022. [arXiv:2105.14491](https://arxiv.org/abs/2105.14491)

### Graph and Temporal Networks

6. Rossi, E. et al. (2020). "Temporal Graph Networks for Deep Learning on Dynamic Graphs." [arXiv:2006.10637](https://arxiv.org/abs/2006.10637)

7. DURENDAL (2023). "Graph deep learning framework for temporal heterogeneous networks." [arXiv:2310.00336](https://arxiv.org/abs/2310.00336)

8. SE-HTGNN (2025). "Simple and Efficient Heterogeneous Temporal Graph Neural Network." [arXiv:2510.18467](https://arxiv.org/abs/2510.18467)

9. Fan, Y. & Ju, M. (2021). "Heterogeneous Temporal Graph Neural Network." [arXiv:2110.13889](https://arxiv.org/abs/2110.13889)

10. HIGFormer (2025). "Player-Team Heterogeneous Interaction Graph Transformer for Soccer Outcome Prediction." KDD 2025. [arXiv:2507.10626](https://arxiv.org/abs/2507.10626)

### Hypergraph and Higher-Order

11. Feng, Y. et al. (2019). "Hypergraph Neural Networks." AAAI 2019. [arXiv:1809.09401](https://arxiv.org/abs/1809.09401)

12. Wang, M. et al. (2024). "Tensorized Hypergraph Neural Networks." SDM 2024. [arXiv:2306.02560](https://arxiv.org/abs/2306.02560)

13. HypOp (2024). "Distributed Constrained Combinatorial Optimization leveraging Hypergraph Neural Networks." Nature Machine Intelligence. [arXiv:2311.09375](https://arxiv.org/abs/2311.09375)

### Bayesian and Interaction Inference

14. Agrawal, R. et al. (2019). "The Kernel Interaction Trick: Fast Bayesian Discovery of Pairwise Interactions in High Dimensions." ICML 2019. [arXiv:1905.06501](https://arxiv.org/abs/1905.06501)

15. Ferrari, F. & Dunson, D. (2021). "Bayesian Factor Analysis for Inference on Interactions." JASA 116(535), 1521-1532. [arXiv:1904.11603](https://arxiv.org/abs/1904.11603)

16. Elmore, R. et al. (2024). "Expected Points Above Average: A Novel NBA Player Metric Based on Bayesian Hierarchical Modeling." [arXiv:2405.10453](https://arxiv.org/abs/2405.10453)

### Player Archetypes and Roster Construction

17. Ke, Y., Bian, R. & Chandra, R. (2024). "A unified machine learning framework for basketball team roster construction: NBA and WNBA." Applied Soft Computing 153, 111298. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1568494624000723)

18. Penner (2025). "Player archetypes within basketball: optimizing roster composition to create a championship team." Frontiers in Sports and Active Living. [Frontiers](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2025.1639431/full)

19. Wei, B. et al. (2025). "Enhancing team dynamics: Unveiling synergy effects through social network analysis in professional basketball." Acta Psychologica. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0001691825010601)

### Sports GNN Surveys and Applications

20. Zhao, H. et al. (2023). "Enhancing Basketball Game Outcome Prediction through Fused GCN and Random Forest Algorithm." Entropy 25(5), 765. [MDPI](https://www.mdpi.com/1099-4300/25/5/765)

21. TacticExpert (2025). "Spatial-Temporal Graph Language Model for Basketball Tactics." [arXiv:2503.10722](https://arxiv.org/abs/2503.10722)

22. Hamilton, W.L., Ying, R. & Leskovec, J. (2017). "Inductive Representation Learning on Large Graphs (GraphSAGE)." NeurIPS 2017. [arXiv:1706.02216](https://arxiv.org/abs/1706.02216)

23. Sports Analytics with GNN Survey (2024). [Preprints.org](https://www.preprints.org/manuscript/202410.0046/v1)

### NBA Analytics Background

24. Medvedovsky, K. DARKO system and lineup stabilization research. [NBAStuffer](https://www.nbastuffer.com/analytics101/regularized-adjusted-plus-minus-rapm/)

25. Squared Statistics. "Applying Tensors to Find Optimal Match-Ups in the NBA." [Blog](https://squared2020.com/2017/07/14/applying-tensors-to-find-optimal-match-ups-in-the-nba/)
