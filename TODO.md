# NBA AI TODO

> **Last Updated**: March 18, 2026
> **Current Phase**: Phase 4 Complete — Generative Autoregressive

---

## Results Summary

| Phase | Best Model | Spread MAE | Win AUC | Win Acc |
|-------|-----------|-----------|---------|---------|
| Phase 1 | Exp 13 (combined) | 12.20 | 0.592 | 57.6% |
| Phase 2 | Exp 5a (fusion residual) | 11.61 | 0.687 | — |
| Phase 3 | **Exp 9 (deep ensemble)** | **10.66** | **0.718** | **66.5%** |
| Phase 4 | Exp 5b (full context + outcome head) | 11.74 | 0.662 | 61.7% |
| Phase 4 | Exp 7 (GPT-5.4-mini LLM) | 11.28 | **0.718** | 65.9% |
| Phase 4 | Exp 7 (GPT-5.4 LLM, 1/3 sample) | 11.16 | **0.726** | **66.6%** |

**Overall best spread**: Phase 3 Exp 9 ensemble (MAE 10.66). **Overall best win prediction**: Phase 4 Exp 7 GPT-5.4 (AUC 0.726, Acc 66.6%). LLM matches/beats our best custom models on win classification but lags ~0.5-1.1 on spread MAE due to variance compression.

---

## Phase 3: Alternative Architectures & Data (Complete — 10/10 experiments)

- [x] Exp 1: Time-aware bidirectional GRU — no improvement (MAE 11.72)
- [x] Exp 2: Self-supervised pre-training — no improvement (MAE 11.84)
- [x] Exp 3a: Full PlayerBox (16 stats + position) — MAE 11.48, AUC 0.707
- [x] Exp 3b: + Extended data (15 seasons) — MAE 11.03, AUC 0.685 (spread ↓, win ↓)
- [x] ~~Exp 3c: + Wider model~~ — skipped (overfitting risk)
- [x] Exp 4: Player interaction self-attention — MAE 10.83, AUC 0.705, ECE 0.0142
- [x] Exp 4b: Multi-query player pooling — no improvement (MAE 10.92)
- [x] Exp 5: Heterogeneous player-game graph — best spread (MAE 10.61), win ↓
- [x] Exp 6: HIGFormer-inspired (pre-training + team GAT) — regressed (MAE 11.52)
- [x] Exp 7: Kitchen sink features (efficiency + GS summaries + flags) — marginal (MAE 10.77), win ↓
- [x] Exp 8: Hybrid transformer + XGBoost — no improvement (MAE 10.85, AUC 0.706)
- [x] Exp 9: Deep ensemble (3 seeds) — **OVERALL BEST** (MAE 10.66, AUC 0.718, Acc 66.5%)

**Key insight**: Variance reduction via ensembling broke the 0.706 AUC ceiling. Five single-model experiments (Exp 1 GRU, Exp 2 pretrain, Exp 6 pretrain+GAT, Exp 7 features, Exp 8 hybrid) all failed to improve both MAE and AUC simultaneously.

## Phase 4: Generative Autoregressive Game State Prediction (Complete — 7 experiments)

- [x] Exp 1: Baseline in-context conditioning (32.3M params) — MAE ~12.5, AUC 0.558 (posterior collapse)
- [x] Exp 1b: Bug fixes + FP32 + ContextScoreBias — AUC 0.565 (context encoder overfit)
- [x] Exp 2: adaLN-Zero + CFG + pre-decoder heads (43.3M params) — MAE 12.62, AUC 0.582
- [x] Exp 3: Simplified context (rolling stats, 157K encoder) + scheduled sampling — MAE 12.28, AUC 0.583
- [x] Exp 4: Scoring-event compression (~110 vs ~487 states) — MAE 11.76, AUC 0.662
- [x] Exp 4b: Uniform class weights ablation — MAE 11.87, AUC 0.667
- [x] Exp 5: Full-context encoder (Phase 3 player-aware, 38.5M params) + outcome head + rules engine — MAE 11.74, AUC 0.662
- [x] Exp 5b: Clock-delta formulation (softplus positive increments) — **BEST** (MAE 11.74, AUC 0.662, fixed rollout drift)

**Progression**: In-context conditioning → adaLN-Zero (fixed decoder ignoring context) → simplified encoder (fixed overfitting) → event compression (eliminated 77% no-score waste) → full context + outcome head (restored Phase 3 data) → clock-delta (fixed rollout monotonicity).

**Key findings**:
- Teacher-forced autoregressive models suffer posterior collapse with prepended context tokens
- adaLN-Zero (DiT-style) forces context usage by modulating every LayerNorm
- 13.1M complex context encoder overfits badly; 157K rolling-stats MLP generalizes
- Scoring-event compression (487→110 steps) gave largest single improvement (+0.079 AUC)
- Outcome head (direct Gaussian spread prediction) outperforms autoregressive rollout for spread MAE
- Clock-delta formulation (softplus positive increments) solves rollout clock stalling; absolute clock predictions oscillate under autoregressive drift
- In-game prefix predictions improve strongly: halftime MAE 10.53 (beats Phase 3 ensemble), end-Q3 MAE 8.93
- Mask convention bug: PyTorch `key_padding_mask` uses True=ignore; collate used True=valid → all context was masked. Fixed by negating masks in FullContextEncoder
- Autoregressive generation is fundamentally harder than direct prediction (~1.1 MAE gap)

### Exp 6: Pre-trained Foundation Models (Planned)

- [ ] Exp 6a: TabPFN 2.5 — tabular foundation model, 62 features/game, zero-shot + fine-tuned (`scripts/exp6a_tabpfn.py`)
- [ ] Exp 6b: Chronos-2 — time-series foundation model with group attention for cross-team matchup dynamics (`scripts/exp6b_chronos2.py`)

### Exp 7: LLM API Prediction (Complete — 3 model tiers)

- [x] gpt-5.4-nano (2120 games, $1.92) — MAE 11.80, AUC 0.693, Acc 65.2%
- [x] gpt-5.4-mini (2120 games, $6.65) — MAE 11.28, AUC 0.718, Acc 65.9%, **ECE 0.0195**
- [x] gpt-5.4 (707 games, $7.64) — MAE 11.16, **AUC 0.726**, **Acc 66.6%**

**Key findings**:
- GPT-5.4-mini matches Phase 3 Exp 9 ensemble on win AUC (0.718) for $6.65
- GPT-5.4 premium beats it (AUC 0.726, Acc 66.6%) — new best win prediction
- Best calibration ever: mini ECE 0.0195 (all bins within 0.04 gap)
- Spread MAE lags by 0.5-1.1 — LLM compresses spread variance (predicted std ~8-10 vs actual ~16)
- Leakage signal: 2024-25 MAE ~0.5-0.9 better than 2025-26 across all tiers (could be partial-season effect)
- LLM has unique advantage: player name recognition + target game roster knowledge
- Total cost for all 3 tiers: $16.21

## Phase 5: Hierarchical Player-to-Game Prediction (Planned)

Goal: bottom-up prediction through 4 hierarchical levels — player → synergy → team → game. Completely separate architecture from Phases 1-4.

### Research
- [ ] Level 1: Player rating systems (EPM/RAPM/DARKO), aging curves, cold-start/rookie projection
- [ ] Level 2: Graph network design, lineup data parsing feasibility, synergy modeling
- [ ] Level 3: Coaching effects, team residual quantification
- [ ] Level 4: Game context features, home court advantage trends
- [ ] Data pipeline: PBP lineup parsing, NBA API player attributes, external sources
- [ ] Architecture: model design per level, inter-level interfaces, training strategy

### Implementation
- [ ] Data: Parse PBP_Logs for lineup/stint data
- [ ] Data: Fetch player attributes (height, weight, age, draft) from NBA API
- [ ] Level 1: Player ability model (hierarchical pre-training)
- [ ] Level 2: Player synergy graph network
- [ ] Level 3: Team residual model
- [ ] Level 4: Game context integration → spread prediction
- [ ] Evaluation: Compare vs Phase 3 Exp 9 ensemble (MAE 10.66, AUC 0.718)

## Phase 6: Final Integration — Betting & ATS (Planned)

Goal: shift from Spread MAE to ATS win rate (beating the Vegas spread). The capstone phase.

- [ ] Exp 1: Include betting data (Vegas spread/total/ML) as transformer features
- [ ] Exp 2: Inject engineered features (43 or 63) into transformer architecture
- [ ] Exp 3: Maximize XGBoost — Optuna-tuned with full feature stack (engineered + betting + transformer embeddings)
- [ ] Exp 4: ATS metric integration — evaluate all models on ATS%, profit/loss at -110, ROI%

**Key data**: Vegas closing spreads available 2007-2026 via `espn_current_spread` (historic) + `covers_closing_spread` (recent). No backfill needed. Vegas MAE ~9.45, our best ~10.66.

**Success criteria**: ATS > 52.4% (breakeven at -110 juice) on test set.

## Future Avenues

- **Player props** — predict individual player statistics
- **Live prediction** — in-game win probability using real-time play-by-play

---

## Completed Sprints

### Sprint 33: Phase 4 Exp 5 — Full Context + Outcome Head + Clock-Delta (Mar 15-18, 2026)

**Summary**: Restored Phase 3 player-aware context encoder (FullContextEncoder, 8.2M params), added Gaussian outcome head for per-position spread prediction, rules engine for deterministic game termination, 6-class events (no game_end), 18-dim enriched state vectors, and overtime handling. Fixed critical mask inversion bug (True=valid vs PyTorch True=ignore) and NaN from empty attention pooling. Implemented clock-delta formulation (softplus positive increments) to fix rollout clock stalling.

**Bugs fixed**:
- `game_mask` / `player_mask` convention: collate used True=valid, but PyTorch `key_padding_mask` expects True=ignore. FullContextEncoder was ignoring all valid games and attending to padding. Fixed by negating masks.
- NaN propagation: games with all-padded players → NaN from empty attention pooling → `0*NaN=NaN` contaminated temporal encoder. Fixed with `torch.nan_to_num`.
- Clock drift: absolute clock predictions oscillated during rollout (training was teacher-forced). Fixed with clock-delta formulation (predict positive increment via softplus instead of absolute progress).

**Result**: Exp 5b is best generative model — MAE 11.74, AUC 0.662, Win Acc 61.7%. Outcome head pre-game prediction outperforms autoregressive rollout. In-game prefix predictions strong: halftime MAE 10.53, end-Q3 MAE 8.93. Rollout score distribution still collapsed (home+2 dominance) — class imbalance remains for future work.

### Sprint 32: Phase 4 Exps 3-4 — Simplified Context + Compression (Mar 14, 2026)

**Summary**: Two experiments targeting identified bottlenecks. Exp 3 replaced the 13.1M complex context encoder with 157K-param rolling-stats MLP (24 features per team), solving context overfitting (train/val gap: 11x → 1.03x). Exp 4 eliminated 77% no-score states by training only on ~110 scoring events per game, with 8-dim state vectors including inter-event time.

**Result**: Exp 4 is best generative model — MAE 11.76, AUC 0.662, Win Acc 61.4%. Exp 4b ablation (uniform weights) showed model learns well-calibrated class distributions regardless of weights. Still ~1.1 MAE gap to Phase 3 ensemble.

### Sprint 31: Phase 4 Exp 1b + Exp 2 — Fixes + adaLN-Zero (Mar 13-14, 2026)

**Summary**: Exp 1b fixed AMP NaN instability (→FP32), added ContextScoreBias shortcut, reduced context dropout. Exp 2 redesigned conditioning: replaced in-context token prepending with adaLN-Zero (DiT-style adaptive LayerNorm), added Classifier-Free Guidance (1.5x), and pre-decoder auxiliary heads for direct gradient to context encoder. 43.3M params.

**Result**: Exp 1b marginal (AUC 0.565); context encoder memorized training set. Exp 2 confirmed adaLN-Zero fixes structural weakness (AUC 0.582), but exposed context encoder overfitting as the new bottleneck (train/val gaps: 3-11x).

### Sprint 30: Phase 4 Exp 1 — Autoregressive Baseline (Mar 12, 2026)

**Summary**: Built complete generative framework (`src/generative/`) — context encoder (13.1M params reusing Phase 2/3 player+temporal encoding), causal decoder (18.9M params, 6 layers), state embedder, and prediction heads for 7-class score events + clock regression. In-context conditioning via prepended context tokens. 32.3M total params.

**Result**: MAE ~12.5, AUC 0.558 (near random). Teacher forcing caused posterior collapse — decoder learned to predict from state history alone, ignoring context tokens. NaN instability at epoch 16 (AMP on RTX 2070 SUPER).

### Sprint 29: Phase 3 Exps 5-9 (Mar 5-10, 2026)

**Summary**: Completed remaining Phase 3 experiments. Exp 5 (heterogeneous graph) achieved best single-model spread (MAE 10.61) but win metrics regressed. Exp 6 (HIGFormer with pre-training + team GAT) regressed. Exp 7 (kitchen sink features) marginal. Exp 8 (hybrid transformer+XGBoost) no improvement. Exp 9 (deep ensemble of 3 Exp 4 seeds) broke the AUC ceiling.

**Result**: Exp 9 is overall best — MAE 10.66, AUC 0.718, Win Acc 66.5%. Only approach to improve both MAE and AUC simultaneously. ECE regressed (0.0378) — fixable with temperature scaling.

### Sprint 28: Phase 3 Exp 5 — Heterogeneous Graph (Mar 5, 2026)

**Summary**: Two-pass message passing (Game→Player + Player→Game) with roster context, trajectory attention, and reinjection cross-attention.

**Result**: Best single-model spread MAE 10.61, but win classification regressed.

### Sprint 27: Phase 3 Exp 4b — Multi-Query Player Pooling (Mar 4, 2026)

**Summary**: Replaced single learned pool query with 4 queries (concat + Linear(1024,256) + LN) in PlayerContributionEncoder. Inspired by temporal module's 8-query MultiQueryAttentionPool. ~263K new params (+0.7%), ~40.3M total. Backward compat via `player_contribution_n_pool_queries=1` default.

**Result**: No improvement. Test: Spread MAE 10.92 (+0.09), RMSE 15.00, Win Acc 64.0% (-1.1pp), AUC 0.685 (-0.020), Brier 0.2255, ECE 0.0349, 90% Coverage 79.1%. Every metric regressed vs Exp 4. Best val MAE 10.53 (raw) at epoch 13, early stopped at epoch 23. Single query is already sufficient for collapsing 15 players — unlike temporal pooling over 82 games, player pooling operates on a small homogeneous set where multi-query fragments the representation without benefit.

### Sprint 26: Phase 3 Exp 4 — Player Interaction Self-Attention (Mar 3-4, 2026)

**Summary**: Added 1-layer TransformerEncoder self-attention (256-d, 4 heads, FF=1024, pre-norm, GELU) between players within each historical game, before attention pooling. ~790K new params (+2% over 3a), ~40M total. Uses 15 training seasons (like 3b). Players now exchange information before pooling, enabling the model to learn player complementarity (e.g., "LeBron + AD together" produces a different representation than encoding them independently).

**Result**: **NEW BEST** across spread metrics while recovering win classification. Spread MAE 10.83 (-0.20 vs 3b, -0.65 vs 3a), RMSE 14.56, Home MAE 9.33, Away MAE 9.46. Win Accuracy 65.1% (recovered from 3b's 62.7%, near 3a's 65.3%), Win AUC 0.705 (recovered from 3b's 0.685, near 3a's 0.707), Brier 0.2180, ECE 0.0142 (best calibration ever), 90% Coverage 82.2%. Best val MAE 10.83 at epoch 9, manually stopped at epoch 15 (overfitting). Key insight: Player interaction fixed 3b's win classification regression while keeping the spread improvement from more data.

### Sprint 25: Phase 3 Exp 3 — Full PlayerBox Integration (Mar 2-3, 2026)

**Summary**: Expanded PlayerContributionEncoder from 1 stat (points) to 16 box score stats (min, pts, oreb, dreb, ast, stl, blk, tov, pf, fga, fgm, fg3a, fg3m, fta, ftm, plus_minus) with position embedding (Guard/Forward/Center/Unknown → 8-d) and plus_minus availability indicator. Stat MLP processes [16 stats + pm_avail] → 64-d, concatenated with player_embed(128) + position(8) = 200 → Linear(200, 256). PlayerFormEncoder similarly expanded to full stats. Three sub-experiments: 3a (feature isolation), 3b (data scaling), 3c (capacity scaling).

**Exp 3a Result**: First improvement beyond the ~11.6 MAE plateau. Spread MAE 11.48 (-0.13), Win AUC 0.707 (+0.020), Win Accuracy 65.3% (+2.3pp), Brier 0.2204 (-0.007), 90% Coverage 78.2% (+5.5pp). Best val MAE 11.14. Confirms feature ceiling was the bottleneck — rebounds, assists, defense, shooting efficiency, and plus/minus provide signal not learnable from game scores alone.

**Exp 3b Result**: Extended to 15 training seasons (2008-2023). Best val MAE 10.62 at epoch 10, test MAE 11.03 (-0.45), AUC 0.685 (-0.022), Win Acc 62.7% (-2.6pp). Early stopped manually at epoch 15 (overfitting — train loss dropping, val loss climbing since epoch 10). More data improved spread significantly but win classification regressed. Exp 3c (wider model) skipped — overfitting with 39M params means 65M would be worse.

### Sprint 24: Phase 3 Exp 2 — Self-Supervised Pre-Training (Mar 1-2, 2026)

**Summary**: BERT-style masked reconstruction on 31K games (25 seasons, 2001-2026). 40% masking, predict team_score/opp_score/margin at masked positions. Pre-trains player_embed + per_game_encoder + temporal_attention. Fine-tune with 3-phase gradual unfreezing (frozen → top block → all) and discriminative LR (0.9x/layer).

**Result**: No improvement. Pre-training converged in 11 epochs (~23s) on 687 team-season samples. Fine-tuning best at epoch 8: val MAE 11.61 (matches baseline), val AUC 0.678 (below 0.687). Test: MAE 11.84, AUC 0.669. Full unfreezing (epochs 11-23) did not improve. Pre-trained representations provide no advantage over random initialization. Combined with Exp 1, confirms the bottleneck is features/data, not encoder quality.

### Sprint 23: Phase 3 Exp 1 — GRU Temporal (Feb 27-28, 2026)

**Summary**: Replaced temporal transformer with a time-aware bidirectional GRU (2 layers, 256 hidden per direction, 64-d calendar-distance embedding, 4-query attention pool). 31M params vs 39M.

**Result**: No improvement. Test MAE 11.72 (vs Exp 5a baseline 11.61), Win AUC 0.677 (vs 0.687). Validation MAE matched baseline (11.34) but didn't generalize — slight overfitting. Temporal module confirmed as not the bottleneck.

### Sprint 22: Phase 2 Final Experiments (Feb 26-27, 2026)

**Summary**: Ran Exp 6 (derived spread + model reduction) and Exp 7 (PLE + cross-attention fusion + Huber loss). Exp 6 regressed to MAE 11.84. Exp 7 regressed to MAE 11.73 / AUC 0.669 (vs Exp 5a: 11.61 / 0.687).

**Result**: Phase 2 plateau confirmed at ~11.6 MAE across 7 experiments. Best model: Exp 5a (Spread MAE 11.61, Win AUC 0.687). Standard transformer approaches exhausted. Transitioned to Phase 3.

### Sprint 21: Phase 2 Implementation & Experimentation (Feb 21-26, 2026)

**Summary**: Designed and implemented the Phase 2 architecture — full-season player-aware transformer with per-game embeddings, attention pooling, and calendar-distance positional encoding. Ran 5 experiments (Exps 1-5a) exploring context length, player form encoding, loss tuning, and fusion fixes.

**Result**: Best model Exp 3 (enhanced) — Spread MAE 11.67 (-0.53 vs Phase 1), Win AUC 0.682 (+0.090 vs Phase 1), Win Acc 63.1% (+5.5pp). Substantial improvement over Phase 1 ceiling. Gap to XGBoost (MAE ~10.1) reduced from ~2.1 to ~1.6.

**Key findings**:
- Full-season context + per-player point contributions broke Phase 1 ceiling
- Player form encoder and team-relative scores improved all metrics (Exp 3)
- Sigma cap and aggressive MSE weight overcorrected (Exp 4 regressed)
- Fusion bottleneck identified: top singular value captures 61% of variance

### Sprint 20: Phase 1 Finalization (Feb 17-21, 2026)

**Summary**: Completed Phase 1b/c, combined models, roster-only ablation, documentation consolidation. 15 experiments total.

**Result**: Best model Exp 13 (combined_v1) — Spread MAE 12.20, Win AUC 0.592, Win Acc 57.6%. See ARCHITECTURE.md.

### Sprint 19: Phase 1a Experimentation (Feb 14-17, 2026)

**Summary**: Trained baseline models and ran ablation studies. Found Spread MAE ceiling at ~12.2.

**Completed**: 6 experiments (baseline v1/v2, N=10 history, 10 seasons, shot features 2/5 seasons)

### Sprint 18: Phase 1 Implementation (Feb 2-14, 2026)

**Summary**: Built complete Phase 1a transformer architecture, data pipeline, training infrastructure, and evaluation tools.

**Completed**:

- Event tokenizer: 10-component tokens from PBP_Logs + GameStates
- Sequence builder: historical game sequences with LRU caching
- PyTorch Dataset/DataLoader with chronological train/val/test splits
- Event Encoder: 4-layer pre-norm Transformer (8 heads, FF=1024)
- Temporal Attention: 2-layer cross-game attention with learned positions
- SimpleFusion MLP: concatenate home + away histories -> matchup representation
- Probabilistic prediction heads: Gaussian (mu, sigma) for spread and scores, win prob via CDF
- Combined loss: NLL + MSE + BCE with configurable weights
- Metrics: MAE, RMSE, AUC, Brier, ECE, coverage, CRPS
- Training loop: AdamW, cosine schedule, early stopping, gradient accumulation, AMP
- Evaluation pipeline: test set evaluation, ablation runner, visualization tools
- YAML-based experiment configuration system

**Architecture**: ~5.7M parameters, two-stream PBP history -> SimpleFusion -> probabilistic heads

### Sprint 17: GenAI Predictor Design (Jan 10 - Feb 2, 2026)

**Summary**: Researched and designed neural architecture for NBA prediction.

**Completed**:
- Evaluated 4 architectures: Encoder-Only Transformer, PTIN, HGT, NBAFM
- Selected Encoder-Only Transformer with direct prediction heads
- Established "minimal context" philosophy: no feature engineering, learn from raw sequences
- Designed phased approach: Phase 1 (supervised) -> Phase 2 (pre-training) -> Phase 3 (foundation)
- Documented architecture decisions in ARCHITECTURE.md

### v0.4.0 Release (Jan 10, 2026)

**Summary**: Public pre-release with complete setup automation and documentation.

### Sprint 16b: Datetime & Timezone Overhaul (Dec 27, 2025)

- Central datetime utilities, Eastern time for NBA operations, user timezone detection

### Sprint 16: Frontend API Optimization (Dec 25, 2025)

- Page load: ~9s -> <0.4s (95%+ improvement), query optimization, live game status sync

### Sprint 15: Pipeline Optimization & Database Consolidation (Dec 19-25, 2025)

- 3-database architecture (current subset of dev subset of full), unified schema, 210 tests passing

### Sprint 13: Cleanup & Testing (Dec 6, 2025)

- Consolidated CLI tools, workflow-aware validation, 14 frontend tests passing

### Sprint 12: Database Consolidation (Dec 6, 2025)

- Schema unification, betting backfill (2021-2023), data availability audit

### Sprint 11.5: Betting Data Integration (Dec 5-6, 2025)

- 3-tier betting system (ESPN -> Covers), 36 tests, 100% coverage for 2024-25

### Sprint 11: Data Infrastructure (Dec 3-4, 2025)

- NBA Official injury PDFs, simplified Players table, 97.6% player ID matching

### Sprint 10: Public Release v0.2.0 (Nov 27, 2025)

- Public release, dependency upgrades, 75 tests passing

### Sprint 9: Traditional ML Models (Nov 26, 2025)

- Ridge/XGBoost/MLP/Ensemble predictors, model registry

### Sprints 1-8 (Nov 24-26, 2025)

- Infrastructure cleanup, prediction engine, live data, database consolidation, web app, data collection
