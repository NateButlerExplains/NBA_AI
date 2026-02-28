# NBA AI TODO

> **Last Updated**: February 28, 2026
> **Current Sprint**: Between sprints — planning Phase 3 Exp 2

---

## Backlog

### Phase 3: Alternative Architectures (1/6 experiments complete)

- [x] Exp 1: Time-aware bidirectional GRU — no improvement (MAE 11.72 vs 11.61)
- [ ] Exp 2: Self-supervised pre-training + fine-tune (masked game prediction on 33K games)
- [ ] Exp 3: Multi-stat player contributions (10 stats + position — blocked on box score backfill)
- [ ] Exp 4: Player interaction graph (self-attention between players within games)
- [ ] Exp 5: Full heterogeneous graph (HIGFormer-inspired multi-pass architecture)
- [ ] Exp 6: Best-of-everything (combine winners)

### Future Avenues

- **Player props model** — predict individual player statistics using PlayerBox data
- **Live prediction** — in-game win probability and score prediction using real-time play-by-play
- **Generative next-state prediction** — predict the next game state rather than final scores

### Cross-Cutting Concerns

- **Data utilization** — expand to 20+ seasons of historical data (2000-2024)
- **Compute scaling** — larger models, longer training, multi-GPU support
- **Temporal freshness / continuous learning** — online updates as new games are played, handle distribution shift across seasons
- **Player/team signal preservation** — better roster encoding, injury impact modeling, player embeddings that transfer across teams

### Phase 1 Improvements (deferred)

- Add spread/score consistency loss
- Wire CRPS into MetricResults
- Add comparison against Vegas closing lines (requires betting data integration)

### Other

- Historical data backfill: PlayerBox/TeamBox (2000-2022), InjuryReports (Dec 2018-2023)

---

## Completed Sprints

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
