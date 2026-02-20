# Phase 1 Experiment Log

> **Hardware**: RTX 2070 SUPER (8GB VRAM)
> **Test Set**: 2024-2025 + 2025-2026 seasons
> **XGBoost Baseline**: Spread MAE ~10.1, Avg Score MAE ~10.1

---

## Summary Table

### PBP Model (Two-stream PBP Transformer → SimpleFusion → Probabilistic Heads)

| # | Experiment | Config | Spread MAE | Avg Score MAE | Win Acc | Win AUC | Val Loss | Best Epoch | Params |
|---|-----------|--------|-----------|---------------|---------|---------|----------|------------|--------|
| 1 | Baseline v1 (5 seasons) | `full_baseline.yaml` | 12.20 | — | — | — | 38.50 | 5 | 5.7M |
| 2 | Baseline v2 (tuned reg) | `baseline_v2.yaml` | 12.26 | — | — | — | 38.49 | 9 | 5.7M |
| 3 | N=10 history games | `ablation_seq10.yaml` | 12.24 | — | — | — | 38.46 | 5 | 5.7M |
| 4 | 10 training seasons | `ablation_10seasons.yaml` | 12.74 | — | — | — | 37.57 | 10 | 5.7M |
| 5 | Shot features (2 seasons) | `fast_2season.yaml` | 12.35 | 10.09 | 52.0% | 0.535 | 39.07 | 13 | 5.85M |
| 6 | Shot features (5 seasons) | `shot_features_5season.yaml` | 12.29 | 10.15 | 52.7% | 0.567 | 38.63 | 12 | 5.85M |

### GameStates Model (Score trajectory only → SimpleFusion → Probabilistic Heads)

| # | Experiment | Config | Spread MAE | Avg Score MAE | Win Acc | Win AUC | Val Loss | Best Epoch | Params |
|---|-----------|--------|-----------|---------------|---------|---------|----------|------------|--------|
| 7 | GameStates (2 seasons) | `gamestates_2season.yaml` | 12.36 | 10.06 | 52.0% | 0.496 | 39.00 | 50 | 5.59M |
| 8 | GameStates (5 seasons) | `gamestates_5season.yaml` | 12.36 | 10.14 | 52.0% | 0.509 | 39.06 | 50 | 5.59M |

### Schedule Model (PBP + Temporal Embeddings → SimpleFusion → Probabilistic Heads)

| # | Experiment | Config | Spread MAE | Avg Score MAE | Win Acc | Win AUC | Val Loss | Best Epoch | Params |
|---|-----------|--------|-----------|---------------|---------|---------|----------|------------|--------|
| 9 | Schedule (2 seasons) | `schedule_2season.yaml` | 12.41 | 10.14 | 51.9% | 0.512 | 39.19 | 9 | 5.93M |
| 10 | Schedule (5 seasons) | `schedule_5season.yaml` | 12.24 | 10.19 | 55.9% | 0.578 | 38.81 | 10 | 5.93M |

**Best Spread MAE**: 12.20 (Experiment 1 — PBP Baseline v1)
**Best Avg Score MAE**: 10.06 (Experiment 7 — GameStates, 2 seasons)

---

## Experiment Details

### Experiment 1: Baseline v1 (Full Baseline)

**Config**: `configs/transformer/full_baseline.yaml`
**Date**: Feb 14, 2026

- **Tokens**: 8 components (actionType, subType, period, clock, team_indicator, score_diff, player, shot_result)
- **Training**: 5 seasons (2018-2023), dropout=0.1, weight_decay=0.01, LR=1e-4, grad_accum=8
- **Result**: Spread MAE 12.20, val_loss 38.50, best epoch 5
- **Notes**: Overfitting began early (epoch 5). Train loss kept dropping while val loss diverged. Indicated need for stronger regularization.

### Experiment 2: Baseline v2 (Tuned Regularization)

**Config**: `configs/transformer/baseline_v2.yaml`
**Date**: Feb 14, 2026

- **Changes from v1**: dropout 0.1→0.3, weight_decay 0.01→0.05, LR 1e-4→5e-5, grad_accum 8→16, patience 15→20
- **Result**: Spread MAE 12.26, val_loss 38.49, best epoch 9
- **Notes**: Regularization extended useful training (5→9 epochs) and slightly improved val_loss, but Spread MAE was marginally worse. The model trains longer without overfitting as quickly.

### Experiment 3: N=10 History Games

**Config**: `configs/transformer/ablation_seq10.yaml`
**Date**: Feb 14, 2026

- **Changes**: n_history_games 5→10
- **Result**: Spread MAE 12.24, val_loss 38.46, best epoch 5
- **Notes**: More history didn't help meaningfully. The temporal attention may not be extracting useful signal beyond the most recent few games, or the additional games add noise.

### Experiment 4: 10 Training Seasons

**Config**: `configs/transformer/ablation_10seasons.yaml`
**Date**: Feb 14, 2026

- **Changes**: Training on 10 seasons instead of 5
- **Result**: Spread MAE 12.74, val_loss 37.57, best epoch 10
- **Notes**: More training data actually hurt test Spread MAE despite improving val_loss. Likely because older seasons (pre-2018) have different play styles and the model learns patterns that don't generalize to 2024-2025.

### Experiment 5: Shot Features (2-Season Fast Test)

**Config**: `configs/transformer/fast_2season.yaml`
**Date**: Feb 17, 2026

- **Changes**: Added shot_distance_bucket (11 bins) and shot_modifier_id (37 entries from descriptor field) as tokens #9 and #10. Trained on 2 seasons only for faster iteration.
- **Token dim**: 224 (up from 192), projected to 256
- **Result**: Spread MAE 12.35, Avg Score MAE 10.09, val_loss 39.07, best epoch 13
- **Notes**: Avg Score MAE of 10.09 beats XGBoost baseline (10.1). Model trained longer (best epoch 13 vs 5-9 in baselines), suggesting the shot features give more to learn. Spread MAE slightly worse due to less training data.

### Experiment 6: Shot Features (5 Seasons)

**Config**: `configs/transformer/shot_features_5season.yaml`
**Date**: Feb 17, 2026

- **Changes**: Same shot features as Exp 5, but full 5-season training data
- **Result**: Spread MAE 12.29, Avg Score MAE 10.15, Win Acc 52.7%, Win AUC 0.567, val_loss 38.63, best epoch 12
- **Training curve**: Val spread MAE reached 12.05 at epoch 12 during training, but test set result was 12.29. Overfitting began after epoch 12.
- **Notes**: Shot features didn't improve Spread MAE vs baseline v1 (12.29 vs 12.20). The model is hitting a ceiling around ~12.2 Spread MAE regardless of PBP token enrichment.

### Experiment 7: GameStates (2 Seasons)

**Config**: `configs/transformer/gamestates_2season.yaml`
**Date**: Feb 18, 2026

- **Model**: GameStates-only (score trajectory) — no PBP, no players, no action types
- **Features**: 5 per row (period, clock_bucket, home_score_bucket, away_score_bucket, margin_bucket)
- **Embedding**: 5×16-d = 80-d → project to 256-d → 4-layer Transformer → mean pool
- **Training**: 2 seasons (2021-2023), same regularization as Exp 5 (dropout=0.3, weight_decay=0.05)
- **Result**: Spread MAE 12.36, Avg Score MAE 10.06, Win Acc 52.0%, Win AUC 0.496, val_loss 39.00, ran all 50 epochs
- **Notes**: Nearly identical Spread MAE to PBP 2-season (12.36 vs 12.35). Val loss converged slowly but never diverged — no overfitting at all. Win AUC below 0.5 (essentially random). The model learned score distributions well (Avg Score MAE 10.06 is best across all experiments) but couldn't extract directional signal.

### Experiment 8: GameStates (5 Seasons)

**Config**: `configs/transformer/gamestates_5season.yaml`
**Date**: Feb 18, 2026

- **Changes**: Same GameStates model with full 5-season training data
- **Result**: Spread MAE 12.36, Avg Score MAE 10.14, Win Acc 52.0%, Win AUC 0.509, val_loss 39.06, ran all 50 epochs
- **Notes**: More training data didn't help. Results are virtually identical to the 2-season run. The GameStates model converges to the same ceiling as PBP regardless of data volume.

### Experiment 9: Schedule (2 Seasons)

**Config**: `configs/transformer/schedule_2season.yaml`
**Date**: Feb 19, 2026

- **Model**: PBP + temporal embeddings (Phase 1c). Adds `days_before_target` (Embed(180, 256)) and `season_game_number` (Embed(110, 256)) per history game, added as residual to game embeddings before TemporalAttention.
- **New params**: +74K (1.3% increase) from embedding tables shared across home/away streams
- **Training**: 2 seasons (2021-2023), same regularization as Exp 5
- **Result**: Spread MAE 12.41, Avg Score MAE 10.14, Win Acc 51.9%, Win AUC 0.512, val_loss 39.19, best epoch 9, early stopped epoch 29
- **Notes**: Schedule features didn't help on the 2-season config. Essentially identical to PBP 2-season baseline (Exp 5: 12.35). The temporal embeddings may not have enough data to learn meaningful patterns with only 2 training seasons.

### Experiment 10: Schedule (5 Seasons)

**Config**: `configs/transformer/schedule_5season.yaml`
**Date**: Feb 19, 2026

- **Changes**: Same schedule model with full 5-season training data
- **Result**: Spread MAE 12.24, Avg Score MAE 10.19, Win Acc 55.9%, Win AUC 0.578, val_loss 38.81, best epoch 10, early stopped epoch 31
- **Notes**: Slight improvement over matched PBP baseline (Exp 6: 12.29 → 12.24) but within noise of Exp 1 (12.20). The temporal embeddings don't break through the ~12.2 ceiling. The model's learned positional encoding already captures most of the temporal signal from game ordering.

---

## Key Findings

1. **Spread MAE ceiling at ~12.2-12.4**: All experiments (PBP and GameStates) converge to approximately the same Spread MAE, regardless of input data type, regularization, history length, or training data volume.

2. **PBP adds no signal beyond score trajectories**: The GameStates-only model (5 features per row: period, clock, scores, margin) achieves the same Spread MAE as the full PBP model (10 features per play: action types, players, shots, etc.). This means the detailed play-by-play information is not contributing predictive signal beyond what the score evolution already provides.

3. **More data isn't always better**: 10 training seasons (PBP Exp 4) performed worse than 5 seasons due to distributional shift. GameStates showed no benefit from 5 vs 2 training seasons either.

4. **Regularization helps training stability, not accuracy**: Tuned regularization (v2) extended training from 5→9 epochs but didn't meaningfully improve final metrics.

5. **Shot features add information but don't break the ceiling**: The model trains longer with richer tokens and achieves competitive score prediction (Avg Score MAE 10.09), but Spread MAE remains stuck.

6. **The bottleneck is not input representation at all**: Since even a minimal 5-feature score trajectory model matches the full PBP model, the limitation is structural:
   - Missing roster/player context (which team's players are actually playing)
   - Missing contextual features (rest days, home/away, back-to-back)
   - Architecture limitations (SimpleFusion may be too simple)
   - The fundamental unpredictability of NBA game outcomes from historical game data alone

7. **Schedule/temporal embeddings don't break the ceiling (Phase 1c)**: Adding raw temporal data (days_before_target, season_game_number) as per-game embeddings produced no meaningful improvement. The model's existing learned positional encoding already captures temporal ordering. Rest effects, B2B fatigue, and tanking signals are either (a) already encoded in game outcomes, (b) too weak relative to game-to-game variance, or (c) need to be combined with roster data to be useful.

---

## Next Steps

Phases 1a and 1c are considered maxed out. The GameStates ablation confirms the bottleneck is NOT in input data representation. Schedule temporal embeddings don't add signal. Potential directions:

- **Phase 1b**: Add roster stream (Set Transformer / RosterEncoder) to incorporate who is actually playing — strongest remaining signal source
- **Architecture changes**: Try different fusion strategies, cross-attention, or deeper temporal models
- **Time2Vec**: Replace discrete clock/period encoding with continuous learned temporal encoding (orthogonal to schedule features)
