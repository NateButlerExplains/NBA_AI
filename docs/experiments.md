# Phase 1a Experiment Log

> **Model**: Two-stream PBP Transformer → SimpleFusion → Probabilistic Heads
> **Hardware**: RTX 2070 SUPER (8GB VRAM)
> **Test Set**: 2024-2025 + 2025-2026 seasons
> **XGBoost Baseline**: Spread MAE ~10.1, Avg Score MAE ~10.1

---

## Summary Table

| # | Experiment | Config | Spread MAE | Avg Score MAE | Win Acc | Win AUC | Val Loss | Best Epoch | Params |
|---|-----------|--------|-----------|---------------|---------|---------|----------|------------|--------|
| 1 | Baseline v1 (5 seasons) | `full_baseline.yaml` | 12.20 | — | — | — | 38.50 | 5 | 5.7M |
| 2 | Baseline v2 (tuned reg) | `baseline_v2.yaml` | 12.26 | — | — | — | 38.49 | 9 | 5.7M |
| 3 | N=10 history games | `ablation_seq10.yaml` | 12.24 | — | — | — | 38.46 | 5 | 5.7M |
| 4 | 10 training seasons | `ablation_10seasons.yaml` | 12.74 | — | — | — | 37.57 | 10 | 5.7M |
| 5 | Shot features (2 seasons) | `fast_2season.yaml` | 12.35 | 10.09 | 52.0% | 0.535 | 39.07 | 13 | 5.85M |
| 6 | Shot features (5 seasons) | `shot_features_5season.yaml` | 12.29 | 10.15 | 52.7% | 0.567 | 38.63 | 12 | 5.85M |

**Best Spread MAE**: 12.20 (Experiment 1 — Baseline v1)
**Best Avg Score MAE**: 10.09 (Experiment 5 — shot features, 2 seasons)

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

---

## Key Findings

1. **Spread MAE ceiling at ~12.2**: All experiments converge to approximately the same Spread MAE (12.2-12.3), regardless of regularization, history length, training data volume, or token richness.

2. **More data isn't always better**: 10 training seasons performed worse than 5 seasons, likely due to distributional shift in older play styles.

3. **Regularization helps training stability, not accuracy**: Tuned regularization (v2) extended training from 5→9 epochs but didn't meaningfully improve final metrics.

4. **Shot features add information but don't break the ceiling**: The model trains longer with richer tokens and achieves competitive score prediction (Avg Score MAE 10.09), but Spread MAE remains stuck.

5. **The bottleneck is likely not PBP token representation**: Since enriching tokens doesn't help, the limitation may be in:
   - Missing roster/player context (which team's players are actually playing)
   - Missing contextual features (rest days, home/away, back-to-back)
   - Architecture limitations (SimpleFusion may be too simple)
   - The fundamental unpredictability of NBA game outcomes at this resolution

---

## Next Steps

Phase 1a is considered maxed out. Potential directions:
- **Phase 1b**: Add roster stream (Set Transformer / RosterEncoder) to incorporate who is actually playing
- **Phase 1c**: Add context stream (rest days, back-to-back, travel distance) with cross-attention fusion
- **Architecture changes**: Try different fusion strategies or deeper temporal models
