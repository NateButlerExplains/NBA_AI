# Phase 4 Generative Model — Architecture & Data Flow

> **Model**: Exp 4 (Compressed Scoring Events) — 30.4M params
> **Result**: Spread MAE 11.76, Win AUC 0.662, Win Acc 61.4%

---

## 1. High-Level Architecture

```
                              CONTEXT PATH                                       GENERATION PATH
                     (encoded once per game)                              (autoregressive, ~110 steps)

 +-----------------+    +-----------------+                       +------------------+
 | TeamBox (DB)    |    | Teams (DB)      |                       | GameStates (DB)  |
 | pts, fga, fgm,  |    | team_id,        |                       | period, clock,   |
 | fg3a, fg3m, fta, |    | abbreviation    |                       | home_score,      |
 | tov, ast, reb,  |    +---------+-------+                       | away_score       |
 | pts_allowed     |              |                               +--------+---------+
 +--------+--------+              |                                        |
          |                       |                                        |
          v                       v                                        v
 +--------+--------+    +--------+-------+                       +--------+---------+
 | Rolling Stats   |    | Team Index     |                       | Compress to      |
 | 20-game window  |    | Embedding      |                       | Scoring Events   |
 | 24 features/team|    | 30 -> 64-d     |                       | ~487 -> ~110     |
 +--------+--------+    +--------+-------+                       | + inter-event dt |
          |                       |                               +--------+---------+
          |     +---------+       |                                        |
          +---->| concat  |<------+                                        v
                | (24+64) |                                       +--------+---------+
                +----+----+                                       | State Embedder   |
                     |                                            | 8-d -> 256 -> 512|
                     v                                            | GELU + LayerNorm |
          +----------+----------+                                 +--------+---------+
          | SimpleContextEncoder|                                          |
          | Linear(88, 256)     |                                          v
          | GELU + Dropout      |                            +-------------+-------------+
          | Linear(256, 512)    |                            |    Causal Decoder          |
          | LayerNorm           |                            |    6 layers x 8 heads      |
          +----+----------+-----+                            |    FF=2048, 512-d           |
               |          |                                  |                             |
               |  (B,2,512)                                  |  adaLN-Zero at every layer  |
               |          |                                  |  gamma, beta, alpha from     |
               v          v                                  |  conditioning vector   <-----+--- cond (B,512)
    +----------+--+ +-----+-------+                          |                             |
    | Context     | | Pre-Decoder |                          | Sinusoidal pos encoding     |
    | Pooling     | | Aux Heads   |                          | KV cache (inference)        |
    | cat(h,a)    | |             |                          +------+-----------+----------+
    | Linear(1024,| | Matchup:    |                                 |           |
    |  512)+SiLU  | | cat(h,a,h-a)|                                 v           v
    +------+------+ | = 1536-d    |                          +------+-----+ +---+--------+
           |        |             |                          | Score Head | | Clock Head |
      cond |        +--+------+--+                          | 512->128->7| | 512->128->1|
      (B,512)          |      |                             +------+-----+ +---+--------+
           |           v      v                                    |           |
           |     +-----+-+ +--+---+                                v           v
           |     |Margin | | Win  |                         +------+-----+ +---+---------+
           |     |Head   | | Head |                         | + ScoreBias| | game_progress|
           |     |MSE    | | BCE  |                         | (B,7)      | | (0 -> 1)    |
           |     +-------+ +------+                         +------+-----+ +---+---------+
           |                                                       |           |
           +----> ContextScoreBias                                 v           v
           |      concat(h,a) -> 128 -> 7                   +-----------+-------------+
           |      = (B,7) added to score logits              |  Multinomial Sample     |
           |                                                 |  temperature=1.0        |
           +----> ContextMarginHead                          |  CFG: guidance_scale=1.5|
                  decoder_out[0] -> 128 -> 1                 +------------+------------+
                  MSE vs final_margin                                     |
                                                                         v
                                                                  +------+------+
                                                                  | Accumulate  |
                                                                  | Scores      |
                                                                  | Check end:  |
                                                                  | game_end or |
                                                                  | progress>=1 |
                                                                  +------+------+
                                                                         |
                                                                   x100 rollouts
                                                                         |
                                                                         v
                                                                  +------+------+
                                                                  | Aggregate   |
                                                                  | mean(spread)|
                                                                  | P(home win) |
                                                                  +-------------+
```

---

## 2. Training Pipeline

```
 +------------------+     +------------------+     +-------------------+
 | build_generative |     | train_generative |     | evaluate_         |
 | _cache.py        | --> | .py              | --> | generative.py     |
 +------------------+     +------------------+     +-------------------+
   GameStates -> .pt        DataLoader(bs=32)        Load best.pt
   TeamBox -> rolling       AdamW lr=3e-4            100 rollouts/game
   stats (24 feat)          Cosine + warmup          Spread MAE, AUC
   PlayerBox -> context     EMA decay=0.999          Win Acc, calibration
                            Grad clip=1.0
                            Early stop patience=15

                     TRAINING FORWARD PASS (per batch)
 ============================================================================

  1. context_encoder(rolling_stats, team_idx) --> context_tokens (B,2,512)

  2. PRE-DECODER HEADS (direct gradient to context encoder):
     matchup = cat(home_ctx, away_ctx, home-away)     (B, 1536)
     pre_margin = PreDecoderMarginHead(matchup)        (B,)  -> MSE vs margin
     pre_win    = PreDecoderWinHead(matchup)            (B,)  -> BCE vs home_win

  3. CONTEXT POOLING:
     cond = Linear(cat(home,away)) + SiLU              (B, 512)
     score_bias = ContextScoreBias(context_tokens)      (B, 7)

  4. CONTEXT DROPOUT (10%):
     With p=0.10, zero out cond AND score_bias          (enables CFG at inference)

  5. STATE EMBEDDING:
     states[:, :-1, :] (teacher forcing)                (B, T-1, 8)
     + optional score jitter (std=0.5)
     StateEmbedder --> state_embeds                     (B, T-1, 512)

  6. CAUSAL DECODING with adaLN-Zero:
     decoder(state_embeds, cond) --> decoder_out        (B, T-1, 512)
     Each layer: LN(x)*(1+gamma)+beta, then alpha*attn/ff residual
     gamma/beta/alpha predicted from cond, zero-initialized

  7. PREDICTION HEADS:
     score_logits = ScoreHead(decoder_out) + score_bias (B, T-1, 7)
     clock_preds  = ClockHead(decoder_out)              (B, T-1)
     ctx_margin   = ContextMarginHead(decoder_out[0])   (B,)

  8. LOSS (5 terms):
     L = 1.0 * CE(score_logits, score_targets)          class-weighted, masked
       + 0.3 * MSE(clock_preds, game_progress_targets)  masked
       + 1.0 * MSE(ctx_margin, final_margin/50)         full batch
       + 1.0 * MSE(pre_margin, final_margin/50)         full batch
       + 0.5 * BCE(pre_win, home_win)                   full batch
```

---

## 3. Data Flow Table

### A. State Sequence (Generation Target)

| Field | Source Table | Raw Value | Normalization | State Index | Description |
|-------|------------|-----------|---------------|-------------|-------------|
| period_norm | GameStates.period | 1-4 | /4.0 | 0 | Quarter number |
| clock_norm | GameStates.clock | "MM:SS" -> seconds | /720.0 (per-quarter) | 1 | Seconds remaining in quarter |
| game_progress | Derived | elapsed/2880 | Already 0-1 | 2 | Monotonic progress through game |
| home_score_norm | GameStates.home_score | 0-150+ | /150.0 | 3 | Cumulative home points |
| away_score_norm | GameStates.away_score | 0-150+ | /150.0 | 4 | Cumulative away points |
| margin_norm | Derived | home-away | /50.0 | 5 | Current point differential |
| total_norm | Derived | home+away | /300.0 | 6 | Combined score |
| inter_event_time | Derived | seconds between scores | /120.0 | 7 | Time gap since last scoring event |

**Compression**: Full GameStates sequence (~487 states) is filtered to scoring events only (~110 events). Non-scoring states are discarded. A `game_end` sentinel (class 6) is appended.

**Score Event Classes (targets)**:

| Class | Event | Typical Frequency |
|-------|-------|-------------------|
| 0 | home+1 (free throw) | ~15.3% |
| 1 | home+2 (field goal) | ~24.6% |
| 2 | home+3 (three-pointer) | ~10.7% |
| 3 | away+1 (free throw) | ~14.6% |
| 4 | away+2 (field goal) | ~24.0% |
| 5 | away+3 (three-pointer) | ~10.0% |
| 6 | game_end | ~0.9% |

### B. Context Features (Rolling Stats — 24 per team)

| Index | Feature | Source | Normalization | Window | Description |
|-------|---------|--------|---------------|--------|-------------|
| 0 | avg_pts | TeamBox.pts | /150.0 | 20-game | Points scored per game |
| 1 | avg_pts_allowed | TeamBox.pts_allowed | /150.0 | 20-game | Points allowed per game |
| 2 | avg_margin | Derived (pts-pts_allowed) | /50.0 | 20-game | Average point differential |
| 3 | avg_total | Derived (pts+pts_allowed) | /300.0 | 20-game | Average combined score |
| 4 | pts_std | TeamBox.pts | /30.0 | 20-game | Scoring variance |
| 5 | margin_std | Derived | /30.0 | 20-game | Margin variance |
| 6 | win_pct | Derived (margin>0) | Already 0-1 | 20-game | Win percentage |
| 7 | streak | Derived | /10.0, clamped [-10,10] | 20-game | Win/loss streak |
| 8 | efg_pct | Derived (fgm+0.5*fg3m)/fga | Already 0-1 | 20-game | Effective field goal % |
| 9 | fg3a_rate | Derived fg3a/fga | Already 0-1 | 20-game | Three-point attempt rate |
| 10 | ft_rate | Derived fta/fga | Already 0-1 | 20-game | Free throw rate |
| 11 | ts_pct | Derived pts/(2*(fga+0.44*fta)) | Already 0-1 | 20-game | True shooting % |
| 12 | pace | Derived fga-0.44*fta+tov | /100.0 | 20-game | Estimated possessions |
| 13 | ast_ratio | Derived ast/fgm | /3.0 | 20-game | Assist-to-FGM ratio |
| 14 | def_rating_proxy | TeamBox.pts_allowed | /150.0 | 20-game | Defensive rating proxy |
| 15 | tov_rate | Derived tov/pace | Already 0-1 | 20-game | Turnover rate |
| 16 | rest_days | Derived from game dates | /7.0, clamped [0,7] | N/A | Days since last game |
| 17 | is_home | Games.home_team | 0 or 1 | N/A | Home court indicator |
| 18 | season_progress | Game ordinal index | /82.0 | N/A | Games played so far |
| 19 | ewm_margin_10 | TeamBox margin | /50.0 | EWM span=10 | Exponential-weighted margin |
| 20 | ewm_margin_20 | TeamBox margin | /50.0 | EWM span=20 | Exponential-weighted margin |
| 21 | ewm_win_10 | Derived | Already 0-1 | EWM span=10 | Exponential-weighted win rate |
| 22 | ewm_ortg | TeamBox.pts | /120.0 | EWM span=10 | EWM offensive rating |
| 23 | ewm_drtg | TeamBox.pts_allowed | /120.0 | EWM span=10 | EWM defensive rating |

### C. Additional Context Inputs

| Field | Source | Shape | Description |
|-------|--------|-------|-------------|
| home_team_idx | Teams.abbreviation -> TEAM_TO_IDX | scalar (0-29) | Home team identity |
| away_team_idx | Teams.abbreviation -> TEAM_TO_IDX | scalar (0-29) | Away team identity |

### D. Training Targets

| Target | Source | Normalization | Loss | Weight |
|--------|--------|---------------|------|--------|
| score_events | Derived from consecutive GameStates scores | Classes 0-6 | Cross-Entropy (class-weighted) | 1.0 |
| clock_targets | GameStates.game_progress at next scoring event | Already 0-1 | MSE | 0.3 |
| final_margin | GameStates final home-away | /50.0 | MSE (ctx + pre-decoder) | 1.0 + 1.0 |
| home_win | Derived (final_margin > 0) | 0 or 1 | BCE (pre-decoder) | 0.5 |

---

## 4. End-to-End Data Flow (Single Game Prediction)

```
  DATABASE                CACHE                    MODEL                      OUTPUT
  ========                =====                    =====                      ======

  GameStates        states/{game_id}.pt
  (period, clock,   (states: Tx7 tensor,        StateEmbedder(8->512)
   home/away_score)  score_events: T-1,    -->  + sinusoidal pos enc    -->  score_logits (T,7)
                     clock_targets: T-1)        + causal masking             clock_pred (T,)
                          |                     + adaLN-Zero(cond)
                          |                          ^
                          | compress                 |
                          | (~487->~110              |
                          |  scoring events)    context_pool(cond)
                          |                          ^
                          |                          |
  TeamBox           rolling_stats.pt            SimpleContextEncoder
  (pts, fga, fgm,   {game_id: {                 (88->256->512, LN)
   fg3a, fg3m, fta,   home_stats: [24],    -->       |
   tov, ast, reb,     away_stats: [24],         (B, 2, 512)
   pts_allowed})       home_team_idx,                |
                       away_team_idx}}          +----+----+---+---+
                                                |         |       |
                                        ContextScore  PreMargin PreWin
                                        Bias (B,7)    Head      Head
                                                |     (MSE)     (BCE)
                                                v
                                          Added to score
                                          logits at every
                                          position

  INFERENCE:
  =========
  Context encoded once --> cond cached
  For each of 100 rollouts:
    Init: state = [period=1/4, clock=1.0, progress=0, h=0, a=0, margin=0, total=0, dt=0]
    Loop (up to 200 steps):
      embed(state) -> decoder_step(embed, cond, kv_cache) -> score_logits + score_bias
      CFG: logits = uncond + 1.5*(cond - uncond)
      sample event from softmax(logits/temp)
      if event == game_end or progress >= 1.0: stop
      update cumulative scores from event delta
      update game_progress from clock_head prediction
    Record final (home_score, away_score)
  Aggregate: mean(spread), P(home_win), std(spread)
```

---

## 5. Module Parameter Breakdown

| Component | Parameters | Description |
|-----------|-----------|-------------|
| SimpleContextEncoder | 157K | Rolling stats MLP + team embedding |
| StateEmbedder | 133K | 8-d state -> 512-d embedding |
| CausalDecoder | 18.9M | 6 adaLN-Zero layers (512-d, 8 heads, FF=2048) |
| ScoreHead | 66K | 512 -> 128 -> 7 classification |
| ClockHead | 66K | 512 -> 128 -> 1 regression |
| ContextMarginHead | 66K | 512 -> 128 -> 1 regression |
| ContextScoreBias | 132K | 1024 -> 128 -> 7 bias |
| PreDecoderMarginHead | 197K | 1536 -> 128 -> 1 regression |
| PreDecoderWinHead | 197K | 1536 -> 128 -> 1 regression |
| ContextPooling | 525K | Linear(1024, 512) + SiLU |
| adaLN modulations | ~10M | 6 layers * Linear(512, 3072) + final Linear(512, 1024) |
| **Total** | **~30.4M** | |
