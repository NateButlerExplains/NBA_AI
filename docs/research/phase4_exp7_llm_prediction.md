# Phase 4 Exp 7: LLM-Based NBA Game Prediction — Research Analysis

> **Date**: March 18, 2026
> **Goal**: Evaluate using LLM APIs (OpenAI GPT-4o, Anthropic Claude) as NBA game predictors
> **Targets**: Point spread (home - away), individual team scores, win probability
> **Baseline**: Phase 3 Exp 9 ensemble — Spread MAE 10.66, Win AUC 0.718, Win Acc 66.5%

---

## 1. Literature Review

### 1.1 LLM Forecasting Accuracy (General)

**"Approaching Human-Level Forecasting with Language Models"** (Halawi et al., Feb 2024, arXiv:2402.18563)
- Built retrieval-augmented LLM system for competitive forecasting questions
- System neared the crowd aggregate of competitive forecasters, sometimes surpassing it
- Used retrieval + reasoning + ensembling of multiple LLM predictions
- Key finding: LLMs can achieve human-competitive forecasting on well-defined binary/probability questions
- Limitation: These were event forecasting questions (geopolitics, economics), not numeric regression tasks like point spreads

**"Leveraging Log Probabilities in Language Models to Forecast Future Events"** (Soru & Marshall, Jan 2025, arXiv:2501.04880)
- Novel approach using token log-probabilities for probability estimation
- Brier score of 0.186 (26pp better than random, 19pp better than existing AI systems)
- Shows LLM internals contain calibrated probability information beyond surface text

**"Future Is Unevenly Distributed"** (Karkar & Chopra, Nov 2025)
- Forecasting ability "varies sharply with domain structure and prompt framing"
- Performance highly variable across domains — sports may be particularly hard due to inherent noise

**"A Test of Lookahead Bias in LLM Forecasts"** (Gao et al., Dec 2025)
- Critical methodological warning: LLMs trained on internet data may have memorized outcomes
- Positive correlation between training data presence and forecast accuracy → must test on truly post-cutoff data
- Our test set (2024-2026 seasons) partially overlaps with training cutoffs of current models

### 1.2 LLM for Sports Specifically

**"Neural Sabermetrics with World Model"** (Ahn et al., Feb 2026, arXiv:2602.07030)
- Trained LLM as world model on 10+ years of MLB tracking data (billions of tokens)
- 64% accuracy on next-pitch prediction, 78% on batter swing decisions
- Key insight: LLM-as-world-model works for sequential sports events
- But: this was a *fine-tuned* model on sports-specific data, not zero-shot prompting

**"AI for Handball: 2024 Olympic Games Prediction"** (Felice, Jul 2024)
- Combined deep learning with LLMs for explainable predictions
- LLMs used for explanation generation, not primary prediction
- Suggests hybrid approach: ML model predicts, LLM explains

**GitHub: chatGPT-sports-handicapper-fast-eddie** (8 stars, 2023)
- JavaScript tool using ChatGPT with "multiple AI personas" for NBA/MLB
- No published accuracy results
- Proof of concept only — no rigorous evaluation

### 1.3 Key Findings from Broader LLM Prediction Research

| Finding | Source | Relevance |
|---------|--------|-----------|
| LLMs approach human forecaster aggregates | Halawi et al. 2024 | High — but for binary questions, not numeric |
| Domain structure matters enormously | Karkar & Chopra 2025 | High — sports have high irreducible noise |
| Log-prob extraction beats surface output | Soru & Marshall 2025 | Medium — could extract calibrated probabilities |
| Lookahead bias inflates accuracy | Gao et al. 2025 | Critical — must control for data leakage |
| Fine-tuned LLMs work for sports sequences | Ahn et al. 2026 | Medium — we'd use zero/few-shot, not fine-tuned |
| Multi-agent debate improves forecasting | Gorur et al. 2025 | Medium — ensemble multiple LLM "perspectives" |
| Human experts outperform LLMs on complex tasks | AgentCaster 2025 | Cautionary — models "overpredict and hallucinate" |

### 1.4 What's Missing

**No published work directly benchmarks LLM zero-shot NBA point spread prediction.** The closest is:
- General event forecasting (Halawi) — binary, not numeric regression
- MLB sequence modeling (Ahn) — fine-tuned, not zero-shot
- The chatGPT sports tool — no published metrics

This means we'd be breaking new ground, which is both interesting and risky.

---

## 2. API Pricing and Batch Processing

### 2.1 Anthropic Claude Pricing

| Model | Standard Input | Standard Output | Batch Input (50% off) | Batch Output (50% off) |
|-------|---------------|-----------------|----------------------|----------------------|
| Claude Sonnet 4.6 | $3/MTok | $15/MTok | $1.50/MTok | $7.50/MTok |
| Claude Haiku 4.5 | $1/MTok | $5/MTok | $0.50/MTok | $2.50/MTok |
| Claude Opus 4.6 | $5/MTok | $25/MTok | $2.50/MTok | $12.50/MTok |

**Anthropic Message Batches API:**
- 50% discount on all input and output tokens
- Max 100,000 requests or 256 MB per batch
- Most batches complete within 1 hour, guaranteed within 24 hours
- Supports all features: tool use, system messages, vision, multi-turn
- All active models supported
- Prompt caching stacks with batch discount (cache hit = 0.1x base, further halved in batch)

### 2.2 OpenAI GPT-5.4 Family Pricing

| Model | Standard Input | Standard Output | Batch Input (50% off) | Batch Output (50% off) |
|-------|---------------|-----------------|----------------------|----------------------|
| gpt-5.4-nano | $0.20/MTok | $1.25/MTok | $0.10/MTok | $0.625/MTok |
| gpt-5.4-mini | $0.75/MTok | $4.50/MTok | $0.375/MTok | $2.25/MTok |
| gpt-5.4 | $2.50/MTok | $15.00/MTok | $1.25/MTok | $7.50/MTok |

**OpenAI Batch API:**
- 50% discount on all tokens
- JSONL file input format
- Completes within 24 hours (often faster)
- Supports structured outputs (`response_format: { type: "json_schema" }`)
- Supports GPT-5.4 family (nano, mini, 5.4)

### 2.3 Cost Estimates for 2,400 Test Games

**Token budget per game (estimated):**
- System prompt + instructions: ~500 tokens
- Game context (team stats, records, recent games): ~1,500-3,000 tokens
- Player data (top 8 players per team, recent stats): ~2,000-4,000 tokens
- Total input: ~4,000-7,500 tokens per game
- Output (structured prediction + optional reasoning): ~200-800 tokens per game

Using **~5,000 input + ~500 output tokens per game** as baseline estimate:

| Model | Mode | Input Cost | Output Cost | Total (2,400 games) |
|-------|------|-----------|-------------|---------------------|
| gpt-5.4-nano | Batch | $1.20 | $0.75 | **$1.95** |
| gpt-5.4-nano | Standard | $2.40 | $1.50 | **$3.90** |
| gpt-5.4-mini | Batch | $4.50 | $2.70 | **$7.20** |
| Claude Haiku 4.5 | Batch | $6.00 | $3.00 | **$9.00** |
| gpt-5.4 | Batch | $15.00 | $9.00 | **$24.00** |
| Claude Sonnet 4.6 | Batch | $18.00 | $9.00 | **$27.00** |
| Claude Opus 4.6 | Batch | $30.00 | $15.00 | **$45.00** |
| gpt-5.4 | Standard | $30.00 | $18.00 | **$48.00** |
| Claude Sonnet 4.6 | Standard | $36.00 | $18.00 | **$54.00** |

**With chain-of-thought (3x output tokens):**
| Model | Mode | Total (2,400 games) |
|-------|------|---------------------|
| gpt-5.4-nano | Batch | **$2.70** |
| gpt-5.4-mini | Batch | **$12.60** |
| Claude Haiku 4.5 | Batch | **$15.00** |
| gpt-5.4 | Batch | **$42.00** |
| Claude Sonnet 4.6 | Batch | **$45.00** |

**With prompt caching (Anthropic, shared system prompt):**
- System prompt cached at 0.1x input price
- Saves ~500 tokens x 0.9 x $rate per game → modest savings (~$1-3 total)
- More valuable if we include large league-average reference data in system prompt

**Multi-prompt ensemble (3 prompts per game):**
- Multiply above costs by 3
- gpt-5.4-nano batch: $5.85-$8.10 for full test set
- gpt-5.4-mini batch: $21.60-$37.80 for full test set
- Claude Sonnet batch: $81-$135 for full test set

**Recommendation**: Start with gpt-5.4-nano batch (~$2-3 per full evaluation) for rapid iteration, then validate best prompts on gpt-5.4-mini or Claude Sonnet.

---

## 3. Structured Output Strategies

### 3.1 Output Schema

Target prediction format:
```json
{
  "home_score": 108,
  "away_score": 103,
  "spread": 5.0,
  "home_win_probability": 0.62,
  "confidence": "medium",
  "reasoning_summary": "Home team's strong 3PT shooting (38.2%) and rest advantage offset away team's superior defense"
}
```

### 3.2 OpenAI Structured Outputs

```python
response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "game_prediction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "home_score": {"type": "integer"},
                "away_score": {"type": "integer"},
                "spread": {"type": "number"},
                "home_win_probability": {"type": "number"},
                "reasoning_summary": {"type": "string"}
            },
            "required": ["home_score", "away_score", "spread", "home_win_probability"],
            "additionalProperties": False
        }
    }
}
```

- GPT-4o and GPT-4o-mini both support `strict: True` JSON schema
- Guarantees valid JSON matching the schema — no parsing failures
- Works with Batch API
- Constrained decoding enforces numeric types

### 3.3 Anthropic Tool Use

```python
tools = [{
    "name": "submit_prediction",
    "description": "Submit NBA game prediction",
    "input_schema": {
        "type": "object",
        "properties": {
            "home_score": {"type": "integer", "description": "Predicted home team score"},
            "away_score": {"type": "integer", "description": "Predicted away team score"},
            "spread": {"type": "number", "description": "Predicted point spread (home - away)"},
            "home_win_probability": {"type": "number", "description": "Home win probability 0-1"},
            "reasoning_summary": {"type": "string"}
        },
        "required": ["home_score", "away_score", "spread", "home_win_probability"]
    }
}]
```

- `tool_choice: {"type": "tool", "name": "submit_prediction"}` forces tool use
- Adds ~346 tokens of system prompt overhead
- Guarantees structured output matching the schema
- Works with Message Batches API

### 3.4 Recommendation

Both approaches reliably produce structured JSON. OpenAI structured outputs are slightly cleaner (native JSON mode vs tool-use wrapper). For cost-sensitive iteration, GPT-4o-mini with structured outputs is the best starting point.

---

## 4. Prompt Engineering Strategies

### 4.1 Recommended Prompt Structure

```
SYSTEM PROMPT (cacheable):
- Role: Expert NBA analyst and quantitative predictor
- Task description with output format
- League averages for calibration (avg score ~112, avg spread ~0, home advantage ~2-3 pts)
- Base rates: home team wins ~57% of games
- Instructions on reasoning approach

USER PROMPT (per-game):
- Game metadata: date, teams, venue
- Home team context:
  - Record (W-L overall, home record)
  - Rolling stats (last 10 games): PPG, OPP_PPG, FG%, 3PT%, FT%, REB, AST, TOV
  - Rest days, back-to-back flag
  - Top 8 players with recent averages (PTS, REB, AST, MIN)
- Away team context: (same structure)
- Matchup data: head-to-head record this season, historical matchup
- Season context: games played, playoff race position
```

### 4.2 Prompting Variants to Test

**Variant A: Direct Prediction (baseline)**
- Minimal system prompt, structured game context, request prediction

**Variant B: Chain-of-Thought**
- "Think step by step about this matchup before making your prediction"
- Analyze offense vs defense, key matchups, situational factors
- Then produce numeric prediction
- Expected: better calibration, higher cost (3x output)

**Variant C: Calibrated with Anchoring**
- Include Vegas line as reference: "The market consensus spread is -5.5 home"
- LLM adjusts from anchor based on its analysis
- Expected: lower MAE (anchored to good baseline), but tests whether LLM adds value

**Variant D: Contrarian/Independent**
- Explicitly withhold Vegas line
- "Make your prediction based solely on the statistical data provided"
- Tests LLM's independent predictive ability

**Variant E: Ensemble of Perspectives**
- Three separate prompts per game with different analytical frames:
  1. "Focus on offensive efficiency and scoring matchups"
  2. "Focus on defensive metrics and pace of play"
  3. "Focus on situational factors: rest, travel, motivation, recent form"
- Average the three predictions

**Variant F: Few-Shot with Examples**
- Include 3-5 example games with known outcomes in system prompt
- Shows desired reasoning pattern and output format
- Risk: may overfit to examples

### 4.3 Feature Selection for Prompts

**High-value features** (include always):
- Team records (W-L, last 10)
- Points scored/allowed (rolling averages)
- Shooting percentages (FG%, 3PT%)
- Rest days / back-to-back
- Home/away context

**Medium-value features** (include if space permits):
- Top player recent averages
- Turnover and assist rates
- Rebound differential
- Free throw rate

**Low-value/noise features** (consider excluding):
- Full 16-stat player box scores (too dense for prompts)
- Historical matchup data beyond current season
- Detailed per-quarter breakdowns
- Advanced metrics the LLM may not reason about well

**Optimal context size**: ~2,000-3,000 tokens of game data. Enough for team-level rolling stats + top 5-8 players per team with key stats (PTS/REB/AST/MIN).

### 4.4 Calibration Techniques

1. **Include base rates explicitly**: "The average NBA game has a total score of ~224 points. Home teams win ~57% of the time. The typical point spread range is -15 to +15."

2. **Temperature tuning**: Lower temperature (0.3-0.5) for more consistent numeric predictions; higher (0.7-1.0) for ensemble diversity.

3. **Post-hoc calibration**: Run LLM predictions on a calibration set (e.g., 200 games), then apply Platt scaling to win probabilities and linear correction to spreads.

4. **Probability extraction via log-probs**: For OpenAI, request `logprobs` on key tokens to extract confidence information beyond the surface prediction.

---

## 5. Hybrid Approaches

### 5.1 LLM as Feature Extractor

The LLM's real advantage over our transformer: it can reason about **qualitative context** that numeric models cannot process:
- Injury reports and return-from-injury narratives
- Team chemistry / trade deadline disruption
- Motivation factors (playoff positioning, rivalry games, rest starters)
- Coaching changes and schematic adjustments
- Travel fatigue, altitude, time zone shifts

**Approach**: Prompt the LLM to output a structured "qualitative assessment" alongside its prediction, then feed both the LLM prediction and its qualitative features into an XGBoost or linear meta-model along with our transformer's prediction.

```json
{
  "prediction": { "spread": 5.0, "home_win_prob": 0.62 },
  "qualitative_features": {
    "injury_impact_home": -0.3,    // -1 to 1 scale
    "injury_impact_away": -0.1,
    "motivation_differential": 0.2,
    "fatigue_factor": -0.1,
    "matchup_advantage": 0.15,
    "overall_edge_assessment": 0.4  // -1 to 1
  }
}
```

### 5.2 LLM + Transformer Ensemble

Most promising approach:
1. Run our Phase 3 Exp 9 ensemble (MAE 10.66)
2. Run LLM predictions on same games
3. Simple average or learned weighted combination
4. If LLM errors are uncorrelated with transformer errors, ensemble improves

**Expected benefit**: Even if LLM alone has MAE ~12-13, its errors may be uncorrelated with our transformer, giving ensemble improvement of 0.1-0.3 MAE.

### 5.3 LLM for Uncertainty Estimation

Use the LLM to flag "unpredictable" games where the transformer's confidence should be reduced:
- Games with major injury uncertainty
- Games where teams are resting starters
- Season opener / post-All-Star break games
- Teams in losing streaks (psychological factors)

This could improve calibration (ECE) and coverage metrics even if it doesn't improve MAE.

---

## 6. Expected Accuracy

### 6.1 Realistic Projections

| Approach | Expected Spread MAE | Expected Win AUC | Reasoning |
|----------|-------------------|------------------|-----------|
| LLM zero-shot (no Vegas line) | 12.5-14.0 | 0.60-0.67 | LLMs lack precise numeric reasoning; high-noise domain |
| LLM zero-shot (with Vegas anchor) | 10.5-11.5 | 0.68-0.72 | Vegas anchor does most work; LLM makes marginal adjustments |
| LLM few-shot with CoT | 12.0-13.5 | 0.62-0.68 | CoT improves reasoning but won't overcome data limitations |
| LLM ensemble (3 prompts) | 11.5-13.0 | 0.63-0.69 | Variance reduction helps modestly |
| LLM + transformer ensemble | 10.3-10.6 | 0.72-0.73 | Best case — uncorrelated errors help |
| Transformer ensemble (current) | 10.66 | 0.718 | Our baseline |
| Vegas closing line | ~9.45 | ~0.75 | Approximate market benchmark |

### 6.2 Why LLMs Alone Won't Beat Our Transformer

1. **Numeric precision**: LLMs generate text, not numbers. Their "numeric reasoning" is pattern matching on training data, not computation. Our transformer directly optimizes MSE loss on spreads.

2. **Training signal**: Our transformer was trained on 30,000+ games with direct gradient signal. The LLM's "training" on sports comes from exposure to articles, commentary, and box scores — indirect and noisy.

3. **Irreducible noise**: NBA games have ~9-10 points of irreducible noise (Vegas MAE ~9.45). Our transformer at 10.66 is already within ~1.2 points of this floor. LLMs won't have better signal.

4. **No proprietary data**: Vegas lines incorporate injury reports, betting market information, and expert analysis that our data includes indirectly. LLMs have access to general knowledge but not real-time game-specific intelligence for historical games.

5. **Calibration**: Our transformer outputs calibrated Gaussian distributions (mu, sigma). Getting calibrated uncertainty from LLMs requires extra work.

### 6.3 Where LLMs Could Add Value

1. **Uncorrelated errors**: LLMs may make different mistakes than our transformer, benefiting an ensemble
2. **Qualitative reasoning**: Injury context, motivation, matchup narratives — things our numeric model misses entirely
3. **Anomaly detection**: Flagging games that are "unusual" (rest patterns, back-to-backs, rivalry games)
4. **Interpretability**: Natural language reasoning about why a prediction was made
5. **Rapid prototyping**: Test new prediction hypotheses without retraining a model

---

## 7. Risks and Limitations

### 7.1 Data Leakage / Lookahead Bias

**Critical risk.** Current LLM knowledge cutoffs:
- Claude Sonnet 4.6: training data through Jan 2026
- GPT-4o: training data through ~Oct 2023 (publicly known), but continuously updated

Our test set spans 2024-2026 seasons. Games from 2024-2025 are almost certainly in LLM training data. The model may have *memorized* outcomes, not predicted them.

**Mitigation**:
- Test on games after the model's training cutoff only
- Compare to a "random baseline" LLM that gets no game context (just team names + date)
- Monitor for suspiciously good performance on specific games
- Use the 2025-2026 season as the primary test set (post-cutoff for most models)

### 7.2 Reproducibility

- LLM outputs are stochastic (even at temperature=0, there can be variation)
- API models may be updated silently, changing behavior
- Must log: model version, temperature, full prompt, raw response, timestamp

### 7.3 Cost Scaling

- 2,400 games at ~$1-2 per run (GPT-4o-mini batch) is cheap for evaluation
- But prompt iteration (50+ variants) could cost $50-100 total
- Production use (daily predictions) is <$1/day — negligible

### 7.4 Latency

- Batch API: 1-24 hours for 2,400 games (not suitable for real-time)
- Standard API: ~1-3 seconds per game, ~40-120 minutes for 2,400 sequential calls
- With parallelism (10 concurrent): ~4-12 minutes for full test set

### 7.5 Model Deprecation

- LLM APIs change models, pricing, and capabilities frequently
- Must pin to specific model versions (e.g., `gpt-4o-2024-08-06`)
- Results may not be reproducible in 6 months

---

## 8. Implementation Plan

### Phase 1: Infrastructure (1-2 days)

1. **Build prompt template system**
   - Jinja2 or f-string templates for system + user prompts
   - Function to extract game context from our existing cache/database
   - Output: `GameContext` dataclass with all features in human-readable format

2. **Build API client with batch support**
   - Anthropic Message Batches client
   - OpenAI Batch API client
   - Common interface: `predict_batch(games: list[GameContext]) -> list[Prediction]`
   - Structured output parsing + validation

3. **Evaluation harness**
   - Reuse existing evaluation metrics (MAE, AUC, calibration)
   - Compare LLM predictions to transformer predictions per-game
   - Correlation analysis between LLM and transformer errors

### Phase 2: Baseline Experiments (2-3 days)

4. **Experiment 7a: GPT-4o-mini zero-shot baseline**
   - Simple prompt with team records + rolling stats
   - No Vegas line, no CoT
   - Batch API for full 2,400 game test set
   - Cost: ~$1-2

5. **Experiment 7b: Prompt variant sweep**
   - Test variants A-F on 200-game calibration subset
   - Identify best prompt strategy
   - Cost: ~$5-10

6. **Experiment 7c: Model comparison**
   - Best prompt on GPT-4o-mini, GPT-4o, Claude Haiku, Claude Sonnet
   - Full test set evaluation
   - Cost: ~$30-50

### Phase 3: Advanced Approaches (2-3 days)

7. **Experiment 7d: Ensemble of prompts**
   - 3 best prompts from 7b, average predictions
   - Multi-temperature runs for diversity

8. **Experiment 7e: LLM + Transformer hybrid**
   - Combine LLM predictions with Phase 3 Exp 9 ensemble
   - Simple average, learned weighting, or meta-model
   - This is the most likely approach to actually improve our numbers

9. **Experiment 7f: LLM qualitative features**
   - Extract qualitative assessment features from LLM
   - Feed into XGBoost alongside transformer embeddings

### Phase 4: Analysis (1 day)

10. **Data leakage audit**
    - Compare accuracy on pre-cutoff vs post-cutoff games
    - Run "no-context" baseline (just team names) to detect memorization

11. **Error analysis**
    - Where does LLM beat transformer? Where does it fail?
    - What types of games benefit from LLM reasoning?
    - Correlation of LLM and transformer residuals

12. **Write-up and decision**
    - Does LLM add value to the ensemble?
    - Cost-benefit for production deployment

### Estimated Total Cost: $50-100 for all experiments

### Estimated Timeline: 6-9 days

---

## 9. Recommended Approach

### Primary Recommendation: Hybrid Ensemble (Exp 7e)

**Rationale**: The LLM alone will almost certainly not beat our transformer (MAE 10.66). However, an LLM + transformer ensemble could improve by 0.1-0.3 MAE if their errors are uncorrelated. This is the same principle that made Phase 3 Exp 9 (deep ensemble) our best model — variance reduction works.

**Implementation**:
1. Use GPT-4o-mini batch for cost-effective iteration ($1-2 per full evaluation)
2. Use structured outputs for reliable numeric predictions
3. Test 3-5 prompt variants on a calibration subset
4. Run best prompt on full test set
5. Combine with transformer via simple average or learned weights
6. Evaluate ensemble for MAE, AUC, and calibration improvements

### Secondary Recommendation: LLM for Qualitative Features (Exp 7f)

**Rationale**: LLMs' unique value is reasoning about things our model cannot see: injuries, motivation, matchup narratives. Extract these as structured features and feed into a meta-model.

### What NOT to Do

- Do not expect LLM-only prediction to beat 10.66 MAE
- Do not use expensive models (Opus, GPT-4) for initial iteration
- Do not skip the data leakage audit
- Do not use standard API when batch API is available (2x cost)
- Do not include too much context — 5,000 tokens of relevant stats beats 20,000 tokens of noise

---

## 10. Quick Reference: Key Numbers

| Metric | Value |
|--------|-------|
| Test set size | ~2,400 games |
| Our best MAE | 10.66 (Phase 3 Exp 9 ensemble) |
| Our best AUC | 0.718 |
| Vegas MAE | ~9.45 |
| Cheapest full evaluation | ~$1.95 (gpt-5.4-nano batch) |
| Most expensive approach | ~$135 (Claude Sonnet 3-prompt ensemble, standard API) |
| Expected LLM standalone MAE | 12.5-14.0 (without Vegas anchor) |
| Expected LLM+Transformer MAE | 10.3-10.6 (optimistic) |
| Tokens per game (input) | ~4,000-7,500 |
| Tokens per game (output) | ~200-800 |
| Batch processing time | 1-24 hours |

---

## 11. Final Design Decisions (March 18, 2026)

1. **Models**: GPT-5.4 family only (nano/mini/5.4). Not GPT-4 series. The GPT-5.4 family offers better performance at competitive pricing and is the current recommended model line from OpenAI.

2. **No Vegas lines**: Predictions use only the same statistical data our Phase 3/4 models use. This ensures a fair apples-to-apples comparison of predictive ability without anchoring to market consensus.

3. **Target game roster included**: Players who actually played in the target game are provided to the LLM. This is consistent with Phase 3 models which also use target game roster (confirmed in `sequence_builder.py` `_extract_roster()`). Notable absences (players averaging >15 MPG but who did not play) are flagged explicitly so the LLM can reason about their impact.

4. **Pre-computed derived stats**: The prompt provides FG%, 3P%, FT%, and total REB instead of raw makes/attempts. This is cleaner for LLM reasoning -- the model does not need to compute ratios from raw counting stats, reducing arithmetic errors and letting it focus on analysis.

5. **Data leakage mitigation**: We accept that some game results may exist in the LLM's training data. Mitigation strategy: (a) instruct the model explicitly not to use outcome knowledge, (b) compute per-season MAE to detect leakage signal (suspiciously better accuracy on older seasons would indicate memorization), (c) no web search tools provided to the model.

6. **No ensemble with transformer**: This experiment evaluates LLM standalone prediction ability. Combining LLM predictions with our Phase 3 transformer ensemble is deferred to a later phase. This keeps the evaluation clean and the results interpretable.

7. **Prompt structure**: System prompt with calibration context (league averages, home advantage base rates, score distribution guidance) + per-game user prompt containing:
   - Team rolling stats across 3 windows (last 5, 10, 20 games)
   - Recent game log (last 10 games with scores and opponents)
   - Player averages for top 10 players per team (using derived stats: PTS, REB, AST, FG%, 3P%, FT%, MIN)
   - Target game roster with flagged absences
   - Head-to-head record for the season

8. **Output format**: JSON with `analysis` (free-text reasoning), `home_score`, `away_score`, and `home_win_probability`. Spread is derived from the score difference (`home_score - away_score`) rather than predicted independently, ensuring internal consistency.
