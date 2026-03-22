# Level 4: Game Context -- Industry & Practical Research Summary

Level 4 takes team-level representations from Levels 1-3 and incorporates game-specific
context (home/away, rest, travel, Vegas lines, injuries) to produce final predictions. This
document surveys industry practice, betting market mechanics, and real-world prediction
systems to inform our implementation.

---

## 1. How Sportsbooks Set NBA Lines

### 1.1 The Line-Setting Process

Modern NBA lines are produced through a hybrid algorithmic-human pipeline:

1. **Power ratings**: Each sportsbook maintains proprietary power ratings for every team,
   updated daily. The basic formula is:
   `Predicted Spread = Team A Rating - Team B Rating + Home Court Adjustment`
   Power ratings incorporate offensive/defensive efficiency, recent form, and roster quality.

2. **Algorithmic first pass**: Statistical models process historical data, player metrics,
   team performance, scheduling factors, and injuries to generate an initial line. This is
   increasingly ML-driven at major books.

3. **Human adjustment**: Experienced oddsmakers review and adjust for factors models may
   miss: team motivation, coaching changes, locker room dynamics, load management patterns.

4. **Market opening**: Lines open at low limits to attract sharp (professional) bettors
   whose action helps discover the true price. Market-making books like Pinnacle and BetCRIS
   open first, establishing the benchmark.

5. **Price discovery**: As money flows in, the line adjusts. Sharp money moves the line
   toward efficiency; public money can create temporary imbalances. By closing, the line
   represents the wealth-weighted consensus of all market participants.

6. **Closing line**: Represents the most accurate pre-game estimate. Studies confirm closing
   lines are significantly more predictive than opening lines, as they incorporate all
   available public and private information.

**Key insight**: The goal is not balanced action on both sides (common misconception).
Market-making books seek to post the most accurate number and profit from volume and vig.
Retail books are more likely to shade toward balanced action.

Sources:
- [How Bookmakers Set Lines](https://themidfield.com/nba-betting/how-bookmakers-set-lines/)
- [Who Sets The Betting Line? The Market Makers](https://unabated.com/articles/who-sets-the-sports-betting-line-market-makers)
- [How Sportsbooks Set Odds: Soft vs Sharp Books](https://help.outlier.bet/en/articles/9922960-how-sportsbooks-set-odds-soft-vs-sharp-books)

### 1.2 Market-Making Hierarchy

Not all sportsbooks are equal. The hierarchy matters for data quality:

| Tier | Example | Role | Characteristics |
|------|---------|------|-----------------|
| Market maker | Pinnacle, BetCRIS | Set opening lines, accept sharp action | Low vig, high limits, fastest adjustment |
| Sharp-accepting | Circa, BookMaker | Early movers, accept large wagers | Lines closely track market makers |
| Retail | FanDuel, DraftKings, Caesars | Copy market maker lines, shade to public | Higher vig, limit sharp bettors |

BetCRIS has become the primary market-maker for NBA in recent years, especially for US
sports, handling the largest decision amounts. Pinnacle remains the gold standard for
efficient closing lines.

**Practical takeaway for our model**: If incorporating Vegas lines as features, prefer
Pinnacle/BetCRIS closing lines for maximum signal. Opening lines contain exploitable
inefficiencies that closing lines largely eliminate.

Sources:
- [Which Sportsbooks Are Sharp?](https://www.pikkit.com/blog/which-sportsbooks-are-sharp)
- [What is a Market Setting Sportsbook?](https://www.sportsinsights.com/blog/what-is-a-market-setting-sportsbook/)

### 1.3 How Lines Move After Injury News

Line response to injury news is measured in **seconds**, not minutes:

- Sportsbooks use automated algorithms that scrape social media, team feeds, and insider
  accounts for keywords like "out," "torn," or "indefinitely."
- For star players, lines can move **3-5 points** within minutes of announcement. A top-five
  NBA player can be worth 4-6 points on a spread.
- Example: If LeBron James is ruled out shortly before tip-off, the Lakers' spread might
  shift from -3.5 to +1.5 (a 5-point swing).
- For major injury news, some books take games **off the board** entirely, re-posting updated
  lines within hours.
- Moneyline follows suit: a -170 favorite might drop to -125 after losing a star player.

**Practical takeaway**: Injury status at lineup lock is critical. Pre-game predictions
without injury data are significantly less accurate. Even "questionable" designations affect
lines by 1-2 points for important players.

Sources:
- [How Injuries Affect Betting Lines](https://www.yardbarker.com/general_sports/articles/how_injuries_affect_betting_lines_a_guide_to_market_movement/s1_17354_43584377)
- [Role of Injuries in NBA Betting](https://hoopheadspod.com/the-role-of-injuries-in-nba-betting-how-line-movements-reflect-player-absences/)
- [Top 5 Ways Player Injuries Affect NBA Betting Lines](https://www.bettoredge.com/post/top-5-ways-player-injuries-affect-nba-betting-lines)

---

## 2. Successful NBA Betting Models (Public Cases)

### 2.1 Haralabos Voulgaris

The most documented case of sustained NBA betting success:

**Early edge (late 1990s-early 2000s)**: Exploited a simple structural flaw -- sportsbooks
split projected total points evenly between halves, but NBA second halves consistently
score more due to fouls and timeouts. He bet second-half overs at ~70% win rate.

**Evolution to quantitative modeling (2005-2018)**:
- Built the "Ewing" model with a math prodigy known as "The Whiz" starting ~2007.
- Ewing was a possession-based simulation model that:
  - Assigned offensive and defensive values to every player
  - Modeled coaching tendencies (timeout usage, play-calling, substitution patterns)
  - Incorporated officiating tendencies
  - Projected final scores through game simulation
- Targeted specific coaches whose tendencies were predictable (Eddie Jordan, Jerry Sloan,
  Byron Scott).
- Achieved **>6% ROI across 1,000+ bets per season** from 2009 onward.
- Bet over $1 million on single days of NBA action.
- His bets moved lines -- sportsbooks began adjusting based on where he placed money.

**Career outcome**: Hired by Dallas Mavericks in 2018 as Director of Quantitative Research
and Development. Left in 2021. Now owns Spanish soccer club CD Castellon, applying
analytical methods.

**Key lessons for our model**:
- Game simulation approaches can find edges invisible to team-level models.
- Coaching tendencies and officiating patterns are underappreciated signal sources.
- Edges erode as markets adapt -- continuous model improvement is essential.
- The shift from simple structural exploitation to sophisticated simulation mirrors the
  evolution of the betting market itself.

Sources:
- [Voulgaris: The Gambler Who Outsmarted Vegas](https://www.opencourt-basketball.com/2025/08/11/haralabos-voulgaris-the-gambler-who-outsmarted-vegas-and-changed-nba-betting-forever/)
- [10 People Who Got Rich on Sports Betting](https://www.tradematesports.com/en/blog/nba-bettor-haralabos-voulgaris-mavericks/)
- [Meet the World's Top NBA Gambler (ESPN)](https://www.espn.com/blog/playbook/dollars/post/_/id/2935/meet-the-worlds-top-nba-gambler)
- [Bob Voulgaris Bio](https://www.legalsportsbetting.com/famous-sports-bettors/bob-voulgaris/)

### 2.2 Documented Historical Edges

**Back-to-back fading**: The classic NBA betting angle.
- Historical record: betting against home favorites on second night of B2B yielded ~58% ATS
  (116-84 in one study from 2005+).
- Overall B2B fading: ATS record 2058-2118 (50.7%), barely above break-even after vig.
- **Current status**: "sharped within an inch of its life" -- the market now fully prices in
  B2B fatigue. The edge, if any, is <1% ROI and requires additional filters.
- Key sub-angles: road B2B + time zone crossing + short rest vs well-rested opponent still
  shows marginal value, but rarely enough to overcome vig alone.

**Early-season totals bias**: Academic research confirms NBA totals lines are significantly
biased early each season (first ~2 weeks), while sides lines show no such bias. Win rate of
56.72% against closing totals during biased periods. Market corrects this by mid-season.

**Closing Line Value (CLV) as edge indicator**: Consistently beating closing lines is the
single best indicator of long-term profitability. Bettors who beat the close 55-60% of the
time typically show positive ROI. Being early to the market yields ~1 point advantage on
spreads, ~2 points on totals, translating to ~55% win rate and ~3% ROI.

**Player props**: Most persistent edge in modern NBA betting. Props are priced with less
resources than game lines, and books cannot dedicate equal attention to hundreds of
individual markets. Sharp money corrects game spreads quickly but props retain
inefficiencies longer.

Sources:
- [NBA Betting Strategy 2026 - Research-Backed Systems](https://www.topendsports.com/betting-guides/sport-specific/nba/strategy.htm)
- [Don't Be Fooled by Blindly Fading B2B Teams (Action Network)](https://www.actionnetwork.com/nba/moore-dont-be-fooled-by-blindly-fading-teams-on-back-to-backs)
- [NBA Betting: Finding Your Path to Profitability](https://unabated.com/articles/nba-betting-path-to-profitability)
- [Pinnacle: Why Betting Early Can Be More Profitable](https://www.pinnacle.com/betting-resources/en/educational/why-betting-early-can-be-more-profitable-nba-opening-vs-closing-line-odds)
- [Learning, Price Formation and the Early Season Bias (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S1544612307000177)

### 2.3 Market Efficiency Benchmarks

The NBA betting market is among the most efficient in sports:

- **Vegas spread MAE**: ~9.1 points historically (2006-2016), rising to ~10.5 points in the
  modern high-variance era (2020-2026) due to increased three-point shooting.
- **ATS margin standard deviation**: 9.2-11.9 points across spread sizes, with a 0.60
  correlation between spread size and accuracy (larger spreads predicted more accurately).
- **Three-point variance as noise floor**: A player's 3PT% from one season explains only
  14.5% of variance the next season (vs 98% for FT%). The increased volume of 3-point
  attempts amplifies game-to-game variance, contributing to the rising MAE.
- **Raymond Sauer study**: Average difference between point spreads and actual point
  differences was less than 0.25 points across six NBA seasons, confirming market efficiency
  at scale.

**Practical takeaway**: Any model that achieves spread MAE consistently below ~10 points in
the modern era is competitive with Vegas. Our Phase 3 best (MAE 10.66) is within range but
has room to improve -- the remaining gap is partly skill, partly irreducible noise from
3PT variance.

Sources:
- [ATS Margin Standard Deviations by Point Spread](https://www.boydsbets.com/ats-margin-standard-deviations-by-point-spread/)
- [Predicting Scores Using Vegas Point Spreads](https://www.boydsbets.com/ats-margin-standard-deviations-by-point-spread/)
- [3-Point Variance in the NBA](https://www.binomialbasketball.com/p/the-nbas-3-point-variance-lie)

---

## 3. Game-Level Features Used by Production Systems

### 3.1 FiveThirtyEight / RAPTOR Model

FiveThirtyEight's NBA prediction system (now maintained by Neil Paine at substack) used a
player-based approach:

- **RAPTOR** (Robust Algorithm using Player Tracking and On/Off Ratings): Combined box score
  stats, player tracking metrics, and plus/minus data to estimate per-100-possession impact.
- **Game predictions**: Aggregated player-level RAPTOR projections using expected minutes for
  each game, then added game-level adjustments for home court, rest, and travel.
- **Monte Carlo simulation**: 50,000 simulations of remaining season to produce win
  probabilities and playoff odds.

**Performance**:
- RAPTOR picked the right game winner ~66.4% of the time.
- Elo (team-level) picked correctly ~67.0% -- essentially tied.
- Combined model (R-squared 0.210) modestly outperformed either alone.
- Both models were outperformed by simple team metrics (net rating) after a 7+ day burn-in
  period early in the season.
- RAPTOR spreads showed poor calibration: games predicted as coin flips (50%) had the home
  team winning only 35%, indicating systematic home team overestimation.

**Key lesson**: Player-based models do not clearly outperform team-level models for game
prediction, but they provide complementary information -- especially early in seasons when
rosters change, or after trades/injuries.

Sources:
- [How Our NBA Predictions Work (FiveThirtyEight)](https://fivethirtyeight.com/methodology/how-our-nba-predictions-work/)
- [FiveThirtyEight's NBA Predictions: RAPTOR vs ELO](https://vandomed.medium.com/fivethirtyeights-nba-predictions-raptor-vs-elo-254de5278645)
- [Evaluating RAPTOR Point Spreads](https://www.walker-harrison.com/posts/2022-05-27-evaluating-the-point-spreads-produced-by-538-s-raptor-metric/)

### 3.2 ESPN Basketball Power Index (BPI)

ESPN's BPI is a team-level metric reflecting how many points above/below average a team is:

- Based on **Real Plus/Minus** (RPM) player ratings rather than RAPTOR.
- Accounts for: game-by-game efficiency, strength of schedule, pace, **days of rest**, game
  location, and preseason expectations.
- Has a self-regulating mechanism to adjust projections when a team over/underperforms the
  combined RPM values of its players.
- Correlation of 0.91 with FiveThirtyEight's preseason ratings, indicating strong agreement
  between major public models.

Source:
- [ESPN's BPI for 2025-2026](https://www.espn.com/nba/story/_/id/46626742/introducing-espn-basketball-power-index-2025-2026-predictions-23-nba-teams)

### 3.3 Quantified Adjustments Used by the Industry

Based on aggregated research and market observations, here are the standard adjustments
sportsbooks and sharp bettors apply:

| Factor | Adjustment | Notes |
|--------|-----------|-------|
| **Home court (league avg)** | +2.0 to +3.0 pts | Down from ~3.5 pre-2014. Currently ~2.1 pts by independent analysis |
| **Denver home** | +4.5 to +5.5 pts | Net rating 8.6 pts better at home vs road (since 2000). Altitude effect adds ~1.5-2.0 pts beyond normal HCA |
| **Utah home** | +3.5 to +4.5 pts | Second-strongest HCA, arena at ~4,300 ft |
| **Oklahoma City home** | +3.5 to +4.5 pts | 83.7% home win rate in 2024-25 but HCA metric ranked only 18th |
| **No rest (B2B)** | -1.25 pts penalty | Per inpredictable.com analysis of 2003-2010 data |
| **Home on B2B vs rested visitor** | HCA reduced to ~2.0 pts | Down from ~3.25 baseline |
| **Rested home vs visitor on B2B** | HCA boosted to ~4.5 pts | +1.25 pts above baseline |
| **Star player out** | 3-6 pts swing | Top-5 player = 4-6 pts; role player = 0.5-1.5 pts |
| **Travel (per 500km)** | -4% win probability | Only statistically significant for Away-Home B2B sequences |
| **Eastward travel** | +3.7% win rate vs westward | 44.5% vs 40.8% winning percentage |
| **3+ timezone crossing** | ~1.5-2.0 pts | Circadian disruption; adapts at ~1 timezone per day |

**Important context**: These adjustments are already incorporated into market lines. To find
edge, a model must either (a) estimate these factors more precisely than the market, or
(b) identify interactions the market misses.

Sources:
- [NBA Home Court Advantage and Rest (inpredictable)](https://www.inpredictable.com/2012/02/nba-home-court-advantage-and-rest.html)
- [NBAsuffer: Rest Days Factor](https://www.nbastuffer.com/rest-days-factor-nba-scheduling/)
- [Impacts of Travel Distance (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8636381/)
- [Time Zones and NBA Results (Taylor & Francis)](https://newsroom.taylorandfrancisgroup.com/time-zones-and-tiredness-strongly-influence-nba-results-study-of-25000-matches-shows/)

---

## 4. Rest and Travel Effects (Detailed)

### 4.1 Rest Days Impact

The Entine & Small (2008) Wharton study found rest accounts for only **9% of home court
advantage** (~0.31 of 3.24 total points). However, this underestimates the full rest effect
because it only measures the differential, not the absolute impact.

**Inpredictable.com analysis (2003-2010)**:

| Home Rest | Road Rest | HCA Value | Sample Size |
|-----------|-----------|-----------|-------------|
| >0 days | >0 days | 3.25 pts | 4,795 games |
| >0 days | 0 days (B2B) | 4.50 pts | 485 games |
| 0 days (B2B) | >0 days | 2.00 pts | 1,887 games |
| 0 days (B2B) | 0 days (B2B) | 3.25 pts | 681 games |

Key insight: Road teams play on zero rest **4x more frequently** than home teams due to NBA
scheduling. This structural advantage is the mechanism by which rest contributes to HCA.

**Physiological basis**: NBA players need 48-72 hours to fully replenish glycogen stores
and allow microtrauma repair after high-intensity games. On B2Bs, athletes start at a
physiological deficit that compounds throughout the game.

Sources:
- [Entine & Small (2008) "The Role of Rest in NBA HCA"](https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Nba.pdf)
- [NBA Home Court Advantage and Rest (inpredictable)](https://www.inpredictable.com/2012/02/nba-home-court-advantage-and-rest.html)

### 4.2 Travel Distance and Direction

**PMC study (2013-2020 NBA seasons, back-to-back games)**:

Three B2B sequence types with significantly different outcomes:
- **Away-Home**: 54.4% win rate (best -- returning home)
- **Away-Away**: 39.2% win rate
- **Home-Away**: 36.8% win rate (worst -- leaving home)

Within Away-Home sequences: every additional **500 km** of travel reduces winning
probability by **~4%** (p=0.038). No significant distance effect in Away-Away or Home-Away
sequences.

**Travel direction**: Eastward travel yields 44.51% win rate vs 40.83% for westward
(p=0.024). This is counterintuitive but explained by circadian rhythm: the body adapts more
easily to longer days (westbound travel makes the day longer, but late-starting West Coast
games push East Coast teams past their circadian peak).

**Time zone crossing effects**: Western home teams show ~10% better win ratio vs eastern
visitors (63.5%) compared to when eastern teams host western visitors (55.0%). The circadian
mechanism: a 7:00 PM Pacific tip-off feels like 10:00 PM Eastern to a visiting East Coast
team.

**NBA scheduling adaptations**: Since 2017, the NBA has eliminated 4-games-in-5-days
schedules and single road games over 2,000 miles away, partially mitigating extreme travel
effects.

Sources:
- [Impacts of Travel Distance on NBA B2B Games (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8636381/)
- [Time Zones and NBA Results (ScienceDaily)](https://www.sciencedaily.com/releases/2024/05/240501091642.htm)
- [How Time Zones Affect NBA Performance](https://scienceblog.com/how-time-zones-affect-nba-performance/)

### 4.3 Feature Engineering Implications

For our model, rest and travel features should include:

1. **Days of rest** for each team (0, 1, 2, 3+)
2. **Rest differential** (home rest - away rest)
3. **B2B sequence type** (Home-Away, Away-Home, Away-Away, not-B2B)
4. **Travel distance** since last game (km or miles)
5. **Timezone crossings** since last game (0, 1, 2, 3)
6. **Direction of travel** (east, west, neutral)
7. **Games in last N days** (schedule density over 5-day and 7-day windows)

---

## 5. Injury Data and Player Availability

### 5.1 NBA Official Injury Report System

**Current designation system** (5 levels):

| Status | Official Probability | Actual Play Rate | Notes |
|--------|---------------------|-------------------|-------|
| Available | 100% | ~99% | Expected to play |
| Probable | 75% | ~90-95% | Very likely to play |
| Questionable | 50% | ~55% | True uncertainty; most volatile |
| Doubtful | 25% | ~3% | Almost certainly out |
| Out | 0% | 0% | Will not play |

The gap between official probabilities and actual play rates is notable: "Probable" players
play at ~95% (not 75%), while "Doubtful" players almost never play (~3%, not 25%). The
"Questionable" designation is the only one close to its stated probability.

### 5.2 Reporting Requirements (2024-25 Rules)

The NBA overhauled injury reporting rules in 2024-25 for gambling integrity:

- **Day-before deadline**: Teams must report by 5:00 PM local time the day before a game.
- **Game-day update**: Additional report required between 8-10 AM (for <=5 PM tip) or 11 AM
  to 1 PM (for later tip).
- **Back-to-back exception**: Second game of B2B reports due by 1:00 PM local time on game
  day.
- **Accuracy enforcement**: League reviews teams that consistently misuse designations
  (e.g., listing players as "Questionable" who always play).

**Practical impact**: These rules create structured reporting windows that can be used as
data collection points. The day-before report captures the most impactful information; the
game-day update catches late changes.

### 5.3 Modeling Player Availability

**Converting injury status to model input**:
- Use actual play rates (not official probabilities) as priors
- Weight by player impact: a "Questionable" Nikola Jokic is worth 3-5 points of spread
  uncertainty; a "Questionable" 12th man is worth ~0
- **Player impact estimation**: BPM (Box Plus/Minus) per minute multiplied by expected
  minutes gives a spread contribution per player. Each point of team differential translates
  to approximately 2.7 wins per season.

**Data sources for our pipeline**:
- Official NBA injury reports (our InjuryReports table, 2021+)
- ESPN injury status feed
- RotoWire injury reports with return timeline projections
- SportsDataIO Process Guide for standardized injury status taxonomy
- BALLDONTLIE API for programmatic access to current rosters and injuries

Sources:
- [NBA Injury Reporting Overhaul](https://dallashoopsjournal.com/p/nba-injury-reporting-rules-overhaul-explained/)
- [NBA Injury Reports for Betting Strategy (VSiN)](https://vsin.com/nba/nba-injury-reports-how-to-use-them-to-inform-your-betting-strategy/)
- [SportsDataIO Injury Process Guide](https://support.sportsdata.io/hc/en-us/articles/9911200480663-Process-Guide-Injuries)

---

## 6. Altitude and Arena-Specific Factors

### 6.1 Denver Altitude Effect (5,280 ft)

Denver has the **strongest home court advantage** in the NBA since 1999-00:

- **Historical home-road gap**: Net rating 8.6 points better at home vs road (league avg:
  6.0 points). This is a **2.6 point additional advantage** beyond normal HCA.
- **All-time home win percentage**: .652 (vs .350 road -- a .302 gap, largest for any active
  franchise in NBA history).
- **Expected home win rate**: 66.1% vs ~62% league average (MSU Denver research).
- **2023 playoffs**: 8-0 home record; led all teams with 19.3 fast-break points per game at
  home.

**Physiological mechanism**: At 5,280 ft, barometric pressure is lower, meaning less oxygen
to working muscles, causing premature fatigue:
- Visiting player average speed: 4.20 mph (Q1) declining to 3.89 mph (Q4)
- Time spent walking/standing: 69.1% (Q1) increasing to 73.0% (Q4)
- Athletes report insomnia, frequent awakening, restless sleep at altitude
- Full acclimatization requires ~6 days (NBA schedule never allows this)
- Estimated extra wins from altitude alone: ~2.5 per season for Denver and Utah

### 6.2 Utah Altitude Effect (~4,300 ft)

Utah consistently ranks as the **second-strongest** home court advantage in the NBA,
attributed to its elevation at roughly 4,300 ft above sea level. The effects are similar to
Denver but at reduced magnitude.

### 6.3 Other Arena Factors

NBA courts are standardized (94 x 50 ft), eliminating the venue-specific quirks found in
baseball or hockey. Remaining factors:

- **Crowd intensity**: Varies significantly by team and season. COVID studies showed crowd
  presence accounts for ~half of HCA effect.
- **Referee behavior**: Home teams receive approximately 1-1.5 additional free throws per
  game on average. Research is mixed on whether this is conscious bias or crowd-induced
  unconscious influence. COVID data suggests crowd influence on effort (rebounding) rather
  than referee bias is the primary mechanism.
- **Arena acoustics**: No quantified research, but some arenas are known to be louder per
  seat capacity.
- **Late-night starts**: West Coast games starting at 10:00+ PM Eastern affect East Coast
  visitors' circadian rhythm (see Section 4.2).

**Practical takeaway for modeling**: Altitude should be an explicit feature for Denver and
Utah games. Other arena factors are better captured implicitly through team-specific HCA
learning rather than explicit features.

Sources:
- [Denver Altitude Advantage (ESPN)](https://www.espn.com/nba/story/_/id/37762170/nba-finals-2023-how-denver-altitude-gives-nuggets-edge)
- [Denver Home Court Advantage (Sportico)](https://www.sportico.com/leagues/basketball/2024/denver-nuggets-home-court-advantage-1234777526/)
- [Is Impact of Denver's Altitude Fact or Fiction? (NBA.com)](https://www.nba.com/news/is-impact-of-denvers-altitude-fact-or-fiction)
- [Quantifying Implicit Biases in NBA Refereeing (Nature)](https://www.nature.com/articles/s41598-023-31799-y)

---

## 7. Ensemble Approaches for Game Prediction

### 7.1 Deep Ensembles (Lakshminarayanan et al., 2017)

The gold standard for uncertainty estimation in neural networks:

- Train M independent networks (typically M=3-10) from different random initializations on
  the same data.
- Final prediction: mean of individual predictions.
- Uncertainty: variance of the mixture (treat ensemble as mixture of Gaussians).
- Produces well-calibrated uncertainty estimates that are as good or better than
  approximate Bayesian neural networks.
- Simple to implement, parallelizable, minimal hyperparameter tuning.

**Our Phase 3 Exp 9 already uses this**: 3 seeds of Exp 4 config with mean aggregation,
achieving our best results (AUC 0.718, MAE 10.66). This validates the approach.

Source:
- [Simple and Scalable Predictive Uncertainty Estimation (arXiv)](https://arxiv.org/abs/1612.01474)

### 7.2 Stacking / Meta-Learning

Recent NBA prediction research (Scientific Reports, 2025) tested stacked ensembles:

| Model | Accuracy | AUC |
|-------|----------|-----|
| Stacking (MLP meta-learner) | 83.27% | 92.13% |
| AdaBoost | 81.10% | 89.97% |
| XGBoost | 81.03% | 90.82% |
| Logistic Regression | 80.49% | 90.28% |
| KNN | 80.15% | 87.92% |
| Naive Bayes | 76.56% | 85.67% |
| Decision Tree | 74.19% | 79.87% |

The stacking approach achieved statistically significant improvement over all individual
base learners (p<0.05). Key: the base learners used were **diverse** -- different algorithm
families with different inductive biases.

**Note**: These accuracy numbers use post-game features (FG%, rebounds, etc.) which are not
available pre-game. Pre-game prediction accuracy is fundamentally lower (~65-70% for win
classification).

Source:
- [Stacked Ensemble Model for NBA Game Outcome Prediction (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12357926/)

### 7.3 When Diversity Helps vs Hurts

Research on ensemble diversity yields clear guidelines:

**When ensembles help**:
- Base models make **independent errors** (low correlation between predictions)
- Models use different architectures, features, or learning algorithms
- Individual accuracy is above random chance (p > 0.5 for binary classification)
- The "ambiguity decomposition": ensemble error = average error - average disagreement

**When ensembles hurt**:
- All models are highly correlated (same architecture, same features, same hyperparameters)
- "Bad diversity" (models that disagree but in uninformative ways) can increase risk
- Adding very weak models dilutes ensemble quality

**Practical guidelines**:
- Pearson correlation between model predictions measures similarity. Maximum diversity when
  correlation = 0.
- Combining a few top performers with a few complementary weaker models often outperforms
  combining only top performers.
- For our project: combining transformer model (Phase 3) with a structurally different model
  (e.g., XGBoost on engineered features, or a simulation-based model) would provide more
  diversity than combining multiple transformer seeds (which we already do).

Sources:
- [A Unified Theory of Diversity in Ensemble Learning (JMLR)](https://jmlr.org/papers/volume24/23-0041/23-0041.pdf)
- [Ensemble Diversity for Machine Learning](https://machinelearningmastery.com/ensemble-diversity-for-machine-learning/)
- [Ensemble Modeling in Sports (Harvard Science Review)](https://harvardsciencereview.org/2025/10/01/ensemble-modeling-in-sports-combining-algorithms-for-stronger-predictions/)

### 7.4 Mixture of Experts for Game Prediction

A Mixture of Experts (MoE) architecture could be particularly suited to NBA prediction:

- **Experts**: Different models specialized for different game contexts (e.g., one expert for
  B2B games, one for well-rested matchups, one for altitude games, one for rivalry games).
- **Gating network**: Learns which expert(s) to trust based on input features (rest days,
  travel, venue, etc.).
- **Advantages**: Only activates relevant experts per input, reducing computational cost and
  allowing specialization.
- **Practical consideration**: Requires enough training data per context type to train
  specialized experts. With ~1,300 games per NBA season, some contexts (e.g., altitude B2B
  games) may have too few examples.

Sources:
- [Mixture of Experts (Wikipedia)](https://en.wikipedia.org/wiki/Mixture_of_experts)
- [MoE with Gating Network (Medium)](https://medium.com/@sudeepdc/mixture-of-experts-using-gating-network-with-ensemble-learning-ea83ea294db8)

---

## 8. Vegas Lines as Model Features

### 8.1 The Case For Incorporating Lines

Vegas closing lines are the most information-dense pre-game signal available:

- They encode team quality, injuries, rest, travel, motivation, and public sentiment.
- They are produced by a market with billions of dollars at stake and sophisticated
  participants.
- Academic research shows they are **unbiased estimators** of game margins (average error
  <0.25 points across large samples).

**Integration approaches**:
1. **As a feature**: Include opening/closing line as an input feature alongside model
   predictions. A meta-learner decides optimal weighting.
2. **As a prior**: Use Vegas line as a Bayesian prior, with model output as the update.
3. **As a benchmark**: Use for calibration and evaluation only, not as input.
4. **Blended prediction**: Walker Harrison's analysis showed that RAPTOR alone doesn't beat
   Vegas, but a blend of RAPTOR + Vegas lines yielded 7.82% ROI on moneylines.

### 8.2 The Case Against

- **Circularity risk**: If our model trains on Vegas lines, we learn to mimic Vegas rather
  than discover independent signal.
- **Availability**: Lines may not be available for all historical games in our training set.
- **Late changes**: Lines move right up to tip-off; our model needs a fixed prediction time.
- **Independence**: A model that adds value beyond Vegas is more useful than one that
  requires Vegas as input.

**Our recommended approach**: Train the model WITHOUT Vegas lines to maximize independent
signal discovery. Use Vegas lines for evaluation and as an optional late-fusion feature at
inference time.

Sources:
- [Evaluating RAPTOR vs Vegas (Walker Harrison)](https://www.walker-harrison.com/posts/2022-05-27-evaluating-the-point-spreads-produced-by-538-s-raptor-metric/)
- [Making Real-Time NBA Predictions by Combining Historical Data and Betting Lines (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S0378437120301618)

---

## 9. Practical Takeaways for Level 4 Implementation

### 9.1 Feature Priority (Ranked by Expected Impact)

1. **Injury/availability status** (highest impact -- 3-6 pts for star players)
2. **Home/away indicator** (2-3 pts baseline; varies by team)
3. **Rest days and B2B status** (1.25 pts per zero-rest day)
4. **Altitude flag** (for Denver/Utah games -- adds 1.5-2.5 pts to home advantage)
5. **Travel distance/direction** (~4% win probability per 500km for B2B sequences)
6. **Timezone crossings** (~1.5-2 pts for 3+ timezone shift)
7. **Schedule density** (games in last 5/7 days)
8. **Vegas line** (optional late-fusion feature for maximum accuracy)

### 9.2 Architecture Recommendations

Based on industry practice:

- **Additive adjustments**: Most sportsbooks start with a team quality differential and add
  context adjustments. This is equivalent to a residual/additive architecture where Level 4
  learns context-dependent offsets to the Level 3 team representation differential.
- **Learned HCA per team**: Do not use a fixed home court value. Let the model learn
  team-specific HCA, ideally with era-awareness (HCA has declined league-wide).
- **Interaction features**: Rest x travel, altitude x B2B, and star player injury x team
  depth are likely underpriced by simple additive models.
- **Calibration**: Apply temperature scaling post-hoc to win probability outputs. Our
  Phase 3 ensemble showed ECE regression (0.0378); temperature scaling is the standard
  industry fix.

### 9.3 Data Availability Summary

| Feature | Source | Coverage |
|---------|--------|----------|
| Home/away | Games table | All seasons |
| Rest days | Computed from schedule | All seasons |
| Injury reports | InjuryReports table | 2021+ only |
| Travel distance | Computed from team cities | All seasons |
| Altitude | Static lookup (Denver=5280, Utah=4265, etc.) | All seasons |
| Timezone | Static lookup by city | All seasons |
| Vegas lines | External data source needed | Variable |
| Player availability | Inferred from PlayerBox (min>0) | All seasons |

### 9.4 Expected Model Improvement

Based on industry benchmarks:
- Adding game context features to a team-quality model typically improves win prediction
  accuracy by **2-5 percentage points** (e.g., from 65% to 67-70%).
- Spread MAE improvement of **0.5-1.5 points** is realistic from context integration.
- The largest gains come from injury data (when available) and rest/scheduling features.
- Diminishing returns: beyond the core features listed above, additional context features
  (weather, referee assignments, motivation) yield marginal-to-zero improvement in practice.
