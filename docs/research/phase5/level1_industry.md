# Level 1: Industry Player Evaluation Systems Research

Research compiled 2026-03-16. Covers production-grade and public NBA player modeling systems, with emphasis on practical implementation details relevant to building our own Level 1 Individual Player Models.

---

## Table of Contents

1. [DARKO (Daily Plus-Minus)](#1-darko-daily-plus-minus)
2. [ESPN's EPM and Net Points](#2-espns-epm-and-net-points)
3. [FiveThirtyEight's RAPTOR and CARMELO](#3-fivethirtyeights-raptor-and-carmelo)
4. [NBA 2K Ratings System](#4-nba-2k-ratings-system)
5. [DraftKings / FanDuel Projection Systems](#5-draftkings--fanduel-projection-systems)
6. [Synergy Sports / Second Spectrum / NBA Tracking Data](#6-synergy-sports--second-spectrum--nba-tracking-data)
7. [Basketball-Reference Advanced Stats (BPM, VORP, Win Shares)](#7-basketball-reference-advanced-stats-bpm-vorp-win-shares)
8. [Other Notable Systems (DRIP, DraftKings Kalman)](#8-other-notable-systems-drip-draftkings-kalman)
9. [Key Takeaways for Our Player Model](#9-key-takeaways-for-our-player-model)

---

## 1. DARKO (Daily Plus-Minus)

**Creator:** Kostya Medvedovsky | **Website:** [apanalytics.shinyapps.io/DARKO](https://apanalytics.shinyapps.io/DARKO/)

DARKO (Daily Adjusted and Regressed Kalman Optimized projections) is one of the best publicly available daily-updating player models. It projects every box-score stat for every NBA player, every day.

### 1.1 Core Architecture

DARKO combines three key components:

1. **Exponential Decay Model**: Each game a player has ever played is weighted by `beta^t`, where `beta` is between 0 and 1 and `t` is the number of days since the game. With `beta = 0.99`, a game from one year ago gets weight `0.99^365 = 0.025`, making it ~40x less important than yesterday's game. This is applied independently to each box-score stat.

2. **Modified Kalman Filter**: A standard time-series technique used in robotics and rocketry, adapted for basketball. The Kalman filter treats each player's true skill as a hidden state and observed box-score stats as noisy measurements. It maintains both an estimate and an uncertainty (confidence interval). The update step adjusts the estimate proportionally to the gap between prediction and observation, weighted by the filter's uncertainty.

3. **Gradient Boosted Decision Trees**: A GBDT model combines the exponential decay projections and Kalman filter projections into final stat predictions. This allows non-linear interaction between the two approaches and likely captures patterns that neither approach handles alone.

### 1.2 What DARKO Projects

- Every box-score stat (points, rebounds, assists, steals, blocks, turnovers, FGA, FGM, 3PA, 3PM, FTA, FTM, minutes, etc.)
- DPM (Daily Plus-Minus) -- an estimate of the player's impact per 100 possessions
- O-DPM and D-DPM (offensive/defensive splits)
- Daily salary projections for DFS
- Win projections when aggregated to team level

### 1.3 Environmental Adjustments

DARKO accounts for context on a stat-by-stat basis:
- **Rest/travel effects**: Back-to-back games, travel distance
- **Home court advantage**: Component-level adjustments
- **Opponent adjustments**: Projects how a specific opponent influences each stat category (e.g., facing a team that allows more rebounds nudges the rebound projection up)

### 1.4 Handling Rookies

DARKO has a significant limitation with rookies: it uses **no pre-NBA data** (no NCAA stats, summer league, or preseason). All rookies are initialized to approximately the same starting point, with minor differences for age. DARKO then "learns" about them as they play. This means:
- Early-season rookie projections are heavily regressed toward the prior
- DARKO cannot distinguish a #1 pick from a second-round pick before they play
- The Bayesian nature means the update size depends on confidence in the prior -- low-minute rookies stay regressed longer

### 1.5 Prediction Accuracy

- Medvedovsky finished 1st place in APBRmetrics prediction contests in 2016 and 2018
- DARKO is measured by mean absolute error per player-game (lower = better)
- DARKO's game-level win projections are competitive with but do not consistently beat betting market lines (markets have ~53-54% accuracy on game outcomes)

### 1.6 Stabilization Rates (from Medvedovsky's research)

Medvedovsky advocates the **"padding method"** for determining how much to trust observed stats vs. priors:
- For 3-point shooting: pad with ~240 attempts of league-average performance. Before January with ~237 actual attempts, roughly half the projection comes from observed data and half from the prior.
- For five-man lineup offensive rating: ~550 possessions to stabilize
- For five-man lineup defensive rating: ~850 possessions to stabilize
- Key insight: **Reliability is a spectrum, not a threshold.** There is no single point where a stat "stabilizes."

### 1.7 Implementation Relevance

**What to adopt:**
- The exponential decay weighting scheme is simple and effective. We already use rolling windows; exponential decay is a strict improvement.
- The Kalman filter framework for per-stat skill estimation with uncertainty tracking is directly applicable.
- Stat-by-stat opponent adjustments (not just "good defense" vs "bad defense" but opponent effect on each specific category).

**What to note:**
- DARKO's no-NCAA-data approach for rookies is a weakness we could improve on by incorporating draft position, college stats, and combine data.

**Sources:**
- [NBAstuffer: DARKO Explained](https://www.nbastuffer.com/analytics101/darko-daily-plus-minus/)
- [Kostya Medvedovsky on DARKO (Podcast)](https://podcasts.apple.com/us/podcast/8-kostya-medvedovsky-darko-and-assorted-nba-analytics/id1469516033?i=1000549030577)
- [Medvedovsky: NBA Stabilization Rates](https://kmedved.com/2020/08/06/nba-stabilization-rates-and-the-padding-approach/)
- [DARKO on BuyMeACoffee](https://buymeacoffee.com/darko)

---

## 2. ESPN's EPM and Net Points

### 2.1 EPM (Estimated Plus-Minus)

**Creator:** Taylor Snarr (data scientist, former Utah Jazz analytics coordinator) | **Website:** [dunksandthrees.com/epm](https://dunksandthrees.com/epm)

EPM is an all-in-one player metric that estimates a player's contribution in points per 100 possessions. It has consistently ranked as one of the most predictive single-number player metrics.

#### Architecture (Two-Stage)

**Stage 1 -- Statistical Plus-Minus (SPM) Model:**
- Uses "Estimated Skills" -- machine-learned projections of each player's true current ability in every statistical category
- Each stat has a unique predictive model optimized via **Differential Evolution** (a genetic optimization algorithm)
- Skills account for: how each stat stabilizes over time, full career performance up to that point, age curves, team/opponent strength, seasonality
- Incorporates player-tracking data (available from 2013-14 onward)
- Uses data from **entire careers**, not just current season

**Stage 2 -- RAPM with SPM Prior:**
- Regularized Adjusted Plus-Minus (RAPM) calculation using the SPM estimate as a Bayesian prior
- This captures impact that doesn't show up in any stat (e.g., gravity, screen-setting quality)
- Ridge regression with the prior pulls the estimate toward the box-score prediction when on/off data is sparse

#### Key Innovation: "Estimated Skills"

Rather than using raw per-game stats, EPM first projects what each stat "should" be given the player's full history. This is critical because:
- Raw stats are noisy game-to-game
- A player's recent 5-game 3PT% is much less informative than their career-adjusted 3PT skill
- Different stats stabilize at different rates (3PT% needs ~750 attempts; rebound rate needs ~200 possessions)
- The Differential Evolution optimizer finds the best parameters for each stat's projection model independently

#### Accuracy

- EPM and ESPN's RPM (which uses a similar RAPM + prior approach) consistently outperform all other public all-in-one metrics in predictive accuracy
- EPM updates nightly and generates 100,000 season simulations for team-level predictions

### 2.2 Net Points

**Creator:** Dean Oliver (ESPN) | **Launched:** March 2025

Net Points is ESPN's newest metric. Unlike EPM (which estimates season-level impact), Net Points is a **single-game impact metric** that assigns credit and blame on every play.

#### How Credit Is Divided

For every possession, Net Points evaluates:

1. **Shots**: Credit divided between shooter and passer. A self-created 3-pointer gives more credit to the shooter than a wide-open catch-and-shoot 3, where credit is split with the passer. Shot difficulty (closest defender distance, shot type, location) affects the credit split.

2. **Rebounds**: Credit depends on rebound difficulty. A contested defensive rebound in traffic is worth more than an uncontested long rebound.

3. **Turnovers**: Blame assigned based on the type and context. A steal-induced turnover assigns some blame to the ball-handler and credit to the defender.

4. **Free Throws**: Credit based on how the foul was drawn (drives to the basket, post-ups, etc.).

#### Data Sources

Net Points uses three data layers:
- Traditional box score
- Play-by-play data (shot type, assist/non-assist, turnover type)
- Player tracking data (shot difficulty, defender proximity)

#### Key Insight: Play-by-Play > Box Score

A key finding from Net Points: **the quality of a shot created by an assist is captured much better by play-by-play data than the box score.** The box score says "assist" but doesn't distinguish between a difficult cross-court pass to a contested shot vs. a simple swing pass to a wide-open corner 3.

#### Limitations

The initial version does not fully incorporate play types, matchups, or help defense. Oliver has stated these will be added in future iterations.

**Sources:**
- [Dunks & Threes: About EPM](https://dunksandthrees.com/about/epm)
- [ESPN: Introducing Net Points](https://www.espn.com/nba/story/_/id/44093220/introducing-net-points-latest-nba-metric-amazing-early-findings)
- [NBAstuffer: EPM Explained](https://www.nbastuffer.com/analytics101/estimated-plus-minus/)
- [NBAstuffer: Net Points Explained](https://www.nbastuffer.com/analytics101/net-points-espn/)
- [Engelmann Substack: DIY Net Points](https://jeremiasengelmann.substack.com/p/espn-just-released-net-points-a-new)

---

## 3. FiveThirtyEight's RAPTOR and CARMELO

### 3.1 RAPTOR

**Full name:** Robust Algorithm (using) Player Tracking (and) On/Off Ratings

RAPTOR measures points above average per 100 possessions, split into offensive and defensive components. It was FiveThirtyEight's primary player metric from 2019 until the site's restructuring.

#### Two Components

**Box Component (~60-70% weight):**
- Uses traditional box-score stats PLUS player-tracking stats and play-by-play derived stats
- Variables chosen by evaluating which predict long-term RAPM (6+ years of data)
- On defense: uses tracking data for nearest-defender frequency, opponent shooting at defender's position, induced offensive fouls
- Key lesson from predecessor DRAYMOND: **don't weight opponent FG% by nearest defender too heavily** -- nearest defender on 3-pointers is often random, and whether the shot goes in adds more randomness

**On-Off Component (~30-40% weight):**
- Three-layer courtmate network:
  1. Team performance when the player is on the floor
  2. How the player's most common courtmates perform WITHOUT the player
  3. How those courtmates' OTHER courtmates perform without those courtmates
- All adjusted for strength of competition
- Weighted by: (possessions courtmate has WITH player) x (possessions courtmate has WITHOUT player)

#### Stabilization

RAPTOR's key advantage: it correlates well with long-term RAPM while stabilizing **much faster**. Pure RAPM needs years of data; RAPTOR becomes useful within ~20-30 games because the box-score component provides a strong prior.

#### Data Available

FiveThirtyEight published historical RAPTOR data on GitHub: [fivethirtyeight/data/nba-raptor](https://github.com/fivethirtyeight/data/tree/master/nba-raptor). The dataset includes raptor_offense, raptor_defense, raptor_total, war_total, and other fields per player-season, going back to 2001.

### 3.2 CARMELO

**Full name:** Career-Arc Regression Model Estimator with Local Optimization

CARMELO projects player careers by finding the most similar historical comparables and using their career arcs as templates.

#### Veteran Projection Features

For established players, CARMELO matches on:
- Box-score stats (per-possession rates)
- Age (treated as absolute for veterans)
- Playing time / role
- Physical profile (height, weight)

#### Rookie Projection Features

For rookies, CARMELO uses a distinct feature set:
- **Age** (slightly flexible -- a 21-year-old pick can match a 20-year-old comparable)
- **Draft position** (strong signal for rookie quality)
- **Height**
- **Weight**
- **Position**
- **College stats** (adjusted for pace and strength of schedule, available since 2001 via ESPN Stats & Info)
- **Scouting rankings** (pre-draft consensus boards)

#### Accuracy

- ~55% of players finished within +/- 1 WAR of CARMELO's projection
- ~80% of players finished within +/- 2 WAR
- Ranked 2nd out of 20 projection systems tracked by APBRmetrics

#### Implementation Relevance

CARMELO's similar-player matching for rookies is conceptually elegant but has a fundamental limitation: it requires a large historical database of players who "look like" the current player, and the NBA changes fast enough that 20-year-old comps may not be relevant. More importantly, the matching happens in a relatively low-dimensional space -- a neural embedding approach could capture more nuanced similarities.

**Sources:**
- [FiveThirtyEight: How RAPTOR Works](https://fivethirtyeight.com/features/how-our-raptor-metric-works/)
- [FiveThirtyEight: Introducing RAPTOR](https://fivethirtyeight.com/features/introducing-raptor-our-new-metric-for-the-modern-nba/)
- [FiveThirtyEight: CARMELO Methodology](https://fivethirtyeight.com/features/how-were-predicting-nba-player-career/)
- [GitHub: RAPTOR Data](https://github.com/fivethirtyeight/data/blob/master/nba-raptor/README.md)
- [NBAstuffer: RAPTOR Explained](https://www.nbastuffer.com/analytics101/raptor/)

---

## 4. NBA 2K Ratings System

### 4.1 Official Methodology

NBA 2K uses ~50 attributes per player that feed into position-specific overall rating formulas.

#### Data Sources

1. **NBA Draft Combine measurements**: Sprint speed, vertical leap, wingspan, etc.
2. **Statistical analysis**: FG%, 3PT%, FT%, rebounds, assists, steals, blocks, etc.
3. **Film study**: Developers watch hours of footage to capture elements not in stats (e.g., comparing combine sprint speed to in-game speed while dribbling)
4. **Contextual analysis**: Not just raw percentages -- they consider HOW shots are taken (off screens vs. pull-up vs. catch-and-shoot), shot difficulty, etc.

#### Position-Specific Formulas

The league became more specialist-oriented, so 2K now has **seven different overall-rating formulas per position**. A player's ~30 sub-attributes are fed through a formula specific to their position and style (e.g., "defensive point guard" vs. "outside-shooting small forward").

#### Tendencies

Beyond attributes, 2K also assigns ~50 "tendencies" that govern AI behavior (how often a player drives, posts up, takes pull-up 3s, etc.). These are largely derived from Synergy Sports play-type data.

### 4.2 Reverse Engineering 2K Ratings from Real Stats

Multiple academic and community efforts have successfully predicted 2K ratings from real NBA stats:

| Approach | Model | Result |
|----------|-------|--------|
| XGBoost | Real stats -> 2K overall | MAE 1.33, MSE 1.75 |
| Random Forest | 24 NBA stats -> 2K rating | Avg error 2.57% |
| KNN | Box-score stats -> rating | Outperformed Lasso and XGBoost in one study |

Key finding: **2K ratings are highly predictable from real stats** (R-squared > 0.90 in most studies), confirming that the rating system, while incorporating subjective elements, is fundamentally grounded in statistical reality.

### 4.3 Implementation Relevance

The 2K system's value for our purposes is limited but offers one insight: **position-specific evaluation matters.** A center's value is captured by different stat patterns than a point guard's. Our model should account for position when interpreting raw stats -- either through position embeddings or position-conditional normalization.

**Sources:**
- [Complex: How NBA 2K Determines Player Rankings](https://www.complex.com/sports/a/kevin-wong/how-nba-2k-determines-player-rankings)
- [Towards Data Science: NBA 2K Ratings Prediction](https://towardsdatascience.com/sport-analytics-nba-2k-ratings-prediction-b7b72e2e72eb/)
- [Springer: NBA 2K20 Rating Predictions with ML](https://link.springer.com/chapter/10.1007/978-3-031-78940-3_14)
- [2KRatings.com: Attribute Definitions](https://www.2kratings.com/nba-2k-attributes-definitions)

---

## 5. DraftKings / FanDuel Projection Systems

### 5.1 Architecture of DFS Projection Models

Daily fantasy projection models generally follow a pipeline:

```
Minutes Prediction -> Per-Minute Stat Rates -> Context Adjustments -> Fantasy Point Total
```

#### Step 1: Minutes Projection (Most Critical)

Minutes are the single most important variable in DFS. Standard approach:

```
Projected Minutes = (Season Avg MPG * 0.65) + (Last 5 Games Avg MPG * 0.35)
```

Adjusted for:
- Injury reports (same-day announcements can shift minutes dramatically)
- Blowout risk (high spread -> starters may sit in 4th quarter)
- Back-to-back rest patterns
- Foul trouble matchups (centers prone to fouls vs. aggressive post players)

Industry insight: **50+ projection updates in the final 30 minutes before slate lock** are common, driven primarily by injury and lineup news.

#### Step 2: Per-Minute Stat Rates

Baseline stat projections use per-minute rates (not per-game totals) to isolate production from opportunity:
- Points per minute
- Rebounds per minute
- Assists per minute
- Steals, blocks, turnovers per minute

These are typically estimated with recency weighting (similar to DARKO's exponential decay).

#### Step 3: Matchup and Context Adjustments

| Factor | Weight | Description |
|--------|--------|-------------|
| Pace | 3-5% | More possessions = more opportunities. Compare team avg pace to projected pace. |
| Team total | 5-7% | Vegas team total vs. season average. High total = more counting stats. |
| Opponent Defense vs. Position (DvP) | 5-10% | How many fantasy points the opponent allows to each position. |
| Spread / blowout risk | 3-5% | Large spreads reduce starters' minutes in garbage time. |
| Usage rate | Key feature | % of possessions used while on floor. Higher usage = more shots and FTA. |
| Rest / travel | 2-3% | Back-to-back fatigue, cross-country travel. |

#### Step 4: Machine Learning Layer

Modern DFS projection platforms use:
- **Neural networks** and **ensemble methods** (random forests, gradient boosting)
- **Bayesian updating** for real-time injury/lineup news
- Features: usage rate, on/off splits, opponent defensive rating, pace, team total, rest days, home/away

### 5.2 Key Insight: Minutes Dominate Everything

The single most valuable finding from the DFS industry: **projected minutes explain more variance in player output than any other single feature.** A mediocre player who plays 38 minutes will almost always outscore a great player who plays 22 minutes. This means our Level 1 model should strongly prioritize minutes prediction.

### 5.3 What DFS Models Get Wrong

DFS projections are optimized for single-game expected value. They struggle with:
- Correctly modeling player correlations within a game (if one player scores 40, teammates get fewer shots)
- Blowout dynamics (garbage time stat padding)
- Coaching decisions (random DNPs, matchup-based rotations)
- These are areas where our model's sequential game-state approach may have an advantage

**Sources:**
- [RotoGrinders: Key Inputs in NBA Projections](https://rotogrinders.com/lessons/key-inputs-in-an-nba-projections-system-1144825)
- [RotoGrinders: Projected Minutes](https://rotogrinders.com/lessons/projected-minutes-the-most-critical-opportunity-stat-in-nba-dfs-3147006)
- [RotoWire: NBA Projected Minutes Explained](https://www.rotowire.com/basketball/article/nba-projected-minutes-explained-fantasy-basketball-97473)
- [Medium: Daily NBA Player Projection Model (Python)](https://medium.com/@jon22anderson/daily-nba-player-projection-model-acb05036702a)

---

## 6. Synergy Sports / Second Spectrum / NBA Tracking Data

### 6.1 Tracking System History

| Era | System | Details |
|-----|--------|---------|
| 2013-2017 | SportVU | 6 cameras per arena, 25 fps, run by STATS LLC |
| 2017-2023 | Second Spectrum | Official NBA optical tracking provider, enhanced with ML |
| 2023-present | Hawk-Eye (Sony) | Replaced Second Spectrum as hardware provider; Second Spectrum continues as analytics provider |

All systems track x/y/z coordinates of all 10 players, referees, and the ball at 25 frames per second.

### 6.2 Publicly Available Tracking Stats (via stats.nba.com)

The NBA makes a subset of tracking data publicly available:

**Speed & Distance:**
- Average speed (off/def), distance covered per game
- Available at: `stats.nba.com/stats/players/speed-distance`

**Touches & Possession:**
- Touches per game, time of possession, points per touch
- Front court / back court touches, elbow / post / paint touches

**Passing:**
- Passes per game, assists per game, potential assists
- Points created by assists, assist-to-pass ratio

**Shot Tracking:**
- Shot distance, closest defender distance at release
- Touch time before shot, dribbles before shot
- Available via `ShotChartDetail` endpoint in nba_api

**Defensive:**
- Defended field goals (at rim, midrange, 3PT)
- Opponent FG% when defending by zone
- Partial tracking of "matchup" minutes

**Rebounding:**
- Contested vs. uncontested rebounds
- Rebound distance, chance percentage

**Hustle Stats:**
- Screen assists, deflections, loose balls recovered
- Charges drawn, contested shots

### 6.3 Accessing the Data

**Python nba_api package** ([github.com/swar/nba_api](https://github.com/swar/nba_api)):
- Most comprehensive open-source interface to stats.nba.com
- Key endpoints: `PlayerDashboardByGameSplits`, `ShotChartDetail`, `PlayerDashPtShots`, `SynergyPlayTypes`, `HustleStatsPlayer`
- Rate limiting: implement delays between requests (HTTP 429 errors)

**Synergy Play Types** (available via NBA.com and SportRadar API):
- Play classifications: Isolation, Transition, PRBallHandler, PRRollman, Postup, Spotup, Handoff, Cut, OffScreen, OffRebound, Misc
- Points per possession, frequency, and efficiency by play type
- Available per player and per team

### 6.4 Second Spectrum Proprietary Metrics

Not publicly available but used by NBA teams:
- **Quantified Shot Quality (qSQ)**: Expected value of each shot based on location, defender proximity, shot type, game context
- **Quantified Shooter Impact (qSI)**: How much a shooter outperforms/underperforms their expected shot quality
- **Matchup data**: Full defensive assignment tracking (who guarded whom on each possession)
- **"Dragon" technology**: Real-time augmented broadcast overlays

### 6.5 Implementation Relevance

**What we can use:**
- Hustle stats, shot tracking, and speed/distance data from stats.nba.com are free and programmatically accessible
- Synergy play-type data provides player-level offensive role characterization
- Shot difficulty (closest defender distance, dribbles, touch time) is excellent for understanding true shooting skill vs. shot selection

**What we cannot use:**
- Full tracking coordinate data is not public
- Second Spectrum's proprietary metrics (qSQ, qSI) are only available to NBA teams and media partners
- Real-time tracking feeds are not accessible

**Sources:**
- [Wikipedia: Player Tracking (NBA)](https://en.wikipedia.org/wiki/Player_tracking_(National_Basketball_Association))
- [NBAstuffer: Second Spectrum](https://www.nbastuffer.com/analytics101/second-spectrum/)
- [NBAstuffer: SportVU Data](https://www.nbastuffer.com/analytics101/sportvu-data/)
- [ESPN: NBA to use Hawk-Eye tracking](https://www.espn.com/nba/story/_/id/35818363/nba-use-hawk-eye-tracking-system-follow-players-ball)
- [GitHub: nba_api](https://github.com/swar/nba_api)
- [NBA.com: Synergy PlayTypes endpoint](https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/synergyplaytypes.md)

---

## 7. Basketball-Reference Advanced Stats (BPM, VORP, Win Shares)

These are the most widely cited "traditional" advanced stats. Understanding their construction reveals both their strengths and their limitations.

### 7.1 Box Plus/Minus (BPM) v2.0

**Creator:** Daniel Myers | **Reference:** [basketball-reference.com/about/bpm2.html](https://www.basketball-reference.com/about/bpm2.html)

#### What It Measures

Points above league average per 100 possessions, estimated from box-score data. League average = 0.0.

#### How It's Calculated

**Step 1 -- Position Assignment:**
- Each player gets two position values on a 1-5 spectrum:
  - Standard position (PG=1, C=5)
  - Offensive "creation" role (Creator=1, Receiver=5)
- Coefficients for each stat vary **linearly** between position 1 and position 5

**Step 2 -- Raw BPM:**
- Linear regression of per-100-possession box-score stats on RAPM
- Inputs: PTS, ORB, DRB, AST, STL, BLK, TOV, PF, and various interaction terms
- Training data: 20 years of player-seasons, weighted by minutes played
- Uses a specially developed Bayesian RAPM as the target variable (not raw +/-)

**Step 3 -- Team Adjustment:**
- Raw BPM for all players on a team is summed (weighted by % of minutes)
- A constant is added to all players so the team total matches the team's actual adjusted efficiency

**Step 4 -- Offensive/Defensive Split:**
- Offensive BPM (OBPM): same regression style, targeting offensive RAPM
- Defensive BPM (DBPM): simply Total BPM minus OBPM

#### Strengths and Limitations

| Strength | Limitation |
|----------|------------|
| Available from 1973-74 onward | Purely box-score based -- misses "gravity," screening, etc. |
| Position-aware coefficients | Defensive rating is crude (Total - Offense) |
| Team-adjusted | Linear model -- can't capture non-linear interactions |
| Stable with moderate samples | Coefficients frozen from training data; doesn't adapt to rule changes |

### 7.2 VORP (Value Over Replacement Player)

VORP converts BPM into a cumulative counting stat:

```
VORP = [BPM - (-2.0)] * (% of possessions played) * (team games / 82)
```

- Replacement level is defined as -2.0 BPM
- Result is in "points above replacement per 100 TEAM possessions over a full season"
- To convert to wins: `VORP * 2.7 = estimated wins over replacement`

#### Implementation Note

VORP is useful as a historical currency for comparing player value, but its dependence on BPM means it inherits all of BPM's limitations. For our purposes, BPM's per-100-possession rate is more directly useful than VORP's cumulative form.

### 7.3 Win Shares

**Original concept:** Bill James (baseball) | **NBA adaptation:** Justin Kubatko, Dean Oliver

#### Offensive Win Shares

1. Calculate **Points Produced** (based on Dean Oliver's offensive rating framework):
   - Credits players for scoring, assist-created points, offensive rebounding
   - Debits for turnovers, missed shots
2. Calculate **Offensive Possessions Used**
3. **Marginal Offense** = Points Produced - 0.92 * (league PPP) * (offensive possessions)
4. **Marginal Points Per Win** = 0.32 * (league PPG) * (team pace / league pace)
5. **OWS** = Marginal Offense / Marginal Points Per Win

#### Defensive Win Shares

1. Start with Dean Oliver's **Defensive Rating** (points allowed per 100 defensive possessions)
2. Calculate the player's share of team defense using contribution weights:
   ```
   Weight = 0.40 * (TRB / Team TRB)
            + 0.25 * (STL / Team STL)
            + 0.25 * (BLK / Team BLK)
            + 0.10 * (AST / Team AST)
   ```
3. Marginal defense is then allocated proportionally

#### Total Win Shares

```
WS = OWS + DWS
WS/48 = WS / (minutes played / 48)  -- rate version
```

#### Critical Limitation: Team Dependency

Win Shares **by construction** sum to team wins. A player on a 60-win team will have more Win Shares than an equally impactful player on a 30-win team. This is a feature for "credit allocation" but a bug for "player quality estimation."

### 7.4 Implementation Relevance

For our Level 1 model, these traditional stats are most useful as:
- **Training targets**: BPM per 100 possessions is a reasonable proxy for player impact
- **Feature inputs**: Not the raw stats themselves, but the underlying box-score data they use (per-possession rates, position-adjusted)
- **Baselines**: Any player model we build should outperform BPM's accuracy in predicting RAPM

**Sources:**
- [Basketball-Reference: About BPM](https://www.basketball-reference.com/about/bpm2.html)
- [Basketball-Reference: Win Shares](https://www.basketball-reference.com/about/ws.html)
- [Sports-Reference Blog: Introducing BPM](https://www.sports-reference.com/blog/2014/10/introducing-box-plusminus-bpm-2/)
- [NBAstuffer: BPM Explained](https://www.nbastuffer.com/analytics101/box-plus-minus/)
- [NBAstuffer: Win Shares Explained](https://www.nbastuffer.com/analytics101/win-share/)

---

## 8. Other Notable Systems

### 8.1 DRIP (Daily-Updated Rating of Individual Performance)

**Creator:** Stats Perform (Opta) | **Reference:** [theanalyst.com](https://theanalyst.com/articles/nba-drip-daily-updated-rating-of-individual-performance)

DRIP is the closest publicly available system to what we're building. It projects player contribution to team +/- per 100 possessions, split into offense and defense.

#### Methodology

1. Takes box score, play-by-play, and lineup data
2. "Now-casts" each stat using time-series techniques (not season averages)
3. Feeds projected stats into a model trained against adjusted +/- as the target

#### Rookie Handling (Best-in-Class Public Approach)

DRIP uses **physical profile + draft position** to initialize rookies:
- Height, weight, age, and draft pick number predict "adjusted game 1" estimates
- A model trained on historical rookies generates the starting point
- The system then updates daily as games are played

#### Recent Improvements

DRIP moved to model types that are less prone to overfitting, which previously caused it to over-weight similarity to other players and miss unique player characteristics.

### 8.2 DraftKings Kalman Filter System

**Reference:** [DraftKings Engineering Blog](https://medium.com/draftkings-engineering/kalman-filters-for-nba-player-ratings-d3bb9365221b)

DraftKings published technical details on their internal player rating system:

#### Architecture

1. **Data**: Split into possessions. Each possession is treated as an observation.
2. **Offense and Defense Separate**: A player's offensive and defensive possessions are modeled independently.
3. **Measurement Matrix**: Describes how each player impacts each possession in the dataset (who was on court, what happened).
4. **Kalman Filter**: Estimates each player's offensive and defensive contribution per possession.
5. **Output**: An offensive rating of 0.1 with 10% usage means the player contributes ~1 point per 100 offensive possessions above average.

#### Key Difference from DARKO

DraftKings' approach operates at the **possession level** (play-by-play data), while DARKO operates at the **game level** (box-score data). The possession-level approach can capture more granular information but requires much more data engineering.

### 8.3 Expected Points Above Average (EPAA)

**Reference:** [arxiv.org/html/2405.10453v1](https://arxiv.org/html/2405.10453v1)

An academic approach using Bayesian Hierarchical Modeling to estimate player impact. Notable for its formal statistical framework with proper uncertainty quantification.

**Sources:**
- [Opta: Introducing DRIP](https://theanalyst.com/articles/nba-drip-daily-updated-rating-of-individual-performance)
- [DraftKings: Kalman Filters for NBA Player Ratings](https://careers.draftkings.com/life-at-draftkings/engineering/how-we-use-kalman-filters-for-nba-player-ratings/)
- [NBAstuffer: DRIP Explained](https://www.nbastuffer.com/analytics101/daily-updated-ranking-of-individual-performance-drip/)

---

## 9. Key Takeaways for Our Player Model

### 9.1 Architecture Patterns That Work

| Pattern | Used By | Why It Works |
|---------|---------|-------------|
| **Kalman filter for skill estimation** | DARKO, DraftKings, EPM | Naturally handles uncertainty, recency weighting, and Bayesian updates |
| **Exponential decay weighting** | DARKO | Simple, effective recency bias. More recent games matter more without discarding history |
| **Two-stage: SPM prior + RAPM** | EPM, RPM, RAPTOR | Box-score model provides strong prior; on/off data captures the rest |
| **"Estimated Skills" (stat projection)** | EPM | Per-stat skill estimation >> raw rolling averages |
| **Position-conditional evaluation** | BPM, 2K | Same stat means different things for PG vs. C |
| **Physical profile for rookies** | DRIP, CARMELO | Height + weight + age + draft position provides a useful cold-start prior |

### 9.2 Features That Matter Most (Ranked)

Based on consensus across systems:

1. **Minutes / playing time** -- Dominates variance in counting stats (DFS finding)
2. **Usage rate** -- How many possessions a player uses when on court
3. **Per-possession stat rates** -- Not per-game, not per-minute, but per-possession
4. **Opponent context** -- Specific opponent effects on each stat category
5. **Pace** -- Team and game pace projections
6. **Age** -- For career arc projection and trajectory estimation
7. **Position / role** -- Determines how to interpret the same stat line
8. **Rest / schedule** -- Back-to-backs, travel, home/away
9. **Shot quality data** -- Closest defender, shot type, dribbles before shot
10. **Play-type distribution** -- Isolation %, PnR %, spot-up % characterizes offensive role

### 9.3 What Our Model Should Do That Others Don't

1. **Unified architecture**: Most systems are multi-stage pipelines (project stats, then convert to impact). Our transformer can learn the full mapping end-to-end.

2. **Game-state conditioning**: No public system conditions player projections on the evolving game state (score, period, momentum). Our generative model already has this infrastructure.

3. **Player interaction modeling**: DARKO and EPM project players independently. Our Phase 3 player interaction attention already models how players' contributions interact. This should extend to Level 1.

4. **Sequence-native**: Kalman filters are sequential but linear. Our transformer encoder naturally handles non-linear sequential patterns in player performance trajectories.

5. **Integrated uncertainty**: Most systems produce point estimates. Our Gaussian output heads already produce (mu, sigma) predictions. This should carry through to player-level estimates.

### 9.4 Practical Implementation Recommendations

**For the Kalman filter / skill estimation component:**
- Implement per-stat exponential decay with learned or tuned beta values
- Use the "padding method" for stabilization (add N attempts of league average)
- Specific padding values to start with: ~240 for 3PT%, ~100 for ORB%, ~200 for AST rate, ~500+ for steal rate

**For rookie cold starts:**
- Use draft position + age + height + weight + college stats (if available)
- Initialize to position-specific league-average priors, then update rapidly
- Consider DRIP's approach: train a separate "game 1 estimate" model on historical rookies

**For data sources to incorporate:**
- Box-score stats (already have in PlayerBox)
- Synergy play-type distributions (available via nba_api)
- Shot tracking: closest defender distance, touch time, dribbles (available via nba_api)
- Hustle stats: deflections, contested shots, screen assists (available via nba_api)
- Speed and distance tracking (available via nba_api)

**For evaluation:**
- Primary target: predict future RAPM (following EPM's lead)
- Secondary target: predict next-game box-score stats (following DARKO)
- Benchmark against: BPM (simple baseline), EPM (state-of-art public metric)

### 9.5 What We Explicitly Should NOT Try To Do

1. **Don't try to beat Vegas on game totals.** Even DARKO + EPM + RAPTOR combined don't consistently beat market lines. Our edge should come from understanding player-level dynamics that feed into game-level predictions, not from building a better game-total model directly.

2. **Don't use Win Shares as a target.** WS is team-win-dependent by construction and conflates playing time with quality.

3. **Don't over-index on defensive metrics from box scores.** BPM's defensive component is unreliable. True defensive impact requires tracking data that we may not have access to at scale.

4. **Don't ignore the stabilization problem.** A player's 10-game sample of 45% 3PT shooting is less informative than it looks. Every stat projection should incorporate appropriate regression toward priors based on sample size.
