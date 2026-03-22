# Level 2: Industry Lineup & Player Synergy Systems Research

Research compiled 2026-03-16. Covers NBA lineup data sources, production analysis tools, lineup sparsity challenges, on/off court methodology, player interaction modeling, and practical considerations for building a Level 2 Player Synergy model.

---

## Table of Contents

1. [NBA Lineup Data Sources](#1-nba-lineup-data-sources)
2. [Reconstructing Lineups from Play-by-Play](#2-reconstructing-lineups-from-play-by-play)
3. [Production Lineup Analysis Tools](#3-production-lineup-analysis-tools)
4. [The Lineup Sparsity Problem](#4-the-lineup-sparsity-problem)
5. [On/Off Court Splits](#5-onoff-court-splits)
6. [Player Interaction & Synergy Modeling](#6-player-interaction--synergy-modeling)
7. [Roster Construction & Trade Analysis](#7-roster-construction--trade-analysis)
8. [Key Takeaways for Our Synergy Model](#8-key-takeaways-for-our-synergy-model)

---

## 1. NBA Lineup Data Sources

### 1.1 NBA.com/stats Lineup Endpoints

NBA.com provides official lineup statistics via two primary endpoints:

**LeagueDashLineups** (`https://stats.nba.com/stats/leaguedashlineups`)
- Returns league-wide lineup statistics across all teams
- Key parameter: `GroupQuantity` -- accepts values `2`, `3`, `4`, or `5` for different combination sizes
- Returned fields: `GROUP_SET`, `GROUP_ID`, `GROUP_NAME`, `TEAM_ID`, `TEAM_ABBREVIATION`, `GP`, `W`, `L`, `W_PCT`, `MIN`, `FGM`, `FGA`, `FG_PCT`, `FG3M`, `FG3A`, `FG3_PCT`, `FTM`, `FTA`, `FT_PCT`, `OREB`, `DREB`, `REB`, `AST`, `TOV`, `STL`, `BLK`, `BLKA`, `PF`, `PFD`, `PTS`, `PLUS_MINUS`, plus corresponding ranking fields
- Supports filtering by `DateFrom/DateTo`, `Conference`, `Division`, `TeamID`, `Location`, `Outcome`, `SeasonSegment`, `ShotClockRange`, and more
- Source: [nba_api LeagueDashLineups docs](https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/leaguedashlineups.md)

**TeamDashLineups** (`https://stats.nba.com/stats/teamdashlineups`)
- Same data but filtered to a specific team via `TeamID` parameter
- Same `GroupQuantity` parameter for 2/3/4/5-man combinations
- Returns both `Lineups` and `Overall` data sets
- Source: [nba_api TeamDashLineups docs](https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/teamdashlineups.md)

**Historical coverage:** Lineup data on NBA.com dates back to the **2007-08 season**, giving ~18 seasons of lineup statistics. The web UI at [nba.com/stats/lineups/advanced](https://www.nba.com/stats/lineups/advanced) allows browsing this data interactively.

### 1.2 nba_api Python Library

The [nba_api](https://github.com/swar/nba_api) library (requires Python 3.10+, uses `requests` + `numpy`) wraps these endpoints. Relevant lineup-related endpoints include:

| Endpoint | Purpose |
|----------|---------|
| `LeagueDashLineups` | League-wide lineup combos (2/3/4/5-man) |
| `TeamDashLineups` | Team-specific lineup combos |
| `PlayByPlayV2` | Raw play-by-play events with substitution tracking |
| `BoxScoreAdvancedV2/V3` | Per-period box scores (useful for period starter identification) |
| `GameRotation` | Player in/out times with `IN_TIME_REAL`, `OUT_TIME_REAL`, `PLAYER_PTS`, `PT_DIFF`, `USG_PCT` |
| `ShotChartLineupDetail` | Shot charts filtered by lineup context |

**Rate limiting:** NBA.com API requests are rate-limited. The nba_api library includes request delays but heavy scraping can still result in blocks. Best practice is to cache aggressively and add random delays between requests.

**Season coverage for play-by-play:** Play-by-play data goes back to **1996-97** on stats.nba.com, though data quality improves significantly from ~2000 onward.

### 1.3 pbpstats (Play-by-Play Stats)

**Website:** [pbpstats.com](https://www.pbpstats.com/) | **Docs:** [pbpstats.readthedocs.io](https://pbpstats.readthedocs.io/) | **GitHub:** [dblackrun/pbpstats](https://github.com/dblackrun/pbpstats)

pbpstats is widely used in the NBA analytics community -- approximately 90% of public NBA analytics products rely on it for possession-level data. Key capabilities:

- **Lineup reconstruction:** Adds lineup-on-floor for all play-by-play events
- **Possession parsing:** Adds detailed possession data including start/end times, score margin, and how the previous possession ended
- **Event reordering:** Fixes order of events for common cases where events are out of order in the raw data
- **Multi-source support:** Uses both `stats.nba.com` and `data.nba.com` endpoints. The `data.nba.com` source includes the offense team ID in all events, making possession tracking easier
- **Substitution handling:** When a substitution occurs mid-possession, only players who finish the possession are credited with the possession count
- **Lineup IDs:** Lineups are represented as hyphen-separated sorted player ID strings (e.g., `101-203-456-789-999`)
- **Coverage:** NBA, WNBA, and G-League

**pbpstats API:** Available at [api.pbpstats.com/docs](https://api.pbpstats.com/docs) (Swagger UI). Provides:
- On/off stats and WOWY combinations
- Lineup and possession-level statistics
- Net ratings for all on/off permutations for player groups
- Code examples at [github.com/dblackrun/pbpstats-api-code-examples](https://github.com/dblackrun/pbpstats-api-code-examples)

### 1.4 nba-on-court

**GitHub:** [shufinskiy/nba-on-court](https://github.com/shufinskiy/nba-on-court) | **PyPI:** [nba-on-court](https://pypi.org/project/nba-on-court/)

A focused Python package for adding on-court player information to NBA play-by-play data:

- **`players_on_court()`**: Takes play-by-play data, returns it with 10 additional columns (5 per team) of `PLAYER_ID` for players on court at each event
- **`players_name()`**: Replaces player IDs with first/last names for readability
- **Algorithm:** Gets the list of players on court at the start of each quarter, then filters to the 10 who actually started the quarter by analyzing substitution events within that quarter
- **Data source:** Includes access to [shufinskiy/nba_data](https://github.com/shufinskiy/nba_data) repository containing play-by-play from three sources (stats.nba.com, pbpstats.com, data.nba.com) plus shot detail for all games since **1996-97 season**

### 1.5 Other Data Sources

- **Basketball-Reference:** Added [play-by-play, lineup, and shooting stats back to 1996-97](https://www.sports-reference.com/blog/2020/04/nba-play-by-play-lineup-and-shooting-stats-added-back-to-1996-97/) in 2020. Provides 2-man and 5-man lineup combination tables.
- **Kaggle datasets:** Pre-built play-by-play datasets with on-court lineups (e.g., 1999-2022 coverage)
- **Sportradar:** Commercial API with historical data, used by official NBA platforms

---

## 2. Reconstructing Lineups from Play-by-Play

### 2.1 The Algorithm

The standard approach for determining which 5 players are on court at any moment:

1. **Identify period starters**: Use the box score endpoint filtered to a specific time window (e.g., BoxScoreAdvancedV2 with start/end times for each quarter) to get all players who appeared during that period
2. **Track substitutions**: Play-by-play events with `EVENTMSGTYPE = 8` are substitutions. `PERSON1` (or `namePlayer1`) is the player leaving; `PERSON2` (or `namePlayer2`) is the player entering
3. **Filter to starters**: Players whose first substitution event in a period is "SUB OUT" were starters. Players whose first event is "SUB IN" came off the bench. Remove all bench entries to get the starting five.
4. **Forward-fill lineups**: Between substitution events, the lineup remains unchanged. Each substitution creates a new `lineupAfter` that persists until the next sub.

**Period time calculations (in deciseconds):**
- Regular quarters: `(720 * (period - 1)) * 10`
- Overtime periods: `(720 * 4 + (OT_number - 1) * 300) * 10`

### 2.2 NBA Play-by-Play Event Types (EVENTMSGTYPE)

| Code | Event Type |
|------|-----------|
| 1 | Field Goal Made |
| 2 | Field Goal Missed |
| 3 | Free Throw |
| 4 | Rebound |
| 5 | Turnover |
| 6 | Foul |
| 7 | Violation |
| 8 | **Substitution** |
| 9 | Timeout |
| 10 | Jump Ball |
| 11 | Ejection |
| 12 | Period Begin |
| 13 | Period End |

### 2.3 Data Quality Issues

Lineup reconstruction from play-by-play is reliable but not perfect. Known issues include:

1. **Missing between-quarter substitutions**: Play-by-play shows every in-game substitution but does NOT show lineup changes made between quarters. If a team ends Q1 with one lineup but starts Q2 with a different one, the between-quarter substitution is invisible in the PBP data.

2. **Invisible players in overtime**: A player could theoretically play an entire 5-minute overtime period without registering in the play-by-play (no shots, rebounds, fouls, etc.). This is rare but undetectable from PBP alone. One analysis found 10 such OT cases requiring manual correction via Basketball-Reference box scores.

3. **Out-of-order events**: Some events have erroneous timestamps, rebounds may appear before the corresponding shot, or substitutions may appear out of chronological order. pbpstats fixes many common cases automatically.

4. **Incorrect player attribution**: Rarely, the wrong player's name is recorded in a substitution event. No algorithmic fix exists -- requires cross-referencing with secondary data sources.

5. **Duplicate events**: Occasional duplicate play entries exist; need deduplication via `distinct(game_id, event_number)`.

6. **Historical data gaps**: The 1996-97 season has significant missing data. Coverage becomes essentially complete from 1997 postseason onward.

7. **Technical fouls from bench**: Players can receive technical fouls while sitting on the bench, which can create false "on-court" detections if not filtered.

### 2.4 Game Structure Statistics

Relevant numbers for understanding the data volume:

- **~95 possessions per team per game** (varies by pace; recent seasons trend slightly higher)
- **~15 distinct 5-man lineups per team per game** (based on 2014-15 season data)
- **Substitutions are unlimited** in the NBA; the frequency varies by coaching philosophy
- **Starters share the court for ~13-17 minutes per game** (out of 48 total)
- Each game produces **~200 total possessions** across both teams

---

## 3. Production Lineup Analysis Tools

### 3.1 Cleaning the Glass

**Website:** [cleaningtheglass.com](https://cleaningtheglass.com/) | **Creator:** Ben Falk (former NBA front office, Philadelphia 76ers and Portland Trail Blazers)

Cleaning the Glass is widely regarded as the gold standard for public-facing NBA analytics. Key methodology features:

**Context filtering:**
- Estimates play context: half court vs. transition vs. off-putbacks
- **Garbage time removal**: Scrubs possessions where the game outcome is decided. Definition: 4th quarter only, with score differential >= 25 (minutes 12-9), >= 20 (minutes 9-6), >= 10 (remainder)
- **End-of-quarter heave removal**: Excludes desperation shots that would distort efficiency metrics

**Position estimation:**
- Estimates position played for all players (not just the 5 traditional positions)
- Allows filtering and sorting lineups by player position combinations

**Lineup display rules:**
- Only lineups with **at least 15 possessions** are displayed
- Orange/blue percentile rankings compare lineups against all lineups with **at least 100 possessions**
- Because lineup samples are generally very small, percentiles for aggregate totals use **player on-court data** instead of raw lineup data to properly convey performance quality

**On/off methodology:**
- Off-court stats include **only games where the player was on the roster** (excludes pre-trade or pre-signing periods)
- Players with **under 100 minutes** are excluded from percentile calculations
- Differences shown as delta from on-court performance (e.g., "+5.2" means team scored 5.2 more points per 100 possessions with player on court)
- Breaks down by Four Factors, shooting, and context (halfcourt/transition/putbacks)

**Subscription:** Approximately $7.50/month or $75/year (as of last known pricing).

### 3.2 NBA.com's Own Lineup Stats

Available at [nba.com/stats/lineups/advanced](https://www.nba.com/stats/lineups/advanced):

- View 5-man, 4-man, 3-man, or 2-man lineup combinations
- Traditional, advanced, scoring, and four factors views
- Filterable by season (back to 2007-08), date range, conference, division
- Minimum minutes filter available
- Powered by the same API endpoints accessible via nba_api

### 3.3 Basketball Index Stabilized Lineup Data

**Website:** [bball-index.com](https://www.bball-index.com/)

Basketball Index addresses lineup reliability using **regression toward the mean**:

- Every lineup starts with "a few hundred minutes of league-average results" as a Bayesian prior
- Actual performance data is layered on top of this baseline
- Strong performance pulls ratings away from average; poor performance does the opposite
- This allows meaningful comparisons between lineups with vastly different playing time

**Example of stabilization in practice:**
- Bucks lineup: 68 minutes, raw +33.1 net rating --> stabilized to **+7.6**
- Nuggets lineup: 171 minutes, raw +20.7 net rating --> stabilized to **+7.5**
- Despite wildly different raw ratings and sample sizes, the stabilized ratings converge to similar values

### 3.4 DARKO Lineup Projections

DARKO publishes estimated ratings for 5-man lineups (split into offensive and defensive) based on blended results from past and present seasons. The standard approach for building lineup projections from player-level metrics:

1. Multiply each player's minutes projection by their O-DPM and D-DPM
2. Sum offensive and defensive values across the 5 players
3. Divide by total projected minutes and multiply by 5 (for 5 players on court)
4. Add offensive and defensive components for net team rating

**Critical assumption:** This is purely **additive** -- it assumes player impacts combine linearly with no interaction effects. This is the standard practice in the industry, with synergy/interaction effects being a known gap.

### 3.5 How Front Offices Evaluate Lineups

Based on reporting and public discussions from NBA analytics departments:

- Teams use **RAPM-based models** (Regularized Adjusted Plus-Minus) as the backbone for lineup evaluation
- Analytics departments employ data scientists, statisticians, and analysts working closely with coaches and executives
- Models specifically account for **team fit**, not just individual player statistics
- Teams run **numerous trade simulations**, telling models what they're looking for and generating different scenarios
- Evaluation criteria extend beyond stats: playstyle compatibility, positional scarcity, leadership, professionalism, and locker room fit
- The Golden State Warriors' front office (documented example) uses AI-powered systems for informed trades and lineup changes

---

## 4. The Lineup Sparsity Problem

This is the central challenge for any lineup-based modeling approach.

### 4.1 Scale of the Problem

**Number of distinct 5-man lineups per team per season:**
- A single NBA team uses **more than 600 distinct 5-man lineups** during a season (per L-RAPM paper, 2023-24 data)
- League-wide, through December of one season: **15,297 total 5-man lineups** across all teams
- Teams use approximately **15 distinct 5-man units per game**

**Minutes/possessions per lineup:**
- Average lineup plays approximately **17 possessions on offense and 17 on defense** (std dev: 56 possessions)
- Only **2.5% of all 5-man units** reach at least 100 possessions
- Most-used lineups (team starters) average only **13-17 minutes per game** together
- The best case (e.g., 2022-23 Nuggets starters) was **17.2 minutes per game**
- Maximum observed for any lineup: approximately **20 minutes per game**

### 4.2 Stabilization Rates

Research by Kostya Medvedovsky established key thresholds:

| Metric | Possessions to Stabilize |
|--------|------------------------|
| Offensive rating | ~550 possessions |
| Defensive rating | ~850 possessions |

**How rare is stabilization?**
- In 2022-23, only **25 lineups league-wide** reached 550 possessions (less than one per team)
- Only **11 lineups** reached 850 possessions

This means virtually no 5-man lineup accumulates enough data for its offensive rating to be statistically reliable, and even fewer for defensive rating.

### 4.3 Selection Bias / Survivorship

A critical confound: lineups that accumulate large samples are systematically biased toward good performance because coaches discontinue underperforming combinations:

| Minutes Played | % with Positive Net Rating | Average Net Rating |
|----------------|---------------------------|-------------------|
| 100+ | 67% | +3.99 |
| 250+ | 80% | +5.75 |
| 500+ | 90% | +5.95 |

In a zero-sum game, the "true" average net rating should be 0.0. The systematic positive skew is entirely due to selection bias -- bad lineups get pulled before accumulating minutes.

### 4.4 Practical Guidance for Analysts

From The Ringer's analysis and industry practice:

- **Only credit net ratings exceeding +/- 10** as potentially meaningful for 5-man units
- **Compare like to like**: Match 3-man combinations against other 3-man data, not across different group sizes
- **Don't cherry-pick** best lineups and compare against league averages
- **2-man and 3-man groups** are generally more useful than 5-man because they accumulate much larger samples
- **Use stabilized/regressed ratings** (like Basketball Index's approach) rather than raw ratings
- **Multi-season data** adds useful information and context, increasing sample size

---

## 5. On/Off Court Splits

### 5.1 How On/Off Splits Work

On/off analysis compares team performance when a specific player is on the court versus when they are off the court:

- **Offensive on/off**: Team offensive rating (points per 100 possessions) with player on court minus offensive rating with player off court
- **Defensive on/off**: Same comparison for defensive rating (note: lower is better for defense, so interpretation is inverted)
- **Net on/off**: Combines both -- the overall point differential impact per 100 possessions

Example: If a team scores 115 per 100 with Player X on court and 98 per 100 with Player X off court, Player X has an offensive on/off of **+17**.

### 5.2 Major Confounding Variables

On/off splits are widely misused because they suffer from severe confounds:

1. **Teammate quality:** The four other players sharing the court are different when the player is on vs. off. A star's "on" minutes feature other starters; their "off" minutes feature the bench. The on/off differential may reflect the starters-vs-bench gap more than the star's individual impact.

2. **Opponent quality:** Stars tend to play against opposing starters; bench players face opposing benches. On/off doesn't adjust for opponent strength.

3. **Staggering effects:** Smart coaches stagger their best players' rest so at least one is always on court. This can deflate one star's on/off while inflating another's, purely due to rotation patterns.

4. **Sample size:** Over limited samples, one very good or bad game can have an inordinate impact on on/off numbers.

5. **Mid-season roster changes:** If a player joins a team mid-season, their "off-court" stats include minutes from before they arrived, when the team had a different composition. Cleaning the Glass addresses this by only including games where the player was on the roster.

6. **Score-state effects:** Starters play more in close games; bench players get more run in blowouts (garbage time). This systematically skews performance context.

### 5.3 Solutions to On/Off Confounds

Several approaches address these issues:

- **RAPM (Regularized Adjusted Plus-Minus):** Solves for all players simultaneously via ridge regression, controlling for teammate and opponent effects. The gold standard but requires large samples.
- **WOWY (With Or Without You):** Examines all pairwise combinations: player A on + player B on, A on + B off, A off + B on, A off + B off. More granular than simple on/off.
- **Context-filtered on/off** (Cleaning the Glass approach): Remove garbage time, filter by play context (halfcourt/transition), require minimum minutes thresholds.
- **Bayesian regression toward mean:** Add 30 games of league-average performance as a prior to combat sample size issues (WOWY research finding for optimal out-of-sample prediction).

### 5.4 RAPM Deep Dive

RAPM is the foundational technique behind most modern player impact metrics:

- **Setup:** Each "stint" (continuous period with the same 10 players on court) is one observation. The design matrix has a column for each player: +1 if on the home team, -1 if on the away team, 0 if not playing.
- **Target:** Point differential per 100 possessions for that stint.
- **Ridge regression:** Adds L2 penalty (lambda * sum of squared coefficients) to handle multicollinearity. Without regularization, players who always play together get wildly unstable coefficients because their effects can't be disentangled.
- **Typical lambda values:** lambda_off = 4000, lambda_def = 6000 (from L-RAPM paper)
- **Multi-year data** (3+ years) produces RAPM estimates roughly **twice as accurate** as single-year
- **Multicollinearity problem:** When two players always play together (or never together), their individual effects can't be separated. Ridge regression penalizes extreme values, producing more stable but slightly biased estimates.

---

## 6. Player Interaction & Synergy Modeling

### 6.1 The Additive Assumption

The standard approach in NBA analytics treats player contributions as **additive**: a lineup's expected net rating is the sum of the five players' individual ratings. Most production systems (DARKO, EPM, BPM) work this way.

The key question for Level 2 modeling: **How much do non-additive interaction effects matter?**

### 6.2 Skills Plus Minus (SPM) Framework

**Paper:** Maymin, Maymin, & Shen (2013), "NBA Chemistry: Positive and Negative Synergies in Basketball" ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1935972))

The SPM framework decomposes each player into six skill components (offensive/defensive x scoring/rebounding/ballhandling) and simulates games to measure lineup effectiveness vs. sum-of-parts.

**Key findings on specific synergies:**
- **Offensive ballhandling has negative synergies with itself (-0.825):** A lineup with one great playmaker doesn't need another; the second one adds diminishing returns
- **Defensive ballhandling has positive synergies with itself (+0.307):** Defenders who create turnovers feed off each other
- **Offensive scoring has negative synergies with itself:** Players must share one ball
- Synergies can explain **as much as 6 wins** per season difference
- Teams are **equally likely** to exhibit positive chemistry as negative chemistry
- The framework identified **more than 200 mutually beneficial trades** where skill synergies were better with the other team

**Implication:** Player value is genuinely context-dependent -- a player's impact depends on the other 9 players on the court.

### 6.3 Archetype-Based Synergy Modeling

A modern approach clusters players into data-driven archetypes and measures synergy at the archetype level rather than individual level:

**Offensive archetypes (8):** Spot Up Shooter, Playmaker, Roll Man, Versatile Big, Movement Shooter, Secondary Ball Handler, Shot Creator, Connector

**Defensive archetypes (7):** Wing Stopper, Low Activity, Mobile Forward, Disruptor, Point-of-Attack Guard, Anchor Big, Mobile Big

**Synergy calculation process:**
1. Luck-adjust for free throw and three-point shooting variance
2. Build a baseline model predicting stint net rating from individual Current Ability differences plus home court
3. The gap between observed and predicted results, grouped by archetype combination, reveals synergy effects
4. Aggregate across thousands of possessions

**Example results:**
- "2 Spot Up Shooters + 1 Playmaker + 1 Roll Man + 1 Versatile Big" = **+9.9** per 100 possessions above expected
- Playmaker presence dramatically improves most archetype combinations
- Small-ball lineups consistently underperform relative to composition-based expectations

Source: [NBA Insights Substack](https://nbainsights.substack.com/p/roster-construction-in-the-nba-using)

### 6.4 L-RAPM: Lineup Regularized Adjusted Plus-Minus

**Paper:** [arxiv.org/abs/2601.15000](https://arxiv.org/abs/2601.15000) (2025)

L-RAPM extends standard RAPM to lineup-level estimation with informed priors:

- **Problem:** 600+ lineups per team, ~17 possessions each on average (std dev: 56)
- **Key innovation:** Rather than shrinking lineup coefficients toward zero, they shrink toward **the sum of individual player RAPM values**. Before seeing any data, a lineup's predicted rating equals league average plus the sum of each player's individual offensive/defensive RAPM.
- **Regularization:** Separate lambdas for offense (4000) and defense (6000)
- **Results:** ~5% relative error improvement over raw ratings; 1.5% improvement translates to approximately **3.4 points per game advantage**
- **Greatest improvement on smallest samples**: The less data a lineup has, the more L-RAPM helps vs. raw ratings

### 6.5 Player Clustering for Lineup Prediction

Multiple research approaches use clustering to reduce lineup dimensionality:

- **K-means clustering** on player box-score features to identify archetypes
- **PCA + Gaussian Mixture Models** for more flexible cluster shapes
- **Neural network lineup models** that predict net rating from archetype composition
- The Sloan Sports Conference paper by Kalman & Bosch specifically addresses "NBA Lineup Analysis on Clustered Player Tendencies"

The key advantage of clustering: it maps the ~450 active NBA players into ~8-15 archetypes, dramatically reducing the combinatorial explosion from C(450,5) possible lineups to a manageable number of archetype combinations.

### 6.6 Network/Graph Approaches

Recent research applies graph theory and social network analysis (SNA) to lineup data:

- **Substitution networks:** Smaller network diameter (fewer "hops" between players) correlates with higher win percentage (r = -0.39), reflecting roster versatility
- **Persistent homology of lineups:** Constructs a simplicial complex from lineup usage weighted by possessions. "Holes" in the complex reveal groups of players who never play together.
- **Player interaction networks:** SNA to identify which player roles (Core Ball Handler, Versatile Big, etc.) enhance teammate performance when co-occurring in lineups

Source: [Wiseman 2025](https://journals.sagepub.com/doi/10.1177/22150218251324877), various ScienceDirect papers

---

## 7. Roster Construction & Trade Analysis

### 7.1 How Front Offices Model Player Addition/Removal

Modern NBA analytics departments evaluate trades by:

1. **Player ratings (individual):** Current Ability (CA), Season Rating (SR), and Potential Ability (PA) components
2. **Archetype classification:** Determine what role the player fills (not just how good they are)
3. **Synergy optimization:** Test which archetype additions improve the team's viable 5-man group combinations
4. **Projection modeling:** Run simulations with hypothetical rosters to estimate win totals

**Example:** Minnesota analysis identified that acquiring a Playmaker archetype player was the optimal addition, with specific trade scenarios projecting Net Rating improvement from +4.4 to +5.3 (~2 additional expected wins).

### 7.2 Cap Space Optimization Tools

Public tools for salary cap analysis:
- **Fanspo Trade Machine:** Validates trades against NBA salary-matching rules, luxury tax thresholds, and roster limitations. Explains why invalid trades fail.
- **Spotrac:** Comprehensive salary cap tracker with trade machine functionality
- **SalarySwish:** NBA salary cap tracker and trade machine
- **ESPN Trade Machine:** The original public trade validator

These tools handle CBA salary matching rules (traded salary must be within allowable range), trade exceptions, and future cap projections.

### 7.3 The 2024 CBA Trade Rules

Key salary matching constraints for trades:
- Teams below the salary cap can absorb salary up to available cap space
- Teams above the cap must match salaries within specified bands (typically 125% + $100K of outgoing salary for smaller contracts)
- Trade exceptions allow acquiring players without sending salary back
- The second apron introduces additional restrictions on sign-and-trade deals, cash considerations, and draft pick trades

---

## 8. Key Takeaways for Our Synergy Model

### 8.1 Data Source Strategy

**Recommended approach for our project:**

1. **Play-by-play reconstruction** is the most flexible path. We already have play-by-play infrastructure via our database. Using substitution events (EVENTMSGTYPE = 8) and period-start box scores, we can reconstruct on-court lineups for every game.
2. **nba-on-court** or custom implementation: Either use the shufinskiy/nba-on-court package or implement the algorithm ourselves (it's straightforward: track period starters + apply substitutions sequentially).
3. **Supplement with LeagueDashLineups API** for pre-aggregated 2-man through 5-man stats as validation/comparison data.
4. **Coverage:** Play-by-play quality is good from ~2001 onward, matching our existing data coverage.

### 8.2 The Fundamental Sparsity Challenge

**The most important finding from this research: 5-man lineup data is almost unusable in its raw form.**

- 600+ distinct lineups per team per season, averaging 17 possessions each
- 550-850 possessions needed for statistical stability
- Fewer than 1 lineup per team reaches the offensive stability threshold
- Selection bias inflates observed net ratings for high-minute lineups

**Implication for Level 2:** We cannot build a model that learns 5-man lineup effects directly. There simply isn't enough data. Instead, we need one of these approaches:

### 8.3 Viable Modeling Approaches (Ordered by Practicality)

1. **Pairwise (2-man) interaction effects:** Much denser data than 5-man. A team's ~12-15 rotation players produce only C(15,2) = 105 pairs, each accumulating far more minutes than any 5-man unit. This is the most practical approach.

2. **Archetype-based synergy:** Cluster players into 8-15 archetypes, then learn interaction effects between archetype pairs. Dramatically reduces dimensionality while capturing meaningful patterns (e.g., "two ball-dominant players have negative synergy").

3. **L-RAPM-style informed priors:** Start with sum-of-individual-ratings as the prior, then let lineup data pull toward observed values proportional to sample size. Most theoretically principled.

4. **On/off differential features:** Simpler but useful -- how a player's on-court stats compare to off-court gives signal about their impact on teammates. Must be filtered for confounds.

5. **Additive player embeddings with learned interaction terms:** Use our existing player embeddings from Phase 3 and add a small interaction network that computes pairwise compatibility scores. Most aligned with our existing architecture.

### 8.4 What the Additive Gap Actually Looks Like

The industry's best systems (DARKO, EPM) use purely additive player models and achieve very strong predictive performance. The synergy/interaction effects are real but relatively small:

- **Synergy effects: up to ~6 wins per season** (Maymin et al.)
- **L-RAPM improvement: ~3.4 points per game** vs. raw lineup ratings (but much of this is better handling of small samples, not interaction effects per se)
- **Archetype synergy: +/- 10 points per 100 possessions** for best/worst combinations, but this includes individual quality effects

This suggests the interaction signal exists but is modest compared to individual player quality. Our Level 2 model should therefore:
- **Start with strong individual player embeddings** (from Level 1)
- **Add lightweight pairwise interaction** rather than trying to model full 5-way interactions
- **Use archetype compression** to share information across similar player pairs
- **Regularize heavily** toward the additive baseline (the L-RAPM insight)

### 8.5 Practical Implementation Notes

- **WOWY as training signal:** For each player pair, compute with-both / with-A-only / with-B-only / with-neither net ratings. The residual (actual - expected if additive) is the pairwise interaction signal to learn.
- **Minimum thresholds:** Use at least 100 minutes (or ~200 possessions) for any pairwise signal to be included in training. This is achievable for most rotation-player pairs within a single season.
- **Bayesian shrinkage:** Always regress interaction estimates toward zero (or league average). The prior should strongly favor additive behavior; only large, consistent deviations should be trusted.
- **Feature enrichment:** Beyond raw on/off data, useful features for predicting interaction include player archetype similarity, offensive usage overlap, positional overlap, and years played together.
- **Temporal stability:** Synergy effects may change as players develop chemistry. Multi-season data helps but should be weighted toward recent seasons.

---

## Sources

### Data Sources & Tools
- [swar/nba_api GitHub](https://github.com/swar/nba_api)
- [nba_api LeagueDashLineups docs](https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/leaguedashlineups.md)
- [nba_api TeamDashLineups docs](https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/teamdashlineups.md)
- [pbpstats GitHub](https://github.com/dblackrun/pbpstats) | [Docs](https://pbpstats.readthedocs.io/)
- [pbpstats API](https://api.pbpstats.com/docs) | [Code examples](https://github.com/dblackrun/pbpstats-api-code-examples)
- [nba-on-court GitHub](https://github.com/shufinskiy/nba-on-court) | [PyPI](https://pypi.org/project/nba-on-court/)
- [nba_data repository](https://github.com/shufinskiy/nba_data)
- [NBA.com Lineups Advanced](https://www.nba.com/stats/lineups/advanced)
- [NBA Play-By-Play Example (rd11490)](https://github.com/rd11490/NBA-Play-By-Play-Example)
- [Sports-Reference lineup stats announcement](https://www.sports-reference.com/blog/2020/04/nba-play-by-play-lineup-and-shooting-stats-added-back-to-1996-97/)

### Analytics Platforms
- [Cleaning the Glass](https://cleaningtheglass.com/) | [On/Off Guide](https://cleaningtheglass.com/stats/guide/player_onoff) | [Lineup Guide](https://cleaningtheglass.com/stats/team/2/lineups)
- [Basketball Index](https://www.bball-index.com/making-more-reliable-lineup-data/)
- [pbpstats.com](https://www.pbpstats.com/)
- [DARKO](https://apanalytics.shinyapps.io/DARKO/) | [beta.darko.app](https://beta.darko.app/)
- [Dunks & Threes EPM](https://dunksandthrees.com/epm) | [About EPM](https://dunksandthrees.com/about/epm)
- [databallr WOWY](https://databallr.com/wowy)
- [CraftedNBA](https://craftednba.com/)
- [Thinking Basketball WOWYR](https://thinkingbasketball.net/metrics/wowyr/)

### Research Papers & Articles
- [L-RAPM: Lineup Regularized Adjusted Plus-Minus (2025)](https://arxiv.org/abs/2601.15000)
- [Maymin et al. "NBA Chemistry: Positive and Negative Synergies"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1935972)
- [Five Reasons NBA Lineup Data Is Lying to You (The Ringer)](https://www.theringer.com/2023/04/13/nba/2023-nba-playoff-preview-how-to-use-and-not-use-lineup-data)
- [Roster Construction in the NBA (NBA Insights Substack)](https://nbainsights.substack.com/p/roster-construction-in-the-nba-using)
- [The Definitive Introduction to Impact Metrics (nbacademic)](https://nbacademic.wordpress.com/2018/12/14/the-definitive-introduction-to-impact-metrics/)
- [Adding lineups to NBA play-by-play data (NBA in R)](https://nbainrstats.netlify.app/post/adding-lineups-to-nba-play-by-play-data/)
- [RAPM deep dive (Squared Statistics)](https://squared2020.com/2017/09/18/deep-dive-on-regularized-adjusted-plus-minus-i-introductory-example/)
- [NBA APM: How to Build It (Royce Webb)](https://www.roycewebb.com/p/nba-adjusted-plus-minus-how-to-build)
- [Learning Stochastic Models for Basketball Substitutions](https://ceur-ws.org/Vol-1970/paper-08.pdf)
- [Persistent Homology of NBA Lineups (Wiseman 2025)](https://journals.sagepub.com/doi/10.1177/22150218251324877)
- [Wharton: Algorithmic NBA Player Acquisition](https://wsb.wharton.upenn.edu/wp-content/uploads/2023/12/Brill_2023_Q.pdf)
- [Unified ML Framework for Basketball Roster Construction (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S1568494624000723)
- [NBA On/Off Stats Explained (CHGO)](https://allchgo.com/nba-on-off-stats-numbers-data-metrics-explained-how-to-best-uses/)
- [RAPM Explained (NBAstuffer)](https://www.nbastuffer.com/analytics101/regularized-adjusted-plus-minus-rapm/)
- [DARKO Explained (NBAstuffer)](https://www.nbastuffer.com/analytics101/darko-daily-plus-minus/)
- [How to Make Team Ratings (basic-nba-tutorials)](https://github.com/anpatton/basic-nba-tutorials/blob/main/team_ratings/how_to_make_team_ratings.md)
- [82games.com Substitution Patterns](http://www.82games.com/simmons2.htm)
