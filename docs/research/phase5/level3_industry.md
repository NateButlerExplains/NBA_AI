# Level 3: Team-Level Modeling -- Industry Research

Research date: 2026-03-16

This document surveys production rating systems, coaching analytics, available NBA data
endpoints, and pace/strength-of-schedule normalization techniques relevant to building
Level 3 (team-level) features for our NBA prediction model. Level 3 captures effects
beyond individual players: coaching schemes, organizational systems, team chemistry,
and contextual adjustments.

---

## 1. ESPN Basketball Power Index (BPI)

### How It Works

BPI is a forward-looking measure of team quality representing how many points above
or below average a team is. It projects each team's point differential against an
average opponent on a neutral court.

**Core methodology:**
- Rates each team's offensive and defensive efficiency independently
- Adjusts for pace (per-possession basis), so a team scoring 100 on 80 possessions
  rates higher than one scoring 100 on 100 possessions
- Factors in opponent strength so that beating a top team is worth more than beating
  a weak one
- Uses 10,000 Monte Carlo simulations to project season outcomes, playoff odds,
  and championship probabilities

**Game-level adjustments:**
- Home-court advantage (location of game)
- Travel distance (including cross-country penalty)
- Days of rest between games
- Altitude effects (Denver at 5,280 ft, Utah at 4,226 ft)

**Preseason priors (college BPI, transferable concepts):**
- Coach's historical track record (adjusted O/D ratings since 2007-08, including prior schools)
- Recruiting talent grades
- Returning roster minutes (injury-adjusted)
- Returning player opponent-adjusted performance
- Combined via Bayesian hierarchical model, with weights varying by returning minutes %

**NBA-specific BPI:**
- Incorporates Vegas expected win totals and prior-year performance as preseason priors
- Down-weights outlier performances in-season
- Accounts for game-by-game efficiency, schedule strength, and pace simultaneously

**Key insight for Level 3:** BPI's Bayesian hierarchical preseason model that weights
coaching track record alongside roster continuity is directly relevant. The concept of
a coach prior that decays as in-season data accumulates is implementable.

Sources:
- [ESPN BPI Methodology](https://www.espn.com/blog/statsinfo/post/_/id/125994/bpi-and-strength-of-record-what-are-they-and-how-are-they-derived)
- [ESPN NBA BPI Explanation](https://www.espn.com/nba/story/_/page/Basketball-Power-Index/espn-nba-basketball-power-index)
- [ESPN NBA BPI 2025-26 Introduction](https://www.espn.com/nba/story/_/id/46626742/introducing-espn-basketball-power-index-2025-2026-predictions-23-nba-teams)


---

## 2. Team Rating Systems

### 2.1 FiveThirtyEight Elo + RAPTOR

**Elo Component:**
- Each franchise starts at 1300 Elo in inaugural season; long-term average is 1500
- K-factor of 20 (relatively responsive to recent results)
- Home-court advantage: +100 Elo points
- Margin of victory matters (with diminishing returns for blowouts)
- Season reset: teams revert 1/4 of the way toward 1505 mean each offseason
- Data source: game scores and venues only

**RAPTOR Player Ratings:**
- Blend of box score stats, player tracking metrics, and plus/minus data
- Measures per-100-possession impact on team offensive/defensive efficiency
- Based on 6-year RAPM sample with expanded box score and luck-adjusted on-off components

**Team Rating Aggregation (critical for Level 3):**
- Weighted average of player O-RAPTOR and D-RAPTOR by expected minutes, multiplied by 5
- **Diminishing returns scalar**: 0.8 in regular season, 0.9 in playoffs -- accounts for
  the gap between individual talent sum and actual on-court team performance
- This 0.8 scalar is effectively the "team chemistry/coaching tax" -- the 20% gap
  represents system effects, chemistry, and coordination costs

**Elo-RAPTOR Blend:**
- Default: 35% Elo + 65% RAPTOR talent
- Varies from 0-55% Elo weight based on **roster continuity** -- how much the current
  depth chart overlaps with the lineup that built the Elo rating
- High continuity = trust Elo more (it reflects this team's actual performance)
- Low continuity (post-trade) = trust RAPTOR more (Elo is stale)

**Handling Roster Changes:**
- Automated ingestion of ESPN injury/trade/transaction data, game-by-game
- Depth-chart algorithm assigns minutes by position for each game
- Two team ratings maintained: "Current Rating" (with injuries) and "Full-Strength Rating"
- Player ratings phase in slowly: no in-season update until 100 minutes played,
  gradual phase-in from 100-1,000 minutes

**Additional game-level factors:**
- Home-court: ~70 rating points boost
- Back-to-back fatigue: -46 rating points penalty
- Travel distance penalty
- Altitude bonus for high-elevation home venues
- Playoff experience: weighted average of career playoff minutes

**Practical takeaway:** The 0.8 diminishing returns scalar is the simplest way to model
the team-level effect. The roster continuity metric for blending historical vs current
ratings is directly implementable. The dual "current vs full-strength" rating concept
handles injuries gracefully.

Sources:
- [FiveThirtyEight NBA Predictions Methodology](https://fivethirtyeight.com/methodology/how-our-nba-predictions-work/)
- [FiveThirtyEight Elo Methodology](https://fivethirtyeight.com/features/how-we-calculate-nba-elo-ratings/)
- [FiveThirtyEight Elo Analysis](https://nicidob.github.io/nba_elo/)
- [Introducing RAPTOR](https://fivethirtyeight.com/features/introducing-raptor-our-new-metric-for-the-modern-nba/)

### 2.2 Neil Paine's Dual-Track Elo (2025-26)

An evolution of FiveThirtyEight's approach with two parallel Elo tracks:

- **Regular-Season Elo**: Reacts quickly to recent results; reflects day-to-day form,
  scheduling noise, injuries
- **Playoff Elo**: Updates heavily from postseason games, moves slowly during regular
  season; captures the longer-term quality that matters in playoffs when rotations tighten

Both tracks interact slightly. Ratings regress toward league average each offseason.
In "Composite Mode," betting market probabilities (FanDuel lines) are blended in,
which implicitly captures injury/roster information the pure Elo model misses.

**Practical takeaway:** The dual-track concept maps to our model -- we could maintain
separate learned representations for "regular season form" vs "playoff capability."

Sources:
- [Neil Paine 2025-26 NBA Elo Forecast](https://neilpaine.substack.com/p/2025-26-nba-elo-forecast-and-player)

### 2.3 Massey Ratings

- Built from a large system of simultaneous equations; only uses score, venue, and date
- Each game creates an equation connecting two teams; the full system is coupled
  so a team's rating depends on the infinite chain of opponents' opponents
- Rating and strength of schedule are interdependent, computed jointly
- Separates offensive and defensive components
- Eliminates noise through the mathematical structure (many redundant equations)

**Key property:** Because SOS is implicit in the model, it accounts for opponent strength
to an effectively infinite number of levels -- no separate SOS calculation needed.

Sources:
- [Massey Ratings Description](https://masseyratings.com/theory/massey.htm)
- [Massey Ratings FAQ](https://masseyratings.com/faq.php)
- [Massey 1997 Paper](https://masseyratings.com/theory/massey97.pdf)

### 2.4 Sagarin Ratings

Three blended components:
1. **PREDICTOR**: Score-based with diminishing returns for blowouts. Comfortable wins
   are rewarded but running up the score is not.
2. **GOLDEN_MEAN**: Separate scoring-based system (proprietary equations)
3. **RECENT**: Recency-weighted version that captures momentum/form

Early-season: uses Bayesian network with starting rankings until the graph of
team-vs-team matchups is well-connected. Once enough cross-play exists, the
Bayesian priors are dropped.

**Practical takeaway:** The Bayesian prior that fades as data accumulates is a clean
pattern for handling season starts and post-trade-deadline roster changes.

Sources:
- [Sagarin NBA Ratings](http://sagarin.com/sports/nbasend.htm)
- [Sagarin Ratings Guide](https://www.pointspreads.com/guides/sagarin-betting-system-guide/)

### 2.5 KenPom (College Basketball, Transferable Concepts)

**Core methodology:**
- Rates offense and defense independently as adjusted efficiency (points per 100 possessions)
- "Adjusted" = performance against average competition at neutral site
- Per-game adjustment: (raw team OE * national average OE) / opponent's adjusted DE
- Final rating: weighted average of adjusted game efficiencies (recency-weighted)
- Accounts for rising average efficiency during the season by benchmarking against
  the national average on the game date

**Adjusted Tempo (AdjT):** Possessions per 40 minutes, captures team pace independent
of opponent pace.

**Transferable insight:** The per-game opponent adjustment formula is elegant and
avoids iterative convergence issues. Recency weighting captures form changes.

Sources:
- [KenPom Ratings Explanation](https://kenpom.com/blog/ratings-explanation/)
- [KenPom Methodology Update](https://kenpom.com/blog/ratings-methodology-update/)

### 2.6 How Rating Systems Handle Roster Changes

**Pure results-based systems (Elo, Massey, Sagarin):**
- Do NOT explicitly handle roster changes
- Elo's K-factor means it adapts in ~10-15 games after a major trade
- Season-start mean reversion partially handles offseason turnover
- Weakness: slow to react to mid-season trades

**Player-based systems (RAPTOR, RPM, DARKO):**
- React immediately to roster changes by re-aggregating player ratings
- DARKO increases learning rate after a player changes teams (gets less confident)
- FiveThirtyEight's depth-chart algorithm knows when a traded player becomes available

**Hybrid approaches (FiveThirtyEight's Elo+RAPTOR blend):**
- Best of both worlds: player-based ratings handle roster changes instantly,
  while Elo provides stability for unchanged rosters
- Roster continuity metric dynamically shifts the blend weight

**Practical takeaway for our model:** We should maintain both a "team-level learned
embedding" (analogous to Elo -- captures coaching, culture, system) and a
"roster-aggregated rating" (from Level 2 player model). A continuity-weighted blend
is the industry standard approach.


---

## 3. Coaching Analytics in Practice

### 3.1 Quantifying Coaching Impact

**Residual-based approaches (the industry standard):**

The dominant method treats coaching as the residual after accounting for roster quality:

1. **ΔtVORP Method** (de Gruyter, 2025):
   - Compute team-adjusted VORP: tVORP = sum(player_VORP * game_minutes / 240)
   - Compare home vs away tVORP (ΔtVORP) to get roster quality differential
   - Model win probability as f(ΔtVORP) per coach using probit monotone Bayesian
     Additive Regression Trees (mBART)
   - **Finding:** 85% of coaches show NO statistically significant scheme advantage
     when teams are equally matched (ΔtVORP = 0)
   - **Elite coaches** like Gregg Popovich win ~60% of even-talent games
   - Some Hall of Fame coaches ranked average in scheme; their success was roster-driven

2. **Fixed Effects Model** (TDS, 2020):
   - Regression of team wins on coach identity with control variables for roster quality
     and injuries across 808 team-seasons (1979-2019), 163 coaches

3. **LightGBM Approach** (Kim & Lee, 2025):
   - Estimate each team's theoretical win probability using only prior-season player
     statistics, deliberately excluding coaching effects
   - Discrepancy between prediction and actual outcome = coaching margin
   - LightGBM achieved 68.50% prediction accuracy
   - **Key finding:** Coaches with higher coaching margins show significantly stronger
     season-level performance outcomes over time

**Practical takeaway:** Coaching effect IS measurable but small (most coaches are average).
For our model, the residual approach is most natural -- our Level 2 player model predicts
game outcomes; the systematic residual per coach/team IS the Level 3 signal.

Sources:
- [Analyzing effects of NBA head coaches](https://www.degruyterbrill.com/document/doi/10.1515/jqas-2025-0025/html)
- [AI evaluation of NBA head coaches](https://journals.sagepub.com/doi/10.1177/22150218251357538)
- [UNC Coaching Impact Research](https://www.unc.edu/wp-content/uploads/2025/06/coach_research.pdf)

### 3.2 Rotation Optimization

**Staggering vs. Syncing Stars:**
- Milwaukee Bucks: three-man stagger (Giannis/Middleton/Holiday rotate out in sequence)
  ensures at least one star is always on court
- Golden State Warriors: Curry and Green are nearly always synced -- they play together
  or sit together, maximizing their combined impact

**Data-driven optimization:**
- Teams analyze net ratings for lineup combinations: both stars together, stars
  separated, both on bench
- Starter-majority lineups (3+ starters) consistently outperform bench-majority units
- The Bucks led starter-majority lineups at +13.38 net per 100 possessions

**For our model:** Rotation patterns create detectable signatures in team performance
data. A team with deep bench (small starter-bench gap) performs differently than one
relying on starters. This is capturable via lineup volatility features.

Sources:
- [FiveThirtyEight: How Best NBA Teams Juggle Lineups](https://fivethirtyeight.com/features/how-the-best-nba-teams-juggle-their-lineups/)
- [Nylon Calculus: Staggering Stars](https://fansided.com/2019/01/22/nylon-calculus-stagger-stars-minutes/)

### 3.3 Timeout Usage Impact

**Counter-intuitive research finding:**
- Multiple studies using causal inference methods find timeouts have NO measurable
  effect on stopping opposing runs
- The perceived benefit is regression to the mean -- after a run, the trailing team
  would have improved anyway
- The magnitude of this null effect varies somewhat by franchise

**Where timeouts DO matter:**
- After timeouts, teams can draw better plays: up to +0.15 points per possession
  (but only if coach waits 10+ seconds to see if team generates a look first)
- In near-jump-ball situations: calling timeout creates ~0.5 point advantage
- Challenging a 3-point foul: worth 1.247 points (highest-value challenge)

**NBA challenge statistics:**
- Average success rate: 59.5%
- Out-of-bounds challenges: 77.7% success
- Foul challenges: 53% success
- Optimal strategy: aggressive on 3-point fouls and goaltending, conservative early
  on 2-point fouls and out-of-bounds calls

**For our model:** Timeout strategy is likely noise rather than signal for game prediction.
However, coach challenge success rate could be a minor feature reflecting coaching
preparation quality.

Sources:
- [Causal Effect of Timeouts](https://arxiv.org/abs/2011.11691)
- [How Coaches Should Use Challenges](https://www.paraballnotes.com/blog/how-should-nba-coaches-use-their-challenges)
- [Stop the Clock: Are Timeout Effects Real?](https://homepages.dcc.ufmg.br/~olmo/wordpress/wp-content/uploads/2020/06/PKDD2020_camera_ready.pdf)


---

## 4. NBA Team Data Available via API

### 4.1 Team Dashboard Endpoints (nba_api)

**Core team stats:**
- `LeagueDashTeamStats` -- comprehensive team stats with MeasureType options:
  - Base: GP, W, L, W_PCT, FGM, FGA, FG_PCT, FG3M, FG3A, FG3_PCT, FTM, FTA,
    FT_PCT, OREB, DREB, REB, AST, TOV, STL, BLK, PF, PTS, PLUS_MINUS (+ ranks)
  - Advanced: OFF_RATING, DEF_RATING, NET_RATING, PACE, PIE, AST_PCT, AST_TO,
    AST_RATIO, OREB_PCT, DREB_PCT, REB_PCT, EFG_PCT, TS_PCT, etc.
  - Four Factors: EFG_PCT, FTA_RATE, TM_TOV_PCT, OREB_PCT (off + def)
  - Misc: PTS_OFF_TOV, PTS_2ND_CHANCE, PTS_FB, PTS_PAINT, OPP_* equivalents
  - Scoring: PCT_FGA_2PT, PCT_FGA_3PT, PCT_PTS_MIDRANGE, PCT_AST_FGM, etc.
  - Opponent: All base stats from opponent's perspective
  - PerMode options: Totals, PerGame, Per48, Per36, Per100Possessions
  - PaceAdjust: Yes/No

**Team dashboard splits:**
- `TeamDashboardByGeneralSplits` -- home/away, wins/losses, by month, pre/post-ASB
- `TeamDashboardByClutch` -- performance in close games (various score differentials)
- `TeamDashboardByGameSplits` -- by half, by quarter
- `TeamDashboardByLastNGames` -- last 5, 10, 15, 20 games
- `TeamDashboardByOpponent` -- vs conference, division, specific opponents
- `TeamDashboardByShootingSplits` -- by shot distance, shot area
- `TeamDashboardByTeamPerformance` -- by score differential buckets
- `TeamDashboardByYearOverYear` -- multi-season comparison

### 4.2 Player Tracking (Second Spectrum)

`LeagueDashPtStats` with PtMeasureType options:
- **SpeedDistance**: distance traveled, average speed
- **Rebounding**: contested/uncontested rebounds, rebound chances
- **Possessions**: touches, time of possession, front court touches
- **CatchShoot**: catch-and-shoot FG%, attempts
- **PullUpShot**: pull-up shooting stats
- **Defense**: contested shots, defensive FG% allowed
- **Drives**: drives per game, drive points, drive assists
- **Passing**: passes made, potential assists, secondary assists
- **ElbowTouch/PostTouch/PaintTouch**: scoring by zone
- **Efficiency**: points per touch, etc.

### 4.3 Hustle Stats

- `LeagueHustleStatsTeam` / `LeagueHustleStatsTeamLeaders`
- Fields: CONTESTED_SHOTS, CONTESTED_SHOTS_2PT, CONTESTED_SHOTS_3PT,
  DEFLECTIONS, CHARGES_DRAWN, LOOSE_BALLS_RECOVERED, SCREEN_ASSISTS

### 4.4 Team Estimated Advanced Stats

Available at NBA.com and via `TeamEstimatedMetrics` endpoint:
- Estimated Offensive Rating, Defensive Rating, Net Rating
- Estimated Pace
- These use NBA's internal models that account for possessions more precisely

### 4.5 Lineup Data

- `LeagueDashLineups` -- performance of specific N-man lineup combinations
- Can query 2-man through 5-man lineups
- Returns NET_RATING, MIN, PLUS_MINUS for each combination

### 4.6 Most Promising Level 3 Features from API Data

**High-value team-level features (not derivable from individual PlayerBox):**
1. Team clutch performance splits (TeamDashboardByClutch)
2. Hustle stats: deflections, contested shots, loose balls (team effort/system)
3. Tracking: team speed/distance (pace style), drives per game (offensive system)
4. Four Factors differentials (eFG%, TOV%, OREB%, FT rate -- off and def)
5. Lineup consistency metrics (from LeagueDashLineups)
6. Catch-and-shoot % vs pull-up % (offensive system indicator)
7. Points in paint / fastbreak points / second-chance points (play style)

Sources:
- [nba_api GitHub](https://github.com/swar/nba_api)
- [nba_api Documentation](https://nba-apidocumentation.knowledgeowl.com/help)
- [hoopR LeagueDashTeamStats](https://hoopr.sportsdataverse.org/reference/nba_leaguedashteamstats.html)
- [NBA.com Teams Advanced](https://www.nba.com/stats/teams/advanced)
- [NBA.com Teams Estimated Advanced](https://www.nba.com/stats/teams/estimated-advanced)


---

## 5. Pace and Style Adjustments

### 5.1 Per-100-Possessions Normalization

The industry standard for meaningful team comparison:
- Offensive efficiency = points scored per 100 possessions
- Defensive efficiency = points allowed per 100 possessions
- Net rating = offensive efficiency - defensive efficiency
- Eliminates pace confound: a team scoring 100 points on 80 possessions (125.0 OE)
  is better than one scoring 100 on 100 possessions (100.0 OE)

**Example:** The Knicks averaged 100.0 PPG (11th in NBA) but were 3rd in offensive
efficiency because they played at the 5th-slowest pace.

**Pace formula:** Possessions per 48 minutes, calculated from FGA, FTA, OREB, TOV

### 5.2 Opponent-Strength Adjustment

**KenPom-style per-game adjustment:**
```
Adjusted OE(game) = Raw OE(game) * League_Avg_OE / Opponent_Adj_DE
```
This is iterative (opponent's adjusted DE depends on their opponents' adjusted OE),
but converges quickly. More recent games are weighted more heavily.

**Simple Rating System (SRS):**
- For each team: Rating = MOV + SOS
- SOS = average of opponents' ratings
- This creates a system of 30 simultaneous equations, solved iteratively
- Equivalent to Massey's approach in the limit

**Regression-based adjustment (RPM/RAPM family):**
- Ridge regression on play-by-play data with player indicators
- Opponent strength is automatically controlled because opponents appear as
  negative indicators in the regression

### 5.3 Strength of Schedule (SOS)

Common approaches:
1. **Win-percentage based**: Average of opponents' W% (simple but circular)
2. **RPI-style**: 25% team W%, 50% opponents' W%, 25% opponents' opponents' W%
3. **Rating-based**: Average of opponents' net ratings (SRS-style)
4. **Venue-adjusted**: Accounts for home/away split in opponent quality

**Advanced SOS (CraftedNBA approach):**
- Rotation-aware blend of projected team strength, adjusted net rating, and raw
  net rating
- Venue and rest adjustments applied per opponent
- Adjusted net rating = raw net rating + SOS difference

### 5.4 Style Normalization

Teams have measurable stylistic signatures worth capturing:
- **Pace** (possessions per game): ranges from ~96 to ~106 in modern NBA
- **Three-point rate** (% of FGA from 3): team offensive philosophy
- **Paint scoring rate**: inside vs outside orientation
- **Fastbreak frequency**: transition vs half-court preference
- **Assist rate**: ball movement vs isolation
- **Defensive scheme**: points allowed in paint, 3PT defense, rim protection

**For Level 3:** Style features are team-level properties that persist across roster
changes (coaching system). A team's 3PT rate and pace often reflect the coach's
philosophy more than individual players.

Sources:
- [Per-100-Possessions Explanation](https://bleacherreport.com/articles/1813902-advanced-nba-stats-for-dummies-how-to-understand-the-new-hoops-math)
- [Pace and Efficiency Analysis](https://thedatajocks.com/models-and-nba-pace-stats/)
- [NBAstuffer SOS](https://www.nbastuffer.com/analytics101/strength-of-schedule-sos/)
- [NBA Stats Team Possessions](https://www.teamrankings.com/nba/stat/possessions-per-game)


---

## 6. Pre-Season Projections and Team Factors

### 6.1 DARKO (Daily Adjusted and Regressed Kalman Optimized)

**Player-level Bayesian projection system:**
- Updates projections for every player, every box-score stat, every day
- Uses combination of classical statistics and machine learning
- Bayesian: update magnitude varies by player and stat
- Independent aging curves per stat (players age differently in different skills)

**Team-level factors:**
- Accounts for opponent influence on each stat component
- Rest/travel/home-court effects applied at component level (not just total)
- After team changes: increases learning rate (gets less confident in prior)
- Daily updates reflecting who plays and who doesn't

**Practical takeaway:** DARKO's per-component rest/travel adjustment (not just a
single scalar) is more sophisticated than most systems.

Sources:
- [DARKO Exploration](https://apanalytics.shinyapps.io/DARKO/)
- [DARKO DPM Explained](https://www.nbastuffer.com/analytics101/darko-daily-plus-minus/)

### 6.2 FiveThirtyEight's Player-to-Team Projection Pipeline

1. **Individual projection**: Find historical comparables, project career trajectory
2. **Minutes projection**: Blend 12.6 games of preseason MPG projection with actual
3. **Team aggregation**: Minutes-weighted RAPTOR * 5, with 0.8 diminishing returns scalar
4. **Blend with Elo**: 35% Elo / 65% RAPTOR (varies by roster continuity, 0-55%)
5. **Game adjustments**: Home court (+70), back-to-back (-46), travel, altitude
6. **Simulation**: 50,000 Monte Carlo runs for season/playoff projections

### 6.3 The Player-to-Team Gap

**What the 0.8 scalar captures (FiveThirtyEight's regular season adjustment):**
- Coordination costs: 5 players must share the ball, reducing individual efficiency
- Defensive scheme effects: team defense > sum of individual defenders
- Coaching system fit: players may not be optimally utilized
- Chemistry and communication
- Role compression: not everyone can be "the guy"

**Research quantification:**
- Team chemistry (synergy) effects can explain up to ~6 wins difference between teams
  with equivalent individual talent (Maymin et al.)
- Teams are equally likely to have positive or negative chemistry
- A quadratic classifier using synergy/adversity matrices between player pairs
  captures interaction effects that linear aggregation misses

### 6.4 What "Team Factors" Systems Add Beyond Player Projections

Based on surveying multiple systems:

| Factor | Who Uses It | Magnitude |
|--------|------------|-----------|
| Coaching scheme | BPI, coaching margin research | ~2-5 wins for elite vs avg coach |
| Roster continuity | FiveThirtyEight (Elo blend weight) | Varies Elo weight 0-55% |
| Diminishing returns | FiveThirtyEight (0.8 scalar) | ~20% talent discount |
| Home court | All systems | 3-5 points / ~70 Elo |
| Rest/fatigue | All production systems | ~1.8 points for B2B |
| Travel distance | BPI, FiveThirtyEight | Distance-dependent |
| Altitude | BPI, FiveThirtyEight | Denver/Utah specific |
| Playoff experience | FiveThirtyEight | Career playoff minutes |
| Betting market prior | Neil Paine, BPI | Blended preseason |
| Analytics investment | MIT/Hosoi 2025 | ~1 win per 4/5 analyst |

Sources:
- [NBA Chemistry Paper](https://philipmaymin.com/papers/Maymin%20Maymin%20and%20Shen%20-%20NBA%20Chemistry%20-%20IJCSS.pdf)
- [Predicting Elite Lineups](https://www.degruyterbrill.com/document/doi/10.1515/jqas-2022-0039/html)
- [Basketball Analytics Investment](https://news.mit.edu/2025/basketball-analytics-investment-nba-wins-and-other-successes-0325)


---

## 7. Dean Oliver's Four Factors

The foundational framework for team-level basketball evaluation, accounting for 96%
of the variance in team wins:

| Factor | What It Measures | Weight |
|--------|-----------------|--------|
| Effective FG% (eFG%) | Shooting quality (includes 3PT bonus) | 40-45% |
| Turnover % (TOV%) | Ball security | 25% |
| Offensive Rebound % (ORB%) | Second-chance opportunities | 20% |
| Free Throw Rate (FTA/FGA) | Getting to the line | 15% |

Applied to both offense AND defense = 8 factors total.

**Cleaning the Glass refinements:**
- Excludes garbage time and end-of-quarter heaves
- Opponent-adjusts each factor: measures performance relative to how the league
  performs against those same opponents
- This "Four Factors Adjusted Rating" is closer to a true team quality metric

**For Level 3:** The four factors are directly computable from our existing TeamBox data.
They provide a natural compression of team offensive/defensive quality into 8 numbers
that explain 96% of win variance. These should be computed as rolling features.

Sources:
- [Basketball-Reference Four Factors](https://www.basketball-reference.com/about/factors.html)
- [Four Factors Revisited (arxiv)](https://arxiv.org/abs/2305.13032)
- [Cleaning the Glass Guide](https://cleaningtheglass.com/stats/guide/league_four_factors)


---

## 8. Player Impact Metrics Landscape

For context on what feeds into team-level aggregation:

| Metric | Inputs | Prediction Error | Notes |
|--------|--------|-----------------|-------|
| EPM | Box + tracking + RAPM | 2.48 | Gold standard (Ilardi endorsement) |
| RPM | Box + tracking + RAPM | ~2.50 | ESPN's metric, similar to EPM |
| RAPTOR | Box + tracking + on/off | 2.63 | FiveThirtyEight, broader data |
| BPM 2.0 | Box score only | 2.71 | Works historically (no tracking needed) |
| DARKO DPM | Box + Bayesian updating | -- | Daily updates, component projections |

**RAPM (Regularized Adjusted Plus-Minus) -- the backbone:**
- Ridge regression on play-by-play data, typically last 3 seasons
- Each row = a stint (time between substitutions)
- Columns = 10 player indicators (+1 for offense, -1 for defense)
- Target = point differential of stint
- Solves for each player's marginal contribution controlling for teammates/opponents

**For our model:** We don't need to replicate these -- our Level 2 player model learns
equivalent representations. The key insight is that the **residual** after our Level 2
predictions is where Level 3 team effects live.

Sources:
- [Dunks & Threes Metric Comparison](https://dunksandthrees.com/blog/metric-comparison)
- [EPM Methodology](https://dunksandthrees.com/epm)
- [RAPM Calculation Tutorial](https://medium.com/@johnchenmbb/calculating-rapm-steps-1-and-2-of-my-summer-plan-1a78e1476b1f)


---

## 9. Home Court Advantage Research

Quantified effects from academic studies:

| Factor | Effect Size | Source |
|--------|------------|--------|
| Overall HCA | +3.24 points (home team) | Entine & Small (Wharton) |
| Rest component of HCA | +0.31 points | Entine & Small |
| Non-rest HCA factors | +2.93 points | Entine & Small |
| B2B penalty (visitor) | -1.77 points | PMC study |
| Home W% (baseline) | ~58% (historical), declining | Multiple |
| Home W% (B2B for home) | ~53% | Entine & Small |
| Home W% (B2B for away) | ~63% | Entine & Small |
| Eastward travel W% | 44.51% | PMC travel study |
| Westward travel W% | 40.83% | PMC travel study |
| Extra rest (>1 day) | +1.1 pts home, +1.6 pts away | PMC study |
| Altitude (Denver/Utah) | Significant, proven | Multiple |

**Key insight:** Only ~10% of HCA is explained by rest/scheduling. The remaining
~90% comes from crowd effects, referee bias, familiarity with court/arena,
and travel fatigue beyond just rest.

Sources:
- [Wharton NBA Rest Study](https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Nba.pdf)
- [PMC Travel Distance Study](https://pmc.ncbi.nlm.nih.gov/articles/PMC8636381/)
- [NBAstuffer HCA](https://www.nbastuffer.com/analytics101/home-court-advantage/)


---

## 10. Practical Takeaways for Level 3 Implementation

### 10.1 Feature Categories to Implement

**A. Team Identity Features (persist across games, capture coaching/system):**
1. Rolling Four Factors (off + def = 8 features, computed from TeamBox)
2. Pace (possessions per game, rolling)
3. Style indicators: 3PT rate, paint scoring rate, fastbreak frequency, assist rate
4. Clutch performance differential (close-game performance vs blowout performance)
5. Hustle stats if available: deflections, contested shots, loose balls
6. Coach tenure / coaching continuity indicator

**B. Contextual Game Features:**
1. Home/away indicator (already have)
2. Days of rest for each team
3. Back-to-back flag
4. Travel distance (calculable from team locations)
5. Altitude differential
6. Time zone change direction (eastward travel is worse)
7. Playoff experience (average career playoff minutes in rotation)

**C. Roster Dynamics Features:**
1. Roster continuity (% of minutes from returning players vs last N games)
2. Recent trade/transaction flag
3. Diminishing returns scalar on aggregated player ratings
4. Starter-bench quality gap (lineup data or estimated from player ratings)

**D. Relative Strength Features:**
1. Rolling team Elo or SRS rating
2. Opponent-adjusted efficiency (KenPom-style)
3. Strength of schedule (rolling, opponent-adjusted)

### 10.2 Architecture Recommendations

1. **Residual modeling**: Level 3 should explicitly model the residual from Level 2
   (player-based) predictions. This is how coaching/system effects are isolated in
   the literature.

2. **Diminishing returns**: Apply a learned scalar (initialized ~0.8) when aggregating
   Level 2 player ratings to team ratings. This captures the coordination cost.

3. **Roster continuity weighting**: When blending Level 3 team embeddings with
   Level 2 aggregated player ratings, weight the blend by roster continuity.
   High continuity = trust team embedding more. Low continuity = trust player
   aggregation more.

4. **Dual-track representation**: Consider maintaining both a "recent form" team
   embedding (updated frequently from recent games) and a "baseline quality"
   embedding (slower-moving, captures stable coaching/organizational effects).

5. **Four Factors as compression**: The Four Factors explain 96% of team win variance
   and compress team performance into 8 numbers. Use as intermediate features in
   the team-level encoder.

### 10.3 What Data We Already Have vs Need to Collect

**Already in our database (TeamBox):**
- All base stats for computing Four Factors, pace, style indicators
- Game-level data for computing rolling features
- Home/away from Games table

**Computable from existing data:**
- Days of rest (from game dates)
- Back-to-back flags
- Travel distance (from team abbreviation + known arena locations)
- Roster continuity (from PlayerBox participation lists)
- Rolling Elo/SRS (from game results)

**Would need to collect (new API calls):**
- Hustle stats (LeagueHustleStatsTeam endpoint)
- Player tracking stats (LeagueDashPtStats)
- Clutch splits (LeagueDashTeamClutch)
- Lineup data (LeagueDashLineups)

**Not available / hard to get:**
- Detailed coaching decision data (timeout timing, challenge usage)
- Practice/film study investment
- Locker room chemistry / organizational culture
- Real-time injury severity (only availability is known)

### 10.4 Expected Impact

Based on the literature:
- The Four Factors alone explain 96% of team win variance (but much of this overlaps
  with player-level stats we already model)
- Coaching scheme effects: ~2-5 wins for best vs average coach (~3-6% accuracy)
- Home court + rest + travel: ~3-5 points per game (significant, partially captured)
- Roster chemistry: up to ~6 wins (but equally likely positive or negative)
- The **marginal improvement** from Level 3 on top of a good Level 2 player model
  is likely 1-3% in win prediction accuracy, mainly from:
  - Better contextual adjustments (rest, travel, altitude)
  - Coaching system effects in the residual
  - Roster continuity and chemistry signals
  - Style matchup interactions

The biggest gains will come from contextual features (rest, travel, home court)
which are well-quantified and straightforward to implement, rather than from the
harder-to-measure coaching/culture effects.
