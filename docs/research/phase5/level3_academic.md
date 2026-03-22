# Level 3: Team-Level Modeling -- Academic Literature Review

> **Purpose**: Level 3 captures what remains after accounting for individual player quality (Level 1) and player interactions/chemistry (Level 2). This is the team-level residual: coaching effects, organizational culture, system effects, rotation philosophy. It is location-agnostic (home court advantage belongs to Level 4).

---

## 1. Coaching Impact Quantification

### 1.1 Berry & Fowler -- "How Much Do Coaches Matter?" (Sloan Sports Analytics Conference)

**Authors**: Christopher Berry, Anthony Fowler (University of Chicago Harris School of Public Policy)

**Method**: Developed Randomization Inference for Leader Effects (RIFLE). Applied across MLB, NBA, NHL, NFL, college football, and college basketball. Estimates coaching effects by examining natural variation in team success within particular teams while accounting for home team advantage and opponent quality.

**Key findings**:
- The right NBA head coach is worth approximately **14 wins per season**.
- Coaches explain about **20--30% of the variation** in a team's success across all sports studied.
- For the NBA specifically, the estimate is approximately **30%** of win variance attributable to coaching.
- Baseball managers affect runs allowed more than runs scored.
- Coaches matter more in college football than in the NFL.

**Practical takeaway for Level 3**: This is the headline number. If ~30% of win variance is coaching, and player talent explains some large chunk (say 50--60%), the remaining 10--20% is noise, organizational factors, and Level 4 context. The 14-win swing between the best and worst coaches suggests a substantial Level 3 signal.

**Reference**: [Sloan Sports Conference](https://www.sloansportsconference.com/research-papers/how-much-do-coaches-matter); [Semantic Scholar PDF](https://www.semanticscholar.org/paper/How-Much-Do-Coaches-Matter-Berry-Fowler/ace1d5028149ca17945c7d1ac12334bbb5b08903)

---

### 1.2 Al-Amine -- "Quantifying the Contribution of NBA Coaches using Fixed Effects" (2020)

**Method**: Regression of team wins on head coach identity (fixed effects), controlling for:
1. **Roster talent** via mean Value Over Replacement Player (VORP) of the top 5 players by minutes (prior season).
2. **Injuries** measured as total games missed by main players.

Covered 808 team-seasons, 163 head coaches from 1979--80 to 2018--19.

**Key findings**:
- Some coaching effects (e.g., Greg Popovich, Steve Kerr, Nick Nurse) are larger than the effects of individual superstars during All-Star seasons.
- The fixed effects model isolates coach-specific contributions after controlling for talent and injuries.

**Practical takeaway for Level 3**: Fixed effects on coach identity are a viable feature. Could encode coaching quality as a learned embedding per coach, or as a derived statistic (career wins above expected).

**Reference**: [Towards Data Science](https://towardsdatascience.com/quantifying-the-contribution-of-nba-coaches-using-fixed-effects-56f77f22153a/)

---

### 1.3 Cannon, Fisher, Fellingham & Page -- "Analyzing the Effects of NBA Head Coaches" (JQAS, 2025)

**Method**: Isolates the effect of a coach's in-game scheme on win probability while controlling for relative team strength via "Team-Adjusted VORP Difference" (delta-tVORP), which accounts for the difference in quality of on-court product between competing teams.

**Key findings**:
- Successfully separates coaching scheme effects from roster talent effects.
- Provides a methodology for attributing in-game tactical decisions to win probability changes.

**Practical takeaway for Level 3**: This is the most methodologically rigorous approach to isolating coaching from talent. The delta-tVORP approach could inform how we construct the coaching residual: predict wins from player quality, and attribute the residual to coaching.

**Reference**: [De Gruyter JQAS](https://www.degruyterbrill.com/document/doi/10.1515/jqas-2025-0025/html)

---

### 1.4 Midseason Coaching Changes -- Regression to the Mean

**Key finding**: Teams usually increase their win percentage after firing their coach, but this improvement is largely explained by **regression to the mean**, not causal coaching improvement. The true causal effect of changing coaches midseason is a **significant decrease** in the likelihood of winning successive games. Teams improve approximately **4 percentage points less** than they would have by keeping the same coach.

Only about **12%** of new coaches significantly outperform their predecessors after controlling for regression effects.

**Practical takeaway for Level 3**: Coaching change is a strong feature, but must be modeled carefully. A new coach should initially be penalized (disruption), with gradual improvement over time. The adaptation curve matters: coaches improve their teams over successive years, with the **3-year improvement** being statistically significant.

**References**: [ResearchGate - Midseason Change](https://www.researchgate.net/publication/285924524_Does_Midseason_Change_of_Coach_Improve_Team_Performance_Evidence_From_the_NBA); [Redalyc](https://www.redalyc.org/journal/710/71051616001/html/)

---

### 1.5 Coaching Tenure Patterns

- Median NBA head coaching tenure: **2.5 seasons**.
- 18.1% of coaches last a single season or less.
- Barely 10% last more than 5 seasons.
- Among non-Spurs teams, average current tenure is just 1.9 seasons.
- Performance improvement per year of tenure is statistically significant by year 3.

**Practical takeaway for Level 3**: Coach tenure should be a feature. Encode as (coach_games_with_team / 82) or similar normalized measure. Expect an inverted-U: improvement through ~3--5 years, then potentially diminishing returns (see Berman et al. on knowledge ossification below).

---

## 2. Team System Effects

### 2.1 Synergy Play Type Classification

Synergy Sports Technology classifies possessions into **11 play types**: cuts, handoffs, isolations, off-screen plays, pick-and-roll ball-handler, pick-and-roll roll-man, post-ups, spot-ups, transition, putbacks, and miscellaneous.

**Key findings on efficiency by play type**:
- **Cuts** are the most efficient play type by a considerable margin (closer to hoop, fewer dribbles).
- **Isolation** is among the least efficient but generates pass-out opportunities.
- **Transition** is highly efficient but volume-limited.
- **Pick-and-roll** is the most frequent and creates the most complex decision trees.

**Surprising finding**: There is "little evidence of a significant relationship between the frequencies with which teams perform or allow certain play types, and the resulting strength of their offensive and defensive efficiency." This suggests that **how well** you execute a system matters far more than **which** system you run.

**Practical takeaway for Level 3**: Play type distribution may be less predictive than play type efficiency. The system effect is about execution quality under a scheme, not scheme choice per se. This means Level 3 should capture "system execution quality" rather than categorical scheme type.

**References**: [Nylon Calculus / FanSided](https://fansided.com/2017/09/08/nylon-calculus-understanding-synergy-play-type-data/); [Play Types and Varying Importance](https://fansided.com/2015/02/16/play-types-varying-importance/)

---

### 2.2 Offensive System Research: Spatial and Temporal Analysis

**MIT Thesis -- "Player motion analysis: automatically classifying NBA plays"**: Used SportVU optical tracking (25fps) to automatically classify NBA plays from player movement data.

**TacticExpert (2025, arXiv 2503.10722)**: A spatial-temporal graph language model for basketball tactics that analyzes player trajectories with event types and locations across sequential time slices.

**Density-Functional Fluctuation Theory (DFFT)** (Scientific Reports, 2025): Uses physics-inspired methods to infer spatial preferences and player-to-player interactions in NBA basketball, offering a novel way to characterize offensive systems through spatial distributions.

**Practical takeaway for Level 3**: Offensive system effects could be captured through spatial tendency features (e.g., shot distribution heatmaps, average touch location, ball movement entropy). Without tracking data, proxy features from box scores include: 3PA/FGA ratio, FTA/FGA, pace, assist-to-FGM ratio, and turnover rate.

**References**: [MIT DSpace](https://dspace.mit.edu/handle/1721.1/100664); [TacticExpert arXiv](https://arxiv.org/html/2503.10722v1); [DFFT Nature](https://www.nature.com/articles/s41598-025-04953-x)

---

### 2.3 Offensive Duration and Pacing

**Research (PMC 12121881)**: Teams that sustained longer possessions in final game stages had more success maintaining leads or mounting comebacks. Offensive success depends not only on speed but on the ability to adjust possession duration to match game context.

**Practical takeaway for Level 3**: Pace is a team-level system choice that interacts with game context. Average pace alone is insufficient; pace variability and context-dependent pace adjustment are more telling of coaching/system quality.

---

## 3. Residual Modeling in Hierarchical Sports Models

### 3.1 The Core Question: What's Left After Player Talent?

This is the central conceptual challenge for Level 3. Several approaches have been taken:

**RAPM (Regularized Adjusted Plus-Minus)**: Decomposes team performance into individual player contributions using ridge regression on plus-minus data. The residual (what RAPM cannot explain) includes: coaching effects, team chemistry, scheme fit, and noise. **RAPM fails to account for the impact of coaching and the synergistic effects of roster construction**, making it fundamentally a player metric rather than a team metric.

**L-RAPM (Lineup Regularized Adjusted Plus-Minus)** (arXiv 2601.15000, January 2026): Extends RAPM to the lineup level, controlling for opposition while using informed priors from individual player ratings. Key insight: an NBA team uses **600+ lineups per season**, averaging only **25--30 possessions per lineup**. L-RAPM shows that lineup-level ratings have predictive power beyond the sum of individual player ratings -- this is the team/chemistry/coaching signal.

**EPM (Estimated Plus-Minus)** by Taylor Snarr (Dunks & Threes): Uses a two-step process: (1) statistical plus-minus model using optimized box score stats, then (2) RAPM calculation with the SPM as a Bayesian prior. EPM consistently outperforms other metrics, but by design it attributes everything to individual players and does not isolate team-level effects.

**Practical takeaway for Level 3**: The gap between sum-of-player-RAPM and actual team performance IS the Level 3 signal. L-RAPM demonstrates this gap is real and predictive. Our model should: (1) predict from player quality, (2) predict from player interactions, (3) model the residual as team-level effects.

**References**: [L-RAPM arXiv](https://arxiv.org/abs/2601.15000); [EPM Dunks & Threes](https://dunksandthrees.com/about/epm); [RAPM NBAstuffer](https://www.nbastuffer.com/analytics101/regularized-adjusted-plus-minus-rapm/)

---

### 3.2 EPAA -- Expected Points Above Average (arXiv 2405.10453, 2024)

**Authors**: Elmore, Williams, Schliep et al. Published in Annals of Applied Statistics (2025).

**Method**: Bayesian hierarchical framework that simultaneously clusters players and teams based on shooting propensities and abilities. EPAA is defined as the difference between a player's cluster distribution and the expected points for an average team.

**Key innovation**: The model explicitly decomposes performance into **player-cluster** and **team-cluster** components, providing uncertainty quantification through full posterior distributions.

**Practical takeaway for Level 3**: This is the closest existing work to our hierarchical decomposition concept. The team-cluster component in EPAA is conceptually similar to our Level 3. The Bayesian framework with posterior distributions provides principled uncertainty estimates.

**Reference**: [arXiv 2405.10453](https://arxiv.org/abs/2405.10453); [Project Euclid](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-19/issue-4/Expected-points-above-average--A-novel-NBA-player-metric/10.1214/25-AOAS2079.short)

---

### 3.3 Mixed-Effects Models and Variance Decomposition

**CMU Capstone Research**: A mixed-effects model with player- and team-level random intercepts reduces residual standard deviation to **10.99 points** and lowers AIC by ~1500 points compared to OLS. Adding random slopes for rest effects yields further AIC improvement of ~320. This confirms that both baseline ability and contextual effects differ meaningfully across players AND teams.

**Key insight**: Even with mixed-effects modeling, **residual SD of ~11 points** indicates a large share of game-to-game variability remains unmodeled. This is consistent with our Phase 3 best MAE of ~10.66.

**Hierarchical approach in elite basketball** (Nature Scientific Reports, 2024): Uses multilevel regression adjusted for hierarchical data (player within position within team over time). Demonstrates that accounting for the nested structure significantly improves model fit.

**Practical takeaway for Level 3**: The ICC (intraclass correlation coefficient) for team-level random effects quantifies how much variance is attributable to team identity after controlling for players. This is directly measurable from our data and should inform the capacity allocated to Level 3.

**References**: [CMU Capstone](https://www.stat.cmu.edu/capstoneresearch/460files_s25/team15.pdf); [Nature - Hierarchical Approach](https://www.nature.com/articles/s41598-024-51232-2)

---

### 3.4 FiveThirtyEight's Hybrid Approach

FiveThirtyEight's NBA model combines:
- **Elo** (team-level, ~35% weight): Head-to-head results, margin of victory, opponent quality.
- **RAPTOR talent** (player-level, ~65% weight): Player tracking + on/off ratings.

**Critical detail**: The weight given to Elo varies from 0--55% based on **roster continuity** between a team's current depth chart and recent lineups. High continuity = more weight on team-level Elo; low continuity = more weight on player-level RAPTOR.

**Practical takeaway for Level 3**: This is exactly the design principle we need. Level 3 should have a continuity-dependent weight. New rosters should rely more on player-level prediction; established rosters should incorporate more team-level identity. FiveThirtyEight found 35% team-level weight optimal on average.

**Reference**: [FiveThirtyEight Methodology](https://fivethirtyeight.com/methodology/how-our-nba-predictions-work/)

---

## 4. Team Efficiency Metrics (Four Factors)

### 4.1 Dean Oliver's Four Factors

The four factors of basketball success, as defined by Dean Oliver (2002, "Basketball on Paper"):

| Factor | Metric | Original Weight | Updated Weight (Poropudas 2023) |
|--------|--------|-----------------|--------------------------------|
| Shooting | eFG% = (FG + 0.5*3P) / FGA | 40% | 47% |
| Turnovers | TOV% = TOV / (FGA + 0.44*FTA + TOV) | 25% | 21% |
| Rebounding | ORB% = OREB / (OREB + OPP_DREB) | 20% | 26% |
| Free Throws | FT Rate = FTM / FGA | 15% | 7% |

### 4.2 Predictive Power

- **Adjusted R-squared = 0.96** for linear regression of Four Factors on wins (2024-25 season), up from 0.86 in 2004-05.
- The Four Factors are "as powerful as ever" and have actually **increased** in explanatory power over time.
- The relationship is **non-linear**: each factor's effect depends on the values of other factors (interaction effects matter).

### 4.3 Poropudas & Halme -- "Dean Oliver's Four Factors Revisited" (arXiv 2305.13032, 2023)

**Key contributions**:
- Derived an equation showing how Four Factors + FG%/FT% exactly determine offensive rating.
- Updated the relative importance weights for 2022-23: **eFG% 47%, ORB% 26%, TOV% 21%, FTR 7%**.
- Shooting matters even more than Oliver thought; free throw rate matters much less.
- The traditional 0.44 coefficient for possession-ending free throws (from Kubatko et al. 2007) is outdated due to changed shooting profiles.

### 4.4 Practical Takeaway for Level 3

The Four Factors are team-level metrics that capture system execution quality:
- **eFG%** reflects shooting scheme quality (shot selection, ball movement creating open looks).
- **TOV%** reflects ball security and offensive discipline (system complexity vs. error rate).
- **ORB%** reflects effort/scheme commitment to offensive boards (coaching decision: crash vs. get back).
- **FTR** reflects aggressiveness/style of play.

These should be computed as rolling averages and fed as Level 3 features. The differential (team Four Factors minus opponent Four Factors) is the most predictive form.

**References**: [arXiv 2305.13032](https://arxiv.org/abs/2305.13032); [Basketball-Reference Four Factors](https://www.basketball-reference.com/about/factors.html); [Statathlon](https://statathlon.com/four-factors-basketball-success/)

---

### 4.5 Pace, Offensive Rating, and Defensive Rating

**Foundational work**: Kubatko, Oliver, Pelton & Rosenbaum (JQAS, 2007) -- "A Starting Point for Analyzing Basketball Statistics." Established the possession concept as central to basketball analysis, with equal possessions for opponents being the key insight.

- **ORtg**: Points scored per 100 possessions.
- **DRtg**: Points allowed per 100 possessions.
- **Net Rating**: ORtg - DRtg (the single best predictor of team quality).
- **Pace**: Possessions per 48 minutes. Offense dictates tempo more than defense.

League offensive efficiency has changed little over 30 years despite pace fluctuations, suggesting efficiency and pace are largely independent dimensions.

**Practical takeaway for Level 3**: Net rating (rolling) is the gold standard team-level feature. Pace is a system characteristic worth encoding. The Four Factors decompose net rating into interpretable components that may help Level 3 learn faster.

**Reference**: [Kubatko et al. 2007](http://vishub.org/officedocs/18024.pdf)

---

## 5. Team Continuity and Roster Stability

### 5.1 Quantitative Findings

**Basketball-Reference continuity metric**: Percentage of a team's regular-season minutes filled by players from the previous season's roster.

- Every **10% increase in returning minutes** yields approximately **3.27 more wins** (4% increase in win percentage).
- There is a **small but positive** correlation between returning-minutes percentage and year-over-year win improvement.
- In 15 of 23 seasons since 2000, both Finals teams had **>60% roster continuity**.
- Teams making conference finals average **73% returning minutes**.
- Teams returning **80%+** of minutes won **42.9%** of championships studied.

### 5.2 The Nuance: Continuity Alone Is Modest

Research (NHSJS 2024) found that Roster Continuity as a standalone predictor has a **P-value of 0.502** -- not statistically significant by itself. However, it interacts with other factors (team quality, coaching stability) to become meaningful.

**Practical takeaway for Level 3**: Continuity is a **moderating feature**, not a standalone predictor. It should modulate the weight given to team-level historical features vs. player-level features (similar to FiveThirtyEight's approach). High continuity = trust team-level trends; low continuity = rely more on player-level data.

**References**: [Basketball-Reference Continuity](https://www.basketball-reference.com/friv/continuity.html); [NBA.com Continuity Rankings](https://www.nba.com/news/2025-continuity-rankings); [The Ringer](https://www.theringer.com/2022/08/16/nba/nba-trades-kevin-durant-roster-continuity)

---

### 5.3 Berman, Down & Hill -- "Tacit Knowledge as a Source of Competitive Advantage in the NBA" (Academy of Management Journal, 2002)

**Key finding**: Shared team experience (a form of tacit knowledge) has an **inverted U-shaped relationship** with performance. Initially, shared experience improves performance, but past some point **knowledge ossification** sets in and the relationship becomes negative.

**Mechanism**: As teams play together longer, they develop effective routines and communication patterns (positive). But eventually, they become predictable and resistant to adaptation (negative).

**Practical takeaway for Level 3**: Model team continuity with diminishing (and eventually negative) returns. A simple feature like `min(continuity_years, 4)` or a learned nonlinear transformation would capture this. This also interacts with coaching change: a new coach resets the ossification clock.

**Reference**: [Academy of Management Journal](https://journals.aom.org/doi/10.5465/3069282)

---

## 6. Organizational Effects

### 6.1 Analytics Investment as Competitive Advantage

**Wang, Sarker & Hosoi (2025) -- MIT**: "The Effect of Basketball Analytics Investment on National Basketball Association (NBA) Team Performance." Published in Journal of Sports Economics.

**Method**: Two-way fixed effects model using 12 years of season-level data (2009--2023). Controls for roster characteristics, injuries, schedule difficulty, team-specific effects, and time-specific effects.

**Key findings**:
- Clubs that invest more in analytics **significantly outperform** competitors after controlling for all observable factors.
- In 2009, only 10 data analysts worked across the entire NBA; by 2023, there were 132.
- Analytics is a **legitimate source of competitive advantage** independent of roster composition, coaching experience, team chemistry, injuries, and unobserved team differences.
- Many teams are still below the optimal analytics headcount.

**Practical takeaway for Level 3**: Analytics investment is a measurable organizational advantage. While we likely cannot get per-team analytics headcount data for features, the MIT study confirms that organizational quality is a real effect beyond coaching and roster.

**Reference**: [Journal of Sports Economics](https://journals.sagepub.com/doi/10.1177/15270025251328264); [MIT News](https://news.mit.edu/2025/basketball-analytics-investment-nba-wins-and-other-successes-0325)

---

### 6.2 Operational Efficiency and Franchise Value

**PLOS ONE (2024)**: Used Data Envelopment Analysis (DEA) to evaluate operational efficiency of NBA teams. Team operating efficiency -- the ability to utilize organizational resources effectively -- is a key determinant of team value. Organizations with resources possessing intrinsic value, rarity, difficulty of imitation, and non-substitutable attributes achieve sustainable performance advantages (resource-based view of the firm).

**Practical takeaway for Level 3**: Organizational quality is persistent and slow-changing. A team-level embedding that is updated seasonally (not game-by-game) could capture this. Alternatively, a rolling 3--5 year team performance trend captures organizational quality implicitly.

**Reference**: [PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0297797)

---

### 6.3 G League and Player Development Pipeline

- As of 2024-25, 31 G League teams exist, with 30 directly affiliated with NBA teams.
- Over 50% of NBA players have G League experience.
- Two-way contracts (up to 3 per team) create a development pipeline.
- Some organizations (e.g., Toronto, Miami) are widely recognized for superior development cultures.

**Practical takeaway for Level 3**: Development pipeline quality is hard to quantify from box scores alone. However, the rate at which a team successfully develops late-draft picks or undrafted players into contributors could serve as a proxy for organizational quality. This is more of a long-term feature than a game-level predictor.

---

## 7. Defensive Scheme Classification

### 7.1 Machine Learning Classification of Defensive Strategies (Scientific Reports, 2025)

**Method**: Hybrid model combining Random Forest (RF), Long Short-Term Memory (LSTM), and Convolutional Neural Networks (CNN) to classify defensive strategies from SportVU tracking data (32,000+ possessions).

**Key findings**:
- Achieved **91.4% classification accuracy** for switches and traps.
- LSTM captures temporal features of player motion; CNN recognizes spatial court patterns.
- Previous work on NCAA data achieved 82% accuracy using LSTM alone.

**Practical takeaway for Level 3**: Defensive scheme is classifiable from tracking data, but we don't have access to tracking data. Proxy features from box scores: opponent 3PA rate, opponent FGA at rim (from shooting splits if available), steal rate, block rate, opponent turnover rate.

**References**: [Nature Scientific Reports](https://www.nature.com/articles/s41598-025-98877-1); [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12015258/)

---

### 7.2 Defensive Coverage Types and Their Effects

The main pick-and-roll coverage types in the NBA:

| Coverage | Description | Strength | Weakness |
|----------|------------|----------|----------|
| **Drop** | Big sits 6-10 ft behind screen, protects rim | Limits rim attacks | Vulnerable to pull-up 3s from elite guards |
| **Switch** | Defenders trade assignments at screen | No gaps created | Creates size/speed mismatches |
| **Blitz/Trap** | Both defenders swarm ball-handler | Forces turnovers | Leaves roller/corner 3 open |
| **ICE/Push** | Force ball-handler away from screen | Predictable recovery | Requires elite wing defenders |
| **Hedge & Recover** | Big shows high, then recovers | Buys time | Requires mobile big + disciplined rotations |

**Second Spectrum "Aggression+" Index**: Measures how aggressive a team's defense is relative to league average across five offensive actions (pick-and-rolls, isolations, post-ups, off-ball screens, dribble handoffs).

**Key finding from Second Spectrum data**: Damian Lillard vs. drop coverage in the bubble achieved **1.373 PPP** -- demonstrating that coverage choice has enormous matchup-dependent effects.

**Practical takeaway for Level 3**: Without tracking data, we can approximate defensive scheme from: opponent 3PA rate (switchy teams allow more 3s), opponent paint points (drop teams allow fewer), steal rate (aggressive schemes generate more steals), and pace of play (aggressive schemes create more transition opportunities for opponents).

---

### 7.3 Franks, Miller, Bornn & Goldsberry -- "Characterizing the Spatial Structure of Defensive Skill in Professional Basketball" (Annals of Applied Statistics, 2015)

**Method**: Combined spatial/spatio-temporal processes, matrix factorization, and hierarchical regression models with player tracking data.

**Key contributions**:
- Detects, characterizes, and quantifies multiple aspects of defensive play.
- Supports some common understandings of defensive effectiveness while challenging others.
- Opens new insights into defensive elements previously unquantifiable.

**Practical takeaway for Level 3**: This is the gold standard for spatial defensive analysis. While we cannot replicate tracking-based features, the paper's framework (decomposing defense into spatial components) informs how we think about defensive system effects. Defensive quality is **not one-dimensional** -- it varies by court region and play type.

**Reference**: [arXiv 1405.0231](https://arxiv.org/abs/1405.0231)

---

## 8. Team Chemistry (Bridging Level 2 and Level 3)

While team chemistry is primarily Level 2 (player interactions), some chemistry effects are team-level/systemic and belong in Level 3.

### 8.1 Horrace, Jung & Sanders -- "Network Competition and Team Chemistry in the NBA" (JBES, 2022)

**Method**: Heterogeneous social interaction model where agents interact with peers within their own network and across opposing networks. Estimated by quasi-maximum likelihood on 2015-16 NBA regular season data.

**Key findings**:
- Significant **positive within-team peer-effects** ("team chemistries") that enhance individual player performance.
- Both **negative and positive opposing-team competitor-effects** ("team rivalries") that can enhance or diminish opposing player performance.
- Chemistry is not just about player pairs -- it's a network-level property.

**Practical takeaway for Level 3**: The within-team peer-effects that are systemic (not specific to player pairs) belong in Level 3. This is the "culture" effect -- some teams systematically elevate all players, others suppress them.

**Reference**: [JBES](https://www.tandfonline.com/doi/full/10.1080/07350015.2020.1773273)

---

### 8.2 Maymin, Maymin & Shen -- "NBA Chemistry: Positive and Negative Synergies in Basketball" (IJCSS, 2013)

**Method**: Skills Plus Minus (SPM) framework evaluating players on offense and defense across scoring, rebounding, and ball-handling. Simulated games using skill ratings of 10 on-court players and calculated synergies.

**Key findings**:
- Synergy differences between teams explain as much as **6 wins**.
- Teams are no more likely to exhibit positive chemistry than negative chemistry.
- **Rare events** (steals, blocks) produce positive synergies; **common events** (defensive rebounds) produce negative synergies.
- Offensive ball-handling + offensive scoring have positive synergies.
- Offensive rebounding has negative synergy with offensive scoring (bad scorers miss more, making offensive rebounding more valuable).
- Found 200+ mutually beneficial trades between NBA teams based on synergy analysis.

**Practical takeaway for Level 3**: The 6-win synergy signal is substantial. However, synergy is partially Level 2 (specific player combinations) and partially Level 3 (coaching decisions about which combinations to use). Level 3 should capture the coaching wisdom of lineup construction.

**Reference**: [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1935972); [PDF](https://philipmaymin.com/papers/Maymin%20Maymin%20and%20Shen%20-%20NBA%20Chemistry%20-%20IJCSS.pdf)

---

## 9. Synthesis: What Level 3 Should Capture

### 9.1 Variance Budget Estimate

Based on the literature reviewed:

| Source | Variance Explained | Notes |
|--------|-------------------|-------|
| Player talent (Level 1) | ~50-60% | RAPM, box score models |
| Player interactions (Level 2) | ~5-10% | Chemistry, lineup synergy (~6 wins) |
| **Coaching + system (Level 3)** | **~15-25%** | Berry & Fowler: ~30% total coaching; minus overlap with talent |
| Context (Level 4) | ~5-10% | Home court, rest, travel, schedule |
| Irreducible noise | ~10-15% | 3PT variance, injuries, motivation |

### 9.2 Concrete Features for Level 3

Based on the literature, Level 3 should encode:

**Coaching identity and quality** (slow-changing):
- Coach embedding (learned) or coach career win% above expected
- Coach tenure with current team (nonlinear: inverted U)
- Coach change flag + games since change

**System execution quality** (medium-changing, rolling):
- Four Factors differentials (eFG%, TOV%, ORB%, FTR) -- rolling 10-20 game windows
- Net rating (ORtg - DRtg) -- rolling
- Pace and pace variability
- Assist rate, 3PA rate (offensive system proxies)

**Defensive scheme proxies** (medium-changing):
- Opponent 3PA rate (low = drop coverage; high = switching)
- Opponent paint points / FGA at rim
- Steal rate, block rate
- Opponent TOV rate

**Organizational stability** (slow-changing):
- Roster continuity percentage
- Front office tenure / stability
- Multi-year team performance trend (captures organizational quality)

**Continuity as a moderator**:
- Weight team-level features more when continuity is high
- Weight player-level features more when continuity is low
- FiveThirtyEight's optimal split: ~35% team-level, ~65% player-level on average

### 9.3 Architectural Recommendations

1. **Dual timescale**: Level 3 features should operate on two timescales: (a) slow features updated seasonally (coach identity, organizational quality, roster continuity), and (b) medium features updated with rolling windows (Four Factors, net rating, defensive scheme indicators).

2. **Residual formulation**: Level 3 should explicitly model `actual_performance - Level1_prediction - Level2_prediction`. This is the residual approach supported by the hierarchical modeling literature.

3. **Continuity-gated attention**: Use roster continuity to gate how much the model relies on team-level history vs. player-level composition, following FiveThirtyEight's insight.

4. **Coaching change handling**: When a coaching change occurs, partially reset team-level features (especially system-related ones) while retaining organizational features. Model an adaptation curve with rapid initial learning that slows over 20-30 games.

5. **Inverted-U for stability**: Following Berman et al. (2002), encode diminishing returns for very high continuity/tenure to capture knowledge ossification.

---

## Key Papers Reference List

| Paper | Year | Key Contribution |
|-------|------|-----------------|
| Berry & Fowler, "How Much Do Coaches Matter?" | 2019 | 14-win coaching effect, RIFLE method |
| Al-Amine, "Quantifying NBA Coaches using Fixed Effects" | 2020 | Coach fixed effects controlling for VORP |
| Cannon et al., "Analyzing Effects of NBA Head Coaches" (JQAS) | 2025 | In-game scheme isolation via delta-tVORP |
| Oliver, "Basketball on Paper" | 2004 | Four Factors framework |
| Poropudas & Halme, "Four Factors Revisited" (arXiv) | 2023 | Updated weights, non-linear relationships |
| Kubatko, Oliver, Pelton & Rosenbaum (JQAS) | 2007 | Possession estimation, advanced metrics foundation |
| Berman, Down & Hill (AMJ) | 2002 | Tacit knowledge inverted-U in NBA |
| Wang, Sarker & Hosoi (J Sports Econ) | 2025 | Analytics investment as competitive advantage |
| Horrace, Jung & Sanders (JBES) | 2022 | Network competition and team chemistry |
| Maymin, Maymin & Shen (IJCSS) | 2013 | Synergy quantification (~6 wins) |
| Elmore et al., EPAA (AOAS) | 2024/25 | Bayesian hierarchical player-team decomposition |
| L-RAPM (arXiv) | 2026 | Lineup ratings with informed priors |
| Franks et al. (AOAS) | 2015 | Spatial defensive skill characterization |
| Machine Learning Defensive Strategies (Sci. Rep.) | 2025 | 91.4% accuracy on switch/trap classification |
| FiveThirtyEight NBA Methodology | 2020 | 35/65 Elo-RAPTOR weighting with continuity |
