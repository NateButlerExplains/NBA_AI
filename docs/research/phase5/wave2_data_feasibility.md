# Wave 2 Data Feasibility Report

Generated: 2026-03-16

This document catalogs every data source needed for the hierarchical NBA prediction system,
with verified endpoint details, coverage, rate limits, and collection estimates.

---

## Current Database Inventory

| Table | Rows | Coverage | Key Fields |
|-------|------|----------|------------|
| Games | 37K | 1999-2026 | game_id, date, home/away team, season |
| PlayerBox | 670K | 2001-2026 | 16 box stats + position per player-game |
| TeamBox | 63K | 2001-2026 | 20 aggregated stats per team-game |
| Players | 5,127 | All-time | person_id, name, from_year, to_year, roster_status, team (NO bio data) |
| PBP_Logs | 16.3M | 2000-2026 (33,553 games) | JSON per play — substitutions, shots, fouls, etc. |
| Betting | ~16K | 2007-2026 | Opening/current/closing spreads, totals, moneylines |
| InjuryReports | 69K | 2021-2025 | Player, team, status, body part, report timestamp |

---

## 1. Player Attributes (Level 1 — Cold Start)

### 1a. `LeagueDashPlayerBioStats` — BEST OPTION FOR BULK BIO DATA

**nba_api class**: `nba_api.stats.endpoints.leaguedashplayerbiostats.LeagueDashPlayerBioStats`

**Fields returned**:
- `PLAYER_ID`, `PLAYER_NAME`, `TEAM_ID`, `TEAM_ABBREVIATION`
- `AGE`, `PLAYER_HEIGHT` (string "6-6"), `PLAYER_HEIGHT_INCHES` (numeric 78)
- `PLAYER_WEIGHT` (numeric 179)
- `COLLEGE`, `COUNTRY`
- `DRAFT_YEAR`, `DRAFT_ROUND`, `DRAFT_NUMBER` (string "Undrafted" if undrafted)
- `GP`, `PTS`, `REB`, `AST`, `NET_RATING`, `OREB_PCT`, `DREB_PCT`, `USG_PCT`, `TS_PCT`, `AST_PCT`

**Coverage**: Verified for 2001-02 through 2024-25. Returns ~440-570 players per season.

**Rate limit**: ~0.6s between calls recommended. 25 seasons = 25 calls = 15 seconds total.

**Verdict**: **Use this as the primary source.** One call per season returns all active players with
height, weight, draft info, country, college, plus advanced rate stats. Covers our entire 2001-2026
range. ~12,000 unique player-seasons across 25 seasons.

### 1b. `CommonPlayerInfo` — PER-PLAYER BIO (SUPPLEMENTAL)

**nba_api class**: `nba_api.stats.endpoints.commonplayerinfo.CommonPlayerInfo`

**Fields returned**:
- `PERSON_ID`, `FIRST_NAME`, `LAST_NAME`, `BIRTHDATE` (not in BioStats!)
- `SCHOOL`, `COUNTRY`, `LAST_AFFILIATION`
- `HEIGHT` (string "6-6"), `WEIGHT` (string "215")
- `SEASON_EXP`, `JERSEY`, `POSITION`, `ROSTERSTATUS`
- `DRAFT_YEAR`, `DRAFT_ROUND`, `DRAFT_NUMBER`
- Also returns `PlayerHeadlineStats`: current season PTS, AST, REB, PIE

**Coverage**: All-time. One call per player.

**Rate limit**: ~0.6s between calls. 2,142 players (from_year >= 2001) = ~21 minutes.

**Unique value**: **BIRTHDATE** — the only source for exact birth date (age calculation).
Also provides `SEASON_EXP` and `POSITION` which may differ from roster position.

**Verdict**: Fetch for all 2,142 players active 2001+. Takes ~21 minutes. Critical for birth date.

### 1c. `CommonTeamRoster` — PLAYER BIO FROM ROSTER CONTEXT

**nba_api class**: `nba_api.stats.endpoints.commonteamroster.CommonTeamRoster`

**Fields returned** (CommonTeamRoster dataset):
- `TeamID`, `SEASON`, `PLAYER`, `PLAYER_ID`
- `POSITION`, `HEIGHT`, `WEIGHT`, `BIRTH_DATE`, `AGE`, `EXP`
- `SCHOOL`, `NUM` (jersey), `PLAYER_SLUG`, `HOW_ACQUIRED`, `NICKNAME`

**Fields returned** (Coaches dataset — see Section 3):
- `COACH_ID`, `COACH_NAME`, `IS_ASSISTANT`, `COACH_TYPE`, `SORT_SEQUENCE`

**Coverage**: Verified 2001-02 through 2024-25. Returns ~17 players + 4-8 coaches per team-season.

**Rate limit**: 30 teams x 25 seasons = 750 calls @ 0.6s = ~7.5 minutes.

**Verdict**: Good supplemental source. Provides `HOW_ACQUIRED` (unique) and roster context.
But `LeagueDashPlayerBioStats` is more efficient for bulk collection.

### 1d. `DraftHistory` — COMPLETE DRAFT DATABASE

**nba_api class**: `nba_api.stats.endpoints.drafthistory.DraftHistory`

**Fields returned**:
- `PERSON_ID`, `PLAYER_NAME`, `SEASON` (draft year)
- `ROUND_NUMBER`, `ROUND_PICK`, `OVERALL_PICK`
- `DRAFT_TYPE`, `TEAM_ID`, `TEAM_CITY`, `TEAM_NAME`, `TEAM_ABBREVIATION`
- `ORGANIZATION`, `ORGANIZATION_TYPE` (College/University, High School, International, etc.)

**Coverage**: 1947-2025, 8,374 total records. Single API call.

**Verdict**: **Trivial to collect.** One call, complete draft history. Use for draft pick features.

### 1e. `DraftCombineStats` — PHYSICAL MEASUREMENTS + ATHLETIC TESTING

**nba_api class**: `nba_api.stats.endpoints.draftcombinestats.DraftCombineStats`

**Fields returned** (47 columns):
- Anthro: `HEIGHT_WO_SHOES`, `HEIGHT_W_SHOES`, `WEIGHT`, `WINGSPAN`, `STANDING_REACH`,
  `BODY_FAT_PCT`, `HAND_LENGTH`, `HAND_WIDTH`
- Athletic: `STANDING_VERTICAL_LEAP`, `MAX_VERTICAL_LEAP`, `LANE_AGILITY_TIME`,
  `MODIFIED_LANE_AGILITY_TIME`, `THREE_QUARTER_SPRINT`, `BENCH_PRESS`
- Shooting: 19 spot-shooting columns (15ft, college 3PT, NBA 3PT from 5 positions + off-dribble)

**Coverage**: 2000-01 through 2024-25. ~45-83 players per season (only draft invitees).

**Rate limit**: 25 calls @ 0.6s = 15 seconds.

**Caveat**: Only ~60% of NBA players attended the combine. Undrafted/international players
often missing. Still valuable for wingspan/standing reach (key defensive features).

**Verdict**: Easy to collect. ~1,500 combine records. Useful for physical archetype features
but will need imputation for ~40% of players.

### 1f. `DraftCombinePlayerAnthro` — ANTHRO SUBSET

Same data as DraftCombineStats anthro columns. Redundant if collecting DraftCombineStats.

---

### Player Attributes: Collection Plan

| Source | Calls | Time | Unique Fields | Priority |
|--------|-------|------|---------------|----------|
| LeagueDashPlayerBioStats | 25 | 15 sec | Height, weight, draft, advanced rates | **P0** |
| CommonPlayerInfo | 2,142 | 21 min | **BIRTHDATE**, season_exp, position | **P0** |
| DraftHistory | 1 | instant | Draft pick, org type, complete history | **P0** |
| DraftCombineStats | 25 | 15 sec | Wingspan, vertical, agility, shooting | **P1** |
| CommonTeamRoster | 750 | 7.5 min | HOW_ACQUIRED, roster context, coaches | **P1** |

**Total estimated collection time: ~30 minutes for all sources.**

---

## 2. Lineup / Stint Data (Level 2)

This is the **highest-value, highest-risk** data gap. Three acquisition strategies evaluated.

### 2a. Strategy A: `GameRotation` Endpoint (RECOMMENDED)

**nba_api class**: `nba_api.stats.endpoints.gamerotation.GameRotation`

**Fields returned** (per player per stint):
- `GAME_ID`, `TEAM_ID`, `TEAM_CITY`, `TEAM_NAME`
- `PERSON_ID`, `PLAYER_FIRST`, `PLAYER_LAST`
- `IN_TIME_REAL`, `OUT_TIME_REAL` (in tenths of seconds from game start)
- `PLAYER_PTS`, `PT_DIFF` (point differential during stint), `USG_PCT`

**Coverage**: Verified back to **2000-01** (tested game_id `0020000001` — returned 31 rows).
Coverage aligns with our PBP_Logs range.

**Row count**: ~30-35 stints per team per game = ~60-70 per game. For 33,822 games = **~2.1M stint records**.

**Rate limit**: 33,822 calls @ 0.6s = **5.6 hours**.

**What this gives us**:
- Exact on/off times for every player in every game
- Can reconstruct which 5 players were on court at any moment
- Can compute stint-level stats by joining with PBP_Logs
- Point differential per stint (built-in plus/minus)

**Verdict**: **Best option.** Clean, structured, excellent historical coverage.
The IN_TIME_REAL/OUT_TIME_REAL format makes lineup reconstruction trivial —
just find which 5 players on each team have overlapping intervals at any timestamp.

### 2b. Strategy B: Parse PBP_Logs Substitution Events

**Our existing PBP_Logs** (16.3M rows, 33,553 games, 2000-2026) contain substitution events:

```json
{
  "actionType": "substitution",
  "subType": "in",  // or "out"
  "personId": 1630552,
  "playerNameI": "J. Johnson",
  "teamTricode": "ATL",
  "description": "SUB in: J. Johnson",
  "period": 1,
  "clock": "PT06M36.00S"
}
```

**Substitution events per season**: ~50K-70K per season (regular season). Paired in/out events.

**Challenge**: Identifying starters requires heuristic:
- Period 1 starters inferred from first actions (jumpball, first shots/rebounds)
- Quarter 2-4 starters = whoever was on court at end of previous period
- Edge cases: technical foul subs, ejections, injuries without sub events

**Verdict**: **Feasible but messy.** We already have the data — no API calls needed.
However, GameRotation is cleaner and provides IN_TIME/OUT_TIME directly.
Use PBP parsing as **validation** for GameRotation data, not primary source.

### 2c. Strategy C: `LeagueDashLineups` — AGGREGATED SEASON STATS

**nba_api class**: `nba_api.stats.endpoints.leaguedashlineups.LeagueDashLineups`

**Parameters**: `group_quantity` = 2, 3, 4, or 5; `measure_type` = Base or Advanced

**Fields returned** (Base): GP, W, L, W_PCT, MIN, full box stats, PLUS_MINUS + ranks
**Fields returned** (Advanced): OFF_RATING, DEF_RATING, NET_RATING, PACE, TS_PCT, PIE, etc.

**Coverage**: Verified back to **2007-08** season. Returns up to **2000 rows** (hard cap).

**Rate limit**: 25 seasons x 4 group_quantities x 2 measure types = 200 calls = 2 minutes.

**Critical limitation**: **2000-row cap.** League-wide 5-man lineups easily exceed 2,000 unique combos.
For 2-man combos it's far worse. Results are sorted by minutes, so low-minute lineups are truncated.

**Alternative**: `TeamDashLineups` (per-team) avoids the cap but requires 30 teams x 25 seasons
x 4 groups = 3,000 calls = ~30 minutes.

**What this gives us**: Season-level lineup chemistry metrics (net rating, offensive rating per combo).
NOT per-game data — cannot predict individual game outcomes from this alone.

**Verdict**: **Supplemental only.** Useful for pre-computed lineup chemistry features, but not a
replacement for per-game stint data. Collect alongside GameRotation.

### 2d. Strategy D: `LeagueLineupViz` — ADVANCED LINEUP METRICS

**nba_api class**: `nba_api.stats.endpoints.leaguelineupviz.LeagueLineupViz`

**Fields returned**: OFF_RATING, DEF_RATING, NET_RATING, PACE, TS_PCT, FTA_RATE,
TM_AST_PCT, shot distribution (2PT/3PT/paint/FT %), OPP_FG3_PCT, OPP_EFG_PCT

**Requires**: `minutes_min` parameter (minimum minutes threshold).

**Verdict**: Useful for enriching lineup features with shooting tendencies and defensive metrics.
Collect as supplement to LeagueDashLineups.

### 2e. Third-Party: `pbpstats` and `nba-on-court`

**pbpstats** (PyPI: `pbpstats`):
- Scrapes and parses NBA/WNBA/G-League PBP data
- Splits PBP into possessions with type labels (transition, halfcourt, etc.)
- Lineup IDs as dash-separated player IDs
- Can use stats.nba.com or data.nba.com as source
- Coverage: from 2000-01 onward

**nba-on-court** (PyPI: `nba-on-court`):
- Adds 10 columns (5 per team) identifying on-court players for every PBP event
- Reconstructs lineups from substitution events within each quarter
- Coverage: 1996-97 onward
- Requires PBP data as input (we have 33K games already)

**Verdict**: These are processing tools, not data sources. Since we already have PBP_Logs
and will collect GameRotation, we can build lineup reconstruction ourselves.
`nba-on-court` logic is a useful reference for edge case handling.

### Lineup Data: Collection Plan

| Source | Calls | Time | Gives Us | Priority |
|--------|-------|------|----------|----------|
| GameRotation | 33,822 | 5.6 hrs | Per-game stint data (in/out times, +/-) | **P0** |
| PBP_Logs parsing | 0 (already have data) | Processing time | Validation + possession-level context | **P0** |
| LeagueDashLineups (Base) | 200 | 2 min | Season-level lineup stats (capped at 2K rows) | **P1** |
| TeamDashLineups (Advanced) | 3,000 | 30 min | Per-team lineup off/def ratings | **P1** |
| LeagueLineupViz | 200 | 2 min | Lineup shooting/defensive tendencies | **P2** |

**Total: ~6.5 hours for core stint data, plus 35 minutes for supplemental lineup metrics.**

---

## 3. Coaching Data (Level 3)

### 3a. `CommonTeamRoster` — COACHES DATASET (PRIMARY)

**Already documented in Section 1c above.** The endpoint returns both a roster dataset AND a coaches dataset.

**Coaches fields**:
- `TEAM_ID`, `SEASON`, `COACH_ID`, `FIRST_NAME`, `LAST_NAME`, `COACH_NAME`
- `IS_ASSISTANT` (1=Head Coach, 2=Assistant, 3=Trainer)
- `COACH_TYPE` ("Head Coach", "Assistant Coach", "Trainer")
- `SORT_SEQUENCE`, `SUB_SORT_SEQUENCE`

**Coverage**: Verified 2001-02 through 2024-25. Returns 4-8 staff per team-season.
Head coach identification uses `IS_ASSISTANT=1`.

**Note**: 2015-16 Lakers returned `IS_ASSISTANT=1` as N/A for head coach — some seasons may use
different conventions. Need to handle edge cases (Byron Scott was HC that year).

**Rate limit**: 30 teams x 25 seasons = 750 calls (shared with roster collection) = ~7.5 minutes.

**Estimated rows**: 30 teams x 25 seasons x ~6 coaches = **~4,500 coach records**.

**Verdict**: **Primary and sufficient source.** Gives us head coach per team per season,
which is the main coaching feature we need. No per-game coaching data needed since
NBA coaching changes are very rare mid-season.

### 3b. Basketball-Reference — SUPPLEMENTAL COACHING STATS

Basketball-Reference has detailed coaching records:
- Win/loss records, playoff appearances, career stats
- Coaching tree (assistant history)
- Historical data back to 1946

**Access**: Web scraping required. Rate-limited. Terms of service may restrict.

**Verdict**: **Not needed for initial implementation.** CommonTeamRoster gives us coach identity;
we can derive coaching features (tenure, win rate) from our own Games table.

---

## 4. Referee Data (Level 4)

### 4a. `BoxScoreSummaryV2` / `BoxScoreSummaryV3` — OFFICIALS DATASET (PRIMARY)

**nba_api class**: `BoxScoreSummaryV2` or `BoxScoreSummaryV3`

**Officials fields**:
- V2: `OFFICIAL_ID`, `FIRST_NAME`, `LAST_NAME`, `JERSEY_NUM`
- V3: `gameId`, `personId`, `name`, `nameI`, `firstName`, `familyName`, `jerseyNum`

**Coverage verified**:
| Season | Officials per game | Status |
|--------|-------------------|--------|
| 2000-01 | 0 | No data |
| 2001-02 | 0 | No data |
| 2002-03 | 0 | No data |
| 2003-04 | 3 | Available |
| 2005-06 | 3 | Available |
| 2007-08 | 3 | Available |
| 2013-14 | 3 | Available |
| 2024-25 | 3 | Available |

**Coverage**: **2003-04 through 2025-26** (~22 seasons, ~28K games).

**Rate limit**: If collecting standalone, 33,822 calls @ 0.6s = 5.6 hours.
**However**, if we collect BoxScoreSummaryV2 alongside GameRotation, we get
officials data for free (already making per-game API calls).

**Note**: V2 has a deprecation warning for games after 4/10/2025 — use V3 for recent games.

**Also provides** (bonus data from same call):
- `InactivePlayers`: Scratched players per game (useful for injury/availability features)
- `GameInfo`: `ATTENDANCE`, `GAME_TIME` (actual game duration)
- `OtherStats`: `PTS_PAINT`, `PTS_2ND_CHANCE`, `PTS_FB`, `LARGEST_LEAD`, `LEAD_CHANGES`,
  `TIMES_TIED`, `TEAM_TURNOVERS`, `PTS_OFF_TO`
- `LastMeeting`: Previous matchup result
- `SeasonSeries`: Head-to-head record

**Estimated rows**: 3 officials x ~28K games = **~84K referee assignment records**.

**Verdict**: **Trivial to collect alongside other per-game data.** Rich bonus data.

### 4b. PBP_Logs — NO REFEREE DATA

Our PBP_Logs JSON does **not** contain referee information. Confirmed by searching for
"official" keyword — zero matches. The NBA PBP v3 endpoint does not include officials.

---

## 5. Advanced Stats / Tracking / Hustle

### 5a. `BoxScoreAdvancedV3` — PER-GAME ADVANCED BOX SCORE

**nba_api class**: `nba_api.stats.endpoints.boxscoreadvancedv3.BoxScoreAdvancedV3`

**Fields returned** (per player per game):
- Ratings: `estimatedOffensiveRating`, `offensiveRating`, `estimatedDefensiveRating`,
  `defensiveRating`, `estimatedNetRating`, `netRating`
- Percentages: `assistPercentage`, `assistToTurnover`, `assistRatio`,
  `offensiveReboundPercentage`, `defensiveReboundPercentage`, `reboundPercentage`,
  `turnoverRatio`, `effectiveFieldGoalPercentage`, `trueShootingPercentage`
- Usage: `usagePercentage`, `estimatedUsagePercentage`
- Pace: `estimatedPace`, `pace`, `pacePer40`, `possessions`, `PIE`

**Also returns** `TeamStats` with same fields at team level.

**Coverage**: Verified **2000-01 through 2024-25** — full historical coverage! All 22 advanced
metrics available even for 2000-01 games.

**Rate limit**: 33,822 calls @ 0.6s = 5.6 hours.

**Estimated rows**: ~25 players x 33,822 games = **~845K player-game records** (similar to PlayerBox).

**Verdict**: **High value.** Offensive/defensive ratings and usage are core features for
hierarchical prediction. Same call volume as GameRotation — could be batched together.

### 5b. `BoxScorePlayerTrackV3` — PLAYER TRACKING (SportVU/Second Spectrum)

**nba_api class**: `nba_api.stats.endpoints.boxscoreplayertrackv3.BoxScorePlayerTrackV3`

**Fields returned** (per player per game):
- Movement: `speed`, `distance`
- Possession: `touches`, `passes`
- Rebounding: `reboundChancesOffensive`, `reboundChancesDefensive`, `reboundChancesTotal`
- Assists: `secondaryAssists`, `freeThrowAssists`
- Contested shots: `contestedFieldGoalsMade/Attempted/Percentage`,
  `uncontestedFieldGoalsMade/Attempted/Percentage`
- Rim defense: `defendedAtRimFieldGoalsMade/Attempted/Percentage`

**Coverage verified**:
| Season | Data Quality |
|--------|-------------|
| 2007-08 | Returns rows but all zeros (no tracking system) |
| 2012-13 | Returns rows but all zeros (SportVU partial) |
| **2013-14** | **Non-zero data (speed, touches, distance)** — first full SportVU season |
| 2014-15+ | Full data |

**Actual coverage**: **2013-14 through 2025-26** (~12 seasons, ~15K games).

**Rate limit**: ~15,000 calls @ 0.6s = 2.5 hours.

**Estimated rows**: ~25 players x 15K games = **~375K player-game tracking records**.

**Verdict**: **Valuable but limited history.** Touches and speed are predictive features not
available from traditional box scores. 12 seasons is enough for a useful training signal.
Pre-2013 games will need these features masked/imputed.

### 5c. `BoxScoreHustleV2` — HUSTLE STATS

**nba_api class**: `nba_api.stats.endpoints.boxscorehustlev2.BoxScoreHustleV2`

**Fields returned** (per player per game):
- `contestedShots`, `contestedShots2pt`, `contestedShots3pt`
- `deflections`, `chargesDrawn`
- `screenAssists`, `screenAssistPoints`
- `looseBallsRecoveredOffensive/Defensive/Total`
- `offensiveBoxOuts`, `defensiveBoxOuts`, `boxOutPlayerTeamRebounds`, `boxOutPlayerRebounds`, `boxOuts`

**Coverage verified**:
| Season | Player-Level Data |
|--------|------------------|
| 2013-14 | Team-level only (2 rows, all zeros) |
| 2015-16 | Team-level only (2 rows, all zeros) |
| **2016-17** | **Full player-level data (25 rows with real values)** |
| 2024-25 | Full player-level data |

**Actual coverage**: **2016-17 through 2025-26** (~9 seasons, ~11K games).

**Rate limit**: ~11,000 calls @ 0.6s = 1.8 hours.

**Estimated rows**: ~25 players x 11K games = **~275K player-game hustle records**.

**Verdict**: **Moderately valuable.** Deflections and contested shots are useful defensive features.
But only 9 seasons of history is limiting. Lower priority than tracking data.

### 5d. `LeagueDashPlayerStats` (MeasureType=Advanced) — SEASON-LEVEL ADVANCED

**nba_api class**: `nba_api.stats.endpoints.leaguedashplayerstats.LeagueDashPlayerStats`

With `measure_type_detailed_defense='Advanced'`, returns season-aggregated advanced stats.

**Coverage**: Full historical (similar to LeagueDashPlayerBioStats).

**Rate limit**: 25 seasons = 25 calls = 15 seconds.

**Verdict**: Useful for season-level features, but `BoxScoreAdvancedV3` gives per-game granularity.
Use as quick supplement if per-game collection is too slow.

### 5e. `BoxScoreMatchupsV3` — DEFENSIVE MATCHUP DATA

**nba_api class**: `nba_api.stats.endpoints.boxscorematchupsv3.BoxScoreMatchupsV3`

**Fields returned**: Per offensive-defensive player pair per game:
- `matchupMinutes`, `partialPossessions`, `switchesOn`
- `playerPoints`, `teamPoints`
- `matchupFieldGoalsMade/Attempted/Percentage`
- `matchupThreePointersMade/Attempted/Percentage`
- `helpBlocks`, `helpFieldGoalsMade/Attempted`
- `shootingFouls`

**Coverage**: Only verified for **2024-25** season. 2013-14 and 2016-17 returned errors.
Likely available from ~2023-24 or 2024-25 onward only.

**Rate limit**: Very limited game count (~2,500 games).

**Verdict**: **Future potential.** Too limited historically. Revisit when more seasons accumulate.

### 5f. `LeagueHustleStatsPlayer` — SEASON-LEVEL HUSTLE

**nba_api class**: `nba_api.stats.endpoints.leaguehustlestatsplayer.LeagueHustleStatsPlayer`

**Coverage verified**:
- 2014-15: 0 players (no data)
- **2015-16: 147 players** (partial — playoff only?)
- **2016-17: 485 players** (full season)
- 2020-21: 538 players
- 2024-25: 567 players

**Rate limit**: 10 seasons = 10 calls = 6 seconds.

**Verdict**: Quick supplement. One call per season for league-wide hustle stats.

---

## 6. Bonus: Data Already Available from BoxScoreSummaryV2

If we fetch BoxScoreSummaryV2/V3 for referee data, we also get for free:

| Dataset | Fields | Value |
|---------|--------|-------|
| Officials | Referee IDs + names | Level 4 feature |
| InactivePlayers | Scratched player IDs | Availability feature |
| GameInfo | Attendance, game duration | Context features |
| OtherStats | PTS_PAINT, PTS_FB, LARGEST_LEAD, LEAD_CHANGES, TIMES_TIED, PTS_OFF_TO | Rich game-level features |
| LastMeeting | Previous H2H result | Already computed, validation |
| SeasonSeries | H2H record entering game | Already computed, validation |
| LineScore | Quarter-by-quarter scores | Game flow features |

---

## 7. Summary: Collection Priority Matrix

### Tier 0 — Must Have (Immediate)

| Data Source | API Calls | Time | New Rows | Value |
|-------------|-----------|------|----------|-------|
| LeagueDashPlayerBioStats | 25 | 15s | ~12K | Player height/weight/draft |
| CommonPlayerInfo | 2,142 | 21 min | ~2,142 | Birth dates (age) |
| DraftHistory | 1 | instant | 8,374 | Complete draft database |
| GameRotation | 33,822 | 5.6 hrs | ~2.1M | Per-game stint data |
| BoxScoreSummaryV2/V3 | 33,822 | 5.6 hrs | ~100K (refs) + bonus | Officials + inactive + game info |

**Combined**: Can share per-game loop for GameRotation + BoxScoreSummary.
**Total: ~5.6 hours** (not additive — same game loop).

### Tier 1 — High Value

| Data Source | API Calls | Time | New Rows | Value |
|-------------|-----------|------|----------|-------|
| BoxScoreAdvancedV3 | 33,822 | 5.6 hrs | ~845K | Off/def ratings, usage, pace |
| DraftCombineStats | 25 | 15s | ~1,500 | Physical measurements |
| CommonTeamRoster | 750 | 7.5 min | ~4,500 coaches + rosters | Coaching data + roster context |
| BoxScorePlayerTrackV3 | ~15,000 | 2.5 hrs | ~375K | Speed, touches, distance (2013+) |
| LeagueDashLineups | 200 | 2 min | ~16K | Season-level lineup chemistry |

### Tier 2 — Nice to Have

| Data Source | API Calls | Time | New Rows | Value |
|-------------|-----------|------|----------|-------|
| BoxScoreHustleV2 | ~11,000 | 1.8 hrs | ~275K | Hustle stats (2016+) |
| LeagueHustleStatsPlayer | 10 | 6s | ~5K | Season-level hustle |
| TeamDashLineups | 3,000 | 30 min | ~90K | Per-team lineup advanced stats |
| LeagueLineupViz | 200 | 2 min | ~16K | Lineup tendencies |
| BoxScoreMatchupsV3 | ~2,500 | 25 min | ~400K | Defensive matchups (2024+ only) |

---

## 8. Optimal Collection Strategy

### Phase 1: Quick Wins (< 30 minutes)

Collect all season-level and small endpoints first:
1. `DraftHistory` — 1 call
2. `LeagueDashPlayerBioStats` x 25 seasons
3. `DraftCombineStats` x 25 seasons
4. `LeagueDashLineups` x 200 calls (4 groups x 2 measures x 25 seasons)
5. `LeagueHustleStatsPlayer` x 10 seasons
6. `CommonTeamRoster` x 750 calls (gets both rosters + coaches)
7. `CommonPlayerInfo` x 2,142 players

### Phase 2: Per-Game Batch (5-6 hours, run overnight)

Single loop over all 33,822 games, making 3 API calls per game with 0.6s delay:
1. `GameRotation(game_id)` — stint data
2. `BoxScoreSummaryV2(game_id)` — officials + inactive + game info
3. `BoxScoreAdvancedV3(game_id)` — advanced box score

At 3 calls x 0.6s = 1.8s per game, total: **~17 hours**.
Or with 2 calls per game (skip advanced initially): **~11 hours**.

### Phase 3: Tracking + Hustle (4-5 hours, second overnight run)

1. `BoxScorePlayerTrackV3` for 2013-14 through 2025-26 (~15K games)
2. `BoxScoreHustleV2` for 2016-17 through 2025-26 (~11K games)

### Error Handling

- NBA API occasionally returns timeouts (observed during testing) — implement retry with backoff
- Some game_ids may return empty data — log and skip
- V2 endpoints have deprecation warnings for 2025+ games — use V3 where available
- Rate limit of 0.6s is conservative; 0.5s may work but risks 429 errors

---

## 9. Database Schema Extensions

New tables needed:

```sql
-- Player attributes (from CommonPlayerInfo + BioStats + DraftCombine)
CREATE TABLE PlayerAttributes (
    person_id INTEGER PRIMARY KEY,
    birth_date TEXT,
    height_inches REAL,
    weight REAL,
    country TEXT,
    college TEXT,
    draft_year INTEGER,
    draft_round INTEGER,
    draft_number INTEGER,
    -- Combine data (nullable — ~40% missing)
    wingspan REAL,
    standing_reach REAL,
    body_fat_pct REAL,
    hand_length REAL,
    hand_width REAL,
    standing_vertical REAL,
    max_vertical REAL,
    lane_agility REAL,
    three_quarter_sprint REAL,
    bench_press INTEGER,
    updated_at TEXT
);

-- Game rotation / stint data
CREATE TABLE GameRotation (
    game_id TEXT,
    team_id INTEGER,
    person_id INTEGER,
    in_time_real REAL,      -- tenths of seconds from game start
    out_time_real REAL,
    player_pts INTEGER,
    pt_diff REAL,
    usg_pct REAL,
    PRIMARY KEY (game_id, team_id, person_id, in_time_real)
);

-- Officials per game
CREATE TABLE GameOfficials (
    game_id TEXT,
    official_id INTEGER,
    first_name TEXT,
    last_name TEXT,
    jersey_num TEXT,
    PRIMARY KEY (game_id, official_id)
);

-- Coaches per team-season
CREATE TABLE CoachHistory (
    team_id INTEGER,
    season TEXT,
    coach_id INTEGER,
    coach_name TEXT,
    is_head_coach BOOLEAN,
    coach_type TEXT,
    PRIMARY KEY (team_id, season, coach_id)
);

-- Advanced box scores per game
CREATE TABLE PlayerBoxAdvanced (
    player_id INTEGER,
    game_id TEXT,
    offensive_rating REAL,
    defensive_rating REAL,
    net_rating REAL,
    usage_pct REAL,
    ts_pct REAL,
    efg_pct REAL,
    ast_pct REAL,
    oreb_pct REAL,
    dreb_pct REAL,
    pace REAL,
    possessions REAL,
    pie REAL,
    PRIMARY KEY (player_id, game_id)
);

-- Player tracking data (2013-14+)
CREATE TABLE PlayerBoxTracking (
    player_id INTEGER,
    game_id TEXT,
    speed REAL,
    distance REAL,
    touches INTEGER,
    passes INTEGER,
    secondary_assists INTEGER,
    contested_fg_made INTEGER,
    contested_fg_attempted INTEGER,
    uncontested_fg_made INTEGER,
    uncontested_fg_attempted INTEGER,
    defended_at_rim_fg_made INTEGER,
    defended_at_rim_fg_attempted INTEGER,
    PRIMARY KEY (player_id, game_id)
);

-- Game-level bonus data (from BoxScoreSummary)
CREATE TABLE GameInfo (
    game_id TEXT PRIMARY KEY,
    attendance INTEGER,
    game_duration TEXT,
    pts_paint_home INTEGER,
    pts_paint_away INTEGER,
    pts_fastbreak_home INTEGER,
    pts_fastbreak_away INTEGER,
    largest_lead_home INTEGER,
    largest_lead_away INTEGER,
    lead_changes INTEGER,
    times_tied INTEGER,
    pts_off_turnovers_home INTEGER,
    pts_off_turnovers_away INTEGER
);

-- Season-level lineup stats
CREATE TABLE LineupStats (
    season TEXT,
    group_quantity INTEGER,    -- 2, 3, 4, or 5
    group_id TEXT,             -- dash-separated player IDs
    group_name TEXT,
    team_id INTEGER,
    gp INTEGER,
    min REAL,
    plus_minus REAL,
    off_rating REAL,
    def_rating REAL,
    net_rating REAL,
    pace REAL,
    ts_pct REAL,
    PRIMARY KEY (season, group_quantity, group_id)
);
```

---

## 10. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| NBA API rate limiting / IP blocks | Medium | 0.6s delay, retry with backoff, rotate user-agents |
| GameRotation timeout for some seasons | Low | Observed one timeout for 2007-08; retry logic handles it |
| BoxScoreSummaryV2 deprecation | Low | Fall back to V3 for games after 4/10/2025 |
| DraftCombine missing for ~40% of players | Medium | Impute from height/weight/position; or train without combine features |
| Tracking data only from 2013-14 | Medium | Mask/impute for 2001-2013 games; model must handle missing features |
| Hustle data only from 2016-17 | Low | Lower priority feature set; same masking approach |
| LineupStats 2000-row cap | Low | Use TeamDashLineups to avoid cap; or accept top-minutes lineups only |
| Multi-day collection time | Low | Run overnight in phases; checkpoint progress; resume on failure |

---

## 11. Key Findings

1. **Player bio data is easy**: Height, weight, draft info, age — all available via bulk endpoints.
   Total collection: ~30 minutes. This completely fills the Players table gap.

2. **Lineup/stint data is solved**: `GameRotation` returns clean in/out times back to 2000-01.
   Combined with our existing PBP_Logs, we can reconstruct per-possession lineups for all 33K+ games.
   This is the single highest-value data addition.

3. **Coaching data comes free**: `CommonTeamRoster` returns coaches alongside roster data.
   Head coach identification is straightforward. No separate collection needed.

4. **Referee data is available from 2003-04**: `BoxScoreSummaryV2` includes officials, inactive players,
   and several bonus game-level stats. Collection shares the per-game loop with GameRotation.

5. **Advanced box scores have full historical coverage**: `BoxScoreAdvancedV3` returns offensive/defensive
   ratings, usage, pace for games back to 2000-01. This is a significant feature upgrade over raw box scores.

6. **Tracking/hustle data is newer but valuable**: Speed, touches, deflections available from 2013+/2016+.
   Partial coverage requires feature masking in the model, but 9-12 seasons is enough for useful signal.

7. **Total collection effort**: ~6 hours for Tier 0 data, ~17 hours for everything.
   Two overnight runs cover the entire dataset.
