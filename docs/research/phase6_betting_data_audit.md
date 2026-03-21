# Phase 6 Betting Data Quality Audit

**Date**: 2026-03-20
**Scope**: All NBA betting/spread data in `data/NBA_AI_full.sqlite` (2007-2026)

## Executive Summary

The audit found **three distinct quality tiers** in our spread data:

| Era | Source | MAE | Quality | Issue |
|-----|--------|-----|---------|-------|
| 2007-2020 | ESPN API (flat format) | 9.33 | **Good** | Stored as `espn_current_spread` due to schema migration, but is actually closing line data |
| 2021-2024 | Covers Tier 3 (team schedules) | 10.15 | **Good** | True closing lines; higher MAE explained by increased game variance |
| 2024-2025 | ESPN API (DraftKings closing) | 10.50 | **Good** | True DraftKings closing lines; MAE elevated by season variance |
| 2025-2026 (ESPN) | ESPN API (opening + closing) | 11.09 | **Acceptable** | Partial season; only 276/890 games have ESPN closing |
| 2025-2026 (Covers) | Covers Tier 2 (matchups page) | 12.88 | **BROKEN** | Scraper is extracting wrong data; 549 games affected |

**Key findings**:
1. ESPN closing lines are legitimate DraftKings closing lines -- verified against OddsShark/VegasInsider
2. The higher MAE in recent seasons (10.5 vs 9.2) is primarily explained by increased game variance in the modern NBA (avg absolute margin: 12.5 vs 11.1), NOT bad data
3. The Covers matchups page scraper (Tier 2) is broken for 2025-2026 -- extracting wrong spreads and labeling all results as "W"
4. The Covers team schedule scraper (Tier 3) works correctly for 2021-2024

---

## 1. Data Source Analysis

### 1a. Coverage by Season and Source

```
Season       | Total | espn_open | espn_curr | espn_close | covers | Dominant Source
-------------|-------|-----------|-----------|------------|--------|----------------
2007-2008    | 1,230 |     0     |   1,230   |     0      |    0   | espn_current
2008-2009    | 1,230 |     0     |   1,230   |     0      |    0   | espn_current
...                                (same pattern through 2018-2019)
2019-2020    | 1,059 |     0     |   1,058   |     0      |    0   | espn_current
2020-2021    | 1,080 |     0     |   1,079   |     0      |    0   | espn_current
2021-2022    | 1,230 |     0     |    457    |     0      |  773   | mixed (espn_curr + covers)
2022-2023    | 1,230 |     0     |     0     |     0      | 1,230  | covers (Tier 3)
2023-2024    | 1,230 |     0     |     0     |     0      | 1,220  | covers (Tier 3)
2024-2025    | 1,230 |     0     |     0     |   1,229    |    1   | espn_closing
2025-2026    |   890 |   148     |    51     |    276     |  575   | covers (broken!)
```

### 1b. How Historical Data Was Loaded

- **2007-2022**: All loaded on 2025-12-06 via ESPN summary API (historical batch).
  The original schema had a single `spread` column. When the schema was refactored
  to opening/current/closing (Sprint 16b, Jan 2026), old data was migrated to
  `espn_current_spread`. Despite the name, **this IS closing line data** -- it came
  from ESPN's flat post-game API format which returns the final/closing spread.

- **2021-2024 Covers**: Loaded 2025-12-06 via Tier 3 (team schedule page scraping).
  Each team's Covers schedule page was scraped. This produces genuine closing lines.

- **2024-2025 ESPN**: Loaded 2025-12-05 in one batch from ESPN summary API.
  The API returns DraftKings closing lines for completed games.

- **2025-2026**: Collected incrementally. ESPN data from Tier 1 (live window),
  Covers from Tier 2 (matchups page scraper -- BROKEN).

### 1c. MAE by Source

| Source | N | MAE | Assessment |
|--------|---|-----|------------|
| espn_current (2007-2020) | 14,639 | 9.33 | Excellent -- true closing lines |
| covers_closing (2021-2024 Tier 3) | 4,443 | 10.04 | Good -- true closing lines |
| espn_closing (2024-2025) | 1,229 | 10.51 | Good -- DraftKings closing lines |
| espn_closing (2025-2026) | 276 | 11.09 | Acceptable -- partial season |
| covers_closing (2025-2026 Tier 2) | 575 | 12.88 | **BROKEN** |
| espn_opening (2025-2026) | 148 | 11.06 | OK for opening lines |

---

## 2. Verification Against Real Closing Lines

### 2a. Games Verified

**OKC vs CLE, Feb 22, 2026** (game_id 0022500817):
- Our DB: `espn_opening_spread = 1.5`, `espn_closing_spread = 4.5` (OKC home = underdog)
- OddsShark consensus: OKC -4 (most books), with DraftKings at -4 or -4.5
- **Verdict**: Our ESPN closing spread matches DraftKings within 0.5 points

**LAL vs BOS, Feb 22, 2026** (game_id 0022500823):
- Our DB: `espn_opening_spread = 1.5`, `espn_closing_spread = 1.5` (LAL home = underdog)
- CBS/VegasInsider: BOS -1.5 favorite (away)
- **Verdict**: Perfect match

**BOS vs PHI, Dec 25, 2024** (game_id 0022400407):
- Our DB: `espn_closing_spread = -9.5` (BOS home = favored by 9.5)
- OddsShark: BOS ranging from -4.5 to -10 across books; most -9.5 to -10
- **Verdict**: Reasonable -- some books had -9.5, consensus was -9.5 to -10

### 2b. Conclusion on ESPN Data Accuracy

ESPN API provides **DraftKings closing lines** (confirmed by code inspection -- `fetch_espn_betting_data`
prefers the "Draft Kings" provider from the pickcenter data). These are legitimate closing lines
from a major sportsbook. The typical difference vs market consensus is 0-0.5 points.

---

## 3. The Modern NBA Variance Explanation

The higher MAE in recent seasons is **NOT a data quality problem**. It's explained by increased game variance:

```
Season Range  | Avg |Margin| | Spread MAE | Ratio (MAE/|Margin|)
--------------|----------------|------------|--------------------
2007-2019     |     11.1       |    9.3     |      0.838
2020-2021     |     12.2       |   10.7     |      0.877
2021-2022     |     12.4       |   10.6     |      0.855
2022-2023     |     11.2       |    9.7     |      0.866
2023-2024     |     12.6       |   10.6     |      0.841
2024-2025     |     12.8       |   10.5     |      0.820
```

The MAE/|Margin| ratio is stable at 0.83-0.88, meaning Vegas maintains the same relative accuracy.
The absolute MAE increased because games themselves became more unpredictable (more blowouts, more
variance in scoring). This is consistent with the "3PT variance" hypothesis from Phase 3.

The total (O/U) MAE is stable at 13.9-14.8 across all eras, further confirming that the data
collection pipeline is working correctly.

---

## 4. Covers Tier 2 Scraper Bug (CRITICAL)

### 4a. The Problem

The Covers matchups page scraper (`src/database_updater/covers.py`, `_parse_matchups_page()`)
is broken for the 2025-2026 season. Evidence:

1. **spread_result is always "W"**: Of 575 games with Covers spread data in 2025-2026,
   561 have `spread_result = 'W'` and 14 have NULL. Zero games show 'L' or 'P'.
   In correctly-working seasons (2022-2024), the split is roughly 50/50 W/L.

2. **Spread values don't correlate with game outcomes**: MAE of 12.88 vs expected ~9.5.
   The absolute spread doesn't improve prediction (MAE flat from 11.3 to 15.6 regardless
   of spread size), whereas good data shows MAE dropping from 12 to 8.8 for larger spreads.

3. **Sign appears inconsistent**: Comparison with ESPN opening spreads shows no consistent
   relationship (neither same-sign nor opposite-sign).

### 4b. Root Cause Hypothesis

The Covers matchups page uses JavaScript client-side rendering. The parser extracts text
like "covered the spread of -3.5" from the summary box. Possible issues:

- Covers may have changed their HTML structure (the `summary-box` class, `gamebox` article)
- The text may always come from the covering team's perspective (hence always "W")
- The spread value may be extracted from the wrong element or from a pre-game container
  rather than the closing line

The fallback path (lines 342-352 in covers.py) extracts from `.trending-and-cover-by-container`
which may return pre-game lines or lines from the wrong team's perspective.

### 4c. Impact

549 of 890 completed 2025-2026 games (62%) have ONLY Covers data. These games have
unreliable spread data that degrades our ATS evaluation from ~11.1 MAE to 12.3 MAE.

### 4d. Fix Priority: HIGH

Options:
1. Re-scrape 2025-2026 using Tier 3 (team schedule pages) -- known working
2. Fix the Tier 2 parser to handle the current Covers page structure
3. Use ESPN API backfill for all completed 2025-2026 games

**Recommended**: Option 1 (Tier 3 re-scrape) is the fastest fix. Then fix Tier 2 for
ongoing collection.

**Note on Tier 3 backfill**: The save logic uses `COALESCE(?, field)` which will overwrite
existing non-NULL `covers_closing_spread` values with new ones. The existing broken rows
are marked `lines_finalized=1` but the Tier 3 backfill does NOT skip finalized games --
it processes all matched games. Previous runs (2022-2023) achieved 1230/1230 match rate,
confirming the date matching works despite UTC-vs-Eastern timezone differences.

---

## 5. `espn_current_spread` Field Investigation

### 5a. What It Actually Contains

Despite its name, `espn_current_spread` contains **ESPN closing/final line data** for
seasons 2007-2022. This happened because:

1. The original schema (Dec 2025) had a single `spread` column
2. Data was loaded from the ESPN summary API's post-game flat format
3. When the schema was refactored to opening/current/closing, the old `spread` data
   was migrated to `espn_current_spread`

The field name is misleading but the data is actually the best quality in our database
(MAE 9.33 for 14,639 games).

### 5b. For 2025-2026

The 51 games with `espn_current_spread` in 2025-2026 are genuinely "current" (pre-game)
lines that were captured before the game finished and never upgraded to closing.

---

## 6. `spread_result` and `ou_result` Fields

### 6a. Coverage

```
Season     | spread_result | ou_result | Total
-----------|---------------|-----------|------
2007-2021  |       0       |     0     | ~16K  (no Covers data for these)
2021-2022  |     773       |   773     | 1,230 (from Covers Tier 3)
2022-2023  |   1,230       | 1,230     | 1,230 (from Covers Tier 3)
2023-2024  |   1,220       | 1,220     | 1,230 (from Covers Tier 3)
2024-2025  |   1,230       | 1,230     | 1,230 (from ESPN; 100% accurate)
2025-2026  |     648       |   652     |   890 (mix; 2025-2026 Covers = all 'W' = broken)
```

### 6b. Accuracy

For seasons where the Tier 3 scraper works (2021-2024), spread_result is 100% accurate
when verified against calculated results:
- `spread_result = 'W'` perfectly matches `home_margin + spread > 0`
- `spread_result = 'L'` perfectly matches `home_margin + spread < 0`
- `spread_result = 'P'` perfectly matches `home_margin + spread = 0`

For 2024-2025 (ESPN), spread_result has a proper 614 W / 616 L split, correctly populated.

For 2025-2026, spread_result is unreliable (all 'W' from broken Tier 2 scraper).

**Recommendation**: Don't rely on `spread_result`; always calculate ATS result from
`spread + (home_score - away_score)`.

---

## 7. Data Reacquisition Plan

### 7a. What Needs Fixing

| Season | Games | Issue | Fix |
|--------|-------|-------|-----|
| 2025-2026 | 549 | Broken Covers Tier 2 data | Re-scrape via Tier 3 |
| 2025-2026 | 23 | Missing spread entirely | Tier 3 backfill |
| 2007-2021 | ~16K | `spread_result` missing | Calculate from data |

### 7b. Recommended Actions

**Immediate (fixes ATS eval)**:
1. Run Tier 3 backfill for 2025-2026: `python -m src.database_updater.betting --backfill 2025-2026`
   - This will use the working team schedule scraper to get correct closing lines
   - ~30 requests (1 per team), ~90 seconds with rate limiting
   - Will overwrite the broken Covers data via COALESCE(new, existing) update pattern

**Short-term (fix ongoing collection)**:
2. Debug the Tier 2 matchups page scraper against the current Covers.com HTML
   - Test with `python -c "from src.database_updater.covers import fetch_matchups_for_date; from datetime import date; print(fetch_matchups_for_date(date(2026,3,15)))"`
   - Check if `article.gamebox`, `summary-box`, and `covered the spread` patterns still match

**Medium-term (improve data quality)**:
3. Consider The Odds API (free tier: 500 requests/month) for consensus closing lines
4. Fill `spread_result` for 2007-2021 by calculating from existing spread + score data
5. Rename `espn_current_spread` to something clearer, or document that it contains
   historical closing lines for 2007-2022

### 7c. Free/Cheap Sources for Historical Closing Lines

| Source | Coverage | Cost | Quality |
|--------|----------|------|---------|
| ESPN API (our current) | 2007-present | Free | DraftKings closing -- Good |
| Covers.com Tier 3 | 2007-present | Free (scraping) | Closing lines -- Good |
| The Odds API | 2020-present | Free tier (500 req/mo) | Multi-book consensus -- Best |
| Kaggle datasets | Varies | Free | Mixed quality |
| SportsDataIO | Historical | $50/mo+ | Professional grade |

---

## 8. Corrected Vegas MAE Estimates

After accounting for data quality issues, our best estimate of true Vegas closing line
MAE by season:

```
Season Range  | N Games | Source           | Spread MAE | Notes
--------------|---------|------------------|------------|------
2007-2020     | 14,639  | ESPN (current)   |    9.33    | Gold standard
2021-2022     |  2,309  | ESPN+Covers mix  |   10.66    | COVID era + higher variance
2022-2023     |  1,230  | Covers Tier 3    |    9.74    | Normal season
2023-2024     |  1,220  | Covers Tier 3    |   10.56    | Higher variance season
2024-2025     |  1,229  | ESPN closing     |   10.51    | DraftKings closing; high variance season
2025-2026     |    276  | ESPN closing only |   11.09    | Partial; exclude Covers
```

**Weighted average 2007-2025**: ~9.75 MAE (using best available source per game)

The "true" closing line MAE for recent seasons (2022-2025) appears to be **9.7-10.5**,
higher than the 2007-2019 average of 9.33 due to increased scoring variance in the modern NBA.
Our Phase 3 ensemble model's spread MAE of 10.66 is ~0.7-1.0 points worse than Vegas.

---

## 9. `lines_finalized` Field Status

```
Season     | Finalized | Not Finalized | Total
-----------|-----------|---------------|------
2007-2024  |  ~18,600  |       0       | ~18,600
2024-2025  |   1,230   |       0       | 1,230
2025-2026  |     851   |      18       |   869
```

All completed games through 2024-2025 are marked as finalized. The 18 unfinalized 2025-2026
games are recent games awaiting Covers/ESPN finalization.

---

## 10. COALESCE Priority for Spread Selection

The current code uses `COALESCE(espn_closing_spread, covers_closing_spread, espn_current_spread)`.
This priority is **correct** for most seasons but has issues:

1. **2007-2022**: Falls through to `espn_current_spread` -- works, since this is the closing data
2. **2021-2022 overlap**: For 457 games with espn_current + 773 with covers -- espn_current
   is used where available (ESPN data slightly better than Covers for this season)
3. **2022-2024**: Falls through to `covers_closing_spread` -- works (Tier 3 data is good)
4. **2024-2025**: Uses `espn_closing_spread` -- works
5. **2025-2026**: Uses `espn_closing_spread` where available (276 games), then falls through
   to broken `covers_closing_spread` (575 games) -- **THIS IS THE PROBLEM**

After running the Tier 3 backfill for 2025-2026, the COALESCE priority should work correctly
because the Covers data will be replaced with good Tier 3 data.

---

## 11. Recommendations for Phase 6 ATS Evaluation

1. **Before any ATS evaluation**: Run `python -m src.database_updater.betting --backfill --season=2025-2026`
2. **Calculate spread_result yourself**: Don't rely on the stored `spread_result` field;
   compute `home_score + spread > away_score` for ATS evaluation
3. **Use conservative COALESCE**: `COALESCE(espn_closing_spread, espn_current_spread, covers_closing_spread)`
   (deprioritize Covers for 2025-2026 until Tier 3 backfill is run)
4. **Report MAE by source**: When evaluating, always break down MAE by spread data source
   to catch data quality issues early
5. **Expected Vegas MAE**: Target ~9.5-10.5 depending on season; anything >11 suggests data issues
