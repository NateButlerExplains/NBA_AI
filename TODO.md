# NBA AI TODO

> **Last Updated**: December 4, 2025  
> **Current Sprint**: Sprint 12 - GenAI Architecture & Prototype

---

## ✅ Recently Completed

### Sprint 11: Data Infrastructure & Simplification (Dec 3-4, 2025) ✅

**Goal**: Simplify data pipeline, integrate injury data, prepare for GenAI development.

**Completed**:
- [x] Switched injury source from ESPN to NBA Official PDFs (simpler, authoritative)
- [x] Simplified Players table (removed biometric columns - position/height/weight/age)
- [x] Simplified InjuryReports schema (removed ESPN columns, added nba_player_id)
- [x] Implemented player ID matching for injuries (97.6% match rate)
- [x] Backfilled injury data for all 3 seasons (15,511 records)
- [x] Renamed database to `NBA_AI_dev.sqlite` (clearer naming)
- [x] Documented 3-database architecture (dev/current/all_seasons)
- [x] Updated DATA_MODEL.md with current schemas
- [x] Updated data_quality.py (added TeamBox, InjuryReports coverage)
- [x] Created test_data_pipeline.py (16 tests, all passing)
- [x] Removed obsolete files (espn_injuries.py, backfill scripts)
- [x] Verified automatic updates working for current season

**Data Status**:
- 4,093 games across 3 seasons (2023-2026)
- 1.58M PBP records, 78K PlayerBox records
- 15,511 injury records with 97.6% player ID matching
- All pipelines automated for current season

---

## 🎯 Active Sprint

### Sprint 12: GenAI Architecture & Prototype (Starting)

**Goal**: Design and build minimal viable GenAI predictor using PBP data.

**Tasks**:
- [ ] Research transformer architectures (TFT, Informer, PatchTST, custom)
- [ ] Define prediction targets (game score, player stats, both?)
- [ ] Define sequence representation (what is one "event"? what context?)
- [ ] Design input representation (tokenization, embeddings)
- [ ] Write Architecture Decision Record (ADR)
- [ ] Define train/val/test split strategy
- [ ] Data pipeline: PBP → model-ready sequences
- [ ] Minimal viable model (prove concept works)
- [ ] Baseline evaluation vs XGBoost (MAE 10.2)

---

## 📋 Backlog

---

### 📊 Post-GenAI (Priority 2)

#### External Prediction Baselines
**Goal**: Measure GenAI against Vegas lines and ESPN predictions for proper comparison

**Available Data Sources**:
1. **Existing Betting Table** (ALL_SEASONS DB): 18,292 games with spreads/O-U/moneylines (2007-2021)
   - `home_spread_at_open`, `home_spread_at_close`, `over_at_open`, `over_at_close`
   - `home_ml`, `away_ml`, `2h_home_spread`, `2h_over`
   - **Status**: Already collected, just needs migration to working DB for recent seasons

2. **ESPN Pickcenter** (via sportsdataverse-py): Live betting data from ESPN API
   - Spreads, O/U, moneylines from multiple sportsbooks
   - `espn_nba_pbp(game_id)['pickcenter']`, `['odds']`, `['againstTheSpread']`
   - **Pros**: Free, live during games, includes ATS results
   - **Cons**: May not have historical depth

3. **ESPN Win Probability**: `espn_nba_pbp(game_id)['espnWP']`
   - ESPN's proprietary win probability model
   - Good for calibration comparison (is our 70% = ESPN's 70%?)

4. **Covers.com** (scraping): Historical lines archive
   - Most comprehensive historical data
   - Requires scraper (see `klane/databall` repo for reference)
   - **Use case**: Fill gaps in existing Betting table (2022-present)

**Implementation Plan**:
- [ ] Migrate existing Betting data to working DB (18K games of free baseline!)
- [ ] Add ESPN pickcenter to data collection pipeline
- [ ] Extend Betting table with recent seasons (scrape or ESPN)
- [ ] Build Vegas baseline predictor (predict using closing line implied probability)
- [ ] Add comparison metrics: vs Vegas ATS%, O/U%, ROI simulation

#### Player Props Model
**Goal**: Player-level predictions (points, rebounds, etc.)  
**Approach**: Either GenAI extension or dedicated model using PlayerBox data

---

### 🔧 Tech Debt & Nice-to-Have (Priority 3 - Defer)

#### Player Enrichment Optimization
**Issue**: Updates all 5,103 players (7+ hours) instead of only changed ones  
**Status**: RESOLVED - Simplified players.py, removed biometric collection (~1 second now)

#### Web App UX Improvements
**Items**: Auto-refresh for live games, mobile responsive, confidence intervals display

#### Logging & Monitoring
**Items**: Structured logging, performance metrics, error alerting

#### Database Consolidation
**Goal**: Eventually make dev DB a strict subset of ALL_SEASONS
- [ ] Backfill injuries to ALL_SEASONS (15K records)
- [ ] Migrate Betting data from ALL_SEASONS to dev (for baseline comparisons)
- [ ] Add PlayerBox/TeamBox to ALL_SEASONS

---

## ✅ Completed Sprints

### Sprint 11: Data Infrastructure & Simplification (Dec 3-4, 2025)
- Switched from ESPN to NBA Official injury PDFs
- Simplified Players table (removed biometrics)
- Player ID matching at 97.6% rate
- Renamed database to NBA_AI_dev.sqlite
- Updated DATA_MODEL.md, data_quality.py
- Created test_data_pipeline.py (16 tests)
- All automatic updates verified working

### Sprint 10: Public Release v0.2.0 (Nov 27, 2025)
- Released v0.2.0 to public GitHub with setup.py automation
- Updated all dependencies (security fixes for Flask, Jinja2, Werkzeug, urllib3)
- Upgraded PyTorch 2.4.0 → 2.8.0, sklearn 1.5.1 → 1.7.2, xgboost 2.1.0 → 3.1.2
- Retrained all models with current package versions (no warnings)
- Installed GitHub CLI, closed all 13 GitHub issues with responses
- Configured git workflow: private repo as default, public for releases only
- 75 tests passing

### Sprint 9: Traditional ML Model Training (Nov 26, 2025)
- Trained Ridge/XGBoost/MLP, created Ensemble predictor
- Built model registry with semantic versioning
- All 5 predictors operational (Baseline, Linear, Tree, MLP, Ensemble)

### Sprint 8: Data Collection & Validation (Nov 26, 2025)
- Complete data for 3 seasons (2,638 games with PBP, GameStates, PlayerBox, TeamBox)
- Database validator with 25+ checks, excellent data quality

### Sprint 7: Web App Testing (Nov 25, 2025)
- Fixed timezone bugs, empty game_states error
- Added player enrichment skip option

### Sprint 5: Database Consolidation (Nov 25, 2025)
- TEXT-based game_id schema unified
- Single data pipeline via database_update_manager.py

### Sprint 4: Data Lineage (Nov 25, 2025)
- ScheduleCache table, timezone-aware datetime handling

### Sprint 3: Live Data Collection (Nov 25, 2025)
- Live game data pipeline, endpoint selection

### Sprint 2: Prediction Engine Refactoring (Nov 25, 2025)
- Base predictor classes, unified training script

### Sprint 1: Infrastructure Cleanup (Nov 24-25, 2025)
- Removed 4 subsystems, requirements cleanup (87→46 packages)
