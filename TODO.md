# NBA AI TODO

> **Last Updated**: December 25, 2025  
> **Current Sprint**: Sprint 16 - GenAI Predictor Design (Planning)

---

## 🎯 Active Sprint

### Sprint 16: Frontend API Optimization (Dec 25, 2025)

**Goal**: Optimize and add logging to frontend API endpoints to reduce page load times.

**Status**: 🚧 IN PROGRESS

**Current Issue**: Page load takes ~9 seconds, but only ~2s is logged by pipeline stages. Missing time is in:
- Database query (`get_normal_data()`) - complex CTE with JOINs, JSON parsing - estimated 6-7s
- Current predictions blending - logged at DEBUG only
- Overhead between stages

**Tasks**:
- [ ] Add logging/timing to `get_normal_data()` query execution
- [ ] Add logging to current predictions generation
- [ ] Analyze database query performance (EXPLAIN QUERY PLAN)
- [ ] Consider query optimizations:
  - [ ] Add indexes if missing (game_id, predictor, play_id)
  - [ ] Simplify CTE (LatestGameStates using MAX subquery)
  - [ ] Limit PBP data returned (only last N plays vs all plays)
- [ ] Consider architectural changes:
  - [ ] Lazy-load full PBP data (send only summary initially)
  - [ ] Cache serialized game data response
  - [ ] Async load non-critical data (betting lines, full PBP)

**Completed**:
- ✅ Fixed date selector to use Eastern Time (NBA schedule timezone)
- ✅ Fixed string comparison for UTC datetimes (ISO format with T separator)
- ✅ Removed duplicate schedule update call
- ✅ Changed verbose logs from INFO to DEBUG level (7 locations)

---

## 📋 Backlog

- **GenAI Predictor Design**: Research transformer architectures for sports sequence data, PBP tokenization strategy, embedding layer design
- **Historical Data Backfill**: PlayerBox/TeamBox (2000-2022, ~30K games), InjuryReports (Dec 2018-2023, ~900 PDFs/season)
- **Player Props Model**: Player-level predictions using PlayerBox data

---

## ✅ Completed Sprints

### Sprint 15: Pipeline Optimization & Database Consolidation (Dec 19-25, 2025)
**Summary**: Comprehensive database optimization, schema unification, and documentation update.

**Database Architecture**:
- Established 3-database architecture: `current ⊂ dev ⊂ full`
- Current: 516MB, 1,302 games (2025-2026 production)
- Dev: 3.0GB, 4,098 games (3 seasons for development)
- Full: 25GB, 37,366 games (27 seasons master archive)
- Created compressed backup: 1.4GB (94% compression ratio)

**Schema Unification**:
- Migrated all databases to unified schema (16 tables)
- Updated Games.status to INTEGER (1=Not Started, 2=In Progress, 3=Final)
- Added Games.status_text for human-readable display
- Fixed InjuryReports season derivation logic
- Synced all data to ensure proper subsetting

**Pipeline Stages Completed**:
- ✅ Stage 1: Schedule - Standardized logging, cache strategy, clean output
- ✅ Stage 2: Players - Standardized logging, cache strategy, clean output
- ✅ Stage 3: Injuries - Standardized logging, 1-hour cache, placeholder rows
- ✅ Stage 4: Betting - Standardized logging, 1-hour cache, 29 tests passing
- ✅ Stage 5: PBP - Standardized logging, leave=False tqdm, 14 tests passing
- ✅ Stage 6: GameStates - Standardized logging, leave=False tqdm, 7 tests passing
- ✅ Stage 7-9: Boxscores, Prior States, Features - Already optimized

**Cleanup**:
- Deleted 5 migration scripts (sync_current_database.py, migrate_full_database.py, etc.)
- Fixed Tier 2 betting log message (uses tqdm only, no duplicate logging)
- Updated DATA_MODEL.md with current schemas and architecture
- Updated copilot-instructions.md with current state
- 210 tests passing, 2 expected failures

### Sprint 13: Cleanup & Testing (Dec 6, 2025)
- Consolidated 3 CLI tools → single database_evaluator.py
- Created workflow-aware validation for all 14 database tables
- Deep review of all 9 pipeline stages
- Frontend tests passing (14/14)
- Removed src/database_migration.py, data_quality.py, database_validator.py, validators/

### Sprint 12: Database Consolidation (Dec 6, 2025)
- Removed unused tables from DEV (BettingLines, PlayerIdMapping)
- Created new tables in ALL_SEASONS (PlayerBox, TeamBox, InjuryReports, ESPNGameMapping, ScheduleCache)
- Migrated ALL_SEASONS Betting to new schema (18,282 rows)
- Synced all DEV data to ALL_SEASONS (DEV is now strict subset)
- Backfilled betting: 2021-2022 (93.4%), 2022-2023 (93.6%) from Covers.com
- Data availability audit: PBP 2000+, Betting 2007+, InjuryReports Dec 2018+
- Updated DATA_MODEL.md: Two-database architecture, unified schema (13 tables)
- Cleaned up data files: removed 227MB of obsolete archives, organized backups
- Removed outdated scripts (betting_backfill_status.py, test_espn_betting_api.py)
- Enhanced data_quality.py: added Betting coverage, database selection flag

### Sprint 11.5: Betting Data Integration (Dec 5-6, 2025)
- Fixed Covers.com scraper (headers, HTML selectors)
- Built 3-tier betting system (ESPN → Covers matchups → Covers schedules)
- Created 36-test suite for betting system
- Backfilled 2023-2024 (1,220 games), updated 2025-2026 (347 games)
- Simplified betting.py (~240 lines removed)
- 2024-2025 at 100% coverage, all results verified

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
