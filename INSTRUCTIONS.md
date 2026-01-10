# NBA AI - Claude Code Instructions

> **Last Updated**: January 9, 2026
> **Current Focus**: Pre-release stabilization and setup tooling verification

---

## Project Mission

**North Star**: Build a GenAI-based prediction engine using play-by-play (PBP) data as the primary source, minimizing human-engineered features and data collection complexity.

**Current Phase**: Pre-release preparation - stabilizing infrastructure before GenAI development begins.

---

## Development Philosophy

### Code Quality
- **Minimal inline documentation**: Keep code self-explanatory, avoid verbose comments
- **Centralized documentation**: README (public), TODO (project management), DATA_MODEL (schemas), this file (AI context)
- **Auto-formatting**: Use automatic formatters for consistency
- **Testing**: Write tests when they add value; prefer overarching tests over granular unit tests
- **Simplicity**: Don't over-engineer; keep solutions minimal and focused

### Git Workflow
- **Branch**: Single `main` branch (solo developer)
- **Commits**: Normal granularity for pre-MVP project
- **IMPORTANT**: Always ask before committing (never auto-commit)
- **Repos**: Private repo for daily work, public repo updated only on releases
- **TODO tracking**: Keep TODO.md current but don't let it bloat

### Virtual Environment
**CRITICAL**: Always activate venv before any Python commands - Copilot forgot this frequently!

```bash
source venv/bin/activate
```

Common issues if venv not activated:
- `ModuleNotFoundError`
- Package version conflicts
- Wrong Python interpreter

---

## Project Architecture

### 5-Stage Data Pipeline

```
1. Schedule Update     → Fetch game schedule from NBA API
2. Game Data Update    → PBP logs, GameStates, PlayerBox/TeamBox
3. Pre-Game Data       → Prior states, feature engineering
4. Predictions         → ML/GenAI predictions
5. Web App / Games API → Live score updates, predictions serving
```

**Core Orchestrator**: `src/database_updater/database_update_manager.py`

### Component Boundaries

- **`src/database_updater/`**: ETL pipeline (NBA API → SQLite)
- **`src/predictions/`**: Feature engineering + predictor engines
- **`src/games_api/`**: REST API for live game data
- **`src/web_app/`**: Flask frontend

### Three-Database Architecture

| Database | Size | Games | Seasons | Purpose | Usage |
|----------|------|-------|---------|---------|-------|
| `NBA_AI_current.sqlite` | 516MB | 1,302 | 2025-2026 | Production | Releases only |
| `NBA_AI_dev.sqlite` | 3.0GB | 4,098 | 2023-2026 | Development | **Default for work** |
| `NBA_AI_full.sqlite` | 25GB | 37,366 | 1999-2026 | Archive | Read-only, use with care |

**Subset Relationship**: `current ⊂ dev ⊂ full`
**Reality**: They drift during active dev (OK), but meant to be subsets long-term

**Set via `.env`**: `DATABASE_PATH=data/NBA_AI_dev.sqlite`

---

## Critical Patterns

### Configuration
- All config in `config.yaml` with `${VAR_NAME}` interpolation from `.env`
- Loaded via `src/config.py` (handles path resolution, env vars, auto-generation)
- **Never hardcode paths** - always use `config["database"]["path"]`

### Datetime & Timezone Strategy

**Principle**: Store UTC, query in Eastern Time (NBA's timezone), display in user's local timezone

**Helper Functions** (`src/utils.py`):
```python
from src.utils import (
    get_utc_now,                    # Current UTC (timezone-aware)
    get_current_eastern_datetime,  # Current Eastern time with DST
    get_current_eastern_date,      # Current Eastern date (NBA operations)
    utc_to_timezone,               # Convert UTC to any timezone (display)
    determine_current_season,      # Uses Eastern for June 30 boundary
)
```

**Usage Guidelines**:
- **Database storage**: Always UTC (`get_utc_now()` for timestamps)
- **Season logic**: Use `determine_current_season()` (handles Eastern automatically)
- **Cache TTL**: Use UTC for consistent comparisons
- **User display**: Get `user_tz` from frontend, use `utc_to_timezone()`

### Logging

Comprehensive logging setup exists - worth reviewing before making changes.

**Setup Pattern** (at entry points):
```python
from src.logging_config import setup_logging
setup_logging(log_level)
```

**Core Rules**:
- Use `logging.getLogger()` in all modules
- Execution timing: Decorate with `@log_execution_time(average_over="game_ids")`
- Rotating file handlers configured
- JSON output support available

### External API Access

**NBA API Rate Limiting**: Rules exist for different endpoints - review when diving into core functionality

**Retry Logic**: All NBA API calls wrapped with `utils.py::requests_retry_session()` (exponential backoff)

**Key Endpoints**:
- Schedule API (NBA Stats)
- PBP: Dual source (CDN primary, Stats API fallback)
- Boxscores via `nba_api` library
- Betting from ESPN API + Covers.com scraping

### Database Operations

**Always use context managers**:
```python
with sqlite3.connect(DB_PATH) as conn:
    # work here
```

**Completion Flags** (see DATA_MODEL.md for details):
- `game_data_finalized`: PbP_Logs + GameStates (with final state)
- `boxscore_data_finalized`: PlayerBox + TeamBox
- `pre_game_data_finalized`: Features created (requires both teams have prior games)

**Chunked Processing**: `database_update_manager.py` processes in 100-game chunks to manage memory

---

## Running the Application

### Web App
```bash
source venv/bin/activate
python start_app.py --predictor=Tree --log_level=INFO
```

Valid predictors: `Baseline`, `Linear`, `Tree`, `MLP`, `Ensemble`

### Database Updates

**Auto-update**: Web app triggers full pipeline when viewing any date (smart - only processes games with `*_finalized=0`)

**Manual batch update**:
```bash
python -m src.database_updater.database_update_manager --season=2024-2025 --predictor=Tree
```

First run for a season: ~1500 API calls, up to 2GB memory, processes in 100-game chunks

### Health Check
```bash
# Full check (pipeline + validation)
python -m src.health_check --season=2024-2025

# Validation only (skip pipeline)
python -m src.health_check --season=2024-2025 --skip-pipeline

# JSON output for automation
python -m src.health_check --season=2024-2025 --skip-pipeline --json
```

**Categories**: Completeness, Structure/Values, Flag Consistency, Referential Integrity, Temporal
**Exit Codes**: 0=pass, 1=warnings, 2=critical failures

---

## Testing Strategy

**Three-Layer Approach**:
1. **Unit/Integration Tests** (pytest, 210 passing)
2. **Inline Validators** (runtime validation during ETL)
3. **Health Check CLI** (season-level database validation)

**Running Tests**:
```bash
pytest                          # All tests
pytest tests/test_api.py        # Specific file
pytest --cov=src --cov-report=html  # With coverage
pytest -x -q                    # Fast mode (stop on first failure)
```

**Test Philosophy** (per project owner):
- Write tests when they add value
- Err on side of less testing
- Prefer overarching tests over specific granular tests
- Keep test setup simple for pre-MVP phase

---

## Predictor Pattern

All predictors implement:
- `make_pre_game_predictions(game_ids)` → Uses Features, returns predictions dict
- `make_current_predictions(game_ids)` → Blends pre-game with current score + time

**Adding New Predictor**:
1. Create class in `src/predictions/prediction_engines/`
2. Add to `PREDICTOR_MAP` in `prediction_manager.py`
3. Add config entry in `config.yaml` under `predictors:`
4. Predictions auto-saved to `Predictions` table as JSON

---

## Common Pitfalls

1. **Virtual environment**: Activate venv FIRST (Copilot forgot this often)
2. **Path resolution**: Run from project root, use `python -m src.module` notation
3. **Season filtering**: Queries need `AND season_type IN ('Regular Season', 'Post Season')` to exclude All-Star
4. **JSON serialization**: Cast NumPy types to native Python before `json.dumps()`
5. **API limits**: `max_game_ids: 20` in config prevents Games API overload
6. **Database subsetting**: `full` database is LARGE and should be accessed with care

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `README.md` | Public-facing project info |
| `TODO.md` | Project management (In Progress / Backlog / Completed) |
| `DATA_MODEL.md` | Database schemas, API endpoints, data structures (single source of truth) |
| `INSTRUCTIONS.md` | This file - AI assistant context |
| `config.yaml` | All configuration with env var interpolation |
| `.env` | Environment variables (DATABASE_PATH, WEB_APP_SECRET_KEY, etc.) |

---

## Current Sprint Focus

**Sprint 17**: GenAI Predictor Design (PLANNING) - **Not started yet**

**Current Priority**: Pre-release stabilization
- Ad hoc testing and validation
- Setup tooling verification (setup.py, requirements.txt, etc.)
- Ensure current database is perfect subset of dev
- Project cleanup and organization

**Next Phase**: Once stable, begin GenAI prediction engine research and design

---

## Working with Claude Code

### What to Expect from Me
- ✅ Always activate venv before Python commands
- ✅ Follow datetime strategy (UTC storage, Eastern queries, user timezone display)
- ✅ Respect North Star (GenAI is high priority, infrastructure is maintenance-only)
- ✅ Use established patterns (config in YAML, logging via setup, chunked processing)
- ✅ Keep TODO.md updated without bloating it
- ✅ **Always ask before committing** (critical - never auto-commit)
- ✅ Minimal inline docs, self-explanatory code
- ✅ Simple test philosophy (value-driven, prefer overarching tests)

### Repository Context
- **Daily work**: Private repo on `main` branch
- **Releases**: Push to public repo
- **Current state**: Pre-release preparation phase
- **Database**: Use `dev` for most work, `full` only when necessary (with care)

---

## Season & Date Utilities

All in `src/utils.py`:
- `game_id_to_season("0022300649")` → `"2023-2024"`
- `date_to_season("2024-03-15")` → `"2023-2024"` (Oct 1 cutoff)
- `validate_game_ids()`, `validate_date_format()` → Raise `ValueError` on invalid

**API restrictions**: `config.yaml` limits valid seasons to avoid accidental bulk updates

---

## Dependencies

**Core Stack** (46 packages in requirements.txt, organized by purpose):
- Web: Flask, python-dotenv, PyYAML, requests
- Database: SQLAlchemy
- Data: numpy, pandas
- ML: scikit-learn, scipy, xgboost, joblib
- DL: torch (CPU version)
- NBA API: nba_api==1.11.3 (critical - BoxScoreTraditionalV3)

**Principle**: Never add transitive dependencies to requirements.txt - pip installs them automatically

---

## Data Directory Structure

```
data/
  NBA_AI_current.sqlite       # 516MB production (releases)
  NBA_AI_dev.sqlite          # 3.0GB development (default)
  NBA_AI_full.sqlite         # 25GB master (use with care)
  backups/
    NBA_AI_full_backup_YYYYMMDD.sqlite.gz
  releases/
```

---

## Quick Reference: Database Schema

See `DATA_MODEL.md` for comprehensive documentation. Key tables:

- **Games**: Master schedule + completion flags
- **PbP_Logs**: Raw play-by-play JSON
- **GameStates**: Parsed snapshots (one per play)
- **PlayerBox/TeamBox**: Boxscore stats
- **Features**: ML feature sets (34 features per game)
- **Predictions**: Model outputs (JSON blob)
- **Betting**: Unified betting lines (opening/current/closing)
- **InjuryReports**: NBA Official injury PDFs
- **Players/Teams**: Reference tables

**Query Pattern**: Use CTEs to get latest GameStates, avoid full table joins (see `games.py::get_normal_data()`)

---

## Development Workflow Example

```bash
# Activate environment (ALWAYS FIRST)
source venv/bin/activate

# Run health check
python -m src.health_check --season=2024-2025 --skip-pipeline

# Run tests
pytest -x

# Update database for current season
python -m src.database_updater.database_update_manager --season=2025-2026 --predictor=Tree

# Start web app
python start_app.py --predictor=Tree --log_level=INFO

# When done, ask Claude before committing!
```

---

*This file is meant to be living documentation - update as patterns change, but keep it focused and current.*
