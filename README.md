<img src='docs/images/nba_ai_header_15x6.png' alt='NBA AI'/>

# NBA AI

## Table of Contents
* [Project Overview](#project-overview)
    * [Architecture](#architecture)
    * [Guiding Principles](#guiding-principles)
* [Web App](#web-app)
* [Prediction Engines](#prediction-engines)
* [Quick Start](#quick-start)
* [Development Status](#development-status)

## Project Overview

#### Using AI to predict the outcomes of NBA games.

This project predicts NBA game spreads and winners using a combination of deep learning models and traditional ML. Unlike my previous project, [NBA Betting](https://github.com/NBA-Betting/NBA_Betting/tree/main), which focused on extensive data collection and feature engineering, this project focuses on building advanced prediction models that learn directly from play-by-play data, box scores, and player tracking — minimizing manual feature engineering in favor of letting the models find the signal.

The system runs a fully automated daily pipeline that collects game data, updates player ability models, and generates pre-game predictions for all upcoming games using 7 different prediction engines. A Flask web app displays predictions alongside Vegas opening lines, with a dashboard for tracking model performance over time.

### Architecture

The system has three main layers:

* **Data Collection** — Automatically collects game data, box scores, play-by-play, injury reports, and betting lines from the NBA API and ESPN into a SQLite database.

* **Prediction Models** — Multiple prediction engines analyze the data and generate pre-game spread and winner predictions. Includes custom deep learning models, traditional ML models, and an ensemble.

* **Web App & Dashboard** — Displays games with predictions alongside Vegas lines and actual results. A dashboard tracks each model's performance over time.

![Project Flowchart](docs/images/project_flowchart.png)

### Guiding Principles

![Project Guiding Principles](docs/images/guiding_principles.png)

- **Time Series Data Inclusive:** Incorporating the sequential nature of events in games and across seasons, recognizing the significance of order and timing in the NBA.
- **Minimal Data Collection:** Streamlining data sourcing to the essentials, aiming for maximum impact with minimal data, thereby reducing time and resource investment.
- **Wider Applicability:** Extending the scope to cover more comprehensive outcomes, moving beyond standard predictions like point spreads or over/unders.
- **Advanced Modeling System:** Developing a system that is not only a learning tool but also potentially novel compared to the methods used by odds setters.
- **Minimal Human Decisions:** Reducing the reliance on human decision-making to minimize errors and the limitations of individual expertise.

## Web App

![Web App Home Page](docs/images/web_app_homepage.png)
![Web App Game Details](docs/images/web_app_game_details.png)

## Prediction Engines

Currently, there are a few basic prediction engines used to predict the outcomes of NBA games. These serve as placeholders for the more advanced DL and GenAI engines that will be implemented in the future. The current engines make pre-game predictions for home and away scores using ML models. These predictions are then used to calculate the win percentage and margin for the home team. Updated (after game start) predictions are based on a combination of the current game score, time remaining, and the pre-game predictions.

### Current Prediction Engines

- **Baseline**: A simple predictor that predicts scores based on teams' PPG and opponents' PPG.
- **Linear**: Ridge Regression model using 34 rolling average features from prior game states.
- **Tree**: XGBoost model using the same features as the Linear model (default, best performance).
- **MLP** *(optional)*: PyTorch MLP model - requires uncommenting PyTorch in requirements.txt.
- **Ensemble** *(optional)*: Weighted average of Linear (30%), Tree (40%), and MLP (30%) - requires PyTorch.


### Performance Metrics

The current metrics are based on pre-game predictions for the home and away team scores, along with downstream metrics such as win percentage and margin. These simple predictors currently outperform the baseline predictor.

In the future, a more challenging baseline based on the Vegas spread will be added when the DL and GenAI models are implemented.

![Prediction Engine Performance Metrics](docs/images/predictor_performance.png)

## Quick Start

### Requirements

- Python 3.10+
- ~2GB disk space (database + models + dependencies)

### Installation

```bash
# Clone the repository
git clone https://github.com/NBA-Betting/NBA_AI.git
cd NBA_AI

# Run automated setup
python setup.py
```

The setup script will:

1. Create a virtual environment
2. Install all dependencies
3. Download the database and trained models from GitHub Releases
4. Create your `.env` configuration file
5. Verify the installation

### Running the Web App

```bash
# Activate the virtual environment
source venv/bin/activate

# Start the web app
python start_app.py
```

Visit `http://localhost:5000` to view games and predictions.

### Command Line Options

```bash
# Use a specific predictor
python start_app.py --predictor=Tree

# Enable debug mode
python start_app.py --debug

# Set log level
python start_app.py --log_level=DEBUG
```

Available predictors: `Baseline`, `Linear`, `Tree` (default), `MLP`*, `Ensemble`*

*Requires PyTorch - uncomment in requirements.txt

---

## Development Status

**This project is in active development.**

The core data pipeline and prediction engines are functional. The focus is now on building advanced DL/GenAI prediction engines using play-by-play data.

### Disclaimer

This is a personal side project provided "as is" with no guarantees of quality, functionality, or ongoing maintenance. I've vibe-coded much of this release and while I'll try to address issues, I can't promise timely responses or fixes.

**For production or commercial use**: Consider using [SportsRadar](https://sportradar.com/), the official NBA data partner. Their API would greatly simplify data management compared to scraping the NBA Stats API. I use this approach only because I can't justify the cost for a personal project.

### Historical Data

The default setup downloads only the current season (2025-2026, ~1,300 games). A development database with 3 seasons (2023-2024 through 2025-2026, ~4,100 games total) is available from [GitHub Releases](https://github.com/NBA-Betting/NBA_AI/releases).

To use it:

1. Download `NBA_AI_dev.zip` from the latest release
2. Extract to `data/NBA_AI_dev.sqlite`
3. Update your `.env`:

```bash
DATABASE_PATH=data/NBA_AI_dev.sqlite
```

### Usage Notes

- **First run for a date**: When viewing a date for the first time, the app fetches data from the NBA API. This initial update may take a few seconds per game. Subsequent views are instant since data is cached in the database.

- **Season restrictions**: By default, the web app allows seasons 2023-2024 through 2025-2026. To restrict or expand this, modify `valid_seasons` in `config.yaml`.

### Technical Notes

- Default focus: 2025-2026 season (current season for public release)
- Database: SQLite with complete pipeline (Schedule → Players → Injuries → Betting → PBP → GameStates → Boxscores → Features → Predictions)
- Built with Python, Flask, SQLite, scikit-learn, XGBoost, and nba_api (PyTorch optional for MLP/Ensemble)


