#!/usr/bin/env python
"""
Phase 6 Exp 4: Comprehensive ATS (Against the Spread) Evaluation.

Evaluates model predictions against Vegas closing spreads to compute ATS win
rate, profit/loss at -110 odds, and confidence-tiered breakdowns. Supports
LLM prediction files (JSONL), naive baselines, and cross-model comparison.

Usage:
    # Evaluate a single LLM result file
    python scripts/phase6_ats_evaluate.py --results data/exp7/test_gpt-5.4-mini_results.jsonl

    # Evaluate all available models
    python scripts/phase6_ats_evaluate.py --all

    # Quick comparison table across all models
    python scripts/phase6_ats_evaluate.py --compare

Sign convention:
    - Vegas spread: negative = home favored (e.g., -5.5 means home favored by 5.5)
    - Our predicted margin: home_score - away_score (positive = home wins)
    - Home covers when actual_margin > -vegas_spread
    - We pick home to cover when predicted_margin > -vegas_spread
"""

import argparse
import json
import logging
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Load environment
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "NBA_AI_full.sqlite"
EXP7_DIR = PROJECT_ROOT / "data" / "exp7"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Standard -110 odds payout
WIN_PAYOUT = 100 / 1.10  # ~$90.91 profit on a $100 bet
LOSS_COST = 100.0  # $100 lost

# Confidence tiers: difference between our predicted margin and Vegas line
TIER_THRESHOLDS = [
    ("High (>5 pts)", 5.0, float("inf")),
    ("Medium (2-5 pts)", 2.0, 5.0),
    ("Low (<2 pts)", 0.0, 2.0),
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class ATSPick:
    """Single ATS evaluation record."""

    game_id: str
    season: str
    home_team: str
    away_team: str
    predicted_margin: float  # home - away (our prediction)
    actual_margin: float  # home - away (actual)
    vegas_spread: float  # negative = home favored
    pick_home: bool  # True if we pick home to cover
    home_covered: bool  # True if home actually covered
    is_push: bool  # True if exact match (push)
    confidence: float  # |predicted_margin - (-vegas_spread)|


@dataclass
class ATSResults:
    """Aggregated ATS evaluation results."""

    model_name: str
    picks: list = field(default_factory=list)

    @property
    def n_games(self) -> int:
        return len(self.picks)

    @property
    def n_pushes(self) -> int:
        return sum(1 for p in self.picks if p.is_push)

    @property
    def n_decided(self) -> int:
        return self.n_games - self.n_pushes

    @property
    def n_wins(self) -> int:
        return sum(
            1 for p in self.picks if not p.is_push and (p.pick_home == p.home_covered)
        )

    @property
    def n_losses(self) -> int:
        return self.n_decided - self.n_wins

    @property
    def ats_pct(self) -> float:
        return self.n_wins / self.n_decided if self.n_decided > 0 else 0.0

    @property
    def profit(self) -> float:
        """Net P/L at -110 on $100 flat bets."""
        return self.n_wins * WIN_PAYOUT - self.n_losses * LOSS_COST

    @property
    def roi_pct(self) -> float:
        """ROI% = profit / total_wagered * 100."""
        wagered = self.n_decided * LOSS_COST
        return (self.profit / wagered * 100) if wagered > 0 else 0.0

    def record_str(self) -> str:
        return f"{self.n_wins}-{self.n_losses}-{self.n_pushes}"

    def filter(self, predicate) -> "ATSResults":
        """Return new ATSResults with only picks matching predicate."""
        filtered = ATSResults(model_name=self.model_name)
        filtered.picks = [p for p in self.picks if predicate(p)]
        return filtered


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
def get_vegas_spreads(db_path: Path, seasons: list[str] | None = None) -> dict:
    """Load Vegas spreads from database.

    Returns dict: game_id -> {vegas_spread, home_team, away_team, season,
                               actual_home_score, actual_away_score, spread_result}
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    query = """
        SELECT
            g.game_id,
            g.home_team,
            g.away_team,
            g.season,
            tbh.pts AS home_score,
            tba.pts AS away_score,
            COALESCE(b.espn_closing_spread, b.covers_closing_spread,
                     b.espn_current_spread) AS vegas_spread,
            b.spread_result
        FROM Games g
        JOIN Betting b ON g.game_id = b.game_id
        JOIN TeamBox tbh ON g.game_id = tbh.game_id
        JOIN Teams th ON tbh.team_id = th.team_id
            AND th.abbreviation = g.home_team
        JOIN TeamBox tba ON g.game_id = tba.game_id
        JOIN Teams ta ON tba.team_id = ta.team_id
            AND ta.abbreviation = g.away_team
        WHERE g.status = 3
          AND g.season_type = 'Regular Season'
          AND COALESCE(b.espn_closing_spread, b.covers_closing_spread,
                       b.espn_current_spread) IS NOT NULL
    """
    params = []
    if seasons:
        placeholders = ", ".join("?" for _ in seasons)
        query += f" AND g.season IN ({placeholders})"
        params = list(seasons)

    rows = conn.execute(query, params).fetchall()
    conn.close()

    result = {}
    for r in rows:
        result[r["game_id"]] = {
            "vegas_spread": r["vegas_spread"],
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "season": r["season"],
            "actual_home_score": r["home_score"],
            "actual_away_score": r["away_score"],
            "spread_result": r["spread_result"],
        }
    return result


# ---------------------------------------------------------------------------
# Prediction loaders
# ---------------------------------------------------------------------------
def load_exp7_results(path: Path) -> list[dict]:
    """Load Exp 7 LLM prediction results from JSONL file.

    Returns list of dicts with: game_id, predicted_home, predicted_away,
                                actual_home, actual_away, season
    """
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not obj.get("success", True):
                continue
            pred = obj.get("prediction", {})
            if pred.get("home_score") is None or pred.get("away_score") is None:
                continue
            records.append(
                {
                    "game_id": obj["game_id"],
                    "predicted_home": pred["home_score"],
                    "predicted_away": pred["away_score"],
                    "actual_home": obj.get("actual_home_score"),
                    "actual_away": obj.get("actual_away_score"),
                    "season": obj.get("season", ""),
                }
            )
    return records


# ---------------------------------------------------------------------------
# ATS evaluation core
# ---------------------------------------------------------------------------
def evaluate_ats(
    predictions: list[dict],
    vegas_data: dict,
    model_name: str,
) -> ATSResults:
    """Evaluate predictions against Vegas spreads.

    predictions: list of dicts with game_id, predicted_home, predicted_away
    vegas_data: dict from get_vegas_spreads()

    ATS logic:
      - predicted_margin = predicted_home - predicted_away
      - vegas_line = -vegas_spread  (the number the home team must beat)
      - If predicted_margin > vegas_line: pick home to cover
      - If predicted_margin < vegas_line: pick away to cover
      - If predicted_margin == vegas_line: pick is ambiguous (treat as home pick)
      - Home covers if actual_margin > vegas_line (strict >)
      - Push if actual_margin == vegas_line
    """
    results = ATSResults(model_name=model_name)
    skipped = 0

    for pred in predictions:
        gid = pred["game_id"]
        if gid not in vegas_data:
            skipped += 1
            continue

        vd = vegas_data[gid]
        vegas_spread = vd["vegas_spread"]
        vegas_line = -vegas_spread  # Points home must win by to cover

        predicted_margin = pred["predicted_home"] - pred["predicted_away"]

        # Use actual scores from DB (more reliable than prediction file)
        actual_home = vd["actual_home_score"]
        actual_away = vd["actual_away_score"]
        actual_margin = actual_home - actual_away

        # Determine if home actually covered
        is_push = actual_margin == vegas_line
        home_covered = actual_margin > vegas_line

        # Our ATS pick
        pick_home = predicted_margin > vegas_line
        # Tie-break: if predicted_margin == vegas_line exactly, we skip (no edge)
        # But this is extremely rare with float predictions. If it happens,
        # default to home pick.

        confidence = abs(predicted_margin - vegas_line)
        season = pred.get("season") or vd.get("season", "")

        results.picks.append(
            ATSPick(
                game_id=gid,
                season=season,
                home_team=vd["home_team"],
                away_team=vd["away_team"],
                predicted_margin=predicted_margin,
                actual_margin=actual_margin,
                vegas_spread=vegas_spread,
                pick_home=pick_home,
                home_covered=home_covered,
                is_push=is_push,
                confidence=confidence,
            )
        )

    if skipped > 0:
        logger.warning(f"  [{model_name}] Skipped {skipped} games with no Vegas spread")

    return results


def make_naive_predictions(vegas_data: dict, home_advantage: float = 0.0) -> list[dict]:
    """Generate naive baseline predictions.

    home_advantage: predicted margin for all games. 0 = predict tie,
                    3.0 = predict home wins by 3 (typical HCA).
    """
    preds = []
    for gid, vd in vegas_data.items():
        preds.append(
            {
                "game_id": gid,
                "predicted_home": 100 + home_advantage / 2,  # arbitrary base
                "predicted_away": 100 - home_advantage / 2,
                "season": vd["season"],
            }
        )
    return preds


def make_vegas_predictions(vegas_data: dict, seed: int = 42) -> list[dict]:
    """Use Vegas spread as our own prediction.

    Since predicted_margin would exactly equal vegas_line (no edge),
    we add a tiny random perturbation (+/- 0.01) to break ties and
    randomly pick each side ~50% of the time. ATS% should be ~50%
    by market efficiency.
    """
    rng = np.random.default_rng(seed)
    preds = []
    for gid, vd in vegas_data.items():
        # Vegas implied margin for home = -vegas_spread
        implied_margin = -vd["vegas_spread"]
        # Tiny random perturbation to break exact ties (pick each side ~50%)
        epsilon = rng.choice([-0.01, 0.01])
        margin = implied_margin + epsilon
        preds.append(
            {
                "game_id": gid,
                "predicted_home": 100 + margin / 2,
                "predicted_away": 100 - margin / 2,
                "season": vd["season"],
            }
        )
    return preds


def make_random_predictions(vegas_data: dict, seed: int = 42) -> list[dict]:
    """Random ATS picks: predict random margin uniformly around Vegas line."""
    rng = np.random.default_rng(seed)
    preds = []
    for gid, vd in vegas_data.items():
        vegas_line = -vd["vegas_spread"]
        # Random margin: vegas_line + uniform(-15, +15)
        noise = rng.uniform(-15, 15)
        margin = vegas_line + noise
        preds.append(
            {
                "game_id": gid,
                "predicted_home": 100 + margin / 2,
                "predicted_away": 100 - margin / 2,
                "season": vd["season"],
            }
        )
    return preds


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------
def validate_ats_against_db(
    results: ATSResults,
    vegas_data: dict,
    db_path: Path = DB_PATH,
) -> tuple[int, int, int]:
    """Cross-check our ATS logic against the DB spread_result column.

    The spread_result in the DB was computed by Covers using covers_closing_spread.
    We only validate against games where:
    1. Our COALESCE spread equals the Covers spread (same source)
    2. The season's spread_result data looks plausible (W/L ratio between 35-65%)

    Returns (matches, mismatches, skipped).
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Load covers spreads for cross-reference
    rows = conn.execute(
        "SELECT game_id, covers_closing_spread FROM Betting "
        "WHERE spread_result IS NOT NULL"
    ).fetchall()
    covers_spreads = {r["game_id"]: r["covers_closing_spread"] for r in rows}

    # Identify seasons with plausible spread_result data (W% between 35-65%)
    season_stats = conn.execute("""
        SELECT g.season,
               SUM(CASE WHEN b.spread_result = 'W' THEN 1 ELSE 0 END) as wins,
               SUM(CASE WHEN b.spread_result = 'L' THEN 1 ELSE 0 END) as losses,
               COUNT(*) as total
        FROM Games g
        JOIN Betting b ON g.game_id = b.game_id
        WHERE g.status = 3 AND g.season_type = 'Regular Season'
        AND b.spread_result IS NOT NULL
        GROUP BY g.season
    """).fetchall()
    conn.close()

    valid_seasons = set()
    for row in season_stats:
        decided = row["wins"] + row["losses"]
        if decided > 0:
            win_pct = row["wins"] / decided
            if 0.35 <= win_pct <= 0.65:
                valid_seasons.add(row["season"])

    matches = 0
    mismatches = 0
    skipped = 0

    for pick in results.picks:
        db_result = vegas_data[pick.game_id].get("spread_result")
        if db_result is None:
            skipped += 1
            continue

        # Skip seasons with corrupt spread_result data
        if pick.season not in valid_seasons:
            skipped += 1
            continue

        # Only validate when our spread matches the Covers spread
        covers_spread = covers_spreads.get(pick.game_id)
        if covers_spread is None or abs(pick.vegas_spread - covers_spread) > 0.01:
            skipped += 1
            continue

        if pick.is_push:
            if db_result == "P":
                matches += 1
            else:
                mismatches += 1
        elif pick.home_covered:
            if db_result == "W":
                matches += 1
            else:
                mismatches += 1
        else:
            if db_result == "L":
                matches += 1
            else:
                mismatches += 1

    return matches, mismatches, skipped


# ---------------------------------------------------------------------------
# Formatting / display
# ---------------------------------------------------------------------------
def print_header(title: str, width: int = 80):
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def print_ats_summary(results: ATSResults, label: str = ""):
    """Print comprehensive ATS summary for one model."""
    title = label or results.model_name
    print_header(title)

    print(f"\n  Overall ATS Performance")
    print(f"  {'-' * 50}")
    print(f"  Record:           {results.record_str()}")
    print(f"  ATS Win Rate:     {results.ats_pct:.1%}")
    print(f"  Games Evaluated:  {results.n_games}")
    print(f"  Decided (no push):{results.n_decided}")
    print(f"  Pushes:           {results.n_pushes}")
    print()
    print(f"  Betting P/L (flat $100 at -110)")
    print(f"  {'-' * 50}")
    print(f"  Total Wagered:    ${results.n_decided * LOSS_COST:,.0f}")
    print(f"  Net Profit/Loss:  ${results.profit:+,.2f}")
    print(f"  ROI:              {results.roi_pct:+.2f}%")
    breakeven = 100 / (100 + WIN_PAYOUT) * 100  # ~52.38%
    print(f"  Breakeven ATS%:   {breakeven:.2f}%")
    gap = results.ats_pct * 100 - breakeven
    print(f"  vs Breakeven:     {gap:+.2f} pp")

    # --- Confidence tiers ---
    print(f"\n  Confidence Tiers")
    print(f"  {'-' * 60}")
    print(f"  {'Tier':<20s} {'Record':>10s} {'ATS%':>8s} {'Profit':>10s} {'ROI':>8s}")
    print(f"  {'-' * 60}")
    for tier_name, lo, hi in TIER_THRESHOLDS:
        tier = results.filter(
            lambda p, lo=lo, hi=hi: not p.is_push and lo <= p.confidence < hi
        )
        if tier.n_decided == 0:
            print(
                f"  {tier_name:<20s} {'0-0-0':>10s} {'N/A':>8s} {'$0':>10s} {'N/A':>8s}"
            )
        else:
            print(
                f"  {tier_name:<20s} {tier.record_str():>10s} "
                f"{tier.ats_pct:>7.1%} {tier.profit:>+10.0f} "
                f"{tier.roi_pct:>+7.1f}%"
            )
    print(f"  {'-' * 60}")

    # --- Season splits ---
    seasons = sorted(set(p.season for p in results.picks if p.season))
    if len(seasons) > 1:
        print(f"\n  Season Breakdown")
        print(f"  {'-' * 60}")
        print(
            f"  {'Season':<15s} {'Record':>10s} {'ATS%':>8s} {'Profit':>10s} {'ROI':>8s}"
        )
        print(f"  {'-' * 60}")
        for season in seasons:
            ssn = results.filter(lambda p, s=season: p.season == s)
            if ssn.n_decided == 0:
                continue
            print(
                f"  {season:<15s} {ssn.record_str():>10s} "
                f"{ssn.ats_pct:>7.1%} {ssn.profit:>+10.0f} "
                f"{ssn.roi_pct:>+7.1f}%"
            )
        print(f"  {'-' * 60}")

    # --- Home/Away favorite ATS ---
    print(f"\n  Home vs Away Favorite")
    print(f"  {'-' * 60}")
    print(f"  {'Category':<25s} {'Record':>10s} {'ATS%':>8s} {'Games':>7s}")
    print(f"  {'-' * 60}")

    # Home is favorite when spread < 0
    home_fav = results.filter(lambda p: p.vegas_spread < 0)
    away_fav = results.filter(lambda p: p.vegas_spread > 0)
    pick_n = results.filter(lambda p: p.vegas_spread == 0)

    for label, sub in [
        ("Home Favored", home_fav),
        ("Away Favored", away_fav),
        ("Pick'em", pick_n),
    ]:
        if sub.n_decided == 0:
            print(f"  {label:<25s} {'0-0-0':>10s} {'N/A':>8s} {sub.n_games:>7d}")
        else:
            print(
                f"  {label:<25s} {sub.record_str():>10s} "
                f"{sub.ats_pct:>7.1%} {sub.n_games:>7d}"
            )
    print(f"  {'-' * 60}")

    # --- Picked home cover vs picked away cover ---
    print(f"\n  Pick Direction (Our Picks)")
    print(f"  {'-' * 60}")
    print(f"  {'We Picked':<25s} {'Record':>10s} {'ATS%':>8s} {'Games':>7s}")
    print(f"  {'-' * 60}")

    picked_home = results.filter(lambda p: p.pick_home)
    picked_away = results.filter(lambda p: not p.pick_home)

    for label, sub in [("Home to Cover", picked_home), ("Away to Cover", picked_away)]:
        if sub.n_decided == 0:
            print(f"  {label:<25s} {'0-0-0':>10s} {'N/A':>8s} {sub.n_games:>7d}")
        else:
            print(
                f"  {label:<25s} {sub.record_str():>10s} "
                f"{sub.ats_pct:>7.1%} {sub.n_games:>7d}"
            )
    print(f"  {'-' * 60}")

    # --- Favorite vs Underdog picks ---
    print(f"\n  Betting Favorites vs Underdogs (Our Picks)")
    print(f"  {'-' * 60}")
    print(f"  {'Pick Type':<25s} {'Record':>10s} {'ATS%':>8s} {'Games':>7s}")
    print(f"  {'-' * 60}")

    # We pick the favorite when our pick aligns with Vegas favorite
    # Home favorite + we pick home = picking favorite
    # Away favorite + we pick away = picking favorite
    def is_picking_favorite(p):
        if p.vegas_spread < 0 and p.pick_home:
            return True  # Home is fav, we pick home
        if p.vegas_spread > 0 and not p.pick_home:
            return True  # Away is fav, we pick away
        return False

    def is_picking_underdog(p):
        if p.vegas_spread < 0 and not p.pick_home:
            return True  # Home is fav, we pick away (underdog)
        if p.vegas_spread > 0 and p.pick_home:
            return True  # Away is fav, we pick home (underdog)
        return False

    pick_fav = results.filter(is_picking_favorite)
    pick_dog = results.filter(is_picking_underdog)
    pick_even = results.filter(lambda p: p.vegas_spread == 0)

    for label, sub in [
        ("Picking Favorite", pick_fav),
        ("Picking Underdog", pick_dog),
        ("Pick'em", pick_even),
    ]:
        if sub.n_decided == 0:
            print(f"  {label:<25s} {'0-0-0':>10s} {'N/A':>8s} {sub.n_games:>7d}")
        else:
            print(
                f"  {label:<25s} {sub.record_str():>10s} "
                f"{sub.ats_pct:>7.1%} {sub.n_games:>7d}"
            )
    print(f"  {'-' * 60}")

    # --- Spread error stats ---
    margins = np.array([p.predicted_margin for p in results.picks])
    actuals = np.array([p.actual_margin for p in results.picks])
    spreads = np.array([-p.vegas_spread for p in results.picks])

    print(f"\n  Spread Prediction Quality")
    print(f"  {'-' * 50}")
    our_mae = np.mean(np.abs(margins - actuals))
    vegas_mae = np.mean(np.abs(spreads - actuals))
    print(f"  Our Spread MAE:   {our_mae:.2f}")
    print(f"  Vegas Spread MAE: {vegas_mae:.2f}")
    print(f"  Avg |Our - Vegas|:{np.mean(np.abs(margins - spreads)):.2f}")
    print(f"  Our predicted margin (mean):  {margins.mean():+.2f}")
    print(f"  Vegas implied margin (mean):  {spreads.mean():+.2f}")
    print(f"  Actual margin (mean):         {actuals.mean():+.2f}")


def print_comparison_table(all_results: list[ATSResults]):
    """Print side-by-side comparison table of all models."""
    print_header("ATS COMPARISON — ALL MODELS", width=110)

    # Sort by ATS% descending
    all_results_sorted = sorted(all_results, key=lambda r: r.ats_pct, reverse=True)

    print()
    hdr = (
        f"  {'Model':<30s} {'Record':>12s} {'ATS%':>7s} "
        f"{'Profit':>10s} {'ROI%':>8s} {'Games':>7s} "
        f"{'Spread MAE':>10s}"
    )
    sep = f"  {'-' * 100}"
    print(hdr)
    print(sep)

    for r in all_results_sorted:
        margins = np.array([p.predicted_margin for p in r.picks])
        actuals = np.array([p.actual_margin for p in r.picks])
        mae = np.mean(np.abs(margins - actuals)) if len(margins) > 0 else 0.0

        ats_str = f"{r.ats_pct:.1%}"
        profit_str = f"${r.profit:+,.0f}"
        roi_str = f"{r.roi_pct:+.1f}%"

        print(
            f"  {r.model_name:<30s} {r.record_str():>12s} {ats_str:>7s} "
            f"{profit_str:>10s} {roi_str:>8s} {r.n_decided:>7d} "
            f"{mae:>10.2f}"
        )

    print(sep)
    breakeven = 100 / (100 + WIN_PAYOUT) * 100
    print(f"\n  Breakeven ATS% at -110 odds: {breakeven:.2f}%")

    # Confidence tier comparison
    print()
    print_header(
        "CONFIDENCE TIER COMPARISON — HIGH CONFIDENCE (>5 pts from Vegas)", width=110
    )
    print()
    hdr2 = (
        f"  {'Model':<30s} {'Record':>12s} {'ATS%':>7s} "
        f"{'Profit':>10s} {'ROI%':>8s} {'Games':>7s}"
    )
    print(hdr2)
    print(sep)

    for r in all_results_sorted:
        hi = r.filter(lambda p: not p.is_push and p.confidence >= 5.0)
        if hi.n_decided == 0:
            print(f"  {r.model_name:<30s} {'N/A -- no high-confidence picks':>40s}")
            continue
        ats_str = f"{hi.ats_pct:.1%}"
        profit_str = f"${hi.profit:+,.0f}"
        roi_str = f"{hi.roi_pct:+.1f}%"
        print(
            f"  {r.model_name:<30s} {hi.record_str():>12s} {ats_str:>7s} "
            f"{profit_str:>10s} {roi_str:>8s} {hi.n_decided:>7d}"
        )

    print(sep)


# ---------------------------------------------------------------------------
# Main evaluation flows
# ---------------------------------------------------------------------------
def evaluate_single_results(
    results_path: Path,
    vegas_data: dict,
    model_name: str | None = None,
) -> ATSResults:
    """Evaluate a single JSONL results file."""
    if model_name is None:
        model_name = results_path.stem

    logger.info(f"Loading predictions from {results_path}")
    predictions = load_exp7_results(results_path)
    logger.info(f"  Loaded {len(predictions)} predictions")

    results = evaluate_ats(predictions, vegas_data, model_name)
    logger.info(
        f"  Matched {results.n_games} games with Vegas spreads "
        f"(skipped {len(predictions) - results.n_games})"
    )

    # Validate ATS logic against DB spread_result
    matches, mismatches, val_skipped = validate_ats_against_db(
        results, vegas_data, db_path=DB_PATH
    )
    if mismatches > 0:
        logger.error(
            f"  ATS VALIDATION: {mismatches} mismatches out of "
            f"{matches + mismatches} checked (likely Covers data error). "
            f"{val_skipped} skipped (different spread source)."
        )
    else:
        logger.info(
            f"  ATS validation passed: {matches} games checked against "
            f"DB spread_result, 0 mismatches"
            f" ({val_skipped} skipped due to different spread source)"
        )

    return results


def evaluate_all(vegas_data: dict) -> list[ATSResults]:
    """Evaluate all available models."""
    all_results = []

    # --- Exp 7 LLM models ---
    exp7_files = [
        ("GPT-5.4-nano (test)", EXP7_DIR / "test_gpt-5.4-nano_results.jsonl"),
        ("GPT-5.4-mini (test)", EXP7_DIR / "test_gpt-5.4-mini_results.jsonl"),
        ("GPT-5.4 (test)", EXP7_DIR / "test_gpt-5.4_results.jsonl"),
    ]

    # Also check for val results
    val_nano = EXP7_DIR / "val_gpt-5.4-nano_results.jsonl"
    if val_nano.exists():
        preds = load_exp7_results(val_nano)
        if len(preds) > 10:  # Only include if meaningful number
            exp7_files.append(("GPT-5.4-nano (val)", val_nano))

    for name, path in exp7_files:
        if path.exists():
            try:
                results = evaluate_single_results(path, vegas_data, model_name=name)
                if results.n_games > 0:
                    all_results.append(results)
            except Exception as e:
                logger.error(f"Failed to evaluate {name}: {e}")

    # --- Phase 3/4 model predictions from data/phase6/ ---
    phase6_dir = PROJECT_ROOT / "data" / "phase6"
    if phase6_dir.exists():
        for pred_file in sorted(phase6_dir.glob("test_*_predictions.jsonl")):
            # Extract model name from filename: test_{model_name}_predictions.jsonl
            stem = pred_file.stem  # e.g. test_phase3_exp4_interaction_predictions
            model_name = stem.replace("test_", "").replace("_predictions", "")
            display_name = (
                model_name.replace("_", " ")
                .replace("phase3 ", "P3 ")
                .replace("gen ", "P4 ")
            )
            try:
                results = evaluate_single_results(
                    pred_file, vegas_data, model_name=display_name
                )
                if results.n_games > 0:
                    all_results.append(results)
            except Exception as e:
                logger.error(f"Failed to evaluate {display_name}: {e}")

    # --- Naive baselines ---
    # Predict spread=0 (every game is a toss-up)
    naive_preds = make_naive_predictions(vegas_data, home_advantage=0.0)
    naive_results = evaluate_ats(naive_preds, vegas_data, "Naive (predict 0)")
    all_results.append(naive_results)

    # Predict home +3 (average HCA)
    hca_preds = make_naive_predictions(vegas_data, home_advantage=3.0)
    hca_results = evaluate_ats(hca_preds, vegas_data, "Naive (home +3)")
    all_results.append(hca_results)

    # --- Vegas as predictor (should be ~50%) ---
    vegas_preds = make_vegas_predictions(vegas_data)
    vegas_results = evaluate_ats(vegas_preds, vegas_data, "Vegas (self-predict)")
    all_results.append(vegas_results)

    # --- Random baseline ---
    random_preds = make_random_predictions(vegas_data, seed=42)
    random_results = evaluate_ats(random_preds, vegas_data, "Random (seed=42)")
    all_results.append(random_results)

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 6 Exp 4: ATS Evaluation Against Vegas Spreads",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate a single result file
    python scripts/phase6_ats_evaluate.py --results data/exp7/test_gpt-5.4-mini_results.jsonl

    # Evaluate all available models
    python scripts/phase6_ats_evaluate.py --all

    # Quick comparison table
    python scripts/phase6_ats_evaluate.py --compare

    # Filter to specific seasons
    python scripts/phase6_ats_evaluate.py --all --seasons 2024-2025 2025-2026
        """,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--results",
        type=Path,
        help="Path to a JSONL results file (e.g., from Exp 7 LLM predictions)",
    )
    mode.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all available models (LLM + baselines)",
    )
    mode.add_argument(
        "--compare",
        action="store_true",
        help="Print comparison table across all models",
    )

    parser.add_argument(
        "--seasons",
        nargs="+",
        type=str,
        default=None,
        help="Filter to specific seasons (e.g., 2024-2025 2025-2026)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DB_PATH,
        help=f"Path to SQLite database (default: {DB_PATH})",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Override model name for --results mode",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Validate DB exists
    if not args.db.exists():
        logger.error(f"Database not found: {args.db}")
        return 1

    # Load Vegas spreads
    logger.info(f"Loading Vegas spreads from {args.db}")
    vegas_data = get_vegas_spreads(args.db, seasons=args.seasons)
    logger.info(f"  Loaded {len(vegas_data)} games with Vegas spreads")

    # Summarize spread coverage by season
    season_counts: dict[str, int] = {}
    for vd in vegas_data.values():
        s = vd["season"]
        season_counts[s] = season_counts.get(s, 0) + 1
    for s in sorted(season_counts):
        logger.info(f"    {s}: {season_counts[s]} games")

    # --- Single file mode ---
    if args.results:
        if not args.results.exists():
            logger.error(f"Results file not found: {args.results}")
            return 1

        results = evaluate_single_results(
            args.results, vegas_data, model_name=args.model_name
        )
        print_ats_summary(results)

        # Also show baselines for context
        print_header("BASELINES FOR COMPARISON")
        vegas_preds = make_vegas_predictions(vegas_data)
        # But filter to only the games in our results
        result_game_ids = {p.game_id for p in results.picks}
        filtered_vegas = {
            gid: vd for gid, vd in vegas_data.items() if gid in result_game_ids
        }

        vegas_r = evaluate_ats(
            make_vegas_predictions(filtered_vegas),
            filtered_vegas,
            "Vegas (same games)",
        )
        naive_r = evaluate_ats(
            make_naive_predictions(filtered_vegas, 0.0),
            filtered_vegas,
            "Naive predict=0 (same games)",
        )
        random_r = evaluate_ats(
            make_random_predictions(filtered_vegas, seed=42),
            filtered_vegas,
            "Random (same games)",
        )

        print()
        hdr = (
            f"  {'Model':<35s} {'Record':>12s} {'ATS%':>7s} "
            f"{'Profit':>10s} {'ROI%':>8s}"
        )
        sep = f"  {'-' * 80}"
        print(hdr)
        print(sep)
        for r in [results, vegas_r, naive_r, random_r]:
            ats_str = f"{r.ats_pct:.1%}"
            profit_str = f"${r.profit:+,.0f}"
            roi_str = f"{r.roi_pct:+.1f}%"
            print(
                f"  {r.model_name:<35s} {r.record_str():>12s} {ats_str:>7s} "
                f"{profit_str:>10s} {roi_str:>8s}"
            )
        print(sep)
        return 0

    # --- All / Compare mode ---
    all_results = evaluate_all(vegas_data)

    if args.compare:
        print_comparison_table(all_results)
    else:
        # Full details for each model
        for results in all_results:
            print_ats_summary(results)
        # Then comparison
        print_comparison_table(all_results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
