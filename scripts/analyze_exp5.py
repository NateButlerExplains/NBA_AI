"""Diagnostic analysis for Phase 4 Exp 5 generative model.

Produces plots and reports across 6 analysis modules:
  1. Rollout distribution analysis (normality, skewness, kurtosis, mode vs mean)
  2. Error trajectory analysis (cumulative error vs step, per-quarter MAE)
  3. Convergence diagnostic (MAE vs N_rollouts elbow curve)
  4. Scoring pattern realism (event type proportions, scoring rate by quarter)
  5. Failure analysis (worst-20, confidence vs accuracy, systematic biases)
  6. Context conditioning (CFG: conditional vs unconditional KL divergence)

All outputs saved to timestamped directory: analysis/exp5_{YYYYMMDD_HHMMSS}/

Usage::

    python scripts/analyze_exp5.py \\
        --config configs/generative/exp5_full_context.yaml \\
        --checkpoint checkpoints/generative/gen_exp5_full_context/best.pt \\
        --split test --n-rollouts 100 --max-games 200

    # Quick pass
    python scripts/analyze_exp5.py \\
        --config configs/generative/exp5_full_context.yaml \\
        --checkpoint checkpoints/generative/gen_exp5_full_context/best.pt \\
        --max-games 50 --n-rollouts 50 --skip-convergence --skip-cfg
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats as scipy_stats
from torch.utils.data import DataLoader

from src.generative.config import GenerativeExperimentConfig
from src.generative.dataset import GenerativeDataset, generative_collate
from src.generative.inference.rollout import AutoregressiveRollout
from src.generative.models.generative_model import GenerativeModel

logger = logging.getLogger(__name__)

# 6-class score event names
EVENT_NAMES = ["h+1", "h+2", "h+3", "a+1", "a+2", "a+3"]
# Deltas per class: (home_delta, away_delta)
EVENT_DELTAS = [(1, 0), (2, 0), (3, 0), (0, 1), (0, 2), (0, 3)]


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def collect_rollout_data(
    rollout_engine: AutoregressiveRollout,
    loader: DataLoader,
    device: torch.device,
    n_rollouts: int,
    temperature: float,
    max_games: int,
) -> list[dict]:
    """Run rollouts and collect per-game data for all analyses.

    Returns list of dicts (one per game) with:
        game_id, true_margin, true_win, spread_mean, spread_std, win_prob,
        outcome_spread, rollout_spreads, rollout_home_scores, rollout_away_scores,
        actual_events (np.ndarray), actual_progress (np.ndarray), actual_states (np.ndarray).
    """
    results = []

    for i, batch in enumerate(loader):
        if i >= max_games:
            break
        if batch is None:
            continue

        batch_dev = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        true_margin = batch_dev["final_margin"].item()
        true_win = 1 if true_margin > 0 else 0

        result = rollout_engine.rollout(
            batch_dev,
            n_rollouts=n_rollouts,
            temperature=temperature,
        )

        # Extract ground truth from the batch (before device move is fine, index 0)
        score_events = batch["score_events"][0].numpy()
        states = batch["states"][0].numpy()

        game_data = {
            "game_id": f"game_{i}",
            "true_margin": true_margin,
            "true_win": true_win,
            "spread_mean": result["spread_mean"],
            "spread_std": result["spread_std"],
            "win_prob": result["win_prob"],
            "outcome_spread": result.get("outcome_spread"),
            "rollout_spreads": [
                h - a for h, a in zip(result["home_scores"], result["away_scores"])
            ],
            "rollout_home_scores": result["home_scores"],
            "rollout_away_scores": result["away_scores"],
            "actual_events": score_events,
            "actual_progress": states[:, 2],  # game_progress column
            "actual_states": states,
        }
        results.append(game_data)

        if (i + 1) % 50 == 0:
            preds = [r["spread_mean"] for r in results]
            trues = [r["true_margin"] for r in results]
            running_mae = np.mean(np.abs(np.array(preds) - np.array(trues)))
            logger.info(
                f"  Collected {i + 1}/{max_games}: running MAE = {running_mae:.2f}"
            )

    return results


# ---------------------------------------------------------------------------
# 1. Rollout Distribution Analysis
# ---------------------------------------------------------------------------


def analyze_distributions(games: list[dict], output_dir: Path) -> None:
    """Normality test, skewness, kurtosis, representative histograms, mode vs mean."""
    print(f"\n{'=' * 60}")
    print("Analysis 1: Rollout Distribution")
    print(f"{'=' * 60}")

    shapiro_pvals = []
    skewnesses = []
    kurtoses = []
    mode_better = 0
    mean_better = 0

    for g in games:
        arr = np.array(g["rollout_spreads"])
        if len(arr) < 8:
            continue

        skewnesses.append(float(scipy_stats.skew(arr)))
        kurtoses.append(float(scipy_stats.kurtosis(arr)))

        # Shapiro-Wilk (capped at 5000 samples)
        try:
            _, pval = scipy_stats.shapiro(arr[:5000])
            shapiro_pvals.append(pval)
        except Exception:
            pass

        # Mode vs mean comparison
        counts, edges = np.histogram(arr, bins=20)
        peak = np.argmax(counts)
        mode_est = (edges[peak] + edges[peak + 1]) / 2.0
        mean_est = float(np.mean(arr))
        true = g["true_margin"]

        if abs(mode_est - true) < abs(mean_est - true):
            mode_better += 1
        else:
            mean_better += 1

    pvals = np.array(shapiro_pvals)
    non_normal_pct = float(np.mean(pvals < 0.05)) * 100 if len(pvals) > 0 else 0.0

    print(f"  Games analyzed:  {len(skewnesses)}")
    print(f"  Non-normal (p<0.05):  {non_normal_pct:.1f}%")
    print(f"  Skewness:  mean={np.mean(skewnesses):.3f}, std={np.std(skewnesses):.3f}")
    print(f"  Kurtosis:  mean={np.mean(kurtoses):.3f}, std={np.std(kurtoses):.3f}")
    print(f"  Mode vs Mean:  mode better {mode_better}, mean better {mean_better}")

    # --- Select 6 representative games ---
    abs_margins = np.array([abs(g["true_margin"]) for g in games])
    abs_errors = np.array([abs(g["spread_mean"] - g["true_margin"]) for g in games])
    stds = np.array([g["spread_std"] for g in games])

    reps = {}
    reps["Close game"] = games[int(np.argmin(abs_margins))]
    reps["Blowout"] = games[int(np.argmax(abs_margins))]
    reps["Lowest std"] = games[int(np.argmin(stds))]
    reps["Highest std"] = games[int(np.argmax(stds))]
    reps["Median error"] = games[int(np.argsort(abs_errors)[len(abs_errors) // 2])]
    reps["Worst error"] = games[int(np.argmax(abs_errors))]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (label, g) in zip(axes.flat, reps.items()):
        spreads = np.array(g["rollout_spreads"])
        ax.hist(spreads, bins=25, alpha=0.7, color="steelblue", edgecolor="white")
        ax.axvline(
            g["true_margin"],
            color="red",
            ls="--",
            lw=2,
            label=f"True: {g['true_margin']:.1f}",
        )
        ax.axvline(
            g["spread_mean"],
            color="green",
            ls="-",
            lw=2,
            label=f"Mean: {g['spread_mean']:.1f}",
        )
        ax.set_title(f"{label} (std={g['spread_std']:.1f})")
        ax.legend(fontsize=7)
        ax.set_xlabel("Spread")

    plt.suptitle("Rollout Spread Distributions (6 Representative Games)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "rollout_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Distribution stats summary plot ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].hist(
        skewnesses, bins=30, edgecolor="black", alpha=0.7, color="steelblue"
    )
    axes[0, 0].axvline(0, color="red", ls="--")
    axes[0, 0].set_title("Skewness Distribution")
    axes[0, 0].set_xlabel("Skewness")

    axes[0, 1].hist(kurtoses, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
    axes[0, 1].axvline(0, color="red", ls="--")
    axes[0, 1].set_title("Excess Kurtosis Distribution")
    axes[0, 1].set_xlabel("Excess Kurtosis")

    if shapiro_pvals:
        axes[1, 0].hist(pvals, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
        axes[1, 0].axvline(0.05, color="red", ls="--", label="p=0.05")
        axes[1, 0].set_title(
            f"Shapiro-Wilk p-values ({100 - non_normal_pct:.0f}% normal)"
        )
        axes[1, 0].set_xlabel("p-value")
        axes[1, 0].legend()
    else:
        axes[1, 0].text(0.5, 0.5, "No data", ha="center", va="center")

    # Mode-Mean diff
    mode_mean_diffs = []
    for g in games:
        arr = np.array(g["rollout_spreads"])
        if len(arr) < 8:
            continue
        counts, edges = np.histogram(arr, bins=20)
        peak = np.argmax(counts)
        mode_est = (edges[peak] + edges[peak + 1]) / 2.0
        mode_mean_diffs.append(mode_est - np.mean(arr))
    axes[1, 1].hist(
        mode_mean_diffs, bins=30, edgecolor="black", alpha=0.7, color="steelblue"
    )
    axes[1, 1].axvline(0, color="red", ls="--")
    axes[1, 1].set_title("Mode - Mean (per game)")
    axes[1, 1].set_xlabel("Points")

    plt.tight_layout()
    plt.savefig(output_dir / "distribution_stats.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: rollout_distributions.png, distribution_stats.png")


# ---------------------------------------------------------------------------
# 2. Error Trajectory Analysis
# ---------------------------------------------------------------------------


def analyze_error_trajectory(games: list[dict], output_dir: Path) -> None:
    """Cumulative error vs step and per-quarter MAE breakdown."""
    print(f"\n{'=' * 60}")
    print("Analysis 2: Error Trajectory")
    print(f"{'=' * 60}")

    # Reconstruct margin-vs-time from actual events, compute error to final
    quarter_errors = {1: [], 2: [], 3: [], 4: []}
    # Collect (progress, abs_error) pairs for averaged curve
    progress_error_pairs = []

    for g in games:
        events = g["actual_events"]
        progress = g["actual_progress"]
        final_margin = g["true_margin"]

        # Reconstruct cumulative scores
        home_cum = [0.0]
        away_cum = [0.0]
        for ev in events:
            h, a = home_cum[-1], away_cum[-1]
            if 0 <= ev <= 5:
                dh, da = EVENT_DELTAS[ev]
                home_cum.append(h + dh)
                away_cum.append(a + da)
            elif ev == 6:
                # game_end (compressed 7-class)
                home_cum.append(h)
                away_cum.append(a)
            elif 1 <= ev <= 6:
                # Full 7-class: 0=none, 1-3=home, 4-6=away
                if ev <= 3:
                    home_cum.append(h + ev)
                    away_cum.append(a)
                else:
                    home_cum.append(h)
                    away_cum.append(a + (ev - 3))
            else:
                home_cum.append(h)
                away_cum.append(a)

        home_arr = np.array(home_cum)
        away_arr = np.array(away_cum)
        margin_arr = home_arr - away_arr

        for t in range(min(len(progress), len(margin_arr))):
            p = progress[t]
            err = abs(margin_arr[t] - final_margin)
            progress_error_pairs.append((p, err))

            # Assign to quarter
            if p < 0.25:
                quarter_errors[1].append(err)
            elif p < 0.50:
                quarter_errors[2].append(err)
            elif p < 0.75:
                quarter_errors[3].append(err)
            else:
                quarter_errors[4].append(err)

    # Per-quarter MAE
    print("  Per-quarter MAE (|margin_t - final_margin|):")
    q_maes = []
    for q in [1, 2, 3, 4]:
        if quarter_errors[q]:
            mae = float(np.mean(quarter_errors[q]))
            q_maes.append(mae)
            print(f"    Q{q}: {mae:.2f} ({len(quarter_errors[q])} positions)")
        else:
            q_maes.append(0.0)
            print(f"    Q{q}: no data")

    # Error growth pattern
    if q_maes[0] > 0 and q_maes[-1] > 0:
        ratio = q_maes[0] / max(q_maes[-1], 0.01)
        print(f"  Q1/Q4 error ratio: {ratio:.2f}")
        if ratio > 2.0:
            print("  >> Error drops substantially Q1->Q4 (early uncertainty, expected)")
        else:
            print("  >> Error relatively flat across quarters")

    # Binned progress-vs-error curve
    n_bins = 20
    bin_edges = np.linspace(0, 1, n_bins + 1)
    binned_mae = []
    bin_centers = []
    for j in range(n_bins):
        lo, hi = bin_edges[j], bin_edges[j + 1]
        errs = [e for p, e in progress_error_pairs if lo <= p < hi]
        if errs:
            binned_mae.append(float(np.mean(errs)))
        else:
            binned_mae.append(0.0)
        bin_centers.append((lo + hi) / 2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: cumulative error curve
    ax = axes[0]
    ax.plot(bin_centers, binned_mae, "o-", color="steelblue", lw=2)
    ax.set_xlabel("Game Progress")
    ax.set_ylabel("Mean |margin_t - final_margin|")
    ax.set_title("Score Margin Error vs Game Progress")
    ax.grid(True, alpha=0.3)

    # Right: per-quarter bars
    ax = axes[1]
    ax.bar([1, 2, 3, 4], q_maes, color="steelblue", edgecolor="white")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Mean |margin_t - final_margin|")
    ax.set_title("Per-Quarter MAE")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])

    plt.tight_layout()
    plt.savefig(output_dir / "error_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: error_trajectory.png")


# ---------------------------------------------------------------------------
# 3. Convergence Diagnostic
# ---------------------------------------------------------------------------


def analyze_convergence(
    rollout_engine: AutoregressiveRollout,
    loader: DataLoader,
    device: torch.device,
    temperature: float,
    max_games: int,
    output_dir: Path,
) -> None:
    """Plot MAE vs N_rollouts, find elbow point."""
    print(f"\n{'=' * 60}")
    print("Analysis 3: Convergence Diagnostic")
    print(f"{'=' * 60}")

    n_values = [1, 5, 10, 25, 50, 100, 200]
    max_n = max(n_values)
    conv_games = min(max_games, 100)

    # Run one pass with max_n rollouts, then subsample
    all_spreads = []  # list of lists: per-game rollout spreads
    all_true = []

    for i, batch in enumerate(loader):
        if i >= conv_games:
            break
        if batch is None:
            continue

        batch_dev = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        result = rollout_engine.rollout(
            batch_dev,
            n_rollouts=max_n,
            temperature=temperature,
        )

        spreads = [h - a for h, a in zip(result["home_scores"], result["away_scores"])]
        all_spreads.append(spreads)
        all_true.append(batch_dev["final_margin"].item())

    if not all_spreads:
        print("  No data collected. Skipping.")
        return

    true_arr = np.array(all_true)

    # Compute MAE for each N by subsampling the first N rollouts
    mae_by_n = {}
    for n in n_values:
        preds = []
        for spreads in all_spreads:
            subset = np.array(spreads[:n])
            preds.append(float(np.mean(subset)))
        pred_arr = np.array(preds)
        mae_by_n[n] = float(np.mean(np.abs(pred_arr - true_arr[: len(pred_arr)])))

    print("  N_rollouts -> MAE:")
    for n in n_values:
        print(f"    N={n:3d}: {mae_by_n[n]:.2f}")

    # Find elbow
    for i in range(2, len(n_values)):
        improvement = mae_by_n[n_values[i - 1]] - mae_by_n[n_values[i]]
        if improvement < 0.1:
            print(f"  Elbow: ~{n_values[i]} rollouts (improvement < 0.1)")
            break
    else:
        print(f"  No clear elbow (MAE still improving at N={n_values[-1]})")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ns = list(mae_by_n.keys())
    maes = [mae_by_n[n] for n in ns]
    ax.plot(ns, maes, "o-", color="steelblue", lw=2, markersize=8)
    ax.set_xlabel("Number of Rollouts")
    ax.set_ylabel("Spread MAE")
    ax.set_title("Convergence: MAE vs N_rollouts")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "convergence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: convergence.png")


# ---------------------------------------------------------------------------
# 4. Scoring Pattern Realism
# ---------------------------------------------------------------------------


def analyze_scoring_patterns(games: list[dict], output_dir: Path) -> None:
    """Event type proportions (actual), scoring rate by quarter, generated vs actual totals."""
    print(f"\n{'=' * 60}")
    print("Analysis 4: Scoring Pattern Realism")
    print(f"{'=' * 60}")

    # Actual event proportions (6-class)
    actual_counts_6 = np.zeros(6)
    actual_by_quarter = {q: np.zeros(6) for q in [1, 2, 3, 4]}
    actual_events_per_quarter = {1: 0, 2: 0, 3: 0, 4: 0}
    n_games_counted = 0

    for g in games:
        events = g["actual_events"]
        progress = g["actual_progress"]
        n_games_counted += 1

        for t, ev in enumerate(events):
            if 0 <= ev <= 5:
                actual_counts_6[ev] += 1
                p = progress[min(t, len(progress) - 1)]
                q = min(int(p * 4) + 1, 4) if p < 1.0 else 4
                actual_by_quarter[q][ev] += 1
                actual_events_per_quarter[q] += 1

    total_events = actual_counts_6.sum()
    if total_events > 0:
        actual_props = actual_counts_6 / total_events
    else:
        actual_props = np.ones(6) / 6

    print(
        f"  Actual event proportions ({int(total_events)} total events, {n_games_counted} games):"
    )
    for i, name in enumerate(EVENT_NAMES):
        print(f"    {name}: {actual_props[i]:.4f} ({int(actual_counts_6[i])} events)")

    print(f"\n  Scoring events per game by quarter (actual):")
    for q in [1, 2, 3, 4]:
        rate = actual_events_per_quarter[q] / max(n_games_counted, 1)
        print(f"    Q{q}: {rate:.1f} events/game")

    # Generated vs actual final scores
    gen_home = [np.mean(g["rollout_home_scores"]) for g in games]
    gen_away = [np.mean(g["rollout_away_scores"]) for g in games]

    # Reconstruct actual final scores from events
    actual_home = []
    actual_away = []
    for g in games:
        h, a = 0.0, 0.0
        for ev in g["actual_events"]:
            if 0 <= ev <= 5:
                dh, da = EVENT_DELTAS[ev]
                h += dh
                a += da
            elif 1 <= ev <= 6 and ev != 6:
                # Full 7-class fallback
                if ev <= 3:
                    h += ev
                else:
                    a += ev - 3
        actual_home.append(h)
        actual_away.append(a)

    print(f"\n  Score comparison (mean per game):")
    print(
        f"    Home: generated {np.mean(gen_home):.1f}, actual {np.mean(actual_home):.1f}"
    )
    print(
        f"    Away: generated {np.mean(gen_away):.1f}, actual {np.mean(actual_away):.1f}"
    )
    print(
        f"    Total: generated {np.mean(gen_home) + np.mean(gen_away):.1f}, actual {np.mean(actual_home) + np.mean(actual_away):.1f}"
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Left: event type proportions
    ax = axes[0]
    x = np.arange(6)
    ax.bar(x, actual_props, color="steelblue", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(EVENT_NAMES)
    ax.set_ylabel("Proportion")
    ax.set_title("Actual Score Event Proportions")

    # Middle: scoring rate by quarter
    ax = axes[1]
    rates = [
        actual_events_per_quarter[q] / max(n_games_counted, 1) for q in [1, 2, 3, 4]
    ]
    ax.bar([1, 2, 3, 4], rates, color="steelblue", edgecolor="white")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Events / game")
    ax.set_title("Scoring Rate by Quarter (Actual)")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])

    # Right: generated vs actual score distributions
    ax = axes[2]
    bins = np.arange(60, 170, 5)
    ax.hist(actual_home, bins=bins, alpha=0.4, label="Actual home", color="red")
    ax.hist(gen_home, bins=bins, alpha=0.4, label="Generated home", color="blue")
    ax.set_xlabel("Team Points")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution: Actual vs Generated")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "scoring_patterns.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: scoring_patterns.png")


# ---------------------------------------------------------------------------
# 5. Failure Analysis
# ---------------------------------------------------------------------------


def analyze_failures(games: list[dict], output_dir: Path) -> None:
    """Worst-20 table, confidence vs accuracy scatter, systematic biases."""
    print(f"\n{'=' * 60}")
    print("Analysis 5: Failure Analysis")
    print(f"{'=' * 60}")

    pred = np.array([g["spread_mean"] for g in games])
    true = np.array([g["true_margin"] for g in games])
    stds = np.array([g["spread_std"] for g in games])
    errors = pred - true
    abs_errors = np.abs(errors)

    # Worst 20
    worst_idx = np.argsort(abs_errors)[-20:][::-1]
    print(f"  Worst 20 games by absolute error:")
    print(
        f"    {'#':>3s}  {'Game ID':<14s}  {'True':>6s}  {'Pred':>6s}  {'Err':>6s}  {'Std':>5s}  {'WinP':>5s}"
    )
    print(
        f"    {'---':>3s}  {'-' * 14}  {'-' * 6}  {'-' * 6}  {'-' * 6}  {'-' * 5}  {'-' * 5}"
    )
    for rank, idx in enumerate(worst_idx, 1):
        g = games[idx]
        print(
            f"    {rank:3d}  {g['game_id']:<14s}  "
            f"{g['true_margin']:6.1f}  {g['spread_mean']:6.1f}  "
            f"{abs_errors[idx]:6.1f}  {g['spread_std']:5.1f}  "
            f"{g['win_prob']:5.2f}"
        )

    # Systematic biases
    mean_signed = float(np.mean(errors))
    print(f"\n  Mean signed error (home bias): {mean_signed:+.2f}")

    home_wins = [g for g in games if g["true_margin"] > 0]
    away_wins = [g for g in games if g["true_margin"] < 0]

    if home_wins:
        hw_mae = np.mean([abs(g["spread_mean"] - g["true_margin"]) for g in home_wins])
        print(f"  Home win MAE ({len(home_wins)} games): {hw_mae:.2f}")
    if away_wins:
        aw_mae = np.mean([abs(g["spread_mean"] - g["true_margin"]) for g in away_wins])
        print(f"  Away win MAE ({len(away_wins)} games): {aw_mae:.2f}")

    close = [g for g in games if abs(g["true_margin"]) <= 5]
    blowouts = [g for g in games if abs(g["true_margin"]) >= 20]
    if close:
        cg_mae = np.mean([abs(g["spread_mean"] - g["true_margin"]) for g in close])
        print(f"  Close (|m|<=5, n={len(close)}): MAE = {cg_mae:.2f}")
    if blowouts:
        bo_mae = np.mean([abs(g["spread_mean"] - g["true_margin"]) for g in blowouts])
        print(f"  Blowouts (|m|>=20, n={len(blowouts)}): MAE = {bo_mae:.2f}")

    # Confidence vs accuracy scatter + signed error histogram
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Scatter: rollout std vs absolute error
    ax = axes[0]
    ax.scatter(stds, abs_errors, alpha=0.4, s=15, color="steelblue")
    ax.set_xlabel("Rollout Std (uncertainty)")
    ax.set_ylabel("Absolute Error")
    ax.set_title("Confidence vs Accuracy")
    if len(stds) > 2:
        corr, pval = scipy_stats.pearsonr(stds, abs_errors)
        ax.annotate(
            f"r = {corr:.3f} (p={pval:.3f})",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            fontsize=9,
            va="top",
        )

    # Signed error distribution
    ax = axes[1]
    ax.hist(errors, bins=40, alpha=0.7, color="steelblue", edgecolor="white")
    ax.axvline(0, color="red", ls="--", lw=1.5)
    ax.axvline(mean_signed, color="green", lw=2, label=f"Mean: {mean_signed:+.2f}")
    ax.set_xlabel("Signed Error (pred - true)")
    ax.set_ylabel("Count")
    ax.set_title("Signed Error Distribution")
    ax.legend()

    # MAE by true margin bucket
    ax = axes[2]
    margin_bins = np.arange(0, 40, 5)
    bin_maes = []
    bin_centers_list = []
    for lo, hi in zip(margin_bins[:-1], margin_bins[1:]):
        mask = (np.abs(true) >= lo) & (np.abs(true) < hi)
        if mask.sum() > 5:
            bin_maes.append(float(np.mean(abs_errors[mask])))
            bin_centers_list.append((lo + hi) / 2)
    ax.bar(bin_centers_list, bin_maes, width=4, color="steelblue", edgecolor="white")
    ax.set_xlabel("|True Margin|")
    ax.set_ylabel("MAE")
    ax.set_title("MAE by Game Closeness")

    plt.tight_layout()
    plt.savefig(output_dir / "failure_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: failure_analysis.png")


# ---------------------------------------------------------------------------
# 6. Context Conditioning (CFG)
# ---------------------------------------------------------------------------


def analyze_context_conditioning(
    rollout_engine: AutoregressiveRollout,
    loader: DataLoader,
    device: torch.device,
    n_rollouts: int,
    temperature: float,
    max_games: int,
    output_dir: Path,
) -> None:
    """Compare conditional vs unconditional rollout distributions via KL divergence."""
    print(f"\n{'=' * 60}")
    print("Analysis 6: Context Conditioning (CFG)")
    print(f"{'=' * 60}")

    if rollout_engine.guidance_scale <= 1.0:
        print("  CFG not active (guidance_scale <= 1.0). Skipping.")
        return

    cfg_games = min(max_games, 50)
    original_scale = rollout_engine.guidance_scale

    kl_divs = []
    cond_maes = []
    uncond_maes = []

    for i, batch in enumerate(loader):
        if i >= cfg_games:
            break
        if batch is None:
            continue

        batch_dev = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        true_margin = batch_dev["final_margin"].item()

        # Conditional (with CFG)
        rollout_engine.guidance_scale = original_scale
        result_cond = rollout_engine.rollout(
            batch_dev,
            n_rollouts=n_rollouts,
            temperature=temperature,
        )

        # Unconditional (no CFG)
        rollout_engine.guidance_scale = 1.0
        result_uncond = rollout_engine.rollout(
            batch_dev,
            n_rollouts=n_rollouts,
            temperature=temperature,
        )

        cond_spreads = np.array(
            [
                h - a
                for h, a in zip(result_cond["home_scores"], result_cond["away_scores"])
            ]
        )
        uncond_spreads = np.array(
            [
                h - a
                for h, a in zip(
                    result_uncond["home_scores"], result_uncond["away_scores"]
                )
            ]
        )

        # KL divergence via histograms
        lo_val = min(cond_spreads.min(), uncond_spreads.min()) - 5
        hi_val = max(cond_spreads.max(), uncond_spreads.max()) + 5
        bins = np.linspace(lo_val, hi_val, 30)
        p_cond, _ = np.histogram(cond_spreads, bins=bins, density=True)
        p_uncond, _ = np.histogram(uncond_spreads, bins=bins, density=True)

        eps = 1e-8
        p_cond = p_cond + eps
        p_uncond = p_uncond + eps
        p_cond = p_cond / p_cond.sum()
        p_uncond = p_uncond / p_uncond.sum()

        kl = float(np.sum(p_cond * np.log(p_cond / p_uncond)))
        kl_divs.append(kl)
        cond_maes.append(abs(result_cond["spread_mean"] - true_margin))
        uncond_maes.append(abs(result_uncond["spread_mean"] - true_margin))

    rollout_engine.guidance_scale = original_scale

    if not kl_divs:
        print("  No games evaluated. Skipping.")
        return

    kl_arr = np.array(kl_divs)
    print(f"  Games analyzed: {len(kl_divs)}")
    print(f"  Mean KL(cond || uncond): {np.mean(kl_arr):.4f}")
    print(f"  KL range: [{np.min(kl_arr):.4f}, {np.max(kl_arr):.4f}]")
    print(f"  Conditional MAE:   {np.mean(cond_maes):.2f}")
    print(f"  Unconditional MAE: {np.mean(uncond_maes):.2f}")
    print(f"  Improvement: {np.mean(uncond_maes) - np.mean(cond_maes):+.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(kl_arr, bins=20, alpha=0.7, color="steelblue", edgecolor="white")
    ax.set_xlabel("KL Divergence")
    ax.set_ylabel("Count")
    ax.set_title("Per-Game KL(conditional || unconditional)")

    ax = axes[1]
    ax.scatter(cond_maes, uncond_maes, alpha=0.4, s=15, color="steelblue")
    lim = max(max(cond_maes), max(uncond_maes)) * 1.05
    ax.plot([0, lim], [0, lim], "r--", lw=1, label="y=x")
    ax.set_xlabel("Conditional MAE")
    ax.set_ylabel("Unconditional MAE")
    ax.set_title("Per-Game: CFG vs No CFG")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "context_conditioning.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: context_conditioning.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Phase 4 Exp 5 generative model"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--n-rollouts", type=int, default=100, help="Rollouts per game")
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--max-games", type=int, default=200, help="Max games to analyze"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: analysis/exp5_{timestamp}/)",
    )
    parser.add_argument(
        "--skip-convergence",
        action="store_true",
        help="Skip convergence analysis (most expensive)",
    )
    parser.add_argument(
        "--skip-cfg",
        action="store_true",
        help="Skip CFG context conditioning analysis",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # ---- Output directory --------------------------------------------------
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("analysis") / f"exp5_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # ---- Config & model ----------------------------------------------------
    config = GenerativeExperimentConfig.from_yaml(args.config)

    use_full = config.model.use_full_context
    use_simple = config.model.use_simplified_context
    use_compressed = config.model.use_scoring_events_only
    max_se = config.model.max_scoring_events

    ds = GenerativeDataset(
        config.data,
        split=args.split,
        use_simplified_context=use_simple,
        use_full_context=use_full,
        use_scoring_events_only=use_compressed,
        max_scoring_events=max_se,
    )
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        collate_fn=generative_collate,
    )

    model = GenerativeModel(config.model)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    max_games = min(args.max_games, len(ds))

    print(f"\n{'=' * 60}")
    print("Phase 4 Exp 5 Analysis")
    print(f"{'=' * 60}")
    print(f"  Split:         {args.split} ({max_games} / {len(ds)} games)")
    print(f"  Rollouts/game: {args.n_rollouts}")
    print(f"  Temperature:   {args.temperature}")
    print(f"  Parameters:    {n_params:,}")
    print(f"  Full context:  {use_full}")
    print(f"  Output:        {output_dir}")

    rollout_engine = AutoregressiveRollout(model, config, device=device)

    # ---- Collect rollout data (single pass) --------------------------------
    logger.info(
        f"Collecting rollout data ({max_games} games, {args.n_rollouts} rollouts each)..."
    )
    games = collect_rollout_data(
        rollout_engine,
        loader,
        device,
        args.n_rollouts,
        args.temperature,
        max_games,
    )

    if not games:
        logger.error("No valid data collected. Exiting.")
        return

    # Overall metrics
    preds = np.array([g["spread_mean"] for g in games])
    trues = np.array([g["true_margin"] for g in games])
    overall_mae = float(np.mean(np.abs(preds - trues)))
    overall_rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    print(f"\n  Overall: MAE = {overall_mae:.2f}, RMSE = {overall_rmse:.2f}")

    # ---- Run all analysis modules ------------------------------------------
    analyze_distributions(games, output_dir)
    analyze_error_trajectory(games, output_dir)

    if not args.skip_convergence:
        conv_loader = DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            collate_fn=generative_collate,
        )
        analyze_convergence(
            rollout_engine,
            conv_loader,
            device,
            args.temperature,
            max_games,
            output_dir,
        )
    else:
        print(f"\n{'=' * 60}")
        print("Analysis 3: Convergence Diagnostic (SKIPPED)")
        print(f"{'=' * 60}")

    analyze_scoring_patterns(games, output_dir)
    analyze_failures(games, output_dir)

    if not args.skip_cfg:
        cfg_loader = DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            collate_fn=generative_collate,
        )
        analyze_context_conditioning(
            rollout_engine,
            cfg_loader,
            device,
            args.n_rollouts,
            args.temperature,
            max_games,
            output_dir,
        )
    else:
        print(f"\n{'=' * 60}")
        print("Analysis 6: Context Conditioning (SKIPPED)")
        print(f"{'=' * 60}")

    # ---- Summary -----------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Analysis Complete")
    print(f"{'=' * 60}")
    print(f"  Output: {output_dir}")
    generated = sorted(output_dir.glob("*.png"))
    if generated:
        print(f"  Files:")
        for f in generated:
            print(f"    - {f.name}")
    print()


if __name__ == "__main__":
    main()
