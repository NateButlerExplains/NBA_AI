"""
Ensemble Combiner: merge predictions from Phase 5 (hierarchical) and Phase 3 (transformer).

Uses weighted averaging for spread/total, log-odds averaging for win probabilities,
and mixture-of-Gaussians for uncertainty (sigma). Degrades gracefully when one model's
prediction is missing for a game.

Usage:
    from src.pipeline.ensemble_combiner import EnsembleCombiner
    ec = EnsembleCombiner()
    combined = ec.combine({"Phase5": phase5_preds, "Phase3": phase3_preds})
"""

from __future__ import annotations

import math
from typing import Optional

DEFAULT_WEIGHTS = {"Phase5": 0.55, "Phase3": 0.45}


class EnsembleCombiner:
    """Combine predictions from multiple models via weighted averaging."""

    def __init__(self, weights: Optional[dict[str, float]] = None):
        self.weights = dict(weights) if weights is not None else dict(DEFAULT_WEIGHTS)
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("Weights must sum to a positive value")
        # Normalize so weights sum to 1
        self.weights = {k: v / total for k, v in self.weights.items()}

    def combine(self, predictions: dict[str, dict[str, dict]]) -> dict[str, dict]:
        """
        Combine predictions from multiple models.

        Args:
            predictions: {model_name: {game_id: pred_dict}}
                Each pred_dict should contain:
                    pred_spread, pred_spread_sigma, pred_total, pred_total_sigma,
                    pred_home_win_pct, pred_home_score, pred_away_score,
                    roster_confidence

        Returns:
            {game_id: combined_pred_dict} with all the same keys plus
            model_contributions showing individual model predictions.
        """
        # Collect all game_ids across all models
        all_game_ids: set[str] = set()
        for model_preds in predictions.values():
            all_game_ids.update(model_preds.keys())

        results = {}
        for game_id in sorted(all_game_ids):
            # Gather available predictions for this game
            available: dict[str, dict] = {}
            for model_name, model_preds in predictions.items():
                if game_id in model_preds:
                    available[model_name] = model_preds[game_id]

            if not available:
                continue

            # Compute effective weights (renormalized over available models)
            raw_weights = {m: self.weights.get(m, 0.0) for m in available}
            w_total = sum(raw_weights.values())
            if w_total <= 0:
                # Fallback: equal weight across available models
                eff_weights = {m: 1.0 / len(available) for m in available}
            else:
                eff_weights = {m: w / w_total for m, w in raw_weights.items()}

            # Weighted average for spread
            spread = sum(
                eff_weights[m] * available[m]["pred_spread"] for m in available
            )

            # Mixture-of-Gaussians sigma for spread
            spread_sigma = _mixture_sigma(
                mus=[available[m]["pred_spread"] for m in available],
                sigmas=[available[m]["pred_spread_sigma"] for m in available],
                weights=[eff_weights[m] for m in available],
                combined_mu=spread,
            )

            # Weighted average for total
            total = sum(eff_weights[m] * available[m]["pred_total"] for m in available)

            # Mixture-of-Gaussians sigma for total
            total_sigma = _mixture_sigma(
                mus=[available[m]["pred_total"] for m in available],
                sigmas=[available[m]["pred_total_sigma"] for m in available],
                weights=[eff_weights[m] for m in available],
                combined_mu=total,
            )

            # Log-odds average for win probability
            win_prob = _logodds_average(
                probs=[available[m]["pred_home_win_pct"] for m in available],
                weights=[eff_weights[m] for m in available],
            )

            # Derive scores from combined spread and total
            # spread = home - away, total = home + away
            # => home = (total + spread) / 2, away = (total - spread) / 2
            # Use negative spread convention: spread = -(home_margin) when home favored
            # Actually spread here is pred_spread = home_score - away_score
            pred_home_score = (total - spread) / 2.0
            pred_away_score = (total + spread) / 2.0

            # Weighted average for roster confidence
            roster_confidence = sum(
                eff_weights[m] * available[m].get("roster_confidence", 0.5)
                for m in available
            )

            # Build model_contributions
            model_contributions = {}
            for m in available:
                model_contributions[m] = {
                    "weight": eff_weights[m],
                    "pred_spread": available[m]["pred_spread"],
                    "pred_total": available[m]["pred_total"],
                    "pred_home_win_pct": available[m]["pred_home_win_pct"],
                }

            results[game_id] = {
                "pred_home_score": pred_home_score,
                "pred_away_score": pred_away_score,
                "pred_home_win_pct": win_prob,
                "pred_spread": spread,
                "pred_spread_sigma": spread_sigma,
                "pred_total": total,
                "pred_total_sigma": total_sigma,
                "roster_confidence": roster_confidence,
                "model_contributions": model_contributions,
            }

        return results


def _logodds_average(probs: list[float], weights: list[float]) -> float:
    """
    Average probabilities in log-odds (logit) space.

    Clamps probabilities to [0.001, 0.999] to avoid infinities.
    """
    EPS = 1e-3
    logits = []
    for p in probs:
        p_clamped = max(EPS, min(1.0 - EPS, p))
        logits.append(math.log(p_clamped / (1.0 - p_clamped)))

    avg_logit = sum(w * l for w, l in zip(weights, logits))
    return 1.0 / (1.0 + math.exp(-avg_logit))


def _mixture_sigma(
    mus: list[float],
    sigmas: list[float],
    weights: list[float],
    combined_mu: float,
) -> float:
    """
    Compute the standard deviation of a Gaussian mixture.

    For a mixture with components (mu_i, sigma_i, w_i), the overall variance is:
        Var = sum_i w_i * (sigma_i^2 + mu_i^2) - combined_mu^2

    This accounts for both within-component variance and between-component spread.
    """
    second_moment = sum(w * (s**2 + m**2) for w, s, m in zip(weights, sigmas, mus))
    variance = second_moment - combined_mu**2
    # Clamp to avoid negative variance from floating point
    return math.sqrt(max(variance, 0.0))
