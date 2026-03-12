"""Autoregressive rollout for pre-game spread / win-probability prediction.

Given context tokens (from the context encoder), this module rolls out game
states autoregressively:
  1. Start from initial state (period=1, clock=720, scores=0-0).
  2. At each step: predict score event + next clock.
  3. Sample score event from the categorical distribution.
  4. Apply deterministic period transitions.
  5. Continue until game over (period >= 4, clock <= 0) or max steps reached.

Multiple rollouts are aggregated to produce spread mean/std and win probability.
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.generative.config import GenerativeExperimentConfig

logger = logging.getLogger(__name__)


class AutoregressiveRollout:
    """KV-cached autoregressive rollout engine for pre-game prediction.

    Parameters
    ----------
    model : nn.Module
        A ``GenerativeModel`` instance (already on device, in eval mode).
    config : GenerativeExperimentConfig
        Experiment config (used for max_rollout_steps, etc.).
    device : str or torch.device
        Device for inference tensors.
    """

    def __init__(
        self,
        model: nn.Module,
        config: GenerativeExperimentConfig,
        device: str | torch.device = "cuda",
    ) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(device) if isinstance(device, str) else device
        self.max_steps = config.training.max_rollout_steps

    @torch.no_grad()
    def rollout(
        self,
        context_data: dict,
        n_rollouts: int = 100,
        temperature: float = 1.0,
    ) -> dict:
        """Run multiple rollouts from the same context and aggregate.

        Args:
            context_data: dict of tensors for the context encoder (batch size 1).
            n_rollouts: number of independent trajectories to sample.
            temperature: softmax temperature for score-event sampling
                (1.0 = calibrated, <1 = greedy-ish).

        Returns:
            dict with ``spread_mean``, ``spread_std``, ``win_prob``,
            ``home_score_mean``, ``away_score_mean``, ``home_scores``,
            ``away_scores``, ``n_ties``.
        """
        self.model.eval()

        # Encode context once (B=1) -> (1, 2, 512)
        context_tokens = self.model.encode_context(context_data)

        # Run batched rollouts in chunks to manage memory
        trajectories = self._batched_rollout(context_tokens, n_rollouts, temperature)
        return self._aggregate(trajectories)

    def _batched_rollout(
        self,
        context_tokens: torch.Tensor,
        n_rollouts: int,
        temperature: float,
    ) -> list[tuple[float, float]]:
        """Run rollouts in chunks, returning list of (home_score, away_score)."""
        chunk_size = min(n_rollouts, 32)
        all_trajectories: list[tuple[float, float]] = []

        for chunk_start in range(0, n_rollouts, chunk_size):
            chunk_n = min(chunk_size, n_rollouts - chunk_start)

            # Expand context for this chunk: (1, 2, D) -> (chunk_n, 2, D)
            ctx = context_tokens.expand(chunk_n, -1, -1)

            # Run the chunk
            chunk_results = self._rollout_chunk(ctx, chunk_n, temperature)
            all_trajectories.extend(chunk_results)

        return all_trajectories

    def _rollout_chunk(
        self,
        ctx: torch.Tensor,
        chunk_n: int,
        temperature: float,
    ) -> list[tuple[float, float]]:
        """Run a single chunk of parallel rollouts.

        This uses step-by-step decoding. If the model's decoder supports
        KV-caching (``init_kv_cache`` / ``decode_step``), those are used.
        Otherwise, it falls back to re-encoding the full sequence at each step
        (slower but always correct).
        """
        decoder = self.model.decoder
        use_kv_cache = hasattr(decoder, "init_kv_cache") and hasattr(decoder, "decode_step")

        # --- Initialise game state ------------------------------------------
        home_scores = torch.zeros(chunk_n, device=self.device)
        away_scores = torch.zeros(chunk_n, device=self.device)
        periods = torch.ones(chunk_n, device=self.device)
        clocks = torch.full((chunk_n,), 720.0, device=self.device)
        active = torch.ones(chunk_n, dtype=torch.bool, device=self.device)

        if use_kv_cache:
            # Prefill KV cache with context tokens
            ctx_output, kv_caches = decoder.init_kv_cache(ctx)
        else:
            # We will accumulate the full embedding sequence
            all_embeds = [ctx]  # start with context tokens (chunk_n, 2, D)

        for step in range(self.max_steps):
            if not active.any():
                break

            # Build current state vector
            state = self._build_state(periods, clocks, home_scores, away_scores)

            # Embed current state: (chunk_n, 7) -> (chunk_n, 1, D)
            state_embed = self.model.state_embedder(state.unsqueeze(1))

            if use_kv_cache:
                output, kv_caches = decoder.decode_step(
                    state_embed, kv_caches, step=step + 2  # +2 for context prefix
                )
            else:
                all_embeds.append(state_embed)
                full_seq = torch.cat(all_embeds, dim=1)  # (chunk_n, 2+step+1, D)
                output = decoder(full_seq)               # (chunk_n, 2+step+1, D)
                output = output[:, -1:, :]               # last position

            # Prediction heads
            h = output.squeeze(1)                        # (chunk_n, D)
            score_logits = self.model.score_head(h)      # (chunk_n, 7)
            clock_pred = self.model.clock_head(h).squeeze(-1)  # (chunk_n,)

            # Sample score event
            score_event = self._sample_score(score_logits, temperature, active)

            # Apply score deltas
            self._apply_score_event(score_event, home_scores, away_scores, active)

            # Deterministic clock / period transition
            next_clock = clock_pred * 720.0
            next_clock = next_clock.clamp(0.0, 720.0)

            period_end = (next_clock <= 0.0) & active
            game_over = period_end & (periods >= 4)
            period_advance = period_end & (periods < 4)

            active[game_over] = False
            periods[period_advance] += 1
            clocks[period_advance] = 720.0
            clocks[~period_end & active] = next_clock[~period_end & active]

        # Collect final scores
        results: list[tuple[float, float]] = []
        for i in range(chunk_n):
            results.append((home_scores[i].item(), away_scores[i].item()))
        return results

    # ---- State helpers -----------------------------------------------------

    def _build_state(
        self,
        periods: torch.Tensor,
        clocks: torch.Tensor,
        home_scores: torch.Tensor,
        away_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Build normalised 7-dim state vector from raw game values.

        Channels: [period_norm, clock_norm, game_progress,
                    home_norm, away_norm, margin_norm, total_norm]
        """
        B = periods.shape[0]
        state = torch.zeros(B, 7, device=self.device)
        state[:, 0] = periods / 4.0
        state[:, 1] = clocks / 720.0
        elapsed = (periods - 1) * 720 + (720 - clocks)
        state[:, 2] = elapsed / 2880.0
        state[:, 3] = home_scores / 150.0
        state[:, 4] = away_scores / 150.0
        state[:, 5] = (home_scores - away_scores) / 50.0
        state[:, 6] = (home_scores + away_scores) / 300.0
        return state

    @staticmethod
    def _sample_score(
        logits: torch.Tensor,
        temperature: float,
        active: torch.Tensor,
    ) -> torch.Tensor:
        """Sample from the score-event distribution.

        Inactive games are forced to ``no_score`` (class 0).
        """
        scaled = logits / max(temperature, 1e-8)
        probs = torch.softmax(scaled, dim=-1)
        events = torch.multinomial(probs, 1).squeeze(-1)
        events[~active] = 0
        return events

    @staticmethod
    def _apply_score_event(
        events: torch.Tensor,
        home_scores: torch.Tensor,
        away_scores: torch.Tensor,
        active: torch.Tensor,
    ) -> None:
        """Apply score deltas to running totals (in-place).

        Class mapping: {0:none, 1:home+1, 2:home+2, 3:home+3,
                        4:away+1, 5:away+2, 6:away+3}
        """
        mask = active.float()

        home_deltas = torch.zeros_like(events, dtype=torch.float)
        away_deltas = torch.zeros_like(events, dtype=torch.float)

        home_deltas[events == 1] = 1.0
        home_deltas[events == 2] = 2.0
        home_deltas[events == 3] = 3.0
        away_deltas[events == 4] = 1.0
        away_deltas[events == 5] = 2.0
        away_deltas[events == 6] = 3.0

        home_scores += home_deltas * mask
        away_scores += away_deltas * mask

    # ---- Aggregation -------------------------------------------------------

    @staticmethod
    def _aggregate(trajectories: list[tuple[float, float]]) -> dict:
        """Aggregate rollout results into summary statistics."""
        home_scores = [t[0] for t in trajectories]
        away_scores = [t[1] for t in trajectories]
        spreads = [h - a for h, a in trajectories]

        return {
            "spread_mean": float(np.mean(spreads)),
            "spread_std": float(np.std(spreads)),
            "win_prob": float(np.mean([s > 0 for s in spreads])),
            "home_score_mean": float(np.mean(home_scores)),
            "away_score_mean": float(np.mean(away_scores)),
            "home_scores": home_scores,
            "away_scores": away_scores,
            "n_ties": sum(1 for s in spreads if s == 0),
        }
