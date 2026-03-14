"""Autoregressive rollout with Classifier-Free Guidance (CFG).

Supports both full mode (7-dim states, ~487 steps) and compressed scoring-event
mode (8-dim states, ~110 steps).

Full mode:
  - Score events: {0:none, 1:h+1, 2:h+2, 3:h+3, 4:a+1, 5:a+2, 6:a+3}
  - Deterministic period transitions at clock=0
  - Game ends at period=4, clock=0

Compressed mode:
  - Score events: {0:h+1, 1:h+2, 2:h+3, 3:a+1, 4:a+2, 5:a+3, 6:game_end}
  - Clock head predicts game_progress (monotonic 0→1) instead of clock_norm
  - Game ends when game_end (class 6) sampled or game_progress >= 1.0
  - Period/clock derived from game_progress
"""

import logging

import numpy as np
import torch
import torch.nn as nn

from src.generative.config import GenerativeExperimentConfig

logger = logging.getLogger(__name__)


class AutoregressiveRollout:
    """KV-cached autoregressive rollout engine with CFG support.

    Parameters
    ----------
    model : nn.Module
        A ``GenerativeModel`` instance (already on device, in eval mode).
    config : GenerativeExperimentConfig
        Experiment config (used for max_rollout_steps, guidance_scale, etc.).
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
        self.compressed = config.model.use_scoring_events_only
        self.max_steps = (
            config.model.max_scoring_events
            if self.compressed
            else config.training.max_rollout_steps
        )
        self.guidance_scale = config.training.guidance_scale

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
            temperature: softmax temperature for score-event sampling.

        Returns:
            dict with spread_mean, spread_std, win_prob, home_score_mean,
            away_score_mean, home_scores, away_scores, n_ties.
        """
        self.model.eval()

        # Encode context once (B=1) → (1, 2, 512)
        context_tokens = self.model.encode_context(context_data)

        # Pool into conditioning vector for adaLN-Zero
        cond = self.model.pool_context(context_tokens)  # (1, 512)

        # Compute score bias from context (direct shortcut)
        score_bias = self.model.compute_score_bias(context_tokens)  # (1, 7)

        # Run batched rollouts in chunks
        trajectories = self._batched_rollout(cond, score_bias, n_rollouts, temperature)
        return self._aggregate(trajectories)

    def _batched_rollout(
        self,
        cond: torch.Tensor,
        score_bias: torch.Tensor,
        n_rollouts: int,
        temperature: float,
    ) -> list[tuple[float, float]]:
        """Run rollouts in chunks, returning list of (home_score, away_score)."""
        chunk_size = min(n_rollouts, 32)
        all_trajectories: list[tuple[float, float]] = []

        for chunk_start in range(0, n_rollouts, chunk_size):
            chunk_n = min(chunk_size, n_rollouts - chunk_start)

            # Expand cond and score bias for this chunk
            c = cond.expand(chunk_n, -1)  # (chunk_n, D)
            bias = score_bias.expand(chunk_n, -1)  # (chunk_n, 7)

            if self.compressed:
                chunk_results = self._rollout_chunk_compressed(
                    c, bias, chunk_n, temperature
                )
            else:
                chunk_results = self._rollout_chunk_full(c, bias, chunk_n, temperature)
            all_trajectories.extend(chunk_results)

        return all_trajectories

    # ---- Full mode rollout (7-dim states) ------------------------------------

    def _rollout_chunk_full(
        self,
        cond: torch.Tensor,
        score_bias: torch.Tensor,
        chunk_n: int,
        temperature: float,
    ) -> list[tuple[float, float]]:
        """Full-mode rollout: 7-dim states, deterministic period transitions."""
        decoder = self.model.decoder
        use_cfg = self.guidance_scale > 1.0

        # --- Initialise game state ---
        home_scores = torch.zeros(chunk_n, device=self.device)
        away_scores = torch.zeros(chunk_n, device=self.device)
        periods = torch.ones(chunk_n, device=self.device)
        clocks = torch.full((chunk_n,), 720.0, device=self.device)
        active = torch.ones(chunk_n, dtype=torch.bool, device=self.device)

        if use_cfg:
            cond_batch = torch.cat(
                [cond, torch.zeros_like(cond)], dim=0
            )  # (2*chunk_n, D)
            kv_caches = decoder.init_kv_cache(2 * chunk_n, self.device)
        else:
            cond_batch = cond
            kv_caches = decoder.init_kv_cache(chunk_n, self.device)

        for step in range(self.max_steps):
            if not active.any():
                break

            # Build current state vector (7-dim)
            state = self._build_state_full(periods, clocks, home_scores, away_scores)

            # Embed current state: (chunk_n, 7) → (chunk_n, 1, D)
            state_embed = self.model.state_embedder(state.unsqueeze(1))

            score_logits, clock_pred = self._decode_step(
                decoder,
                state_embed,
                cond_batch,
                kv_caches,
                step,
                score_bias,
                use_cfg,
                chunk_n,
            )

            # Sample score event
            score_event = self._sample_score(score_logits, temperature, active)

            # Apply score deltas (full mapping)
            self._apply_score_event_full(score_event, home_scores, away_scores, active)

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

    # ---- Compressed mode rollout (8-dim states) ------------------------------

    def _rollout_chunk_compressed(
        self,
        cond: torch.Tensor,
        score_bias: torch.Tensor,
        chunk_n: int,
        temperature: float,
    ) -> list[tuple[float, float]]:
        """Compressed-mode rollout: 8-dim states, game_end class detection."""
        decoder = self.model.decoder
        use_cfg = self.guidance_scale > 1.0

        # --- Initialise game state ---
        home_scores = torch.zeros(chunk_n, device=self.device)
        away_scores = torch.zeros(chunk_n, device=self.device)
        game_progress = torch.zeros(chunk_n, device=self.device)
        active = torch.ones(chunk_n, dtype=torch.bool, device=self.device)

        if use_cfg:
            cond_batch = torch.cat([cond, torch.zeros_like(cond)], dim=0)
            kv_caches = decoder.init_kv_cache(2 * chunk_n, self.device)
        else:
            cond_batch = cond
            kv_caches = decoder.init_kv_cache(chunk_n, self.device)

        prev_progress = torch.zeros(chunk_n, device=self.device)

        for step in range(self.max_steps):
            if not active.any():
                break

            # Build current state vector (8-dim)
            inter_event_time = (game_progress - prev_progress) * 2880.0 / 120.0
            if step == 0:
                inter_event_time.zero_()

            state = self._build_state_compressed(
                game_progress, home_scores, away_scores, inter_event_time
            )

            # Embed current state: (chunk_n, 8) → (chunk_n, 1, D)
            state_embed = self.model.state_embedder(state.unsqueeze(1))

            score_logits, progress_pred = self._decode_step(
                decoder,
                state_embed,
                cond_batch,
                kv_caches,
                step,
                score_bias,
                use_cfg,
                chunk_n,
            )

            # Sample score event
            score_event = self._sample_score(score_logits, temperature, active)

            # Check for game_end (class 6)
            game_end = (score_event == 6) & active
            active[game_end] = False

            # Apply score deltas for active non-end events (compressed mapping)
            self._apply_score_event_compressed(
                score_event, home_scores, away_scores, active
            )

            # Update game progress (clock_head predicts game_progress in compressed mode)
            prev_progress = game_progress.clone()
            # Clamp to be monotonically increasing, max 1.0
            new_progress = torch.max(progress_pred, game_progress).clamp(max=1.0)
            game_progress = torch.where(active, new_progress, game_progress)

            # End game if progress reaches 1.0
            time_up = (game_progress >= 1.0) & active
            active[time_up] = False

        results: list[tuple[float, float]] = []
        for i in range(chunk_n):
            results.append((home_scores[i].item(), away_scores[i].item()))
        return results

    # ---- Shared decode step --------------------------------------------------

    def _decode_step(
        self,
        decoder: nn.Module,
        state_embed: torch.Tensor,
        cond_batch: torch.Tensor,
        kv_caches: list,
        step: int,
        score_bias: torch.Tensor,
        use_cfg: bool,
        chunk_n: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one decode step, returning score logits and clock/progress prediction."""
        if use_cfg:
            state_embed_batch = state_embed.repeat(2, 1, 1)
            output, kv_caches[:] = decoder.decode_step(
                state_embed_batch, cond_batch, kv_caches, step
            )

            cond_out, uncond_out = output.chunk(2, dim=0)
            h_cond = cond_out.squeeze(1)
            h_uncond = uncond_out.squeeze(1)

            # CFG on score logits
            cond_logits = self.model.score_head(h_cond)
            uncond_logits = self.model.score_head(h_uncond)
            score_logits = uncond_logits + self.guidance_scale * (
                cond_logits - uncond_logits
            )
            score_logits = score_logits + score_bias

            # Clock from conditional path only
            clock_pred = self.model.clock_head(h_cond).squeeze(-1)
        else:
            output, kv_caches[:] = decoder.decode_step(
                state_embed, cond_batch, kv_caches, step
            )
            h = output.squeeze(1)
            score_logits = self.model.score_head(h) + score_bias
            clock_pred = self.model.clock_head(h).squeeze(-1)

        return score_logits, clock_pred

    # ---- State builders ------------------------------------------------------

    def _build_state_full(
        self,
        periods: torch.Tensor,
        clocks: torch.Tensor,
        home_scores: torch.Tensor,
        away_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Build normalised 7-dim state vector from raw game values."""
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

    def _build_state_compressed(
        self,
        game_progress: torch.Tensor,
        home_scores: torch.Tensor,
        away_scores: torch.Tensor,
        inter_event_time: torch.Tensor,
    ) -> torch.Tensor:
        """Build normalised 8-dim state vector for compressed mode.

        Derives period/clock from game_progress (monotonic 0→1).
        """
        B = game_progress.shape[0]
        state = torch.zeros(B, 8, device=self.device)

        # Derive period and clock from game_progress
        # game_progress ∈ [0, 1] maps to 4 quarters
        quarter_index = (game_progress * 4).clamp(0, 3.99).floor()  # 0-3
        quarter_frac = game_progress * 4 - quarter_index  # 0-1 within quarter
        period = quarter_index + 1  # 1-4
        clock_norm = 1.0 - quarter_frac  # 1.0 at start, 0.0 at end of quarter

        state[:, 0] = period / 4.0
        state[:, 1] = clock_norm
        state[:, 2] = game_progress
        state[:, 3] = home_scores / 150.0
        state[:, 4] = away_scores / 150.0
        state[:, 5] = (home_scores - away_scores) / 50.0
        state[:, 6] = (home_scores + away_scores) / 300.0
        state[:, 7] = inter_event_time
        return state

    # ---- Score sampling and application --------------------------------------

    @staticmethod
    def _sample_score(
        logits: torch.Tensor,
        temperature: float,
        active: torch.Tensor,
    ) -> torch.Tensor:
        """Sample from the score-event distribution.

        Inactive games are forced to class 0.
        """
        scaled = logits / max(temperature, 1e-8)
        probs = torch.softmax(scaled, dim=-1)
        events = torch.multinomial(probs, 1).squeeze(-1)
        events[~active] = 0
        return events

    @staticmethod
    def _apply_score_event_full(
        events: torch.Tensor,
        home_scores: torch.Tensor,
        away_scores: torch.Tensor,
        active: torch.Tensor,
    ) -> None:
        """Apply score deltas — full mode.

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

    @staticmethod
    def _apply_score_event_compressed(
        events: torch.Tensor,
        home_scores: torch.Tensor,
        away_scores: torch.Tensor,
        active: torch.Tensor,
    ) -> None:
        """Apply score deltas — compressed mode.

        Class mapping: {0:home+1, 1:home+2, 2:home+3,
                        3:away+1, 4:away+2, 5:away+3, 6:game_end}
        Game end (class 6) is handled before this call.
        """
        mask = active.float()

        home_deltas = torch.zeros_like(events, dtype=torch.float)
        away_deltas = torch.zeros_like(events, dtype=torch.float)

        home_deltas[events == 0] = 1.0
        home_deltas[events == 1] = 2.0
        home_deltas[events == 2] = 3.0
        away_deltas[events == 3] = 1.0
        away_deltas[events == 4] = 2.0
        away_deltas[events == 5] = 3.0

        home_scores += home_deltas * mask
        away_scores += away_deltas * mask

    # ---- Aggregation ---------------------------------------------------------

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
