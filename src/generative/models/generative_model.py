"""Top-level generative model with adaLN-Zero conditioning.

Supports three context encoder modes:
  - Full context (Exp 5): FullContextEncoder with Phase 3 player-aware signals
  - Simplified (Exp 3-4): SimpleContextEncoder with rolling stats
  - Original (Exp 1-2): ContextEncoder with player-level history

Training forward pass:
  1. context_encoder(context_data) → (B, 2, 512)
  2. Pre-decoder auxiliary heads on raw context (margin + win prediction)
  3. context_pooling(concat(home, away)) → cond (B, 512) for adaLN-Zero
  4. context_score_bias(context_tokens) → (B, n_score_classes) — direct shortcut
  5. Apply context dropout (zero out cond AND score bias with p)
  6. state_embedder(states[:, :-1]) → (B, T-1, 512) — teacher forcing
  7. Apply score jitter to input states during training
  8. Optional scheduled sampling: mix model predictions into input
  9. decoder(state_embeds, cond) → (B, T-1, 512) — adaLN-Zero modulation
  10. score_head(decoder_out) + score_bias → (B, T-1, n_score_classes)
  11. clock_head(decoder_out) → (B, T-1)
  12. context_margin_head(position 0 output) → (B, 1)
  13. outcome_head(decoder_out) → (spread_mu, spread_sigma) each (B, T-1)  [Exp 5 only]
"""

import torch
import torch.nn as nn

from src.generative.config import GenerativeModelConfig
from src.generative.models.state_embedder import StateEmbedder
from src.generative.models.causal_decoder import CausalDecoder
from src.generative.models.prediction_heads import (
    ScoreHead,
    ClockHead,
    ContextMarginHead,
    ContextScoreBias,
    PreDecoderMarginHead,
    PreDecoderWinHead,
)
from src.generative.models.outcome_head import OutcomeHead

# Score event value mapping: class index → (home_delta, away_delta)
# Full mode: {0:none, 1:h+1, 2:h+2, 3:h+3, 4:a+1, 5:a+2, 6:a+3}
_SCORE_DELTAS_FULL = torch.tensor(
    [
        [0, 0],  # 0: no_score
        [1, 0],  # 1: home+1
        [2, 0],  # 2: home+2
        [3, 0],  # 3: home+3
        [0, 1],  # 4: away+1
        [0, 2],  # 5: away+2
        [0, 3],  # 6: away+3
    ],
    dtype=torch.float32,
)

# Compressed mode: {0:h+1, 1:h+2, 2:h+3, 3:a+1, 4:a+2, 5:a+3, 6:game_end}
_SCORE_DELTAS_COMPRESSED = torch.tensor(
    [
        [1, 0],  # 0: home+1
        [2, 0],  # 1: home+2
        [3, 0],  # 2: home+3
        [0, 1],  # 3: away+1
        [0, 2],  # 4: away+2
        [0, 3],  # 5: away+3
        [0, 0],  # 6: game_end (no score change)
    ],
    dtype=torch.float32,
)

# Exp 5 mode: 6-class (no game_end — termination handled by rules engine)
# {0:h+1, 1:h+2, 2:h+3, 3:a+1, 4:a+2, 5:a+3}
_SCORE_DELTAS_EXP5 = torch.tensor(
    [
        [1, 0],  # 0: home+1
        [2, 0],  # 1: home+2
        [3, 0],  # 2: home+3
        [0, 1],  # 3: away+1
        [0, 2],  # 4: away+2
        [0, 3],  # 5: away+3
    ],
    dtype=torch.float32,
)


class GenerativeModel(nn.Module):
    """Top-level generative NBA game model with adaLN-Zero conditioning.

    Context modulates every decoder layer via Adaptive Layer Normalization,
    making it impossible for the decoder to ignore team context.
    """

    def __init__(self, config: GenerativeModelConfig) -> None:
        super().__init__()
        self.config = config

        # Sub-modules — context encoder chosen by config
        if config.use_full_context:
            from src.generative.models.full_context_encoder import (
                FullContextEncoder,
            )

            self.context_encoder = FullContextEncoder(config)
        elif config.use_simplified_context:
            from src.generative.models.simple_context_encoder import (
                SimpleContextEncoder,
            )

            self.context_encoder = SimpleContextEncoder(config)
        else:
            from src.generative.models.context_encoder import ContextEncoder

            self.context_encoder = ContextEncoder(config)

        self.state_embedder = StateEmbedder(config)
        self.decoder = CausalDecoder(config)
        self.score_head = ScoreHead(
            config.hidden_dim, config.head_hidden_dim, n_classes=config.n_score_classes
        )
        self.clock_head = ClockHead(
            config.hidden_dim, config.head_hidden_dim, use_delta=config.use_clock_delta
        )
        self.context_margin_head = ContextMarginHead(
            config.hidden_dim, config.head_hidden_dim
        )
        self.context_score_bias = ContextScoreBias(
            config.hidden_dim, config.head_hidden_dim, config.n_score_classes
        )

        # Context pooling: concat(home, away) → cond vector for adaLN-Zero
        self.context_pool = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.SiLU(),
        )

        # Pre-decoder auxiliary heads (direct gradient to context encoder)
        self.pre_margin_head = PreDecoderMarginHead(
            config.hidden_dim, config.head_hidden_dim
        )
        self.pre_win_head = PreDecoderWinHead(config.hidden_dim, config.head_hidden_dim)

        # Outcome head (Exp 5: endpoint guidance via Gaussian spread prediction)
        self.outcome_head = None
        if config.use_full_context:
            self.outcome_head = OutcomeHead(config)

    def forward(self, batch: dict, teacher_forcing_ratio: float = 1.0) -> dict:
        """Training forward pass with teacher forcing and adaLN-Zero conditioning.

        Args:
            batch: dict containing context data keys + "states" (B, T, D).
                D=7 for Exp 1-4 (full states) or 8 (compressed), 18 for Exp 5 enriched.
            teacher_forcing_ratio: 1.0 = full teacher forcing, <1.0 = mix
                model predictions into input states (scheduled sampling).

        Returns:
            dict with score_logits, clock_preds, context_margin_pred,
            pre_margin_pred, pre_win_pred, and optionally outcome_mu/outcome_sigma.
        """
        # 1. Encode context
        context_tokens = self.context_encoder(batch)  # (B, 2, 512)

        # 2. Pre-decoder auxiliary predictions (always, no dropout)
        home_ctx = context_tokens[:, 0, :]  # (B, 512)
        away_ctx = context_tokens[:, 1, :]  # (B, 512)
        matchup = torch.cat(
            [home_ctx, away_ctx, home_ctx - away_ctx], dim=-1
        )  # (B, 1536)
        pre_margin_pred = self.pre_margin_head(matchup).squeeze(-1)  # (B,)
        pre_win_pred = self.pre_win_head(matchup).squeeze(-1)  # (B,)

        # 3. Context pooling → cond vector for adaLN-Zero
        ctx_flat = context_tokens.reshape(context_tokens.shape[0], -1)  # (B, 1024)
        cond = self.context_pool(ctx_flat)  # (B, 512)

        # 4. Compute score bias from full context (before dropout)
        score_bias = self.context_score_bias(context_tokens)  # (B, n_score_classes)

        # 5. Context dropout: zero out cond AND score bias
        if self.training and self.config.context_dropout > 0:
            B = cond.shape[0]
            drop_mask = torch.rand(B, device=cond.device) < self.config.context_dropout
            cond = cond.masked_fill(drop_mask.unsqueeze(-1), 0.0)
            score_bias = score_bias.masked_fill(drop_mask.unsqueeze(-1), 0.0)

        # 6. State embedding with teacher forcing
        states = batch["states"]  # (B, T, D)
        input_states = states[:, :-1, :]  # (B, T-1, D)

        # 7. Score jitter during training (applied to score channels at indices 3-6)
        if self.training:
            jitter_std = getattr(self.config, "score_jitter_std", None)
            if jitter_std is None:
                jitter_std = batch.get("score_jitter_std", 0.0)
            if jitter_std and jitter_std > 0:
                noise = torch.randn_like(input_states[:, :, 3:7]) * (jitter_std / 150.0)
                input_states = input_states.clone()
                input_states[:, :, 3:7] = input_states[:, :, 3:7] + noise

        # 8. Scheduled sampling: mix model predictions into score channels
        if self.training and teacher_forcing_ratio < 1.0:
            input_states = self._apply_scheduled_sampling(
                input_states, cond, score_bias, teacher_forcing_ratio
            )

        state_embeds = self.state_embedder(input_states)  # (B, T-1, 512)

        # 9. Decode with adaLN-Zero conditioning (state tokens only, no context prefix)
        decoder_out = self.decoder(state_embeds, cond)  # (B, T-1, 512)

        # 10. Prediction heads
        score_logits = self.score_head(decoder_out) + score_bias.unsqueeze(
            1
        )  # (B, T-1, n_score_classes)
        clock_preds = self.clock_head(decoder_out).squeeze(-1)  # (B, T-1)

        # 11. Context margin head on first state position
        context_margin_pred = self.context_margin_head(decoder_out[:, 0, :]).squeeze(
            -1
        )  # (B,)

        # 12. Outcome head (Exp 5 endpoint guidance)
        outcome_mu, outcome_sigma = None, None
        if self.outcome_head is not None:
            outcome_mu, outcome_sigma = self.outcome_head(decoder_out)  # each (B, T-1)

        result = {
            "score_logits": score_logits,
            "clock_preds": clock_preds,
            "context_margin_pred": context_margin_pred,
            "pre_margin_pred": pre_margin_pred,
            "pre_win_pred": pre_win_pred,
        }

        if outcome_mu is not None:
            result["outcome_mu"] = outcome_mu
            result["outcome_sigma"] = outcome_sigma

        return result

    def _apply_scheduled_sampling(
        self,
        input_states: torch.Tensor,
        cond: torch.Tensor,
        score_bias: torch.Tensor,
        teacher_forcing_ratio: float,
    ) -> torch.Tensor:
        """Mix model-predicted scores into teacher-forced input states.

        Does a no-grad forward pass to get score predictions, then replaces
        score channels (indices 3-6) at randomly selected positions with
        cumulative scores derived from predicted events.

        Period/clock channels (indices 0-2) stay from ground truth since they
        are deterministic given game structure.

        Supports 7-class (Exp 1-4) and 6-class (Exp 5) score event modes.
        """
        B, T, D = input_states.shape
        device = input_states.device

        with torch.no_grad():
            state_embeds = self.state_embedder(input_states)
            decoder_out = self.decoder(state_embeds, cond)
            score_logits = self.score_head(decoder_out) + score_bias.unsqueeze(1)
            predicted_events = score_logits.argmax(dim=-1)  # (B, T)

        # Convert predicted events to cumulative scores
        # Choose delta table based on config: 6-class (Exp 5), 7-class compressed, or 7-class full
        n_classes = self.config.n_score_classes
        if n_classes == 6:
            delta_table = _SCORE_DELTAS_EXP5
        elif self.config.use_scoring_events_only:
            delta_table = _SCORE_DELTAS_COMPRESSED
        else:
            delta_table = _SCORE_DELTAS_FULL

        deltas = delta_table.to(device)  # (n_classes, 2)
        # Clamp predicted events to valid range
        predicted_events_clamped = predicted_events.clamp(0, n_classes - 1)
        event_deltas = deltas[
            predicted_events_clamped
        ]  # (B, T, 2) = (home_delta, away_delta)
        cum_home = event_deltas[:, :, 0].cumsum(dim=1)  # (B, T)
        cum_away = event_deltas[:, :, 1].cumsum(dim=1)  # (B, T)

        # Build predicted score channels (normalized same as ground truth)
        pred_scores = input_states.clone()
        pred_scores[:, :, 3] = cum_home / 150.0
        pred_scores[:, :, 4] = cum_away / 150.0
        pred_scores[:, :, 5] = (cum_home - cum_away) / 50.0
        pred_scores[:, :, 6] = (cum_home + cum_away) / 300.0

        # Mix: per-position Bernoulli decides teacher forcing vs model prediction
        mix_mask = (
            torch.rand(B, T, 1, device=device) > teacher_forcing_ratio
        )  # True = use model prediction
        input_states = torch.where(mix_mask, pred_scores, input_states)

        return input_states

    def encode_context(self, context_data: dict) -> torch.Tensor:
        """Encode context only (for inference caching).

        Returns:
            (B, 2, 512) context tokens.
        """
        return self.context_encoder(context_data)

    def pool_context(self, context_tokens: torch.Tensor) -> torch.Tensor:
        """Pool context tokens into conditioning vector for adaLN-Zero.

        Args:
            context_tokens: (B, 2, 512) [home, away] context.

        Returns:
            (B, 512) conditioning vector.
        """
        ctx_flat = context_tokens.reshape(context_tokens.shape[0], -1)
        return self.context_pool(ctx_flat)

    def compute_score_bias(self, context_tokens: torch.Tensor) -> torch.Tensor:
        """Compute score event bias from context tokens (for rollout).

        Returns:
            (B, n_score_classes) score logit bias.
        """
        return self.context_score_bias(context_tokens)
