"""Top-level generative model with adaLN-Zero conditioning (Exp 2).

Training forward pass:
  1. context_encoder(context_data) → (B, 2, 512)
  2. Pre-decoder auxiliary heads on raw context (margin + win prediction)
  3. context_pooling(concat(home, away)) → cond (B, 512) for adaLN-Zero
  4. context_score_bias(context_tokens) → (B, 7) — direct shortcut
  5. Apply context dropout (zero out cond AND score bias with p)
  6. state_embedder(states[:, :-1]) → (B, T-1, 512) — teacher forcing
  7. Apply score jitter to input states during training
  8. decoder(state_embeds, cond) → (B, T-1, 512) — adaLN-Zero modulation
  9. score_head(decoder_out) + score_bias → (B, T-1, 7)
  10. clock_head(decoder_out) → (B, T-1)
  11. context_margin_head(position 0 output) → (B, 1)
"""

import torch
import torch.nn as nn

from src.generative.config import GenerativeModelConfig
from src.generative.models.context_encoder import ContextEncoder
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


class GenerativeModel(nn.Module):
    """Top-level generative NBA game model with adaLN-Zero conditioning.

    Context modulates every decoder layer via Adaptive Layer Normalization,
    making it impossible for the decoder to ignore team context.
    """

    def __init__(self, config: GenerativeModelConfig) -> None:
        super().__init__()
        self.config = config

        # Sub-modules
        self.context_encoder = ContextEncoder(config)
        self.state_embedder = StateEmbedder(config)
        self.decoder = CausalDecoder(config)
        self.score_head = ScoreHead(config.hidden_dim, config.head_hidden_dim)
        self.clock_head = ClockHead(config.hidden_dim, config.head_hidden_dim)
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

    def forward(self, batch: dict) -> dict:
        """Training forward pass with teacher forcing and adaLN-Zero conditioning.

        Args:
            batch: dict containing context data keys + "states" (B, T, 7).

        Returns:
            dict with score_logits, clock_preds, context_margin_pred,
            pre_margin_pred, pre_win_pred.
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
        score_bias = self.context_score_bias(context_tokens)  # (B, 7)

        # 5. Context dropout: zero out cond AND score bias
        if self.training and self.config.context_dropout > 0:
            B = cond.shape[0]
            drop_mask = torch.rand(B, device=cond.device) < self.config.context_dropout
            cond = cond.masked_fill(drop_mask.unsqueeze(-1), 0.0)
            score_bias = score_bias.masked_fill(drop_mask.unsqueeze(-1), 0.0)

        # 6. State embedding with teacher forcing
        states = batch["states"]  # (B, T, 7)
        input_states = states[:, :-1, :]  # (B, T-1, 7)

        # 7. Score jitter during training
        if self.training:
            jitter_std = getattr(self.config, "score_jitter_std", None)
            if jitter_std is None:
                jitter_std = batch.get("score_jitter_std", 0.0)
            if jitter_std and jitter_std > 0:
                noise = torch.randn_like(input_states[:, :, 3:7]) * (jitter_std / 150.0)
                input_states = input_states.clone()
                input_states[:, :, 3:7] = input_states[:, :, 3:7] + noise

        state_embeds = self.state_embedder(input_states)  # (B, T-1, 512)

        # 8. Decode with adaLN-Zero conditioning (state tokens only, no context prefix)
        decoder_out = self.decoder(state_embeds, cond)  # (B, T-1, 512)

        # 9. Prediction heads
        score_logits = self.score_head(decoder_out) + score_bias.unsqueeze(
            1
        )  # (B, T-1, 7)
        clock_preds = self.clock_head(decoder_out).squeeze(-1)  # (B, T-1)

        # 10. Context margin head on first state position
        context_margin_pred = self.context_margin_head(decoder_out[:, 0, :]).squeeze(
            -1
        )  # (B,)

        return {
            "score_logits": score_logits,
            "clock_preds": clock_preds,
            "context_margin_pred": context_margin_pred,
            "pre_margin_pred": pre_margin_pred,
            "pre_win_pred": pre_win_pred,
        }

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
            (B, 7) score logit bias.
        """
        return self.context_score_bias(context_tokens)
