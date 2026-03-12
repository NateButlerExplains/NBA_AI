"""Top-level generative model: wires context encoder, state embedder, decoder, and heads.

Training forward pass:
  1. context_encoder(context_data) → (B, 2, 512)
  2. Apply context dropout (zero out context tokens with p=0.1)
  3. state_embedder(states[:, :-1]) → (B, T-1, 512)  — teacher forcing
  4. Apply score jitter to input states during training
  5. decoder(context_tokens, state_embeds) → (B, 2+T-1, 512)
  6. Extract state outputs (positions 2:) → (B, T-1, 512)
  7. score_head(state_outputs) → (B, T-1, 7)
  8. clock_head(state_outputs) → (B, T-1, 1)
  9. context_margin_head(position 2 output) → (B, 1)
"""

import torch
import torch.nn as nn

from src.generative.config import GenerativeModelConfig
from src.generative.models.context_encoder import ContextEncoder
from src.generative.models.state_embedder import StateEmbedder
from src.generative.models.causal_decoder import CausalDecoder
from src.generative.models.prediction_heads import ScoreHead, ClockHead, ContextMarginHead


class GenerativeModel(nn.Module):
    """Top-level generative NBA game model.

    Combines context encoding (team history), state embedding (game state sequence),
    causal decoding, and prediction heads for score events, clock, and margin.
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
        self.context_margin_head = ContextMarginHead(config.hidden_dim, config.head_hidden_dim)

    def forward(self, batch: dict) -> dict:
        """Training forward pass with teacher forcing.

        Args:
            batch: dict containing:
                - All context_data keys (see ContextEncoder.forward)
                - "states": (B, T, 7) full game state sequence
                - "score_jitter_std": float (from data config, passed through)

        Returns:
            dict with:
                score_logits: (B, T-1, 7) score event logits
                clock_preds: (B, T-1) clock predictions
                context_margin_pred: (B,) expected final margin
        """
        # 1. Encode context
        context_tokens = self.context_encoder(batch)  # (B, 2, 512)

        # 2. Context dropout: zero out context tokens with probability p during training
        if self.training and self.config.context_dropout > 0:
            B = context_tokens.shape[0]
            # Per-sample dropout: same mask for both home/away tokens
            drop_mask = torch.rand(B, 1, 1, device=context_tokens.device) < self.config.context_dropout
            context_tokens = context_tokens.masked_fill(drop_mask, 0.0)

        # 3. State embedding with teacher forcing: use states[0:T-1] as input
        states = batch["states"]  # (B, T, 7)
        input_states = states[:, :-1, :]  # (B, T-1, 7) — shift right

        # 4. Score jitter during training: add noise to score-related dims (indices 3,4,5,6)
        if self.training:
            jitter_std = getattr(self.config, "score_jitter_std", None)
            if jitter_std is None:
                # Fall back to batch-provided value
                jitter_std = batch.get("score_jitter_std", 0.0)
            if jitter_std > 0:
                noise = torch.randn_like(input_states[:, :, 3:7]) * (jitter_std / 150.0)
                input_states = input_states.clone()
                input_states[:, :, 3:7] = input_states[:, :, 3:7] + noise

        state_embeds = self.state_embedder(input_states)  # (B, T-1, 512)

        # 5. Decode: context + state sequence
        decoder_out = self.decoder(context_tokens, state_embeds)  # (B, 2+T-1, 512)

        # 6. Extract state outputs (positions 2 onward, after 2 context tokens)
        state_outputs = decoder_out[:, 2:, :]  # (B, T-1, 512)

        # 7. Prediction heads
        score_logits = self.score_head(state_outputs)    # (B, T-1, 7)
        clock_preds = self.clock_head(state_outputs)     # (B, T-1, 1)
        clock_preds = clock_preds.squeeze(-1)            # (B, T-1)

        # 8. Context margin head: operates on position 2 (first state position)
        context_margin_pred = self.context_margin_head(decoder_out[:, 2, :])  # (B, 1)
        context_margin_pred = context_margin_pred.squeeze(-1)                 # (B,)

        return {
            "score_logits": score_logits,
            "clock_preds": clock_preds,
            "context_margin_pred": context_margin_pred,
        }

    def encode_context(self, context_data: dict) -> torch.Tensor:
        """Encode context only (for inference caching).

        Args:
            context_data: dict with all context keys (see ContextEncoder.forward).

        Returns:
            (B, 2, 512) context tokens.
        """
        return self.context_encoder(context_data)
