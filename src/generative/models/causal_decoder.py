"""Causal decoder with adaLN-Zero conditioning.

Instead of prepending context tokens to the sequence (which the decoder learns
to ignore), context modulates every LayerNorm in every layer via Adaptive
Layer Normalization. This makes context impossible to ignore.

Reference: Peebles & Xie, "Scalable Diffusion Models with Transformers" (ICCV 2023)

adaLN-Zero: modulation weights initialized to zero so initial behavior is
identity (standard pre-norm transformer). Context influence grows organically
during training.

Supports both full-sequence training and incremental inference with KV cache.
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from src.generative.config import GenerativeModelConfig  # noqa: F401


def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Upper-triangular causal mask: -inf above diagonal, 0 on/below."""
    return nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)


class AdaLNCausalDecoderLayer(nn.Module):
    """Decoder layer with adaLN-Zero conditioning.

    Pre-norm architecture where LayerNorm parameters are predicted from
    the context conditioning vector:
      gamma1, beta1, alpha1 = modulate(cond)  # for attention block
      gamma2, beta2, alpha2 = modulate(cond)  # for FF block

      h = LN(x) * (1 + gamma1) + beta1
      x = x + alpha1 * self_attn(h)
      h = LN(x) * (1 + gamma2) + beta2
      x = x + alpha2 * ff(h)

    alpha gates the residual connection (zero-initialized → identity at start).
    """

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float, cond_dim: int
    ) -> None:
        super().__init__()
        # LayerNorm without learnable affine — adaLN provides scale/shift
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        # adaLN modulation: cond → (gamma1, beta1, alpha1, gamma2, beta2, alpha2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * d_model),
        )
        # Zero-initialize so initial behavior is identity
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with adaLN-Zero conditioning and optional KV cache.

        Args:
            x: (B, T, D) input — full sequence (training) or single token (inference).
            cond: (B, D) context conditioning vector.
            attn_mask: optional (T, T) causal mask for training.
            kv_cache: optional (cached_keys, cached_values), each (B, S, D).

        Returns:
            (output, updated_kv_cache) — cache is None during training.
        """
        # Compute modulation parameters from context
        modulation = self.adaLN_modulation(cond)  # (B, 6*D)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = modulation.chunk(6, dim=-1)

        # --- Modulated pre-norm attention ---
        h = self.norm1(x) * (1 + gamma1.unsqueeze(1)) + beta1.unsqueeze(1)

        new_kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None

        if kv_cache is not None:
            # Inference mode: single query token attending to cache + self
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, h], dim=1)
            v = torch.cat([cached_v, h], dim=1)
            attn_out, _ = self.self_attn(query=h, key=k, value=v)
            new_kv_cache = (k, v)
        else:
            # Training mode: full sequence with causal mask
            attn_out, _ = self.self_attn(
                query=h,
                key=h,
                value=h,
                attn_mask=attn_mask,
            )

        x = x + alpha1.unsqueeze(1) * attn_out

        # --- Modulated pre-norm FF ---
        h = self.norm2(x) * (1 + gamma2.unsqueeze(1)) + beta2.unsqueeze(1)
        h = self.ff(h)
        x = x + alpha2.unsqueeze(1) * h

        return x, new_kv_cache


class CausalDecoder(nn.Module):
    """Stack of AdaLNCausalDecoderLayers with positional encoding.

    Context is injected via adaLN-Zero modulation at every layer, NOT as
    prepended tokens. The sequence contains only state tokens.

    Sinusoidal positional encoding (max_len=800).
    """

    def __init__(self, config: GenerativeModelConfig) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim

        self.layers = nn.ModuleList(
            [
                AdaLNCausalDecoderLayer(
                    d_model=config.hidden_dim,
                    n_heads=config.decoder_heads,
                    d_ff=config.decoder_ff_dim,
                    dropout=config.decoder_dropout,
                    cond_dim=config.hidden_dim,
                )
                for _ in range(config.decoder_layers)
            ]
        )

        # Final norm with adaLN modulation
        self.norm = nn.LayerNorm(config.hidden_dim, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 2 * config.hidden_dim),
        )
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)

        pe = self._build_sinusoidal_pe(config.decoder_max_seq_len, config.hidden_dim)
        self.register_buffer("pos_encoding", pe)

        # Pre-compute causal mask for max sequence length (avoids recomputation per forward)
        causal = _causal_mask(config.decoder_max_seq_len, device=torch.device("cpu"))
        self.register_buffer("_causal_mask_buf", causal)

    @staticmethod
    def _build_sinusoidal_pe(max_len: int, dim: int) -> torch.Tensor:
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * -(math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(
        self,
        state_embeds: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Training mode: full sequence with causal masking and adaLN conditioning.

        Args:
            state_embeds: (B, T, 512) from state embedder.
            cond: (B, 512) context conditioning vector.

        Returns:
            (B, T, 512) decoder output for all state positions.
        """
        B, T, D = state_embeds.shape

        # Add positional encoding (no context prefix offset)
        x = state_embeds + self.pos_encoding[:T].unsqueeze(0)

        # Slice pre-computed causal mask to current sequence length
        mask = self._causal_mask_buf[:T, :T]

        for layer in self.layers:
            x, _ = layer(x, cond, attn_mask=mask, kv_cache=None)

        # Final adaLN-modulated norm
        gamma, beta = self.final_adaLN(cond).chunk(2, dim=-1)
        x = self.norm(x) * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        return x

    def init_kv_cache(
        self, batch_size: int, device: torch.device
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Initialize empty KV caches (no context prefix to process).

        Args:
            batch_size: number of parallel rollouts.
            device: target device.

        Returns:
            List of (empty_keys, empty_values) per layer.
        """
        return [
            (
                torch.empty(batch_size, 0, self.hidden_dim, device=device),
                torch.empty(batch_size, 0, self.hidden_dim, device=device),
            )
            for _ in self.layers
        ]

    def decode_step(
        self,
        token_embed: torch.Tensor,
        cond: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]],
        step: int,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Inference mode: single token with KV cache and adaLN conditioning.

        Args:
            token_embed: (B, 1, 512) current token embedding.
            cond: (B, 512) context conditioning vector.
            kv_caches: list of (cached_k, cached_v) per layer.
            step: current position in sequence (0-indexed).

        Returns:
            (output, updated_kv_caches)
        """
        x = token_embed + self.pos_encoding[step : step + 1].unsqueeze(0)

        updated_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer, cache in zip(self.layers, kv_caches):
            x, new_cache = layer(x, cond, attn_mask=None, kv_cache=cache)
            assert new_cache is not None
            updated_caches.append(new_cache)

        # Final adaLN-modulated norm
        gamma, beta = self.final_adaLN(cond).chunk(2, dim=-1)
        x = self.norm(x) * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        return x, updated_caches
