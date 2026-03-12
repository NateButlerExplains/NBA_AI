"""State embedder: 7-dim game state vector → 512-d embedding.

Linear(7, 256) → LayerNorm → GELU → Linear(256, 512) → LayerNorm
"""

import torch
import torch.nn as nn

from src.generative.config import GenerativeModelConfig


class StateEmbedder(nn.Module):
    """Embed 7-dim game state vector into hidden_dim (512-d)."""

    def __init__(self, config: GenerativeModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.state_input_dim, config.state_hidden_dim),   # 7 → 256
            nn.LayerNorm(config.state_hidden_dim),
            nn.GELU(),
            nn.Linear(config.state_hidden_dim, config.hidden_dim),        # 256 → 512
            nn.LayerNorm(config.hidden_dim),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Embed game state vectors.

        Args:
            states: (B, T, 7) game state vectors.

        Returns:
            (B, T, 512) state embeddings.
        """
        return self.net(states)
