"""Phase 4: Autoregressive Game State Prediction.

Clean-slate generative approach — predicts next game state autoregressively,
analogous to next-token prediction. Supports pre-game (full rollout from 0-0)
and in-game (condition on observed states) prediction.

No imports from src/transformer/. Shared infrastructure only: src/database, config.yaml.
"""

__version__ = "0.1.0"
