"""
Neural network architectures for the transformer-based prediction model.

Components:
    - EventEncoder: Transformer for PBP sequences
    - TemporalAttention: Attention over historical games
    - SimpleFusion: Combine team history streams with MLP
    - PredictionHeads: Probabilistic outputs (μ, σ)
    - Phase1Model: Complete model
"""

from src.transformer.models.event_encoder import (
    EventEncoder,
    EventEmbedding,
    PositionalEncoding,
)
from src.transformer.models.temporal_attention import (
    TemporalAttention,
    TemporalAttentionLayer,
    LearnedPositionalEncoding,
    SinusoidalRecencyEncoding,
)
from src.transformer.models.fusion import (
    SimpleFusion,
    CrossAttentionFusion,
)
from src.transformer.models.roster_encoder import (
    RosterEncoder,
)
from src.transformer.models.prediction_heads import (
    PredictionHeads,
    GamePrediction,
    GaussianHead,
    ScoreHead,
)
from src.transformer.training.config import ModelConfig
from src.transformer.models.phase1_model import (
    Phase1Model,
    create_model_from_tokenizer,
)

__all__ = [
    # Event Encoder
    "EventEncoder",
    "EventEmbedding",
    "PositionalEncoding",
    # Temporal Attention
    "TemporalAttention",
    "TemporalAttentionLayer",
    "LearnedPositionalEncoding",
    "SinusoidalRecencyEncoding",
    # Fusion
    "SimpleFusion",
    "CrossAttentionFusion",
    # Roster Encoder
    "RosterEncoder",
    # Prediction Heads
    "PredictionHeads",
    "GamePrediction",
    "GaussianHead",
    "ScoreHead",
    # Complete Model
    "Phase1Model",
    "ModelConfig",
    "create_model_from_tokenizer",
]
