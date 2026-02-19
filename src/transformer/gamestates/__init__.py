"""
GameStates-only model for NBA game prediction.

Uses score trajectory data from the GameStates table (period, clock, scores, margin)
instead of full play-by-play data. Tests whether score evolution alone is sufficient
for game prediction, or whether detailed PBP information contributes signal.

Architecture: GameStateEncoder → TemporalAttention → SimpleFusion → PredictionHeads
"""

from src.transformer.gamestates.tokenizer import GameStateTokenizer
from src.transformer.gamestates.sequence_builder import GameStateSequenceBuilder
from src.transformer.gamestates.dataset import GameStatesDataset, collate_gamestates
from src.transformer.gamestates.model import GameStateModel

__all__ = [
    "GameStateTokenizer",
    "GameStateSequenceBuilder",
    "GameStatesDataset",
    "collate_gamestates",
    "GameStateModel",
]
