"""
Datasets for L2 Player Synergy Network training.

Two dataset types:
1. PairDataset — samples of player pairs with WOWY targets (primary training signal)
2. LineupDataset — samples of 5-man lineups with net rating targets (secondary)

Both datasets load from:
- L1 ability vectors (pre-computed per player per game)
- WOWY cache (pairwise stats from stint extraction)
- PlayerAttributes (for position, age features)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .l2_config import L2Config

logger = logging.getLogger(__name__)

L2_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "l2_cache"
L1_VECTORS_DIR = L2_CACHE_DIR / "l1_vectors"


def load_l2_metadata() -> dict:
    """Load L2 cache metadata."""
    path = L2_CACHE_DIR / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"L2 metadata not found at {path}. Build caches first.")
    with open(path) as f:
        return json.load(f)


class PairDataset(Dataset):
    """
    Dataset of player pairs with WOWY synergy targets.

    Each sample provides:
    - Two players' L1 ability vectors (at the time of their shared games)
    - Their archetype weights
    - Pairwise edge features (shared_minutes, years_together, etc.)
    - WOWY residual target (the pure interaction effect)
    - Shared possessions (for loss weighting)
    """

    def __init__(
        self,
        pair_data: list[dict],
        l1_vectors: dict[int, dict],
        player_to_idx: dict[int, int],
        cfg: L2Config,
    ):
        """
        Args:
            pair_data: list of dicts with keys:
                player_a, player_b, season,
                wowy_residual, shared_possessions,
                minutes_together, years_together, etc.
            l1_vectors: {player_id: {ability: (N,32), game_ids: (N,), archetype: (10,)}}
            player_to_idx: {player_id: embedding_index} for FM lookup
            cfg: L2Config
        """
        self.pairs = pair_data
        self.l1_vectors = l1_vectors
        self.player_to_idx = player_to_idx
        self.cfg = cfg
        logger.info(f"PairDataset: {len(self.pairs)} pairs")

    def __len__(self) -> int:
        return len(self.pairs)

    def _get_ability(
        self, player_id: int, season: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get player's ability vector and archetype for a season (use latest game in season)."""
        data = self.l1_vectors.get(player_id)
        if data is None:
            return np.zeros(self.cfg.d_ability, dtype=np.float32), np.zeros(
                self.cfg.n_archetypes, dtype=np.float32
            )

        # Find games in this season
        game_ids = data["game_ids"]
        season_start = int(season.split("-")[0]) - 2000
        season_mask = []
        for gid in game_ids:
            try:
                code = int(str(gid)[3:5])
                season_mask.append(code == season_start)
            except (ValueError, IndexError):
                season_mask.append(False)

        season_mask = np.array(season_mask)
        if season_mask.any():
            # Use last game in the season
            last_idx = np.where(season_mask)[0][-1]
            ability = data["ability"][last_idx]
        else:
            # Fallback: use last available game before this season
            ability = data["ability"][-1]

        archetype = data.get(
            "archetype_weights", np.zeros(self.cfg.n_archetypes, dtype=np.float32)
        )
        return ability.astype(np.float32), archetype.astype(np.float32)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        pair = self.pairs[index]
        pid_a = pair["player_a"]
        pid_b = pair["player_b"]
        season = pair["season"]

        # L1 ability vectors
        ability_a, arch_a = self._get_ability(pid_a, season)
        ability_b, arch_b = self._get_ability(pid_b, season)

        # FM embedding indices
        idx_a = self.player_to_idx.get(pid_a, 0)
        idx_b = self.player_to_idx.get(pid_b, 0)

        # Edge features (7 precomputed + 2 computed from L1)
        ability_cosine = float(
            np.dot(ability_a, ability_b)
            / (np.linalg.norm(ability_a) * np.linalg.norm(ability_b) + 1e-8)
        )
        positional_overlap = float(
            np.dot(arch_a, arch_b)
        )  # archetype similarity as proxy

        return {
            "ability": torch.tensor(
                np.stack([ability_a, ability_b]), dtype=torch.float32
            ),  # (2, 32)
            "archetypes": torch.tensor(
                np.stack([arch_a, arch_b]), dtype=torch.float32
            ),  # (2, 10)
            "player_idx": torch.tensor([idx_a, idx_b], dtype=torch.long),  # (2,)
            "target_wowy": torch.tensor(pair["wowy_residual"], dtype=torch.float32),
            "shared_possessions": torch.tensor(
                pair["shared_possessions"], dtype=torch.float32
            ),
            # Precomputed edge features (for logging/analysis, not used in pairwise prediction)
            "minutes_together": torch.tensor(
                pair.get("minutes_together", 0.0), dtype=torch.float32
            ),
            "years_together": torch.tensor(
                pair.get("years_together", 0), dtype=torch.float32
            ),
        }


class LineupDataset(Dataset):
    """
    Dataset of 5-man lineups with net rating targets.

    Each sample provides a full team lineup (padded to max_roster)
    with all players' L1 vectors and pairwise edge features.
    """

    def __init__(
        self,
        lineup_data: list[dict],
        l1_vectors: dict[int, dict],
        player_to_idx: dict[int, int],
        pair_features: dict[tuple[int, int, str], dict],
        cfg: L2Config,
    ):
        """
        Args:
            lineup_data: list of dicts with keys:
                player_ids (list of 5), team_id, season,
                net_rating, possessions
            l1_vectors: {player_id: {ability, game_ids, archetype_weights}}
            player_to_idx: {player_id: embedding_index}
            pair_features: {(player_a, player_b, season): {shared_minutes, wowy_net_rtg, ...}}
            cfg: L2Config
        """
        self.lineups = lineup_data
        self.l1_vectors = l1_vectors
        self.player_to_idx = player_to_idx
        self.pair_features = pair_features
        self.cfg = cfg
        logger.info(f"LineupDataset: {len(self.lineups)} lineups")

    def __len__(self) -> int:
        return len(self.lineups)

    def _get_ability(
        self, player_id: int, season: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get ability, uncertainty, archetype for a player in a season."""
        data = self.l1_vectors.get(player_id)
        if data is None:
            return (
                np.zeros(self.cfg.d_ability, dtype=np.float32),
                np.ones(self.cfg.d_ability, dtype=np.float32) * 0.5,
                np.ones(self.cfg.n_archetypes, dtype=np.float32)
                / self.cfg.n_archetypes,
            )

        game_ids = data["game_ids"]
        season_start = int(season.split("-")[0]) - 2000
        season_mask = np.array([int(str(gid)[3:5]) == season_start for gid in game_ids])

        if season_mask.any():
            last_idx = np.where(season_mask)[0][-1]
        else:
            last_idx = len(game_ids) - 1

        ability = data["ability"][last_idx].astype(np.float32)
        uncertainty = data.get("uncertainty", np.ones_like(ability) * 0.5)
        if uncertainty.ndim == 2:
            uncertainty = uncertainty[last_idx].astype(np.float32)
        archetype = data.get(
            "archetype_weights",
            np.ones(self.cfg.n_archetypes, dtype=np.float32) / self.cfg.n_archetypes,
        )
        return ability, uncertainty, archetype.astype(np.float32)

    def _get_edge_features(
        self, pid_a: int, pid_b: int, season: str, ab_a: np.ndarray, ab_b: np.ndarray
    ) -> np.ndarray:
        """Build 9-d edge feature vector for a pair."""
        key = (min(pid_a, pid_b), max(pid_a, pid_b), season)
        pf = self.pair_features.get(key, {})

        # Features 0-1: archetype_syn and fm_syn are filled by the model
        # Features 2-8: precomputed
        features = np.zeros(self.cfg.n_edge_features, dtype=np.float32)
        features[2] = np.log1p(pf.get("shared_minutes", 0.0))  # log-scaled
        features[3] = min(pf.get("years_together", 0), 5) / 5.0  # capped, normalized
        features[4] = pf.get("wowy_net_rtg", 0.0) / 10.0  # scaled
        # Features 5-6: positional overlap and minutes overlap
        features[5] = 0.0  # placeholder — computed from archetype in model
        features[6] = pf.get("minutes_overlap_pct", 0.0)
        # Feature 7: ability cosine similarity
        norm_a = np.linalg.norm(ab_a)
        norm_b = np.linalg.norm(ab_b)
        features[7] = float(np.dot(ab_a, ab_b) / (norm_a * norm_b + 1e-8))
        # Feature 8: age gap (placeholder — need birth years)
        features[8] = 0.0

        return features

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        lineup = self.lineups[index]
        player_ids = lineup["player_ids"]
        season = lineup["season"]
        A = len(player_ids)
        max_A = self.cfg.max_roster

        # Pad to max_roster
        abilities = np.zeros((max_A, self.cfg.d_ability), dtype=np.float32)
        uncertainties = np.zeros((max_A, self.cfg.d_ability), dtype=np.float32)
        archetypes = np.zeros((max_A, self.cfg.n_archetypes), dtype=np.float32)
        player_idx = np.zeros(max_A, dtype=np.int64)
        mask = np.zeros(max_A, dtype=bool)

        for i, pid in enumerate(player_ids):
            ab, unc, arch = self._get_ability(pid, season)
            abilities[i] = ab
            uncertainties[i] = unc
            archetypes[i] = arch
            player_idx[i] = self.player_to_idx.get(pid, 0)
            mask[i] = True

        # Build pairwise edge features (max_A × max_A × n_edge)
        edge_features = np.zeros(
            (max_A, max_A, self.cfg.n_edge_features), dtype=np.float32
        )
        for i in range(A):
            for j in range(A):
                if i != j:
                    edge_features[i, j] = self._get_edge_features(
                        player_ids[i],
                        player_ids[j],
                        season,
                        abilities[i],
                        abilities[j],
                    )

        return {
            "ability": torch.tensor(abilities),  # (max_A, 32)
            "uncertainty": torch.tensor(uncertainties),  # (max_A, 32)
            "archetypes": torch.tensor(archetypes),  # (max_A, 10)
            "player_idx": torch.tensor(player_idx),  # (max_A,)
            "mask": torch.tensor(mask),  # (max_A,)
            "edge_features": torch.tensor(edge_features),  # (max_A, max_A, 9)
            "target_net_rtg": torch.tensor(lineup["net_rating"], dtype=torch.float32),
            "possessions": torch.tensor(lineup["possessions"], dtype=torch.float32),
            "n_players": torch.tensor(A, dtype=torch.long),
        }
