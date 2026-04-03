"""
Phase 5 Hierarchical Predictor: L1→L2→L3→L4 inference pipeline.

Implements the BasePredictor interface for integration with the existing
prediction system. For each game: assembles roster, loads L1 vectors,
runs L2→L3→L4 forward pass, returns spread/win/total predictions.

Usage:
    from src.pipeline.phase5_predictor import Phase5Predictor
    predictor = Phase5Predictor()
    preds = predictor.make_pre_game_predictions(["0022500900"])
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from src.phase5.cache_builder import ALTITUDE_MAP, BOX_STAT_COLUMNS, CONTEXT_COLUMNS
from src.phase5.config import NKEHConfig
from src.phase5.dataset import Normalizer, load_metadata, load_profiles
from src.phase5.l2_config import L2Config
from src.phase5.l2_model import PlayerSynergyNetwork
from src.phase5.l3_config import L3Config
from src.phase5.l3_model import TeamModel
from src.phase5.l4_config import L4Config
from src.phase5.l4_model import GamePredictor
from src.phase5.model import NKEH
from src.pipeline.roster_assembler import RosterAssembler
from src.pipeline.team_features import TeamFeatureComputer
from src.predictions.prediction_engines.base_predictor import BasePredictor

logger = logging.getLogger(__name__)

DB_PATH = PROJECT_ROOT / "data" / "NBA_AI_full.sqlite"
L1_VECTORS_DIR = PROJECT_ROOT / "data" / "l2_cache" / "l1_vectors"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "phase5"
PHASE_B_CACHE = PROJECT_ROOT / "data" / "phase_b_cache"
L1_CACHE_DIR = PROJECT_ROOT / "data" / "phase5_cache"

# Maximum players per team for L2 (must match training)
MAX_PLAYERS = 13


class Phase5Predictor(BasePredictor):
    """Phase 5 Hierarchical model predictor."""

    def __init__(self, model_paths=None):
        super().__init__(model_paths)
        self._models_loaded = False
        self._device = None
        self._l2_model = None
        self._l3_model = None
        self._l4_model = None
        self._l2_cfg = None
        self._roster_assembler = RosterAssembler()
        self._team_features = TeamFeatureComputer()
        self._l1_updater = None  # Lazy-loaded on first freshness check

        # Normalization stats and player index mapping
        self._l3_mean = None
        self._l3_std = None
        self._l4_mean = None
        self._l4_std = None
        self._rs_mean = None
        self._rs_std = None
        self._player_to_idx = None
        self._archetype_centroids = None

        # L1 model for player stat predictions (lazy-loaded)
        self._l1_model = None
        self._l1_normalizer = None
        self._l1_metadata = None
        self._l1_profiles_data = None
        self._l1_profile_idx = None
        self._l1_box_mean = None
        self._l1_box_std = None

    def _ensure_models_loaded(self):
        """Lazy-load models on first prediction call."""
        if self._models_loaded:
            return

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load configs
        self._l2_cfg = L2Config()
        l3_cfg = L3Config()
        l4_cfg = L4Config()

        # Load Phase B metadata for normalization and player mapping
        meta_path = PHASE_B_CACHE / "metadata.json"
        meta = json.loads(meta_path.read_text())
        norm = meta["normalization"]
        self._l3_mean = np.array(norm["l3_mean"], dtype=np.float32)
        self._l3_std = np.array(norm["l3_std"], dtype=np.float32)
        self._l4_mean = np.array(norm["l4_mean"], dtype=np.float32)
        self._l4_std = np.array(norm["l4_std"], dtype=np.float32)
        self._rs_mean = np.array(norm["roster_summary_mean"], dtype=np.float32)
        self._rs_std = np.array(norm["roster_summary_std"], dtype=np.float32)

        # Player-to-index mapping for L2 FM residual embeddings
        idx_path = PHASE_B_CACHE / "player_to_idx.json"
        if idx_path.exists():
            self._player_to_idx = {
                int(k): v for k, v in json.loads(idx_path.read_text()).items()
            }
        else:
            self._player_to_idx = {}

        # Ensure L2 has enough embedding slots
        if self._player_to_idx:
            self._l2_cfg.n_players = max(
                self._l2_cfg.n_players, max(self._player_to_idx.values()) + 1
            )

        # Load archetype centroids for soft assignment
        arch_path = PROJECT_ROOT / "data" / "phase5_cache" / "archetypes.npz"
        if arch_path.exists():
            self._archetype_centroids = np.load(str(arch_path))["centroids"]  # (K, 32)
        else:
            self._archetype_centroids = None

        # Create models
        self._l2_model = PlayerSynergyNetwork(self._l2_cfg).to(self._device)
        self._l3_model = TeamModel(l3_cfg).to(self._device)
        self._l4_model = GamePredictor(l4_cfg).to(self._device)

        # Load checkpoints
        l2_ckpt = torch.load(
            str(CHECKPOINT_DIR / "l2_best.pt"),
            map_location=self._device,
            weights_only=False,
        )
        self._l2_model.load_state_dict(l2_ckpt["model_state_dict"], strict=False)

        c_ckpt = torch.load(
            str(CHECKPOINT_DIR / "phase_c_best.pt"),
            map_location=self._device,
            weights_only=False,
        )
        self._l3_model.load_state_dict(c_ckpt["l3_state_dict"], strict=False)
        self._l4_model.load_state_dict(c_ckpt["l4_state_dict"], strict=False)

        self._l2_model.eval()
        self._l3_model.eval()
        self._l4_model.eval()

        logger.info(
            f"Phase 5 models loaded on {self._device}: "
            f"L2={sum(p.numel() for p in self._l2_model.parameters()):,}, "
            f"L3={sum(p.numel() for p in self._l3_model.parameters()):,}, "
            f"L4={sum(p.numel() for p in self._l4_model.parameters()):,}"
        )
        self._models_loaded = True

    def _ensure_l1_fresh(self):
        """Update L1 vectors if any players have new games since last update."""
        try:
            if self._l1_updater is None:
                from src.pipeline.l1_updater import L1IncrementalUpdater

                self._l1_updater = L1IncrementalUpdater(
                    db_path=str(DB_PATH), device="cpu"
                )
            results = self._l1_updater.update_all_new()
            if results:
                total = sum(results.values())
                logger.info(
                    f"L1 vectors updated: {len(results)} players, {total} new games"
                )
        except Exception as e:
            logger.warning(f"L1 freshness check failed (using cached vectors): {e}")

    def make_pre_game_predictions(self, game_ids: list[str]) -> dict:
        """
        Generate pre-game predictions for the given games.

        Automatically updates L1 vectors if new games have been played since
        the last update, ensuring fresh player ability estimates.

        Returns {game_id: {
            pred_home_score, pred_away_score, pred_home_win_pct,
            pred_spread, pred_spread_sigma, pred_total, pred_total_sigma,
            pred_ats_prob, roster_confidence
        }}
        """
        self._ensure_l1_fresh()
        self._ensure_models_loaded()

        # Step 1: Assemble rosters
        rosters = self._roster_assembler.get_projected_rosters(game_ids)

        # Step 2: Compute team features and game context
        team_feats = self._team_features.compute_for_games(game_ids)

        # Step 3: For each game, run L1→L2→L3→L4
        predictions = {}
        for game_id in game_ids:
            if game_id not in rosters or game_id not in team_feats:
                logger.warning(f"Missing data for game {game_id}, skipping")
                continue

            try:
                pred = self._predict_single_game(
                    game_id, rosters[game_id], team_feats[game_id]
                )
                predictions[game_id] = pred
            except Exception as e:
                logger.warning(f"Error predicting game {game_id}: {e}")

        return predictions

    def _predict_single_game(self, game_id: str, roster: dict, features: dict) -> dict:
        """Run the full L2→L3→L4 forward pass for a single game."""
        device = self._device

        # Load L1 vectors for all players
        home_data = self._load_player_data(roster["home_players"])
        away_data = self._load_player_data(roster["away_players"])

        # Normalize team features and game context
        home_tf = self._normalize(
            features["home_team_features"], self._l3_mean, self._l3_std
        )
        away_tf = self._normalize(
            features["away_team_features"], self._l3_mean, self._l3_std
        )
        game_ctx = self._normalize(
            features["game_context"], self._l4_mean, self._l4_std
        )

        # Compute roster summaries (12-d: mean ability stats)
        home_rs = self._compute_roster_summary(home_data)
        away_rs = self._compute_roster_summary(away_data)
        home_rs_norm = self._normalize(home_rs, self._rs_mean, self._rs_std)
        away_rs_norm = self._normalize(away_rs, self._rs_mean, self._rs_std)

        # Continuity from team features (index 32, raw)
        home_cont = features["home_team_features"][32]
        away_cont = features["away_team_features"][32]

        # Convert to tensors
        with torch.no_grad():
            # L2 forward
            home_l2_out = self._l2_model(
                ability=home_data["abilities"].unsqueeze(0).to(device),
                uncertainty=home_data["uncertainties"].unsqueeze(0).to(device),
                archetypes=home_data["archetypes"].unsqueeze(0).to(device),
                mask=home_data["mask"].unsqueeze(0).to(device),
                player_idx=home_data["player_idx"].unsqueeze(0).to(device),
            )
            away_l2_out = self._l2_model(
                ability=away_data["abilities"].unsqueeze(0).to(device),
                uncertainty=away_data["uncertainties"].unsqueeze(0).to(device),
                archetypes=away_data["archetypes"].unsqueeze(0).to(device),
                mask=away_data["mask"].unsqueeze(0).to(device),
                player_idx=away_data["player_idx"].unsqueeze(0).to(device),
            )

            home_l2 = home_l2_out["team_vector"]  # (1, 134)
            away_l2 = away_l2_out["team_vector"]

            # L3 forward
            home_repr = self._l3_model(
                l2_team=home_l2,
                team_features=torch.tensor(home_tf, dtype=torch.float32)
                .unsqueeze(0)
                .to(device),
                roster_summary=torch.tensor(home_rs_norm, dtype=torch.float32)
                .unsqueeze(0)
                .to(device),
                coach_idx=torch.tensor([0], dtype=torch.long).to(device),
                roster_continuity=torch.tensor([[home_cont]], dtype=torch.float32).to(
                    device
                ),
                coach_games=torch.tensor([0.0], dtype=torch.float32).to(device),
            )
            away_repr = self._l3_model(
                l2_team=away_l2,
                team_features=torch.tensor(away_tf, dtype=torch.float32)
                .unsqueeze(0)
                .to(device),
                roster_summary=torch.tensor(away_rs_norm, dtype=torch.float32)
                .unsqueeze(0)
                .to(device),
                coach_idx=torch.tensor([0], dtype=torch.long).to(device),
                roster_continuity=torch.tensor([[away_cont]], dtype=torch.float32).to(
                    device
                ),
                coach_games=torch.tensor([0.0], dtype=torch.float32).to(device),
            )

            # L4 forward
            preds = self._l4_model(
                team_home=home_repr,
                team_away=away_repr,
                l2_home=home_l2,
                l2_away=away_l2,
                context=torch.tensor(game_ctx, dtype=torch.float32)
                .unsqueeze(0)
                .to(device),
            )

        # Extract predictions
        spread_mu = preds["spread_mu"].item()
        spread_sigma = preds["spread_sigma"].item()
        win_prob = preds["win_prob"].item()
        total_mu = preds["total_mu"].item()
        total_sigma = preds["total_sigma"].item()

        # Convert spread/total to home/away scores
        home_score = (total_mu + spread_mu) / 2.0
        away_score = (total_mu - spread_mu) / 2.0

        # Determine if this is a live prediction (before game) or backtest
        from src.database import get_db

        with get_db(str(DB_PATH)) as conn:
            row = conn.execute(
                "SELECT status, home_team, away_team, date_time_utc, season "
                "FROM Games WHERE game_id = ?",
                (game_id,),
            ).fetchone()
        game_status = row[0] if row else None
        home_team = row[1] if row else ""
        away_team = row[2] if row else ""
        date_time_utc = row[3] if row else ""
        season = row[4] if row else ""
        is_live = game_status == 1  # status=1 means scheduled (not yet played)

        # Player-level stat predictions from L1 decoder
        pred_players = self._predict_player_stats(
            roster, home_team, away_team, date_time_utc, season
        )

        return {
            "pred_home_score": round(home_score, 1),
            "pred_away_score": round(away_score, 1),
            "pred_home_win_pct": round(win_prob, 4),
            "pred_spread": round(spread_mu, 2),
            "pred_spread_sigma": round(spread_sigma, 2),
            "pred_total": round(total_mu, 1),
            "pred_total_sigma": round(total_sigma, 2),
            "prediction_type": "live" if is_live else "backtest",
            "roster_confidence": min(
                roster.get("home_confidence", 1.0),
                roster.get("away_confidence", 1.0),
            ),
            "pred_players": pred_players,
        }

    def _load_player_data(self, player_ids: list[int]) -> dict:
        """
        Load L1 vectors for a list of player IDs.

        Returns dict with abilities, uncertainties, archetypes, mask, player_idx
        all as tensors of shape (MAX_PLAYERS, ...).
        """
        abilities = np.zeros((MAX_PLAYERS, 32), dtype=np.float32)
        uncertainties = np.ones((MAX_PLAYERS, 32), dtype=np.float32)
        archetypes = np.zeros((MAX_PLAYERS, 10), dtype=np.float32)
        mask = np.zeros(MAX_PLAYERS, dtype=bool)
        player_idx = np.zeros(MAX_PLAYERS, dtype=np.int64)

        for j, pid in enumerate(player_ids[:MAX_PLAYERS]):
            npz_path = L1_VECTORS_DIR / f"{pid}.npz"
            if npz_path.exists():
                data = np.load(str(npz_path), allow_pickle=True)
                # Use the most recent ability vector
                abilities[j] = data["ability"][-1]
                uncertainties[j] = data["uncertainty"][-1]
                # Archetype weights are produced by the L1 model (static per player)
                archetypes[j] = data["archetype_weights"]
                mask[j] = True
                player_idx[j] = self._player_to_idx.get(pid, 0)
            else:
                # Player has no L1 vectors — use zero (masked out)
                mask[j] = False

        return {
            "abilities": torch.tensor(abilities, dtype=torch.float32),
            "uncertainties": torch.tensor(uncertainties, dtype=torch.float32),
            "archetypes": torch.tensor(archetypes, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "player_idx": torch.tensor(player_idx, dtype=torch.long),
        }

    def _compute_archetypes(self, ability: np.ndarray) -> np.ndarray:
        """Compute soft archetype assignment from ability vector."""
        if self._archetype_centroids is None:
            return np.ones(10, dtype=np.float32) / 10.0

        # Euclidean distance to each centroid
        dists = np.linalg.norm(
            self._archetype_centroids - ability[np.newaxis, :], axis=1
        )
        # Soft assignment via negative distance softmax
        logits = -dists
        logits = logits - logits.max()  # numerical stability
        weights = np.exp(logits)
        weights /= weights.sum() + 1e-8
        return weights.astype(np.float32)

    def _compute_roster_summary(self, player_data: dict) -> np.ndarray:
        """Compute 12-d roster summary from player abilities."""
        abilities = player_data["abilities"].numpy()  # (A, 32)
        mask = player_data["mask"].numpy()  # (A,)
        uncertainties = player_data["uncertainties"].numpy()  # (A, 32)

        n_valid = mask.sum()
        if n_valid == 0:
            return np.zeros(12, dtype=np.float32)

        valid_ab = abilities[mask]  # (N, 32)
        valid_unc = uncertainties[mask]

        summary = np.zeros(12, dtype=np.float32)
        # Mean ability across dims (take first 6 principal dims as summary)
        summary[0:6] = valid_ab.mean(axis=0)[:6]
        # Std of ability norms (spread of talent)
        norms = np.linalg.norm(valid_ab, axis=1)
        summary[6] = norms.mean()
        summary[7] = norms.std() if len(norms) > 1 else 0.0
        # Mean uncertainty
        summary[8] = valid_unc.mean()
        # Number of valid players (normalized)
        summary[9] = n_valid / 13.0
        # Max ability norm (star power)
        summary[10] = norms.max() if len(norms) > 0 else 0.0
        # Min ability norm (depth)
        summary[11] = norms.min() if len(norms) > 0 else 0.0

        return summary

    @staticmethod
    def _normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Z-score normalization."""
        return (x - mean) / (std + 1e-8)

    # ------------------------------------------------------------------
    # L1 decoder: player-level stat predictions
    # ------------------------------------------------------------------

    def _ensure_l1_loaded(self):
        """Lazy-load the NKEH model and L1 normalization infrastructure."""
        if self._l1_model is not None:
            return

        device = self._device or torch.device("cpu")

        # Load metadata and normalizer
        self._l1_metadata = load_metadata()
        self._l1_normalizer = Normalizer(self._l1_metadata)
        self._l1_box_mean = np.array(self._l1_metadata["box_mean"], dtype=np.float32)
        self._l1_box_std = np.array(self._l1_metadata["box_std"], dtype=np.float32)

        # Load profiles
        self._l1_profiles_data = load_profiles()
        self._l1_profile_idx = {
            int(pid): i for i, pid in enumerate(self._l1_profiles_data["person_ids"])
        }

        # Load NKEH model
        cfg = NKEHConfig(
            n_box_stats=len(self._l1_metadata["box_stat_columns"]),
            n_pbp_stats=len(self._l1_metadata["pbp_stat_columns"]),
            n_context=len(self._l1_metadata["context_columns"]),
            n_profile=len(self._l1_metadata["profile_columns"]),
        )
        model = NKEH(cfg)
        ckpt_path = CHECKPOINT_DIR / "phase2_best.pt"
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.to(device)
        model.eval()
        self._l1_model = model

        logger.info(
            f"L1 NKEH model loaded for player stat predictions "
            f"(epoch {ckpt.get('epoch', '?')}, device={device})"
        )

    def _get_l1_profile_tensor(self, player_id: int) -> torch.Tensor | None:
        """Get normalized profile tensor for a player."""
        if player_id in self._l1_profile_idx:
            idx = self._l1_profile_idx[player_id]
            profile_cols = self._l1_metadata["profile_columns"]
            vals = [float(self._l1_profiles_data[col][idx]) for col in profile_cols]
            raw = torch.tensor(vals, dtype=torch.float32).unsqueeze(0)
            return self._l1_normalizer.normalize_profile(raw)

        # Fallback: load from DB for players not in cache
        from src.database import get_db
        from src.phase5.cache_builder import PROFILE_COLUMNS, _load_player_profile

        with get_db(str(DB_PATH)) as conn:
            profile_dict = _load_player_profile(conn, player_id)

        if not profile_dict:
            return None

        vec = np.array(
            [profile_dict.get(col, 0) for col in PROFILE_COLUMNS],
            dtype=np.float32,
        )
        raw = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
        return self._l1_normalizer.normalize_profile(raw)

    def _build_prediction_context(
        self,
        player_id: int,
        is_home: bool,
        home_team: str,
        away_team: str,
        date_time_utc: str,
        season: str,
        n_career_games: int | None = None,
    ) -> np.ndarray:
        """
        Build a 12-d context vector for next-game prediction.

        Uses simplified defaults where exact rolling stats are not available,
        mirroring the CONTEXT_COLUMNS order from cache_builder.
        """
        ctx = np.zeros(len(CONTEXT_COLUMNS), dtype=np.float32)

        # Parse game date
        game_date = date_time_utc[:10] if date_time_utc else ""
        game_year = int(game_date[:4]) if len(game_date) >= 4 else 2026
        game_month = int(game_date[5:7]) if len(game_date) >= 7 else 1

        # [0] age_at_game: estimate from profile birth_year
        birth_year = 1995.0  # default
        if player_id in self._l1_profile_idx:
            idx = self._l1_profile_idx[player_id]
            birth_year = float(self._l1_profiles_data["birth_year"][idx])
        ctx[0] = game_year - birth_year + (game_month - 6) / 12.0

        # [1] rest_days: default 1 (typical NBA schedule)
        ctx[1] = 1.0

        # [2] home_flag
        ctx[2] = 1.0 if is_home else 0.0

        # [3] opponent_drtg: use league average
        ctx[3] = 110.0

        # [4] team_pace: use league average
        ctx[4] = 200.0

        # [5] minutes_share: estimate from career game count
        # Players with more games tend to play more; default to average rotation share
        if n_career_games is not None:
            ctx[5] = min(0.20, 0.08 + 0.0005 * n_career_games)
        else:
            ctx[5] = 0.12

        # [6] season_progress: compute from game month
        # NBA season: Oct (0.0) to Apr (1.0), playoffs May-Jun
        if game_month >= 10:
            ctx[6] = (game_month - 10) / 7.0  # Oct=0, Nov~0.14, Dec~0.28
        elif game_month <= 6:
            ctx[6] = min(1.0, (game_month + 2) / 7.0)  # Jan~0.43, Feb~0.57, ...
        else:
            ctx[6] = 0.5  # offseason default

        # [7] career_games_played
        ctx[7] = float(n_career_games) if n_career_games is not None else 50.0

        # [8] days_since_last_game: default 2 (typical NBA schedule)
        ctx[8] = 2.0

        # [9] opponent_pace: league average
        ctx[9] = 200.0

        # [10] team_ortg: league average
        ctx[10] = 110.0

        # [11] altitude_ft: from ALTITUDE_MAP using home team
        ctx[11] = float(ALTITUDE_MAP.get(home_team, 0))

        return ctx

    def _predict_player_stats(
        self,
        roster: dict,
        home_team: str,
        away_team: str,
        date_time_utc: str,
        season: str,
    ) -> dict:
        """
        Predict per-player box stats using the L1 NKEH decoder's next_game_head.

        Returns:
            {"home": {player_id: {"pred_points": X, "pred_rebounds": Y, ...}},
             "away": {player_id: {...}}}
        """
        try:
            self._ensure_l1_loaded()
        except Exception as e:
            logger.warning(f"Could not load L1 model for player stats: {e}")
            return {"home": {}, "away": {}}

        device = self._device or torch.device("cpu")
        result = {"home": {}, "away": {}}

        for side in ("home", "away"):
            player_ids = roster.get(f"{side}_players", [])
            is_home = side == "home"

            for pid in player_ids:
                # Load L1 vectors (ability, game count for context)
                npz_path = L1_VECTORS_DIR / f"{pid}.npz"
                if not npz_path.exists():
                    continue
                l1_data = np.load(str(npz_path), allow_pickle=True)
                ability = l1_data["ability"][-1]  # (32,) most recent
                n_career_games = len(l1_data["game_ids"])

                # Build context
                ctx_raw = self._build_prediction_context(
                    pid,
                    is_home,
                    home_team,
                    away_team,
                    date_time_utc,
                    season,
                    n_career_games=n_career_games,
                )
                # Normalize context
                ctx_t = self._l1_normalizer.normalize_context(
                    torch.tensor(ctx_raw, dtype=torch.float32).unsqueeze(0)
                ).to(device)

                # Get normalized profile
                profile_t = self._get_l1_profile_tensor(pid)
                if profile_t is None:
                    continue
                profile_t = profile_t.to(device)

                # Ability tensor
                ability_t = (
                    torch.tensor(ability, dtype=torch.float32).unsqueeze(0).to(device)
                )

                # Run decoder
                with torch.no_grad():
                    dec_out = self._l1_model.decoder(ability_t, ctx_t, profile_t)
                    next_game_norm = dec_out["next_game"][0].cpu().numpy()  # (16,)

                # Denormalize: predicted = normalized * std + mean
                pred_raw = next_game_norm * self._l1_box_std + self._l1_box_mean

                # Clamp non-negative (all stats except plus_minus must be >= 0)
                for i in range(15):  # indices 0-14 are non-negative stats
                    pred_raw[i] = max(0.0, pred_raw[i])

                # Map to named predictions
                # BOX_STAT_COLUMNS: min, pts, oreb, dreb, ast, stl, blk, tov, pf,
                #                   fga, fgm, fg3a, fg3m, fta, ftm, plus_minus
                player_preds = {
                    "pred_minutes": round(float(pred_raw[0]), 1),
                    "pred_points": round(float(pred_raw[1]), 1),
                    "pred_rebounds": round(float(pred_raw[2] + pred_raw[3]), 1),
                    "pred_oreb": round(float(pred_raw[2]), 1),
                    "pred_dreb": round(float(pred_raw[3]), 1),
                    "pred_assists": round(float(pred_raw[4]), 1),
                    "pred_steals": round(float(pred_raw[5]), 1),
                    "pred_blocks": round(float(pred_raw[6]), 1),
                    "pred_turnovers": round(float(pred_raw[7]), 1),
                    "pred_fouls": round(float(pred_raw[8]), 1),
                    "pred_fga": round(float(pred_raw[9]), 1),
                    "pred_fgm": round(float(pred_raw[10]), 1),
                    "pred_fg3a": round(float(pred_raw[11]), 1),
                    "pred_fg3m": round(float(pred_raw[12]), 1),
                    "pred_fta": round(float(pred_raw[13]), 1),
                    "pred_ftm": round(float(pred_raw[14]), 1),
                    "pred_plus_minus": round(float(pred_raw[15]), 1),
                }
                result[side][str(pid)] = player_preds

        n_home = len(result["home"])
        n_away = len(result["away"])
        logger.debug(f"Player stat predictions: {n_home} home, {n_away} away players")
        return result
