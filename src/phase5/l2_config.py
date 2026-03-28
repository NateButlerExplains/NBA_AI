"""
L2 Player Synergy Network configuration.
"""

from dataclasses import dataclass


@dataclass
class L2Config:
    """Configuration for Level 2: Player Synergy Network."""

    # --- Dimensions (must match L1) ---
    d_ability: int = 32  # from L1 NKE-H
    n_archetypes: int = 10  # K soft archetypes from L1

    # --- FM Synergy Vectors ---
    d_synergy: int = 32  # synergy vector dimensionality
    fm_hidden: int = 64  # MLP hidden dim for ability→synergy projection
    n_players: int = 3000  # max player embedding table size
    fm_residual_init_gate: float = -2.0  # sigmoid(-2)=0.12, MLP dominates initially
    fm_l2_lambda: float = 0.01  # L2 regularization on residual embeddings

    # --- GATv2 ---
    n_gat_heads: int = 4
    n_edge_features: int = 9  # archetype_syn, fm_syn, shared_min, years_together,
    #                           wowy_last20, positional_overlap, minutes_overlap_pct,
    #                           ability_cosine, age_gap
    gat_dropout: float = 0.1

    # --- Aggregation ---
    d_team_player: int = 64  # player aggregation output
    d_team_synergy: int = 64  # synergy aggregation output
    n_meta_features: int = 6  # team_uncertainty, n_players, ability_std,
    #                           archetype_entropy, archetype_dominant, max_ability_norm
    synergy_mlp_hidden: int = 128  # hidden dim for synergy aggregation MLP

    # --- Roster ---
    max_roster: int = 15  # max active players per team (padded)

    # --- Training ---
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256  # pair samples for 2-man loss
    lineup_batch_size: int = 64  # lineup samples for 5-man loss
    gradient_clip: float = 1.0
    epochs: int = 30
    patience: int = 10

    # --- Loss weights ---
    w_2man: float = 1.0  # primary: pairwise WOWY residual
    w_5man: float = 0.3  # secondary: lineup net rating
    w_consistency: float = 0.1  # additive consistency regularization
    w_fm_l2: float = 0.01  # L2 penalty on FM residual embeddings

    # --- Curriculum control (tuned: Experiment C best) ---
    curriculum_warmup: int = 10  # epochs of 2-man only before introducing 5-man
    lineup_ramp_epochs: int = 10  # epochs over which w_5man ramps from 0 to full value
    lineup_min_possessions: int = (
        200  # minimum possessions for lineup inclusion (was 50, noisy)
    )
    freeze_fm_for_lineup: bool = False  # if True, freeze FM vectors during lineup phase

    # --- Time decay ---
    time_decay_beta: float = 0.997  # half-life ~231 days (1 season)

    # --- Data splits (same as L1 for pre-training) ---
    pretrain_seasons: tuple = ("2007-2008", "2016-2017")  # start, end
    val_season: str = "2016-2017"
    train_seasons: tuple = ("2017-2018", "2023-2024")

    @property
    def d_l2_output(self) -> int:
        """Total L2 output dimensionality per team."""
        return self.d_team_player + self.d_team_synergy + self.n_meta_features  # 134
