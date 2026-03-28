"""
L3 Team Model configuration.
"""

from dataclasses import dataclass


@dataclass
class L3Config:
    """Configuration for Level 3: Team Model.

    L3 captures coaching, system, and organizational effects that are NOT
    explained by player talent (L1) and player synergy (L2). It operates on
    the aggregated L2 team vector — it never sees individual player vectors.
    """

    # --- Input dimensions ---
    d_l2_input: int = 134  # from L2 (64 player + 64 synergy + 6 meta)
    d_team_features: int = 34  # dual-scale rolling + coaching + org + defense
    d_roster_summary: int = 12  # from L1 vectors (experience, position coverage)
    d_coach_embed: int = 16  # learned per-coach embedding

    # --- Coach embedding ---
    n_coaches: int = 200  # max coach embedding table size
    coach_shrinkage_midpoint: float = 40.0  # games at which gate = 0.5

    # --- Architecture ---
    d_team: int = 128  # output dimensionality (team representation)
    mlp_hidden: int = 256  # hidden dim in residual blocks
    n_residual_blocks: int = 2  # number of residual MLP blocks
    dropout: float = 0.1

    # --- Loss weights for Phase B ---
    w_team_residual: float = 1.0  # MSE on team performance residual after L1+L2

    # --- Training ---
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 64  # team-game samples
    gradient_clip: float = 1.0
    epochs: int = 40
    patience: int = 15

    # --- Data splits (same window as L4) ---
    train_seasons: tuple = ("2017-2018", "2022-2023")
    val_season: str = "2023-2024"

    @property
    def d_input(self) -> int:
        """Total input dimensionality before projection."""
        return (
            self.d_l2_input
            + self.d_team_features
            + self.d_roster_summary
            + self.d_coach_embed
        )  # 196
