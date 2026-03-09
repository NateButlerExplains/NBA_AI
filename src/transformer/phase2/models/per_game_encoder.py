"""
Phase 2 Per-Game Encoder.

Encodes each historical game as a holistic embedding containing:
scores + opponent + location + player contributions + (optional) dynamics.
"""

import torch
import torch.nn as nn


class PlayerContributionEncoder(nn.Module):
    """
    Attention-pooled player contributions within a single game.

    Per player: concat([player_emb(128), norm_points(1)]) -> Linear(129, 256) -> LN + GELU
    Attention pool over players: learned query + MultiheadAttention.

    Input: player_ids (B, G, P), player_points (B, G, P), player_mask (B, G, P)
    Output: (B, G, 256)
    """

    def __init__(
        self,
        player_embed: nn.Embedding,
        contribution_dim: int = 256,
        n_heads: int = 4,
        dropout: float = 0.2,
        n_player_stats: int = 0,
        stat_hidden_dim: int = 64,
        n_positions: int = 4,
        position_dim: int = 8,
        interaction_layers: int = 0,
        interaction_heads: int = 4,
        interaction_ff_dim: int = 1024,
        interaction_dropout: float = 0.2,
        n_pool_queries: int = 1,
    ):
        super().__init__()

        self.player_embed = player_embed  # Shared reference
        embed_dim = player_embed.embedding_dim  # 128
        self.n_player_stats = n_player_stats
        self.contribution_dim = contribution_dim
        self.n_pool_queries = n_pool_queries

        if n_player_stats > 0:
            # Full stats path: stat MLP + position embedding + wider projection
            self.stat_mlp = nn.Sequential(
                nn.Linear(n_player_stats + 1, stat_hidden_dim),
                nn.LayerNorm(stat_hidden_dim),
                nn.GELU(),
                nn.Linear(stat_hidden_dim, stat_hidden_dim),
                nn.LayerNorm(stat_hidden_dim),
                nn.GELU(),
            )
            self.position_emb = nn.Embedding(n_positions, position_dim)
            proj_input_dim = embed_dim + stat_hidden_dim + position_dim
        else:
            # Legacy path: points only
            proj_input_dim = embed_dim + 1

        self.player_projection = nn.Sequential(
            nn.Linear(proj_input_dim, contribution_dim),
            nn.LayerNorm(contribution_dim),
            nn.GELU(),
        )

        # Player interaction self-attention (Exp 4+)
        self.interaction = None
        if interaction_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=contribution_dim,
                nhead=interaction_heads,
                dim_feedforward=interaction_ff_dim,
                dropout=interaction_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.interaction = nn.TransformerEncoder(
                encoder_layer, num_layers=interaction_layers
            )

        # Attention pool over players
        self.pool_queries = nn.Parameter(torch.randn(1, n_pool_queries, contribution_dim))
        self.pool_attention = nn.MultiheadAttention(
            contribution_dim, n_heads, dropout=dropout, batch_first=True
        )
        if n_pool_queries > 1:
            self.pool_projection = nn.Sequential(
                nn.Linear(n_pool_queries * contribution_dim, contribution_dim),
                nn.LayerNorm(contribution_dim),
            )

    def forward(
        self,
        player_ids: torch.Tensor,
        player_points: torch.Tensor,
        player_mask: torch.Tensor,
        player_stats: torch.Tensor = None,
        player_positions: torch.Tensor = None,
        player_pm_available: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            player_ids: (B, G, P) int64
            player_points: (B, G, P) float32 (normalized by /30.0)
            player_mask: (B, G, P) bool, True=padding
            player_stats: (B, G, P, 16) float32, optional — normalized box score stats
            player_positions: (B, G, P) int64, optional — position index 0-3
            player_pm_available: (B, G, P) float32, optional — binary indicator

        Returns:
            (B, G, contribution_dim)
        """
        B, G, P = player_ids.shape

        # Flatten to (B*G, P)
        ids_flat = player_ids.reshape(B * G, P)
        mask_flat = player_mask.reshape(B * G, P)

        # Embed players
        emb = self.player_embed(ids_flat)  # (B*G, P, 128)

        if self.n_player_stats > 0 and player_stats is not None:
            # Full stats path
            stats_flat = player_stats.reshape(B * G, P, -1)  # (B*G, P, 16)
            pm_flat = player_pm_available.reshape(B * G, P)  # (B*G, P)
            pos_flat = player_positions.reshape(B * G, P)  # (B*G, P)

            stat_input = torch.cat([stats_flat, pm_flat.unsqueeze(-1)], dim=-1)  # (B*G, P, 17)
            stat_repr = self.stat_mlp(stat_input)  # (B*G, P, stat_hidden_dim)
            pos_repr = self.position_emb(pos_flat)  # (B*G, P, position_dim)

            combined = torch.cat([emb, stat_repr, pos_repr], dim=-1)  # (B*G, P, 128+64+8=200)
        else:
            # Legacy path
            points_flat = player_points.reshape(B * G, P)
            points_feat = points_flat.unsqueeze(-1)  # (B*G, P, 1)
            combined = torch.cat([emb, points_feat], dim=-1)  # (B*G, P, 129)

        x = self.player_projection(combined)  # (B*G, P, contribution_dim)

        # Player interaction self-attention (before pooling)
        if self.interaction is not None:
            x = self.interaction(x, src_key_padding_mask=mask_flat)

        # Attention pool
        queries = self.pool_queries.expand(B * G, -1, -1)  # (B*G, N, dim)
        pooled, _ = self.pool_attention(
            queries, x, x, key_padding_mask=mask_flat, need_weights=False
        )
        # pooled: (B*G, N, dim)
        if self.n_pool_queries > 1:
            pooled = pooled.reshape(B * G, self.n_pool_queries * self.contribution_dim)
            pooled = self.pool_projection(pooled)  # (B*G, dim)
        else:
            pooled = pooled.squeeze(1)  # (B*G, contribution_dim)

        return pooled.reshape(B, G, -1)


class PerGameEncoder(nn.Module):
    """
    Encodes each historical game into a 512-d embedding.

    Components:
    - Score: Linear(4, 128) + LN + GELU
    - Opponent: Embedding(30, 64)
    - Location: Embedding(2, 32)
    - Player contributions: PlayerContributionEncoder -> (B, G, 256)
    - Context combine: concat([128, 64, 32, 256]) = 480 -> Linear(480, 512) -> LN
    - Dynamics merge: concat([context(512), dynamics(512)]) -> Linear(1024, 512) -> LN

    Input: scores, opponent_ids, location, player_ids/points/mask, dynamics, is_recent
    Output: game_reprs (B, G, 512)
    """

    def __init__(
        self,
        player_embed: nn.Embedding,
        hidden_dim: int = 512,
        score_dim: int = 128,
        opponent_dim: int = 64,
        location_dim: int = 32,
        contribution_dim: int = 256,
        contribution_heads: int = 4,
        contribution_dropout: float = 0.2,
        n_teams: int = 30,
        use_ple: bool = False,
        n_ple_bins: int = 16,
        n_player_stats: int = 0,
        stat_hidden_dim: int = 64,
        n_positions: int = 4,
        position_dim: int = 8,
        interaction_layers: int = 0,
        interaction_heads: int = 4,
        interaction_ff_dim: int = 1024,
        interaction_dropout: float = 0.2,
        n_pool_queries: int = 1,
        has_team_gat: bool = False,
        n_efficiency_features: int = 0,
        efficiency_hidden_dim: int = 64,
        gs_summary_dim: int = 32,
        flag_dim: int = 16,
    ):
        super().__init__()
        self.n_efficiency_features = n_efficiency_features

        # Score projection
        if use_ple:
            from src.transformer.phase2.models.piecewise_linear import PiecewiseLinearEncoding
            self.score_proj = PiecewiseLinearEncoding(
                n_features=4, n_bins=n_ple_bins, output_dim=score_dim,
            )
        else:
            self.score_proj = nn.Sequential(
                nn.Linear(4, score_dim),
                nn.LayerNorm(score_dim),
                nn.GELU(),
            )

        # Opponent embedding
        self.opponent_emb = nn.Embedding(n_teams, opponent_dim)

        # Location embedding
        self.location_emb = nn.Embedding(2, location_dim)

        # Player contribution encoder
        self.player_encoder = PlayerContributionEncoder(
            player_embed=player_embed,
            contribution_dim=contribution_dim,
            n_heads=contribution_heads,
            dropout=contribution_dropout,
            n_player_stats=n_player_stats,
            stat_hidden_dim=stat_hidden_dim,
            n_positions=n_positions,
            position_dim=position_dim,
            interaction_layers=interaction_layers,
            interaction_heads=interaction_heads,
            interaction_ff_dim=interaction_ff_dim,
            interaction_dropout=interaction_dropout,
            n_pool_queries=n_pool_queries,
        )

        # Efficiency feature encoders (Phase 3 Exp 7+)
        self.efficiency_encoder = None
        self.gs_summary_encoder = None
        self.flag_encoder = None
        efficiency_out_dim = 0
        gs_summary_out_dim = 0
        flag_out_dim = 0
        if n_efficiency_features > 0:
            efficiency_out_dim = efficiency_hidden_dim
            self.efficiency_encoder = nn.Sequential(
                nn.Linear(n_efficiency_features, efficiency_hidden_dim),
                nn.LayerNorm(efficiency_hidden_dim),
                nn.GELU(),
                nn.Linear(efficiency_hidden_dim, efficiency_hidden_dim),
                nn.LayerNorm(efficiency_hidden_dim),
                nn.GELU(),
            )
            gs_summary_out_dim = gs_summary_dim
            self.gs_summary_encoder = nn.Sequential(
                nn.Linear(6, gs_summary_dim),
                nn.LayerNorm(gs_summary_dim),
                nn.GELU(),
            )
            flag_out_dim = flag_dim
            self.flag_encoder = nn.Sequential(
                nn.Linear(2, flag_dim),
                nn.LayerNorm(flag_dim),
                nn.GELU(),
            )

        # Context combine
        context_dim = (score_dim + opponent_dim + location_dim + contribution_dim
                       + efficiency_out_dim + gs_summary_out_dim + flag_out_dim)
        self.context_combine = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # H2H team GAT gated residual (Phase 3 Exp 6+)
        self.has_team_gat = has_team_gat
        if has_team_gat:
            self.opp_gate = nn.Sequential(
                nn.Linear(opponent_dim * 2, opponent_dim),
                nn.Sigmoid(),
            )

        # Gated dynamics merge (for recent games with GameStates)
        # Instead of concat-with-zeros (which poisons 94% of games with zero input),
        # use an additive gate: context + gate * proj(dynamics) for recent games only
        self.dynamics_gate = nn.Linear(hidden_dim, hidden_dim)
        self.dynamics_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(
        self,
        scores: torch.Tensor,
        opponent_ids: torch.Tensor,
        location: torch.Tensor,
        player_ids: torch.Tensor,
        player_points: torch.Tensor,
        player_mask: torch.Tensor,
        dynamics: torch.Tensor = None,
        is_recent: torch.Tensor = None,
        player_stats: torch.Tensor = None,
        player_positions: torch.Tensor = None,
        player_pm_available: torch.Tensor = None,
        h2h_team_repr: torch.Tensor = None,
        efficiency_features: torch.Tensor = None,
        gs_summary_features: torch.Tensor = None,
        context_flags: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            scores: (B, G, 4) float32 — normalized [home, away, margin, total]
            opponent_ids: (B, G) int64 — team index 0-29
            location: (B, G) int64 — 0=away, 1=home
            player_ids: (B, G, P) int64
            player_points: (B, G, P) float32
            player_mask: (B, G, P) bool, True=padding
            dynamics: (B, G, hidden_dim) or None — GameStates dynamics for recent games
            is_recent: (B, G) bool — True for games with dynamics data

        Returns:
            game_reprs: (B, G, hidden_dim)
        """
        B, G = scores.shape[:2]

        # Score features
        score_repr = self.score_proj(scores)  # (B, G, 128)

        # Opponent features
        opp_repr = self.opponent_emb(opponent_ids)  # (B, G, 64)

        # H2H team GAT gated residual
        if self.has_team_gat and h2h_team_repr is not None:
            # h2h_team_repr: (B, 30, 64) -> index by opponent_ids (B, G) -> (B, G, 64)
            D = h2h_team_repr.shape[-1]
            idx = opponent_ids.unsqueeze(-1).expand(-1, -1, D)  # (B, G, D)
            h2h_opp = h2h_team_repr.gather(1, idx)  # (B, G, D)
            gate = self.opp_gate(torch.cat([opp_repr, h2h_opp], dim=-1))  # (B, G, 64)
            opp_repr = opp_repr + gate * h2h_opp

        # Location features
        loc_repr = self.location_emb(location)  # (B, G, 32)

        # Player contribution features
        player_repr = self.player_encoder(
            player_ids, player_points, player_mask,
            player_stats=player_stats,
            player_positions=player_positions,
            player_pm_available=player_pm_available,
        )  # (B, G, 256)

        # Combine context features
        context_parts = [score_repr, opp_repr, loc_repr, player_repr]

        if self.efficiency_encoder is not None and efficiency_features is not None:
            context_parts.append(self.efficiency_encoder(efficiency_features))
        if self.gs_summary_encoder is not None and gs_summary_features is not None:
            context_parts.append(self.gs_summary_encoder(gs_summary_features))
        if self.flag_encoder is not None and context_flags is not None:
            context_parts.append(self.flag_encoder(context_flags))

        context = torch.cat(context_parts, dim=-1)
        context = self.context_combine(context)  # (B, G, hidden_dim)

        # Gated dynamics merge for recent games only
        # Non-recent games (94%) pass through as pure context — no zero-concat pollution
        game_reprs = context

        if dynamics is not None and is_recent is not None:
            context = context.to(dynamics.dtype)
            game_reprs = context.clone()

            # Compute gate and projection over all positions
            gate = torch.sigmoid(self.dynamics_gate(dynamics))  # (B, G, h)
            projected = self.dynamics_proj(dynamics)  # (B, G, h)

            # Apply only at recent positions (where dynamics is non-zero)
            mask = is_recent.unsqueeze(-1).float()  # (B, G, 1)
            game_reprs = game_reprs + mask * gate * projected

        return game_reprs
