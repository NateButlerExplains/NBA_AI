"""
Pre-training Model for Phase 3 Experiment 2.

BERT-style masked reconstruction: encode full season, mask 40% of positions,
predict scores at masked positions via reconstruction head.

Pre-trains: player_embed, per_game_encoder, temporal_attention.
"""

import torch
import torch.nn as nn

from src.transformer.phase2.config import Phase2ModelConfig
from src.transformer.phase2.models.per_game_encoder import PerGameEncoder
from src.transformer.phase2.models.temporal_attention import Phase2TemporalAttention


class PretrainModel(nn.Module):
    """
    Pre-training model with masked score reconstruction.

    Architecture:
        1. per_game_encoder(all games) → (B, G, 512) game embeddings
        2. Replace masked positions with learned mask_token
        3. temporal_attention.forward_positions(masked_seq) → (B, G, 512)
        4. Extract masked positions → recon_head → (N_masked, 3)
        5. MSE loss vs actual [team_score, opp_score, margin]
    """

    def __init__(self, config: Phase2ModelConfig):
        super().__init__()
        self.config = config
        h = config.hidden_dim

        # Shared player embedding (will transfer to Phase2Model)
        self.player_embed = nn.Embedding(
            config.n_players, config.player_embed_dim, padding_idx=0
        )

        # Per-game encoder (no dynamics for pre-training)
        self.per_game_encoder = PerGameEncoder(
            player_embed=self.player_embed,
            hidden_dim=h,
            score_dim=config.score_dim,
            opponent_dim=config.opponent_dim,
            location_dim=config.location_dim,
            contribution_dim=config.player_contribution_dim,
            contribution_heads=config.player_contribution_heads,
            contribution_dropout=config.player_contribution_dropout,
            n_teams=config.n_teams,
            use_ple=config.use_ple,
            n_ple_bins=config.n_ple_bins,
        )

        # Temporal attention (transformer only, will transfer to Phase2Model)
        self.temporal_attention = Phase2TemporalAttention(
            hidden_dim=h,
            num_layers=config.temporal_layers,
            num_heads=config.temporal_heads,
            ff_dim=config.temporal_ff_dim,
            dropout=config.temporal_dropout,
            max_days=config.temporal_max_days,
            n_pool_queries=config.temporal_n_pool_queries,
            pos_encoding=config.temporal_pos_encoding,
        )

        # Learned mask token
        self.mask_token = nn.Parameter(torch.randn(h))

        # Reconstruction head: (512) -> (3) [team_score, opp_score, margin]
        self.recon_head = nn.Sequential(
            nn.Linear(h, 128),
            nn.GELU(),
            nn.Linear(128, 3),
        )

    def forward(self, batch: dict) -> dict:
        """
        Forward pass for masked reconstruction.

        Returns dict with 'predictions' (N_total_masked, 3) and
        'targets' (N_total_masked, 3).
        """
        scores = batch["scores"]          # (B, G, 4)
        opponent_ids = batch["opponent_ids"]  # (B, G)
        location = batch["location"]      # (B, G)
        player_ids = batch["player_ids"]  # (B, G, P)
        player_points = batch["player_points"]  # (B, G, P)
        player_mask = batch["player_mask"]  # (B, G, P)
        days_between = batch["days_between"]  # (B, G)
        game_mask = batch["game_mask"]    # (B, G) bool, True=padding
        mask_indices = batch["mask_indices"]  # (B, M)
        mask_padding = batch["mask_padding"]  # (B, M) bool, True=padding
        target_scores = batch["target_scores"]  # (B, G, 3)

        B, G = scores.shape[:2]
        h = self.config.hidden_dim

        # 1. Encode all games (no dynamics, no is_recent)
        game_embeddings = self.per_game_encoder(
            scores=scores,
            opponent_ids=opponent_ids,
            location=location,
            player_ids=player_ids,
            player_points=player_points,
            player_mask=player_mask,
        )  # (B, G, h)

        # 2. Replace masked positions with mask_token
        masked_embeddings = game_embeddings.clone()
        for b in range(B):
            valid_mask = ~mask_padding[b]
            indices = mask_indices[b][valid_mask]
            masked_embeddings[b, indices] = self.mask_token

        # 3. Temporal attention: get per-position outputs
        contextualized = self.temporal_attention.forward_positions(
            masked_embeddings, days_between, game_mask
        )  # (B, G, h)

        # 4. Extract masked positions and predict
        all_preds = []
        all_targets = []
        for b in range(B):
            valid_mask = ~mask_padding[b]
            indices = mask_indices[b][valid_mask]
            if len(indices) > 0:
                masked_repr = contextualized[b, indices]  # (n_masked, h)
                preds = self.recon_head(masked_repr)  # (n_masked, 3)
                targets = target_scores[b, indices]  # (n_masked, 3)
                all_preds.append(preds)
                all_targets.append(targets)

        if all_preds:
            predictions = torch.cat(all_preds, dim=0)
            targets = torch.cat(all_targets, dim=0)
        else:
            predictions = torch.zeros(0, 3, device=scores.device)
            targets = torch.zeros(0, 3, device=scores.device)

        return {
            "predictions": predictions,
            "targets": targets,
        }

    def get_transferable_state_dict(self) -> dict:
        """Extract state dict for components that transfer to Phase2Model."""
        state = {}
        for name, param in self.named_parameters():
            if name.startswith(("player_embed.", "per_game_encoder.", "temporal_attention.")):
                state[name] = param.data.clone()
        for name, buf in self.named_buffers():
            if name.startswith(("player_embed.", "per_game_encoder.", "temporal_attention.")):
                state[name] = buf.clone()
        return state
