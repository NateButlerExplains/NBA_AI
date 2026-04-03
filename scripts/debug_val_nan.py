"""Debug NaN in validation: run one batch and check each stage."""

import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.generative.config import GenerativeExperimentConfig
from src.generative.dataset import GenerativeDataset, generative_collate
from src.generative.models.generative_model import GenerativeModel
from torch.utils.data import DataLoader


def check_nan(name, t):
    if t is None:
        return False
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    if has_nan or has_inf:
        nan_frac = torch.isnan(t).float().mean().item()
        finite = t[torch.isfinite(t)]
        if len(finite) > 0:
            logger.warning(
                f"  {name}: shape={list(t.shape)}, NaN={nan_frac:.1%}, finite_range=[{finite.min().item():.4f}, {finite.max().item():.4f}]"
            )
        else:
            logger.warning(f"  {name}: shape={list(t.shape)}, ALL NaN/Inf!")
        return True
    else:
        logger.info(
            f"  {name}: OK shape={list(t.shape)}, range=[{t.min().item():.4f}, {t.max().item():.4f}]"
        )
        return False


def main():
    config = GenerativeExperimentConfig.from_yaml(
        "configs/generative/exp5_full_context.yaml"
    )

    val_ds = GenerativeDataset(
        config.data,
        split="val",
        use_full_context=config.model.use_full_context,
        use_simplified_context=config.model.use_simplified_context,
        use_scoring_events_only=config.model.use_scoring_events_only,
        max_scoring_events=config.model.max_scoring_events,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=generative_collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GenerativeModel(config.model).to(device)
    model.eval()

    batch = next(iter(val_loader))
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }

    logger.info("=== Tracing context encoder internals ===")
    enc = model.context_encoder

    with torch.no_grad():
        for side in ["home", "away"]:
            logger.info(f"\n--- {side.upper()} side ---")
            player_mask = batch[f"{side}_player_mask"]
            player_pm_available = torch.ones_like(player_mask, dtype=torch.float32)
            game_mask = batch[f"{side}_game_mask"]
            scores = batch[f"{side}_scores"]
            opponents = batch[f"{side}_opponents"]
            locations = batch[f"{side}_locations"]
            player_ids = batch[f"{side}_player_ids"]
            player_stats = batch[f"{side}_player_stats"]
            player_positions = batch[f"{side}_player_positions"]
            days_before = batch[f"{side}_days_before"]
            rest_days = batch[f"{side}_rest_days"]
            rolling_stats = batch[f"{side}_rolling_stats"]
            team_idx = batch[f"{side}_team_idx"]

            B, G, P = player_ids.shape
            logger.info(f"  B={B}, G={G}, P={P}")

            # Convert masks
            padding_mask = ~game_mask
            player_padding_mask = ~player_mask
            logger.info(
                f"  game_mask valid count per sample: {game_mask.sum(dim=1).tolist()}"
            )
            logger.info(
                f"  player_mask valid per sample (sum over G,P): {player_mask.sum(dim=(1,2)).tolist()}"
            )

            # Player encoder
            player_points = player_stats[:, :, :, 1]
            player_repr = enc.player_encoder(
                player_ids=player_ids,
                player_points=player_points,
                player_mask=player_padding_mask,
                player_stats=player_stats,
                player_positions=player_positions,
                player_pm_available=player_pm_available,
            )
            check_nan(f"{side}_player_repr", player_repr)

            # Per-game encoding
            score_repr = enc.score_proj(scores)
            check_nan(f"{side}_score_repr", score_repr)
            opp_repr = enc.opp_embed(opponents)
            loc_repr = enc.loc_embed(locations)

            game_repr = torch.cat([score_repr, opp_repr, loc_repr, player_repr], dim=-1)
            game_repr = enc.game_combine(game_repr)
            check_nan(f"{side}_game_combine", game_repr)

            # Temporal encoder
            game_repr = enc.temporal_pos(game_repr, days_before)
            check_nan(f"{side}_temporal_pos", game_repr)

            game_repr = enc.temporal_encoder(
                game_repr, src_key_padding_mask=padding_mask
            )
            check_nan(f"{side}_temporal_encoder", game_repr)

            game_repr = enc.temporal_norm(game_repr)
            check_nan(f"{side}_temporal_norm", game_repr)

            # Attention pooling
            queries = enc.pool_queries.expand(B, -1, -1)
            pooled, _ = enc.pool_attention(
                queries,
                game_repr,
                game_repr,
                key_padding_mask=padding_mask,
                need_weights=False,
            )
            check_nan(f"{side}_pool_attention", pooled)

            n_pool = enc.config.full_context_temporal_pool_queries
            pooled_flat = pooled.reshape(B, n_pool * enc.config.hidden_dim)
            temporal_repr = enc.pool_projection(pooled_flat)
            check_nan(f"{side}_temporal_repr", temporal_repr)

            # Rolling stats
            rolling_repr = enc.rolling_proj(rolling_stats)
            check_nan(f"{side}_rolling_repr", rolling_repr)

            # Team identity
            team_repr = enc.team_id_embed(team_idx)
            rest_clamped = rest_days.clamp(0, enc.config.max_rest_days)
            rest_repr = enc.rest_embed(rest_clamped)

            combined = torch.cat(
                [temporal_repr, rest_repr, rolling_repr, team_repr], dim=-1
            )
            team_output = enc.team_combine(combined)
            check_nan(f"{side}_team_output", team_output)


if __name__ == "__main__":
    main()
