"""
Prediction Heads for NBA Game Prediction.

Produces probabilistic outputs from matchup representations:
- Spread prediction: (μ_spread, σ_spread)
- Score predictions: (μ_home, σ_home), (μ_away, σ_away)
- Win probability (derived from score distributions)

Uses softplus to ensure σ > 0 for valid probability distributions.

Usage:
    heads = PredictionHeads(hidden_dim=256)
    predictions = heads(matchup_repr)
"""

import math
from typing import Optional, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GamePrediction(NamedTuple):
    """
    Container for all prediction outputs.

    A NamedTuple is like a regular tuple but with named fields, so you can
    access predictions.spread_mean instead of predictions[0]. This makes
    the code much more readable.

    The model outputs PROBABILISTIC predictions -- not just a single number,
    but a DISTRIBUTION described by a mean (best guess) and standard deviation
    (how uncertain the model is). This is far more useful than point estimates
    because you know when the model is confident vs. when it's unsure.
    """

    # Spread prediction (home_score - away_score)
    # Example: spread_mean=+3.5 means model predicts home wins by ~3.5 points
    spread_mean: torch.Tensor  # mu (best guess for the spread)
    spread_std: torch.Tensor   # sigma (uncertainty -- higher = less confident)

    # Score predictions for each team (also as distributions)
    home_score_mean: torch.Tensor  # mu_home (e.g., 110.5)
    home_score_std: torch.Tensor   # sigma_home (e.g., 12.0)
    away_score_mean: torch.Tensor  # mu_away (e.g., 107.0)
    away_score_std: torch.Tensor   # sigma_away (e.g., 11.5)

    # Win probability: DERIVED from the score distributions (not a separate prediction).
    # P(home wins) = P(home_score > away_score), computed using Gaussian math.
    home_win_prob: torch.Tensor  # Value between 0.0 and 1.0


class GaussianHead(nn.Module):
    """
    Prediction head that outputs Gaussian (normal distribution) parameters: mean and std.

    WHY PROBABILISTIC PREDICTIONS?
    Instead of predicting a single number (e.g., "spread = +3"), we predict a
    probability distribution: "spread is centered at +3 with uncertainty of +/- 8."
    This tells us BOTH the model's best guess AND how confident it is.

    KEY DESIGN CHOICES:
    - softplus(x) = log(1 + exp(x)) ensures standard deviation is always POSITIVE
      (you can't have negative uncertainty). softplus is a smooth approximation
      of ReLU that is always > 0.
    - min_std prevents the model from becoming overconfident (std >= min_std always).
      Without this, the model might predict sigma close to 0 for some games,
      meaning it's "100% certain" -- which is never realistic for sports.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        min_std: float = 1.0,
        max_std: Optional[float] = None,
        dropout: float = 0.1,
    ):
        """
        Initialize Gaussian head.

        Args:
            input_dim: Input dimension (from fusion layer, e.g., 256)
            hidden_dim: Hidden layer dimension (128)
            min_std: Minimum standard deviation floor -- sigma will always be >= this.
                     For spread: 1.0 (at least 1 point of uncertainty)
                     For scores: 5.0 (at least 5 points of uncertainty)
            max_std: Maximum standard deviation cap. Prevents sigma from inflating
                     to absorb all prediction error (the NLL sigma trap).
            dropout: Dropout probability
        """
        super().__init__()

        self.min_std = min_std
        self.max_std = max_std

        # Shared hidden layers that process the matchup representation
        # before splitting into separate mean and std outputs
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Two separate output layers: one for mean, one for standard deviation.
        # Each outputs a single number (1 dimension) per sample.
        self.mean_head = nn.Linear(hidden_dim, 1)  # Predicts mu (no constraints)
        self.std_head = nn.Linear(hidden_dim, 1)   # Raw value, softplus applied later

        # Initialize the std_head bias so that the initial standard deviation
        # output is approximately 5.0 (before adding min_std).
        # We compute softplus_inverse(5.0) = log(exp(5) - 1) to set the bias.
        # This gives the model a reasonable starting uncertainty rather than random.
        with torch.no_grad():
            self.std_head.bias.fill_(math.log(math.exp(5.0) - 1))  # softplus^-1(5)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Gaussian parameters.

        Args:
            x: Input tensor of shape (batch, input_dim)

        Returns:
            (mean, std) where each has shape (batch,)
        """
        # Shared layers process the matchup representation
        h = self.shared_layers(x)              # (batch, hidden_dim)

        # Mean: no constraints, can be any real number (positive or negative spread)
        mean = self.mean_head(h).squeeze(-1)   # (batch, 1) -> (batch,)

        # Std: must be positive.
        # 1. self.std_head(h) outputs a raw unconstrained value
        # 2. F.softplus() applies log(1 + exp(x)), guaranteeing output > 0
        # 3. + self.min_std adds the minimum floor (e.g., 1.0 for spread)
        # squeeze(-1) removes the trailing size-1 dimension: (batch, 1) -> (batch,)
        std = F.softplus(self.std_head(h)).squeeze(-1) + self.min_std

        # Cap sigma to prevent the NLL sigma inflation trap where the model
        # inflates sigma to reduce NLL gradient on mu, killing spread predictions
        if self.max_std is not None:
            std = torch.clamp(std, max=self.max_std)

        return mean, std


class ScoreHead(nn.Module):
    """
    Prediction head for absolute game scores (home and away).

    Similar to GaussianHead but outputs FOUR values: mean and std for both
    the home score and away score. A deeper shared network (2 layers instead
    of 1) is used because predicting absolute scores is harder than spreads.

    min_std=5.0 means the model always predicts at least 5 points of uncertainty
    per team's score, which is realistic for NBA games.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        min_std: float = 5.0,  # NBA scores typically have std of ~10-15
        dropout: float = 0.1,
    ):
        super().__init__()

        self.min_std = min_std

        # Deeper shared layers (2 hidden layers) because score prediction
        # is harder than spread prediction
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Separate output heads for home and away scores (mean + std each)
        self.home_mean = nn.Linear(hidden_dim, 1)
        self.home_std = nn.Linear(hidden_dim, 1)
        self.away_mean = nn.Linear(hidden_dim, 1)
        self.away_std = nn.Linear(hidden_dim, 1)

        # Initialize biases to realistic NBA score values so the model starts
        # with reasonable predictions instead of random garbage.
        # Home team bias = 110 (slight home court advantage)
        # Away team bias = 108
        # Initial std = softplus^-1(8) so initial uncertainty is ~8 + 5 = ~13 points
        with torch.no_grad():
            self.home_mean.bias.fill_(110.0)
            self.away_mean.bias.fill_(108.0)
            self.home_std.bias.fill_(math.log(math.exp(8.0) - 1))
            self.away_std.bias.fill_(math.log(math.exp(8.0) - 1))

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute score distribution parameters.

        Args:
            x: Input tensor of shape (batch, input_dim)

        Returns:
            (home_mean, home_std, away_mean, away_std) each of shape (batch,)
        """
        h = self.shared_layers(x)  # Shared processing of matchup representation

        # Home score distribution: mean and std (same softplus + min_std approach)
        home_mean = self.home_mean(h).squeeze(-1)                       # (batch,)
        home_std = F.softplus(self.home_std(h)).squeeze(-1) + self.min_std  # >= 5.0

        # Away score distribution
        away_mean = self.away_mean(h).squeeze(-1)                       # (batch,)
        away_std = F.softplus(self.away_std(h)).squeeze(-1) + self.min_std  # >= 5.0

        return home_mean, home_std, away_mean, away_std


class PredictionHeads(nn.Module):
    """
    Complete prediction heads for NBA game prediction.

    Produces:
    - Spread prediction (μ, σ)
    - Home/away score predictions (μ, σ for each)
    - Win probability (derived from score distributions)
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        spread_min_std: float = 1.0,
        spread_max_std: Optional[float] = None,
        score_min_std: float = 5.0,
        dropout: float = 0.1,
        derive_spread: bool = False,
    ):
        """
        Initialize prediction heads.

        Args:
            input_dim: Dimension of matchup representation
            hidden_dim: Hidden layer dimension for heads
            spread_min_std: Minimum std for spread prediction
            spread_max_std: Maximum std for spread prediction (caps sigma inflation)
            score_min_std: Minimum std for score predictions
            dropout: Dropout probability
            derive_spread: If True, derive spread from score heads instead of
                using a separate spread head. spread_mean = home_mean - away_mean,
                spread_std = sqrt(home_std^2 + away_std^2). Eliminates the spread
                collapse failure mode and routes all gradient through score heads.
        """
        super().__init__()

        self.derive_spread = derive_spread

        # Spread head (only if not deriving from scores)
        if not derive_spread:
            self.spread_head = GaussianHead(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                min_std=spread_min_std,
                max_std=spread_max_std,
                dropout=dropout,
            )

        # Score head
        self.score_head = ScoreHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            min_std=score_min_std,
            dropout=dropout,
        )

    def forward(self, matchup_repr: torch.Tensor) -> GamePrediction:
        """
        Compute all predictions.

        Args:
            matchup_repr: Matchup representation of shape (batch, input_dim)

        Returns:
            GamePrediction namedtuple with all outputs
        """
        # --- Predict individual score distributions ---
        home_mean, home_std, away_mean, away_std = self.score_head(matchup_repr)

        # --- Predict spread distribution ---
        if self.derive_spread:
            # Derive spread directly from score predictions. This eliminates the
            # separate spread head (which suffered from collapse to near-zero) and
            # routes all spread gradient through the score heads.
            spread_mean = home_mean - away_mean
            spread_std = torch.sqrt(home_std**2 + away_std**2)
        else:
            spread_mean, spread_std = self.spread_head(matchup_repr)

        # --- Derive win probability from the score distributions ---
        # This is elegant: we don't need a separate win prediction model!
        # If home ~ N(mu_h, sigma_h) and away ~ N(mu_a, sigma_a) are independent,
        # then (home - away) ~ N(mu_h - mu_a, sqrt(sigma_h^2 + sigma_a^2)).
        # P(home wins) = P(home - away > 0) = CDF of the difference distribution.
        diff_mean = home_mean - away_mean
        # Standard deviation of the difference (variances add for independent vars)
        diff_std = torch.sqrt(home_std**2 + away_std**2)
        # Gaussian CDF using the error function (erf):
        # P(X > 0) = 0.5 * (1 + erf(mean / (std * sqrt(2))))
        # When diff_mean > 0 (home predicted higher), probability > 0.5
        # When diff_mean < 0 (away predicted higher), probability < 0.5
        home_win_prob = 0.5 * (1 + torch.erf(diff_mean / (diff_std * math.sqrt(2))))

        # Bundle all predictions into a named tuple for clean access
        return GamePrediction(
            spread_mean=spread_mean,
            spread_std=spread_std,
            home_score_mean=home_mean,
            home_score_std=home_std,
            away_score_mean=away_mean,
            away_score_std=away_std,
            home_win_prob=home_win_prob,
        )

    def predict_spread(self, matchup_repr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convenience method for spread prediction only."""
        return self.spread_head(matchup_repr)

    def predict_scores(
        self, matchup_repr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convenience method for score predictions only."""
        return self.score_head(matchup_repr)


def test_prediction_heads():
    """Test prediction heads with sample data."""
    import logging

    logging.basicConfig(level=logging.INFO)

    input_dim = 256
    batch_size = 8

    # Create sample matchup representation
    matchup_repr = torch.randn(batch_size, input_dim)

    # Test PredictionHeads
    print("Testing PredictionHeads...")
    heads = PredictionHeads(
        input_dim=input_dim,
        hidden_dim=128,
        spread_min_std=1.0,
        score_min_std=5.0,
    )
    heads.eval()

    print(f"  Parameters: {sum(p.numel() for p in heads.parameters()):,}")

    with torch.no_grad():
        predictions = heads(matchup_repr)

    print(f"\n  Spread: mean={predictions.spread_mean.mean():.2f}, "
          f"std={predictions.spread_std.mean():.2f}")
    print(f"  Home score: mean={predictions.home_score_mean.mean():.2f}, "
          f"std={predictions.home_score_std.mean():.2f}")
    print(f"  Away score: mean={predictions.away_score_mean.mean():.2f}, "
          f"std={predictions.away_score_std.mean():.2f}")
    print(f"  Home win prob: {predictions.home_win_prob.mean():.3f}")

    # Verify shapes
    assert predictions.spread_mean.shape == (batch_size,), "Spread mean shape error"
    assert predictions.spread_std.shape == (batch_size,), "Spread std shape error"
    assert predictions.home_score_mean.shape == (batch_size,), "Home mean shape error"
    assert predictions.home_score_std.shape == (batch_size,), "Home std shape error"
    assert predictions.away_score_mean.shape == (batch_size,), "Away mean shape error"
    assert predictions.away_score_std.shape == (batch_size,), "Away std shape error"
    assert predictions.home_win_prob.shape == (batch_size,), "Win prob shape error"

    # Verify constraints
    assert (predictions.spread_std >= 1.0).all(), "Spread std below minimum"
    assert (predictions.home_score_std >= 5.0).all(), "Home std below minimum"
    assert (predictions.away_score_std >= 5.0).all(), "Away std below minimum"
    assert ((predictions.home_win_prob >= 0) & (predictions.home_win_prob <= 1)).all(), \
        "Win prob out of range"

    print("\n  All constraints satisfied!")

    # Test convenience methods
    print("\nTesting convenience methods...")
    with torch.no_grad():
        spread_mean, spread_std = heads.predict_spread(matchup_repr)
        print(f"  predict_spread: mean shape={spread_mean.shape}")

        home_m, home_s, away_m, away_s = heads.predict_scores(matchup_repr)
        print(f"  predict_scores: home mean shape={home_m.shape}")

    print("\nAll tests passed!")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(test_prediction_heads())
