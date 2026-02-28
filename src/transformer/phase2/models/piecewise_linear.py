"""
Piecewise Linear Encoding (PLE) for numerical features.

Based on Gorishniy et al., 2022 — "On Embeddings for Numerical Features in
Tabular Deep Learning". Discretizes each numerical feature into learned bins
and encodes values by their position within bins. This captures non-linear
relationships that a simple Linear projection misses.

For NBA score features (home_score, away_score, margin, total), PLE lets the
model learn that e.g. a 10-point margin is qualitatively different from a
2-point margin, rather than treating the relationship as purely linear.
"""

import torch
import torch.nn as nn


class PiecewiseLinearEncoding(nn.Module):
    """
    PLE for multiple numerical features.

    For each feature, defines n_bins learnable boundaries. A value x is encoded
    as a sparse vector where at most two adjacent bins are active, with weights
    determined by linear interpolation within the bin.

    Input: (B, G, n_features) float32
    Output: (B, G, output_dim)
    """

    def __init__(self, n_features: int, n_bins: int = 16, output_dim: int = 128):
        super().__init__()
        self.n_features = n_features
        self.n_bins = n_bins

        # Learnable bin boundaries per feature: (n_features, n_bins + 1)
        # Initialized to cover normalized score range (-3 to +3 std)
        init_boundaries = torch.linspace(-3.0, 3.0, n_bins + 1).unsqueeze(0).expand(n_features, -1).clone()
        self.boundaries = nn.Parameter(init_boundaries)

        # Project concatenated PLE features to output_dim
        self.projection = nn.Sequential(
            nn.Linear(n_features * n_bins, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, G, n_features) — normalized numerical features

        Returns:
            (B, G, output_dim)
        """
        B, G, F = x.shape
        assert F == self.n_features

        # Sort boundaries to maintain monotonicity during training
        boundaries, _ = self.boundaries.sort(dim=-1)  # (F, n_bins + 1)

        # x: (B, G, F) -> (B, G, F, 1) for broadcasting against boundaries (F, n_bins+1)
        x_expanded = x.unsqueeze(-1)  # (B, G, F, 1)

        # Left and right boundaries for each bin: (F, n_bins)
        left = boundaries[:, :-1]   # (F, n_bins)
        right = boundaries[:, 1:]   # (F, n_bins)

        # Compute bin widths, clamped to avoid division by zero
        widths = (right - left).clamp(min=1e-6)  # (F, n_bins)

        # Compute how far x is into each bin: (x - left) / width
        # Values < 0 mean x is below this bin, > 1 means x is above
        t = (x_expanded - left) / widths  # (B, G, F, n_bins)

        # Clamp to [0, 1] — this gives the piecewise linear encoding:
        # 0 for bins entirely below or above x, linear interpolation within
        activations = t.clamp(0.0, 1.0)  # (B, G, F, n_bins)

        # Reshape: (B, G, F * n_bins)
        ple_features = activations.reshape(B, G, F * self.n_bins)

        return self.projection(ple_features)
