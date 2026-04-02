"""
models.py

Shared neural network model definitions for training and inference.

Classes:
- MLP: Multi-layer perceptron for NBA score prediction (v0.4 architecture).
- MLPv2: Enhanced MLP with BatchNorm (v0.5 architecture).

Usage:
    from src.model_training.models import MLP, MLPv2

    model = MLP(input_size=43, hidden_sizes=[64, 32], dropout=0.2)
    model_v2 = MLPv2(input_size=43, hidden_sizes=[256, 128, 64], dropout=0.2)
    predictions = model(features_tensor)
"""

import torch
from torch import nn


class MLP(nn.Module):
    """
    Multi-layer Perceptron for NBA score prediction (v0.4).

    Architecture:
        input -> hidden_1 -> ReLU -> Dropout -> ... -> hidden_n -> ReLU -> Dropout -> 2

    Predicts [home_score, away_score] from game features.

    Args:
        input_size: Number of input features (default: 43)
        hidden_sizes: List of hidden layer sizes (default: [64, 32])
        dropout: Dropout probability (default: 0.2)

    Example:
        >>> model = MLP(input_size=43)
        >>> x = torch.randn(32, 43)  # batch of 32 games
        >>> predictions = model(x)  # shape: (32, 2)
    """

    def __init__(self, input_size, hidden_sizes=None, dropout=0.2):
        super(MLP, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = [64, 32]

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout

        # Build network layers
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 2))  # Output: [home_score, away_score]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Tensor of shape (batch_size, 2) with [home_score, away_score]
        """
        return self.network(x)


class MLPv2(nn.Module):
    """
    Enhanced MLP with BatchNorm for NBA score prediction (v0.5).

    Architecture:
        input -> [Linear -> ReLU -> BN -> Dropout] x (N-1) -> [Linear -> ReLU] -> Linear -> 2

    BatchNorm and Dropout are applied on all hidden layers except the last,
    which only has Linear -> ReLU before the output projection.

    Args:
        input_size: Number of input features (default: 43)
        hidden_sizes: List of hidden layer sizes (default: [256, 128, 64])
        dropout: Dropout probability (default: 0.2)

    Example:
        >>> model = MLPv2(input_size=43, hidden_sizes=[256, 128, 64])
        >>> x = torch.randn(32, 43)
        >>> predictions = model(x)  # shape: (32, 2)
    """

    def __init__(self, input_size=43, hidden_sizes=None, dropout=0.2):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout

        layers = []
        prev = input_size
        for i, h in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            # BatchNorm + Dropout on all but last hidden layer
            if i < len(hidden_sizes) - 1:
                layers.append(nn.BatchNorm1d(h))
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Tensor of shape (batch_size, 2) with [home_score, away_score]
        """
        return self.network(x)
