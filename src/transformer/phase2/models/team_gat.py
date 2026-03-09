"""
Team Interaction GAT (Phase 3 Experiment 6).

Graph Attention Network over 30 NBA teams with head-to-head edge features.
Pure PyTorch implementation (no PyG) since the graph is trivially dense (30 nodes).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """
    GAT layer with edge feature bias.

    Attention: softmax(Q*K^T / sqrt(d) + edge_bias) * V
    Edge bias: Linear(3, n_heads) applied per directed edge.
    """

    def __init__(self, hidden_dim: int = 64, n_heads: int = 4, dropout: float = 0.1,
                 edge_features: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.norm = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Edge feature -> attention bias per head
        self.edge_bias = nn.Linear(edge_features, n_heads, bias=False)

        # FFN
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) node features, N=30 teams
            edge_features: (B, N, N, 3) directed edge features

        Returns:
            (B, N, D) updated node features
        """
        B, N, D = x.shape
        H = self.n_heads
        d = self.head_dim

        # Pre-norm attention
        h = self.norm(x)
        Q = self.q_proj(h).view(B, N, H, d).transpose(1, 2)  # (B, H, N, d)
        K = self.k_proj(h).view(B, N, H, d).transpose(1, 2)
        V = self.v_proj(h).view(B, N, H, d).transpose(1, 2)

        # Attention scores with edge bias
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (d ** 0.5)  # (B, H, N, N)
        bias = self.edge_bias(edge_features).permute(0, 3, 1, 2)  # (B, H, N, N)
        attn = attn + bias

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # (B, H, N, d)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        x = x + self.dropout(out)

        # Pre-norm FFN
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))

        return x


class TeamInteractionGAT(nn.Module):
    """
    Multi-layer GAT for learning team interaction representations.

    Input: H2H feature matrix (B, 30, 30, 3)
    Output: Team embeddings (B, 30, hidden_dim)
    """

    def __init__(
        self,
        n_teams: int = 30,
        hidden_dim: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
        edge_features: int = 3,
    ):
        super().__init__()
        self.node_embed = nn.Embedding(n_teams, hidden_dim)
        self.layers = nn.ModuleList([
            GATLayer(hidden_dim, n_heads, dropout, edge_features)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h2h_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h2h_features: (B, 30, 30, 3) directed H2H edge features

        Returns:
            (B, 30, hidden_dim) team embeddings
        """
        B = h2h_features.shape[0]
        device = h2h_features.device

        # Initialize node features from learned embeddings
        team_ids = torch.arange(30, device=device)
        x = self.node_embed(team_ids).unsqueeze(0).expand(B, -1, -1)  # (B, 30, D)

        for layer in self.layers:
            x = layer(x, h2h_features)

        return self.final_norm(x)
