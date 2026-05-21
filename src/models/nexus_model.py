"""
Nexus Multi-Modal Architecture — unified deep-learning model.

Replaces the 6 independent CatBoost models and the isolated Transformer
with a single end-to-end network.

Architecture
------------
* Temporal Backbone  – Mamba-2-style SSM (simplified pure-PyTorch implementation).
* Tabular Backbone   – FT-Transformer for scalar / contextual features.
* Relational Backbone– Lightweight Graph Attention Network (GAT) for lineup synergy.
* Fusion & Copula Head
    * Concatenates backbone representations.
    * Final layer returns a 6-dimensional mean vector and a 6x6
      Cholesky-decomposed covariance matrix.

Hardware caveats
----------------
The production Mamba-2 layer uses fast CUDA kernels (mamba_ssm).  The
fallback here is a mathematically-similar SSM block built with standard
PyTorch ops so training / inference still works on CPU and macOS while
preserving the exact parameter shapes for weight-porting later.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.bool()


def _init_xavier(m: nn.Module) -> None:
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# 1. Temporal Backbone – Simplified Mamba-2 SSM
# ---------------------------------------------------------------------------

class SimplifiedSSMBlock(nn.Module):
    """SSM block that approximates Mamba-2 using only standard PyTorch ops.

    The canonical Mamba block is:
        x -> linear -> short conv -> SiLU -> linear -> gated residual
    We keep the same shapes so weights from the CUDA kernel version can
    be ported in later if desired.
    """

    def __init__(self, d_model: int, d_state: int = 64, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=3,
            padding=1,
            groups=self.d_inner,
            bias=True,
        )

        self.ssm_A = nn.Parameter(torch.randn(d_state, d_state) * 0.01)
        self.ssm_B = nn.Linear(self.d_inner, d_state, bias=False)
        self.ssm_C = nn.Linear(d_state, self.d_inner, bias=False)
        self.ssm_D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.apply(_init_xavier)

    def _ssm_scan(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        bx = self.ssm_B(x)
        states = []
        h = torch.zeros(B, self.d_state, device=x.device, dtype=x.dtype)
        for t in range(L):
            h = h @ self.ssm_A.t() + bx[:, t, :]
            states.append(h)
        h_seq = torch.stack(states, dim=1)
        out = self.ssm_C(h_seq) + self.ssm_D * x
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        projected = self.in_proj(x)
        x_conv, gate = projected.chunk(2, dim=-1)
        x_conv = x_conv.transpose(1, 2)
        x_conv = self.conv1d(x_conv)
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)
        x_ssm = self._ssm_scan(x_conv)
        x_gated = x_ssm * F.silu(gate)
        out = self.out_proj(self.dropout(x_gated))
        return out + residual


class TemporalBackbone(nn.Module):
    """Stack of SSM blocks for multivariate 82-game sequences."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        num_layers: int = 4,
        d_state: int = 64,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList(
            [
                SimplifiedSSMBlock(d_model, d_state=d_state, expand=expand, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.apply(_init_xavier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=1)
        return self.norm(x)


# ---------------------------------------------------------------------------
# 2. Tabular Backbone – FT-Transformer
# ---------------------------------------------------------------------------

class FeatureTokenizer(nn.Module):
    """Embed each scalar feature into a token vector."""

    def __init__(self, num_features: int, d_model: int = 128):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.feature_embeddings = nn.ModuleList(
            [nn.Linear(1, d_model) for _ in range(num_features)]
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_bias = nn.Parameter(torch.randn(1, num_features + 1, d_model) * 0.02)
        self.apply(_init_xavier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        tokens = []
        for i, emb in enumerate(self.feature_embeddings):
            feature = x[:, i:i + 1]
            token = emb(feature)
            tokens.append(token)
        tokens = torch.stack(tokens, dim=1)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_bias
        return tokens


class FTTransformerBlock(nn.Module):
    """Transformer encoder block for tabular tokens."""

    def __init__(self, d_model: int, nhead: int = 8, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.apply(_init_xavier)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=mask)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class FTTransformerBackbone(nn.Module):
    """Feature-Tokenizer + Transformer for scalar contextual features."""

    def __init__(
        self,
        num_features: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(num_features, d_model)
        self.blocks = nn.ModuleList(
            [
                FTTransformerBlock(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.apply(_init_xavier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        for block in self.blocks:
            tokens = block(tokens)
        cls = tokens[:, 0, :]
        return self.norm(cls)


# ---------------------------------------------------------------------------
# 3. Relational Backbone – Lightweight GAT
# ---------------------------------------------------------------------------

class GATLayer(nn.Module):
    """Single Graph Attention layer (GATv1-style) with multi-head attention.
    Always outputs (B, N, out_features) regardless of num_heads / concat.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,
        add_self_loops: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.add_self_loops = add_self_loops

        self.head_dim = out_features // num_heads if concat else out_features
        self.linear = nn.Linear(in_features, num_heads * self.head_dim, bias=False)

        self.att_src = nn.Parameter(torch.randn(1, num_heads, 1, 1))
        self.att_dst = nn.Parameter(torch.randn(1, num_heads, 1, 1))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        if concat:
            self.proj = nn.Linear(num_heads * self.head_dim, out_features)
        else:
            self.proj = None
        self.apply(_init_xavier)

    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, _ = x.shape
        h = self.linear(x)
        h = h.view(B, N, self.num_heads, self.head_dim)
        h = h.permute(0, 2, 1, 3)

        src = (h * self.att_src).sum(dim=-1, keepdim=True)
        dst = (h * self.att_dst).sum(dim=-1, keepdim=True)
        scores = src + dst.transpose(-2, -1)
        scores = self.leaky_relu(scores)

        if adj is not None:
            adj = adj.unsqueeze(1)
            scores = scores.masked_fill(adj == 0, float("-inf"))

        alpha = F.softmax(scores, dim=-1)
        alpha = self.dropout(alpha)

        out = alpha @ h
        out = out.permute(0, 2, 1, 3).contiguous()
        if self.concat:
            out = out.reshape(B, N, -1)
            out = self.proj(out)
        else:
            out = out.mean(dim=2)

        return out


class RelationalBackbone(nn.Module):
    """Lightweight GAT that pools lineup/matchup synergy into a graph-level vector."""

    def __init__(
        self,
        node_features: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        dims = [node_features] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList(
            [
                GATLayer(dims[i], dims[i + 1], num_heads=num_heads, dropout=dropout)
                for i in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.pool = nn.Linear(hidden_dim, hidden_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.apply(_init_xavier)

    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer, norm in zip(self.layers, self.norms):
            x = layer(x, adj)
            x = F.gelu(norm(x))
        pooled, _ = x.max(dim=1)
        pooled = self.pool(pooled)
        return self.out_norm(pooled)


# ---------------------------------------------------------------------------
# 4. Copula Head – Multivariate Gaussian with Cholesky covariance
# ---------------------------------------------------------------------------

class CopulaHead(nn.Module):
    """Outputs a 6-d mean vector and a Cholesky-decomposed 6x6 covariance.

    The covariance is parameterized via the lower-triangular Cholesky factor L
    so that  Sigma = L @ L.T  is guaranteed positive-definite.
    """

    NUM_STATS = 6  # PTS, REB, AST, STL, BLK, TOV

    def __init__(self, in_features: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.mean_net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.NUM_STATS),
        )
        self.chol_diag_net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.NUM_STATS),
        )
        self.chol_off_net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.NUM_STATS * (self.NUM_STATS - 1) // 2),
        )
        self.apply(_init_xavier)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean_net(x)
        diag_raw = self.chol_diag_net(x)
        diag = F.softplus(diag_raw) + 1e-5
        off = self.chol_off_net(x)

        B = x.shape[0]
        L = torch.zeros(B, self.NUM_STATS, self.NUM_STATS, device=x.device, dtype=x.dtype)
        L[:, range(self.NUM_STATS), range(self.NUM_STATS)] = diag
        tril_indices = torch.tril_indices(self.NUM_STATS, self.NUM_STATS, offset=-1, device=x.device)
        L[:, tril_indices[0], tril_indices[1]] = off

        return mean, L


# ---------------------------------------------------------------------------
# 5. Nexus Model – Assembly of all backbones
# ---------------------------------------------------------------------------

class NexusModel(nn.Module):
    """Unified multi-modal model for NBA stat-line prediction.

    Inputs
    ------
    temporal : (B, seq_len, temporal_dim)
        Rolling / sequential features per player (e.g. last 82 games).
    tabular  : (B, tabular_dim)
        Scalar contextual features (rest days, pace, opponent strength, etc.).
    lineup   : Optional (B, num_players, node_features)
        Lineup / matchup node embeddings for relational reasoning.
    adj      : Optional (B, num_players, num_players)
        Adjacency matrix for the lineup graph.

    Output
    ------
    mean : (B, 6)       predicted mean for PTS, REB, AST, STL, BLK, TOV
    cov  : (B, 6, 6)    full covariance matrix ( Sigma = L @ L.T )
    """

    def __init__(
        self,
        temporal_dim: int = 128,
        tabular_dim: int = 64,
        node_features: int = 32,
        seq_len: int = 82,
        d_model: int = 128,
        temporal_layers: int = 4,
        tabular_layers: int = 4,
        relational_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_relational: bool = True,
    ):
        super().__init__()
        self.temporal_dim = temporal_dim
        self.tabular_dim = tabular_dim
        self.node_features = node_features
        self.use_relational = use_relational
        self.num_stats = CopulaHead.NUM_STATS

        self.temporal_backbone = TemporalBackbone(
            input_dim=temporal_dim,
            d_model=d_model,
            num_layers=temporal_layers,
            dropout=dropout,
        )

        self.tabular_backbone = FTTransformerBackbone(
            num_features=tabular_dim,
            d_model=d_model,
            nhead=num_heads,
            num_layers=tabular_layers,
            dropout=dropout,
        )

        rel_repr_dim = 0
        if use_relational:
            self.relational_backbone = RelationalBackbone(
                node_features=node_features,
                hidden_dim=d_model // 2,
                num_layers=relational_layers,
                num_heads=max(num_heads // 2, 2),
                dropout=dropout,
            )
            rel_repr_dim = d_model // 2
        else:
            self.relational_backbone = None

        fusion_dim = d_model + d_model + rel_repr_dim
        self.fusion_proj = nn.Sequential(
            nn.Linear(fusion_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

        self.copula_head = CopulaHead(in_features=d_model, hidden_dim=d_model, dropout=dropout)

    def forward(
        self,
        temporal: torch.Tensor,
        tabular: torch.Tensor,
        lineup: Optional[torch.Tensor] = None,
        adj: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t_repr = self.temporal_backbone(temporal)
        tab_repr = self.tabular_backbone(tabular)

        if self.use_relational and lineup is not None:
            rel_repr = self.relational_backbone(lineup, adj)
            fused = torch.cat([t_repr, tab_repr, rel_repr], dim=-1)
        else:
            fused = torch.cat([t_repr, tab_repr], dim=-1)

        fused = self.fusion_proj(fused)
        mean, L = self.copula_head(fused)
        cov = L @ L.transpose(-2, -1)
        return mean, cov

    def predict_stats(
        self,
        temporal: torch.Tensor,
        tabular: torch.Tensor,
        lineup: Optional[torch.Tensor] = None,
        adj: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            mean, _ = self.forward(temporal, tabular, lineup, adj)
        return mean.detach().cpu().numpy()
