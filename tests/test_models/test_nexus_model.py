"""Tests for the Nexus multi-modal model."""

# pyright: reportPrivateImportUsage=false

import numpy as np
import pytest
import torch

from src.models.nexus_model import (
    CopulaHead,
    FeatureTokenizer,
    FTTransformerBackbone,
    GATLayer,
    NexusModel,
    RelationalBackbone,
    SimplifiedSSMBlock,
    TemporalBackbone,
)


class TestSimplifiedSSMBlock:
    def test_forward_shape(self):
        block = SimplifiedSSMBlock(d_model=32, d_state=16, expand=2)
        x = torch.randn(2, 5, 32)
        out = block(x)
        assert out.shape == (2, 5, 32)

    def test_residual_addition(self):
        block = SimplifiedSSMBlock(d_model=16)
        x = torch.randn(1, 3, 16)
        out = block(x)
        # Residual path should keep gradients flowing
        assert out.requires_grad is True

    def test_ssm_scan_statefulness(self):
        block = SimplifiedSSMBlock(d_model=16, d_state=8)
        x = torch.randn(1, 4, 32)
        out = block._ssm_scan(x)
        assert out.shape == (1, 4, 32)


class TestTemporalBackbone:
    def test_pooled_output(self):
        bb = TemporalBackbone(input_dim=8, d_model=32, num_layers=2)
        x = torch.randn(4, 10, 8)
        out = bb(x)
        assert out.shape == (4, 32)

    def test_grad_flow(self):
        bb = TemporalBackbone(input_dim=4, d_model=16, num_layers=1)
        x = torch.randn(2, 6, 4, requires_grad=True)
        out = bb(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestFeatureTokenizer:
    def test_token_count(self):
        tok = FeatureTokenizer(num_features=5, d_model=16)
        x = torch.randn(3, 5)
        tokens = tok(x)
        # num_features + CLS
        assert tokens.shape == (3, 6, 16)


class TestFTTransformerBackbone:
    def test_cls_output(self):
        ft = FTTransformerBackbone(num_features=5, d_model=32, num_layers=2)
        x = torch.randn(4, 5)
        out = ft(x)
        assert out.shape == (4, 32)

    def test_attention_mask_ignored_for_full_self(self):
        ft = FTTransformerBackbone(num_features=3, d_model=16, num_layers=1)
        x = torch.randn(2, 3)
        out = ft(x)
        assert out.shape == (2, 16)


class TestGATLayer:
    def test_forward_shape(self):
        layer = GATLayer(in_features=8, out_features=16, num_heads=4)
        x = torch.randn(2, 5, 8)
        adj = torch.ones(2, 5, 5)
        out = layer(x, adj)
        assert out.shape == (2, 5, 16)

    def test_forward_without_adj(self):
        layer = GATLayer(in_features=4, out_features=8, num_heads=2)
        x = torch.randn(3, 4, 4)
        out = layer(x)
        assert out.shape == (3, 4, 8)


class TestRelationalBackbone:
    def test_pooling(self):
        gat = RelationalBackbone(node_features=4, hidden_dim=8, num_layers=2, num_heads=2)
        x = torch.randn(3, 6, 4)
        adj = torch.ones(3, 6, 6)
        out = gat(x, adj)
        assert out.shape == (3, 8)

    def test_no_adj(self):
        gat = RelationalBackbone(node_features=2, hidden_dim=4, num_layers=1, num_heads=2)
        x = torch.randn(2, 3, 2)
        out = gat(x)
        assert out.shape == (2, 4)


class TestCopulaHead:
    def test_mean_shape(self):
        head = CopulaHead(in_features=16)
        x = torch.randn(4, 16)
        mean, L = head(x)
        assert mean.shape == (4, 6)
        assert L.shape == (4, 6, 6)

    def test_cholesky_lower_triangular(self):
        head = CopulaHead(in_features=8)
        x = torch.randn(2, 8)
        _, L = head(x)
        # Upper triangle (excluding diagonal) should be ~zero
        upper = L.triu(diagonal=1)
        assert torch.allclose(upper, torch.zeros_like(upper))

    def test_positive_diagonal(self):
        head = CopulaHead(in_features=8)
        x = torch.randn(3, 8)
        _, L = head(x)
        diag = torch.diagonal(L, dim1=-2, dim2=-1)
        assert (diag > 0).all()

    def test_cov_positive_definite(self):
        head = CopulaHead(in_features=8)
        x = torch.randn(3, 8)
        _, L = head(x)
        cov = L @ L.transpose(-2, -1)
        eig = torch.linalg.eigvalsh(cov)
        assert (eig > 0).all()


class TestNexusModel:
    def test_forward_all_backbones(self):
        model = NexusModel(
            temporal_dim=12,
            tabular_dim=8,
            node_features=6,
            seq_len=10,
            d_model=32,
            temporal_layers=2,
            tabular_layers=2,
            relational_layers=1,
            num_heads=4,
            use_relational=True,
        )
        temporal = torch.randn(3, 10, 12)
        tabular = torch.randn(3, 8)
        lineup = torch.randn(3, 5, 6)
        adj = torch.ones(3, 5, 5)
        mean, cov = model(temporal, tabular, lineup, adj)
        assert mean.shape == (3, 6)
        assert cov.shape == (3, 6, 6)

    def test_forward_no_relational(self):
        model = NexusModel(
            temporal_dim=12,
            tabular_dim=8,
            use_relational=False,
            d_model=32,
            temporal_layers=1,
            tabular_layers=1,
        )
        temporal = torch.randn(2, 10, 12)
        tabular = torch.randn(2, 8)
        mean, cov = model(temporal, tabular)
        assert mean.shape == (2, 6)
        assert cov.shape == (2, 6, 6)

    def test_predict_stats_returns_numpy(self):
        model = NexusModel(
            temporal_dim=4,
            tabular_dim=3,
            use_relational=False,
            d_model=16,
            temporal_layers=1,
            tabular_layers=1,
        )
        temporal = torch.randn(2, 5, 4)
        tabular = torch.randn(2, 3)
        preds = model.predict_stats(temporal, tabular)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (2, 6)

    def test_backward(self):
        model = NexusModel(
            temporal_dim=4,
            tabular_dim=3,
            use_relational=False,
            d_model=16,
            temporal_layers=1,
            tabular_layers=1,
        )
        temporal = torch.randn(2, 5, 4, requires_grad=True)
        tabular = torch.randn(2, 3, requires_grad=True)
        mean, cov = model(temporal, tabular)
        loss = mean.sum() + cov.sum()
        loss.backward()
        assert temporal.grad is not None
        assert tabular.grad is not None

    @pytest.mark.gpu
    def test_device_transfer(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        device = torch.device("cuda")
        model = NexusModel(temporal_dim=4, tabular_dim=3, use_relational=False, d_model=16)
        model = model.to(device)
        temporal = torch.randn(1, 5, 4, device=device)
        tabular = torch.randn(1, 3, device=device)
        mean, cov = model(temporal, tabular)
        assert mean.device.type == "cuda"
        assert cov.device.type == "cuda"
