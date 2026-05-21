"""Tests for the Nexus model loss functions."""

# pyright: reportPrivateImportUsage=false

import math

import numpy as np
import pytest
import torch

from src.training.nexus_loss import GaussianNLLLoss, NexusLoss


class TestGaussianNLLLoss:
    def test_shapes(self):
        loss_fn = GaussianNLLLoss()
        mean = torch.randn(4, 6)
        L = torch.tril(torch.randn(4, 6, 6))
        L[:, range(6), range(6)] = torch.abs(torch.diagonal(L, dim1=-2, dim2=-1)) + 1.0
        target = torch.randn(4, 6)
        loss = loss_fn(mean, L, target)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_perfect_prediction_zero_maha(self):
        """If target == mean, Mahalanobis = 0 and loss reduces to logdet term."""
        loss_fn = GaussianNLLLoss()
        mean = torch.zeros(2, 3)
        # Identity Cholesky -> Sigma = I, logdet = 0
        L = torch.eye(3).unsqueeze(0).expand(2, 3, 3).float()
        target = mean.clone()
        loss = loss_fn(mean, L, target)
        expected = 0.5 * 3 * math.log(2 * math.pi)
        assert np.isclose(loss.item(), expected, atol=1e-4)

    def test_mask_filters_samples(self):
        loss_fn = GaussianNLLLoss()
        mean = torch.randn(3, 6)
        L = torch.tril(torch.randn(3, 6, 6))
        L[:, range(6), range(6)] = torch.abs(torch.diagonal(L, dim1=-2, dim2=-1)) + 1.0
        target = torch.randn(3, 6)
        mask = torch.tensor([True, False, True])
        loss = loss_fn(mean, L, target, mask=mask)
        # Should only average over 2 samples
        assert loss.ndim == 0

    def test_handles_empty_mask(self):
        loss_fn = GaussianNLLLoss()
        mean = torch.randn(2, 6)
        L = torch.tril(torch.randn(2, 6, 6))
        L[:, range(6), range(6)] = torch.abs(torch.diagonal(L, dim1=-2, dim2=-1)) + 1.0
        target = torch.randn(2, 6)
        mask = torch.tensor([False, False])
        loss = loss_fn(mean, L, target, mask=mask)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_psd_covariance(self):
        """Ensure Cholesky gives positive-definite covariances."""
        loss_fn = GaussianNLLLoss()
        mean = torch.randn(3, 6)
        L = torch.tril(torch.randn(3, 6, 6))
        L[:, range(6), range(6)] = torch.abs(torch.diagonal(L, dim1=-2, dim2=-1)) + 1.0
        target = torch.randn(3, 6)
        loss = loss_fn(mean, L, target)
        cov = L @ L.transpose(-2, -1)
        eig = torch.linalg.eigvalsh(cov)
        assert (eig > 0).all()

    def test_gradient_flow(self):
        loss_fn = GaussianNLLLoss()
        mean = torch.randn(3, 4, requires_grad=True)
        L = torch.tril(torch.randn(3, 4, 4))
        L[:, range(4), range(4)] = torch.abs(torch.diagonal(L, dim1=-2, dim2=-1)) + 1.0
        L = L.clone().detach().requires_grad_(True)
        target = torch.randn(3, 4)
        loss = loss_fn(mean, L, target)
        loss.backward()
        assert mean.grad is not None
        assert L.grad is not None


class TestNexusLoss:
    def test_composite_loss(self):
        loss_fn = NexusLoss(nll_weight=1.0, mse_weight=0.5, cov_div_weight=0.1)
        mean = torch.randn(3, 6)
        L = torch.tril(torch.randn(3, 6, 6))
        L[:, range(6), range(6)] = torch.abs(torch.diagonal(L, dim1=-2, dim2=-1)) + 1.0
        target = torch.randn(3, 6)
        total, metrics = loss_fn(mean, L, target)
        assert isinstance(total, torch.Tensor)
        assert "nll" in metrics
        assert "mse" in metrics
        assert "cov_div" in metrics
        assert "total" in metrics

    def test_nll_only(self):
        loss_fn = NexusLoss(nll_weight=1.0, mse_weight=0.0, cov_div_weight=0.0)
        mean = torch.randn(2, 6)
        L = torch.tril(torch.randn(2, 6, 6))
        L[:, range(6), range(6)] = torch.abs(torch.diagonal(L, dim1=-2, dim2=-1)) + 1.0
        target = torch.randn(2, 6)
        total, metrics = loss_fn(mean, L, target)
        assert metrics["mse"] == 0.0
        assert metrics["cov_div"] == 0.0

    def test_backward(self):
        loss_fn = NexusLoss(nll_weight=1.0, mse_weight=0.2)
        mean = torch.randn(2, 6, requires_grad=True)
        L = torch.tril(torch.randn(2, 6, 6))
        L[:, range(6), range(6)] = torch.abs(torch.diagonal(L, dim1=-2, dim2=-1)) + 1.0
        L = L.clone().detach().requires_grad_(True)
        target = torch.randn(2, 6)
        total, _ = loss_fn(mean, L, target)
        total.backward()
        assert mean.grad is not None
        assert L.grad is not None
