"""
Nexus Model Loss Functions.

Provides the Negative Log-Likelihood (NLL) loss for a multivariate
Gaussian distribution parameterized by a mean vector and a Cholesky-
decomposed covariance matrix.

This natively penalizes the model for predicting mathematically
impossible stat combinations (e.g. negative variances or correlation
matrices that are not positive semi-definite).
"""

from __future__ import annotations

# pyright: reportPrivateImportUsage=false

import math
from typing import Tuple

import torch
import torch.nn as nn


class GaussianNLLLoss(nn.Module):
    """Multivariate Gaussian NLL with Cholesky covariance.

    Given predicted mean  mu  (B, K)  and Cholesky factor  L  (B, K, K),
    the covariance is  Sigma = L @ L.T  which is guaranteed SPD.

    The negative log-likelihood is:

        NLL = 0.5 * [ K * log(2*pi)
                      + log(det(Sigma))
                      + (y - mu)^T @ Sigma^{-1} @ (y - mu) ]

    Using Cholesky factorisation:
        det(Sigma)      = prod(diag(L)) ** 2
        log(det(Sigma)) = 2 * sum(log(diag(L)))
        Sigma^{-1}      = L^{-T} @ L^{-1}

    The Mahalanobis term is computed via solving L * z = (y - mu) and
    taking  ||z||^2 .
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.log_2pi = math.log(2.0 * math.pi)

    def _mahalanobis(self, residual: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """Compute (y-mu)^T @ Sigma^{-1} @ (y-mu) efficiently.

        Args:
            residual: (B, K)
            L:        (B, K, K) lower-triangular
        Returns:
            (B,) Mahalanobis distance squared.
        """
        # Solve L z = residual  (forward substitution because L is lower-tri)
        z = torch.linalg.solve_triangular(L, residual.unsqueeze(-1), upper=False)
        # ||z||^2
        return z.squeeze(-1).pow(2).sum(dim=-1)

    def forward(
        self,
        mean: torch.Tensor,
        L: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            mean:   (B, K) predicted means
            L:      (B, K, K) lower-triangular Cholesky factors
            target: (B, K) ground-truth stat lines
            mask:   Optional (B,) bool tensor where True = keep sample.
        Returns:
            Scalar loss (mean over batch).
        """
        residual = target - mean  # (B, K)
        K = mean.shape[-1]

        # log(det(Sigma)) = 2 * sum(log(diag(L)))
        diag_L = torch.diagonal(L, dim1=-2, dim2=-1)  # (B, K)
        logdet = 2.0 * torch.sum(torch.log(diag_L + self.eps), dim=-1)  # (B,)

        # Mahalanobis distance
        maha = self._mahalanobis(residual, L)  # (B,)

        # NLL per sample
        nll = 0.5 * (K * self.log_2pi + logdet + maha)  # (B,)

        if mask is not None:
            nll = nll[mask]
            if nll.numel() == 0:
                return torch.tensor(0.0, device=nll.device, dtype=nll.dtype)

        return nll.mean()


class NexusLoss(nn.Module):
    """Composite loss for the Nexus model.

    Combines:
    1. Multivariate Gaussian NLL (primary)
    2. Optional per-stat MSE regulariser (stabilises early training)
    3. Optional covariance-diversity bonus (prevents collapse to diagonal)
    """

    def __init__(
        self,
        nll_weight: float = 1.0,
        mse_weight: float = 0.0,
        cov_div_weight: float = 0.0,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.nll_loss = GaussianNLLLoss(eps=eps)
        self.nll_weight = nll_weight
        self.mse_weight = mse_weight
        self.cov_div_weight = cov_div_weight

    def forward(
        self,
        mean: torch.Tensor,
        L: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            mean:   (B, K)
            L:      (B, K, K)
            target: (B, K)
            mask:   (B,)
        Returns:
            total_loss, dict of component values
        """
        loss_nll = self.nll_loss(mean, L, target, mask=mask)

        loss_mse = torch.tensor(0.0, device=mean.device, dtype=mean.dtype)
        if self.mse_weight > 0.0:
            diff = (mean - target).pow(2)
            if mask is not None:
                diff = diff[mask]
            if diff.numel() > 0:
                loss_mse = diff.mean()

        loss_cov_div = torch.tensor(0.0, device=mean.device, dtype=mean.dtype)
        if self.cov_div_weight > 0.0:
            # Encourage non-diagonal covariance by penalising very small off-diagonals
            cov = L @ L.transpose(-2, -1)
            off_mask = ~torch.eye(cov.shape[-1], dtype=torch.bool, device=cov.device)
            off_vals = cov[:, off_mask]
            loss_cov_div = -off_vals.abs().mean()  # negative == bonus for larger abs values

        total = (
            self.nll_weight * loss_nll
            + self.mse_weight * loss_mse
            + self.cov_div_weight * loss_cov_div
        )

        metrics: dict[str, float] = {
            "nll": float(loss_nll.item()),
            "mse": float(loss_mse.item()),
            "cov_div": float(loss_cov_div.item()),
            "total": float(total.item()),
        }
        return total, metrics
