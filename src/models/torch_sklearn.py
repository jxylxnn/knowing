import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class TorchRidgeRegressor:
    """GPU-accelerated Ridge Regression using PyTorch."""
    def __init__(self, alpha: float = 1.0, device: str = 'cuda'):
        self.alpha = alpha
        self.device = torch.device('cpu') # Force CPU for RTX 5070 Ti compatibility
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X, y, sample_weight: Optional[np.ndarray] = None):
        # Convert to tensors
        if isinstance(X, pd.DataFrame): X = X.values
        if isinstance(y, (pd.Series, pd.DataFrame)): y = y.values
        
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.float32).to(self.device).view(-1, 1)
        
        # Add bias term for intercept
        ones = torch.ones(X_t.shape[0], 1).to(self.device)
        X_b = torch.cat([ones, X_t], dim=1)
        
        # Apply sample weights if provided
        if sample_weight is not None:
            w_t = torch.tensor(sample_weight, dtype=torch.float32).to(self.device).sqrt().view(-1, 1)
            X_b = X_b * w_t
            y_t = y_t * w_t
        
        # Closed-form solution: w = (X^T X + alpha*I)^-1 X^T y
        XTX = torch.mm(X_b.t(), X_b)
        eye = torch.eye(X_b.shape[1]).to(self.device)
        # Don't regularize the intercept (first column)
        eye[0, 0] = 0
        
        # Add small epsilon for numerical stability and apply ridge regularization
        identity_reg = self.alpha * eye + 1e-6 * torch.eye(X_b.shape[1]).to(self.device)
        
        try:
            w = torch.linalg.solve(XTX + identity_reg, torch.mm(X_b.t(), y_t))
        except torch._C._LinAlgError:
            # Fallback to least squares if the matrix is still singular
            logger.warning("Ridge solver fallback to linalg.lstsq due to singularity.")
            w, _, _, _ = torch.linalg.lstsq(XTX + identity_reg, torch.mm(X_b.t(), y_t))
        
        weights = w.cpu().numpy()
        self.intercept_ = weights[0, 0]
        self.coef_ = weights[1:].flatten()
        return self

    def predict(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame): X = X.values
        return X @ self.coef_ + self.intercept_

    def get_params(self, deep=True):
        return {"alpha": self.alpha, "device": self.device}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

class TorchMLPRegressor:
    """GPU-accelerated MLP using PyTorch with sklearn-compatible interface."""
    def __init__(self, hidden_dims: List[int] = [256, 128], max_iter: int = 1000, 
                 lr: float = 0.001, weight_decay: float = 1e-5, device: str = 'cuda'):
        self.hidden_dims = hidden_dims
        self.max_iter = max_iter
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = torch.device('cpu') # Force CPU for RTX 5070 Ti compatibility
        self.model = None
        self.is_trained = False

    def _build_model(self, input_dim):
        layers = []
        curr_dim = input_dim
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.ReLU())
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, 1))
        return nn.Sequential(*layers).to(self.device)

    def fit(self, X, y, sample_weight: Optional[np.ndarray] = None):
        if isinstance(X, pd.DataFrame): X = X.values
        if isinstance(y, (pd.Series, pd.DataFrame)): y = y.values
        
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.float32).to(self.device).view(-1, 1)
        w_t = None
        if sample_weight is not None:
            w_t = torch.tensor(sample_weight, dtype=torch.float32).to(self.device).view(-1, 1)
        
        self.model = self._build_model(X_t.shape[1])
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        self.model.train()
        for _ in range(self.max_iter):
            optimizer.zero_grad()
            outputs = self.model(X_t)
            if w_t is not None:
                loss = (w_t * (outputs - y_t)**2).mean()
            else:
                loss = nn.functional.mse_loss(outputs, y_t)
            loss.backward()
            optimizer.step()
            
            if loss.item() < 1e-6: # Early convergence check
                break
                
        self.is_trained = True
        return self

    def predict(self, X) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained.")
        if isinstance(X, pd.DataFrame): X = X.values
        
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            return self.model(X_t).cpu().numpy().flatten()

    def get_params(self, deep=True):
        return {
            "hidden_dims": self.hidden_dims, "max_iter": self.max_iter, 
            "lr": self.lr, "weight_decay": self.weight_decay, "device": self.device
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
