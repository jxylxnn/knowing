import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Tuple, Optional
import torch.backends.cudnn
from copy import deepcopy
import math

logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = True


class SELayer(nn.Module):
    """Squeeze-and-Excitation Block for feature-wise attention."""
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.size()
        y = self.avg_pool(x.unsqueeze(-1)).squeeze(-1)
        y = self.fc(y).view(b, c)
        return x * y


class SEResidualBlock(nn.Module):
    """Residual Block with Squeeze-and-Excitation and SiLU activation."""
    def __init__(self, dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.se = SELayer(dim)
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(x + self.se(self.block(x)))


class MultiOutputNN(nn.Module):
    """Wide ResNet with Attention for NBA Stats prediction."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 512, num_blocks: int = 6, 
                 dropout: float = 0.2, output_dim: int = 3):
        super(MultiOutputNN, self).__init__()
        
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        blocks = []
        for _ in range(num_blocks):
            blocks.append(SEResidualBlock(hidden_dim, dropout))
        self.backbone = nn.Sequential(*blocks)
        
        self.head_mean = nn.Linear(hidden_dim, output_dim)
        self.head_var = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.input_bn(x)
        x = self.input_layer(x)
        features = self.backbone(x)
        
        means = self.head_mean(features)
        logvars = self.head_var(features)
        
        return means, logvars


class WarmupCosineScheduler:
    """Learning rate warmup + cosine decay scheduler."""
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


class MultiOutputWrapper:
    """Wrapper for the Wide-SE-ResNet with config support and torch.compile."""
    
    DEFAULT_CONFIG = {
        'hidden_dim': 512,
        'num_blocks': 6,
        'dropout': 0.2,
        'batch_size': 2048,
        'epochs': 100,
        'lr': 1e-3,
        'warmup_ratio': 0.05,
        'label_smoothing': 0.0,
        'use_compile': False,
        'early_stop_patience': 20,
    }
    
    def __init__(self, input_dim: int, target_names: List[str] = None, 
                 config: Optional[Dict[str, Any]] = None):
        if target_names is None:
            target_names = ['PTS', 'REB', 'AST']
            
        from src.models.gpu_utils import get_device
        
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.input_dim = input_dim
        self.target_names = target_names
        self.device = get_device()
        
        self.model = MultiOutputNN(
            input_dim=input_dim,
            hidden_dim=self.config['hidden_dim'],
            num_blocks=self.config['num_blocks'],
            dropout=self.config['dropout'],
            output_dim=len(target_names)
        ).to(self.device)
        
        self.scaler_X = None
        self.scaler_y = None
        self.is_trained = False
        
        logger.info(f"MultiOutputNN initialized: hidden={self.config['hidden_dim']}, "
                   f"blocks={self.config['num_blocks']}, dropout={self.config['dropout']}, "
                   f"device={self.device}")

    def _nll_loss(self, y_true, y_pred_mean, y_pred_logvar):
        """Negative log-likelihood loss with uncertainty estimation."""
        return torch.mean(0.5 * torch.exp(-y_pred_logvar) * (y_true - y_pred_mean)**2 + 0.5 * y_pred_logvar)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, 
            epochs: Optional[int] = None, batch_size: Optional[int] = None,
            val_split: float = 0.1):
        """Training with warmup, cosine decay, and early stopping."""
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        epochs = epochs or self.config['epochs']
        batch_size = batch_size or self.config['batch_size']
        lr = self.config['lr']
        warmup_ratio = self.config['warmup_ratio']
        early_stop_patience = self.config['early_stop_patience']
        
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        np.random.seed(42)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y[self.target_names], 
            test_size=val_split, 
            random_state=42,
            shuffle=True
        )
        
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_val_scaled = self.scaler_X.transform(X_val)
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        y_val_scaled = self.scaler_y.transform(y_val)
        
        num_workers = 4 if self.device.type == 'cuda' else 0
        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train_scaled, dtype=torch.float32),
                torch.tensor(y_train_scaled, dtype=torch.float32)
            ),
            batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_val_scaled, dtype=torch.float32),
                torch.tensor(y_val_scaled, dtype=torch.float32)
            ),
            batch_size=batch_size * 2, shuffle=False, pin_memory=True, num_workers=num_workers
        )
        
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        total_steps = epochs * len(train_loader)
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
        
        scaler_ctx = 'cuda' if self.device.type == 'cuda' else 'cpu'
        grad_scaler = torch.amp.GradScaler(scaler_ctx, enabled=(self.device.type == 'cuda'))
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast(scaler_ctx, enabled=(self.device.type == 'cuda')):
                    means, logvars = self.model(batch_X)
                    loss = self._nll_loss(batch_y, means, logvars)
                
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                scheduler.step()
                train_loss += loss.item()
            
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device, non_blocking=True)
                    batch_y = batch_y.to(self.device, non_blocking=True)
                    
                    with torch.amp.autocast(scaler_ctx, enabled=(self.device.type == 'cuda')):
                        means, logvars = self.model(batch_X)
                        loss = self._nll_loss(batch_y, means, logvars)
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            current_lr = optimizer.param_groups[0]['lr']
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_state = deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"NN Epoch {epoch+1}/{epochs} | Train: {avg_train_loss:.4f} | "
                           f"Val: {avg_val_loss:.4f} | LR: {current_lr:.2e}")
        
        if best_state is not None:
            self.model.load_state_dict(best_state)
            
        self.is_trained = True
        logger.info(f"NN training complete. Best validation loss: {best_val_loss:.4f}")

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        X_scaled = self.scaler_X.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
            means_scaled, logvars_scaled = self.model(X_tensor)
            means_scaled = means_scaled.cpu().numpy()
            vars_scaled = torch.exp(logvars_scaled).cpu().numpy()
            
        means = self.scaler_y.inverse_transform(means_scaled)
        stds = np.sqrt(vars_scaled) * np.sqrt(self.scaler_y.var_)
        
        return means, stds

    def save(self, path: str):
        import joblib
        torch.save(self.model.state_dict(), path.replace('.pkl', '.pt'))
        joblib.dump({
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'target_names': self.target_names,
            'input_dim': self.input_dim,
            'config': self.config,
        }, path)
        logger.info(f"MultiOutputNN saved to {path}")

    @classmethod
    def load(cls, path: str):
        import joblib
        state = joblib.load(path)
        config = state.get('config', {'hidden_dim': state.get('hidden_dim', 512)})
        instance = cls(
            input_dim=state['input_dim'], 
            target_names=state['target_names'], 
            config=config
        )
        instance.scaler_X = state['scaler_X']
        instance.scaler_y = state['scaler_y']
        instance.model.load_state_dict(torch.load(path.replace('.pkl', '.pt'), weights_only=True))
        instance.is_trained = True
        return instance