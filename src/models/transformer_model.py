import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint
import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Dict, Any, Optional
import math
import torch.backends.cudnn as cudnn

logger = logging.getLogger(__name__)

cudnn.benchmark = True


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerModel(nn.Module):
    """Transformer for processing long player stat trajectories with Self-Attention."""
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 8, 
                 num_layers: int = 4, output_dim: int = 3, dropout: float = 0.1,
                 dim_feedforward: int = 512, grad_checkpoint: bool = False):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        self.grad_checkpoint = grad_checkpoint
        
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=dim_feedforward, 
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.fc = nn.Linear(d_model, output_dim)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, return_attention=False):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        if self.grad_checkpoint and self.training:
            x = checkpoint(self.transformer_encoder, x, use_reentrant=False)
        else:
            x = self.transformer_encoder(x)
        
        out = self.fc(x[:, -1, :])
        return out


class WarmupScheduler:
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


class TransformerWrapper:
    """Wrapper for handling sequence processing and training for the Transformer."""
    
    DEFAULT_CONFIG = {
        'd_model': 128,
        'nhead': 8,
        'num_layers': 4,
        'dim_feedforward': 512,
        'dropout': 0.1,
        'batch_size': 32,
        'epochs': 50,
        'lr': 5e-4,
        'warmup_ratio': 0.1,
        'grad_checkpoint': False,
        'use_compile': False,
    }
    
    def __init__(self, input_dim: int, seq_len: int = 50, config: Optional[Dict[str, Any]] = None):
        from src.models.gpu_utils import get_device
        self.seq_len = seq_len
        self.device = get_device()
        
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        
        self.model = TransformerModel(
            input_dim=input_dim,
            d_model=self.config['d_model'],
            nhead=self.config['nhead'],
            num_layers=self.config['num_layers'],
            dim_feedforward=self.config['dim_feedforward'],
            dropout=self.config['dropout'],
            grad_checkpoint=self.config['grad_checkpoint']
        ).to(self.device)
        
        self.input_dim = input_dim
        self.is_trained = False
        self.feat_mean = None
        self.feat_std = None
        
        logger.info(f"Transformer initialized: d_model={self.config['d_model']}, "
                   f"heads={self.config['nhead']}, layers={self.config['num_layers']}, "
                   f"ff_dim={self.config['dim_feedforward']}, grad_ckpt={self.config['grad_checkpoint']}, "
                   f"device={self.device}")

    def _create_sequences(self, df: pd.DataFrame, feature_cols: List[str], target_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for player_id, group in df.groupby('PLAYER_ID'):
            if len(group) < self.seq_len:
                continue
            group = group.sort_values('GAME_DATE')
            features = group[feature_cols].values
            targets = group[target_cols].values
            for i in range(len(features) - self.seq_len):
                X.append(features[i:i + self.seq_len])
                y.append(targets[i + self.seq_len])
        return np.array(X), np.array(y)

    def fit(self, df: pd.DataFrame, feature_cols: List[str], target_cols: List[str], 
            epochs: Optional[int] = None):
        """Train with warmup + cosine decay and gradient clipping."""
        epochs = epochs or self.config['epochs']
        batch_size = self.config['batch_size']
        lr = self.config['lr']
        warmup_ratio = self.config['warmup_ratio']
        
        logger.info(f"Generating sequences (len={self.seq_len}) for Transformer...")
        X, y = self._create_sequences(df, feature_cols, target_cols)
        
        if len(X) == 0:
            logger.warning("Not enough data to create sequences for Transformer.")
            return
        
        self.feat_mean = X.mean(axis=(0, 1))
        self.feat_std = X.std(axis=(0, 1)) + 1e-6
        X = (X - self.feat_mean) / self.feat_std
        
        from src.models.lstm_model import PlayerSequenceDataset
        dataset = PlayerSequenceDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
        
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        total_steps = epochs * len(loader)
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = WarmupScheduler(optimizer, warmup_steps, total_steps)
        criterion = nn.MSELoss()
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)
                preds = self.model(batch_X)
                loss = criterion(preds, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(loader)
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Transformer Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")
        
        self.is_trained = True
        logger.info(f"Transformer training complete. Final loss: {total_loss/len(loader):.4f}")

    def predict(self, sequence: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            return None
        self.model.eval()
        sequence = (sequence - self.feat_mean) / self.feat_std
        seq_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            preds = self.model(seq_tensor).cpu().numpy()
        return preds

    def save(self, path: str):
        import joblib
        state = {
            'model_state': self.model.state_dict(),
            'feat_mean': self.feat_mean,
            'feat_std': self.feat_std,
            'input_dim': self.input_dim,
            'seq_len': self.seq_len,
            'config': self.config,
        }
        joblib.dump(state, path)
        logger.info(f"Transformer model saved to {path}")

    @classmethod
    def load(cls, path: str):
        import joblib
        state = joblib.load(path)
        
        config = state.get('config', {})
        instance = cls(
            input_dim=state['input_dim'], 
            seq_len=state['seq_len'],
            config=config
        )
        instance.model.load_state_dict(state['model_state'])
        instance.feat_mean = state['feat_mean']
        instance.feat_std = state['feat_std']
        instance.is_trained = True
        return instance