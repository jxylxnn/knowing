import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Dict, Any, Optional
import torch.backends.cudnn as cudnn
import math

logger = logging.getLogger(__name__)

cudnn.benchmark = True


class PlayerSequenceDataset(Dataset):
    """Dataset for sequence-based player stat prediction."""
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class LSTMModel(nn.Module):
    """LSTM for temporal pattern recognition in player stats."""
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, 
                 output_dim: int = 3, dropout: float = 0.2, bidirectional: bool = False):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.input_dim = input_dim
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_output_dim, output_dim)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
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
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


class LSTMWrapper:
    """Wrapper for handling sequence processing and training for the LSTM."""
    
    DEFAULT_CONFIG = {
        'hidden_dim': 128,
        'num_layers': 2,
        'bidirectional': False,
        'dropout': 0.2,
        'batch_size': 32,
        'epochs': 50,
        'lr': 1e-3,
        'warmup_ratio': 0.1,
        'use_compile': False,
    }
    
    def __init__(self, input_dim: int, seq_len: int = 10, config: Optional[Dict[str, Any]] = None):
        from src.models.gpu_utils import get_device
        self.seq_len = seq_len
        self.device = get_device()
        
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        
        self.model = LSTMModel(
            input_dim=input_dim,
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout'],
            bidirectional=self.config['bidirectional']
        ).to(self.device)
        
        self.input_dim = input_dim
        self.is_trained = False
        self.feat_mean = None
        self.feat_std = None
        
        logger.info(f"LSTM initialized: hidden={self.config['hidden_dim']}, "
                   f"layers={self.config['num_layers']}, bidirectional={self.config['bidirectional']}, "
                   f"device={self.device}")

    def _create_sequences(self, df: pd.DataFrame, feature_cols: List[str], target_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Creates sliding window sequences for each player."""
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
        """Trains the LSTM on sequential data with warmup + cosine decay."""
        epochs = epochs or self.config['epochs']
        batch_size = self.config['batch_size']
        lr = self.config['lr']
        warmup_ratio = self.config['warmup_ratio']
        
        logger.info(f"Generating sequences for {len(df)} records...")
        X, y = self._create_sequences(df, feature_cols, target_cols)
        
        if len(X) == 0:
            logger.warning("Not enough data to create sequences for LSTM.")
            return
        
        self.feat_mean = X.mean(axis=(0, 1))
        self.feat_std = X.std(axis=(0, 1)) + 1e-6
        X = (X - self.feat_mean) / self.feat_std
        
        dataset = PlayerSequenceDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
        
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
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
                logger.info(f"LSTM Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")
        
        self.is_trained = True
        logger.info(f"LSTM training complete. Final loss: {total_loss/len(loader):.4f}")

    def predict(self, sequence: np.ndarray) -> np.ndarray:
        """Predicts the next game based on a sequence of games."""
        if not self.is_trained:
            return None
        
        self.model.eval()
        sequence = (sequence - self.feat_mean) / self.feat_std
        seq_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            preds = self.model(seq_tensor).cpu().numpy()
        return preds

    def save(self, path: str):
        """Saves the model and normalization params."""
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
        logger.info(f"LSTM model saved to {path}")

    @classmethod
    def load(cls, path: str):
        """Loads the LSTM model."""
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