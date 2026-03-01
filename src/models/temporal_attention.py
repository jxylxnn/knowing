import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Optional, Dict, Any
import math
import joblib

logger = logging.getLogger(__name__)


class TemporalAttentionDataset(Dataset):
    """Dataset for temporal attention prediction, requiring history and next context."""
    def __init__(self, history: np.ndarray, next_context: np.ndarray, targets: np.ndarray):
        self.history = torch.tensor(history, dtype=torch.float32)
        self.next_context = torch.tensor(next_context, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.history)

    def __getitem__(self, idx):
        return self.history[idx], self.next_context[idx], self.targets[idx]


class TemporalAttentionModel(nn.Module):
    """Attention over recent games with learned game importance, conditioned on next game context."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_heads: int = 4, 
                 dropout: float = 0.1, output_dim: int = 3):
        super().__init__()
        
        self.game_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.recency_embedding = nn.Parameter(torch.randn(50, hidden_dim) * 0.02)
        
        self.context_encoder = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout)
        
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, history: torch.Tensor, next_game_context: torch.Tensor):
        batch_size, seq_len, _ = history.shape
        
        game_embeds = self.game_encoder(history)
        
        recency = self.recency_embedding[-seq_len:].unsqueeze(0)
        game_embeds = game_embeds + recency
        
        query = self.context_encoder(next_game_context).unsqueeze(1)
        
        attended, attn_weights = self.attention(query, game_embeds, game_embeds)
        attended = attended.squeeze(1)
        
        combined = torch.cat([attended, query.squeeze(1)], dim=-1)
        
        return self.output_head(combined), attn_weights


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


class TemporalAttentionWrapper:
    """Wrapper for handling sequence processing and training for the TemporalAttentionModel."""
    
    DEFAULT_CONFIG = {
        'hidden_dim': 128,
        'num_heads': 4,
        'dropout': 0.1,
        'batch_size': 256,
        'epochs': 50,
        'lr': 5e-4,
        'warmup_ratio': 0.1,
        'output_dim': 3,
    }
    
    def __init__(self, input_dim: int, seq_len: int = 20, config: Optional[Dict[str, Any]] = None):
        from src.models.gpu_utils import get_device
        
        self.seq_len = seq_len
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.device = get_device()
        self.input_dim = input_dim
        
        self.model = TemporalAttentionModel(
            input_dim=input_dim,
            hidden_dim=self.config['hidden_dim'],
            num_heads=self.config['num_heads'],
            dropout=self.config['dropout'],
            output_dim=self.config['output_dim']
        ).to(self.device)
        
        self.is_trained = False
        self.feat_mean = None
        self.feat_std = None
        
        logger.info(f"TemporalAttention initialized: hidden={self.config['hidden_dim']}, "
                   f"heads={self.config['num_heads']}, dropout={self.config['dropout']}, "
                   f"device={self.device}")

    def _create_sequences(self, df: pd.DataFrame, feature_cols: List[str], target_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Creates sliding window sequences for each player, including next game context (vectorized)."""
        df_sorted = df.sort_values(['PLAYER_ID', 'GAME_DATE'])
        features = df_sorted[feature_cols].values
        targets = df_sorted[target_cols].values
        player_ids = df_sorted['PLAYER_ID'].values

        player_boundaries = np.flatnonzero(np.diff(player_ids) != 0) + 1
        starts = np.concatenate([[0], player_boundaries])
        ends = np.concatenate([player_boundaries, [len(player_ids)]])
        group_lens = ends - starts

        valid_mask = group_lens >= (self.seq_len + 1)
        valid_starts = starts[valid_mask]
        valid_lens = group_lens[valid_mask]

        total_seqs = int(np.sum(valid_lens - self.seq_len))
        if total_seqs == 0:
            return np.array([]), np.array([]), np.array([])

        n_feat = features.shape[1]
        n_tgt = targets.shape[1]
        hist = np.empty((total_seqs, self.seq_len, n_feat), dtype=np.float32)
        ctx = np.empty((total_seqs, n_feat), dtype=np.float32)
        y = np.empty((total_seqs, n_tgt), dtype=np.float32)

        idx = 0
        for s, gl in zip(valid_starts, valid_lens):
            n_seq = gl - self.seq_len
            end = s + gl
            feat_window = np.lib.stride_tricks.sliding_window_view(
                features[s:end], (self.seq_len, n_feat)
            ).reshape(-1, self.seq_len, n_feat)[:n_seq]
            hist[idx:idx + n_seq] = feat_window
            ctx[idx:idx + n_seq] = features[s + self.seq_len:end]
            y[idx:idx + n_seq] = targets[s + self.seq_len:end]
            idx += n_seq

        return hist, ctx, y

    def fit(self, df: pd.DataFrame, feature_cols: List[str], target_cols: List[str], 
            epochs: Optional[int] = None):
        """Trains the TemporalAttentionModel with warmup + cosine decay."""
        epochs = epochs or self.config['epochs']
        batch_size = self.config['batch_size']
        lr = self.config['lr']
        warmup_ratio = self.config['warmup_ratio']
        
        logger.info(f"Generating sequences (len={self.seq_len}) for Temporal Attention...")
        hist, ctx, y = self._create_sequences(df, feature_cols, target_cols)
        
        if len(hist) == 0:
            logger.warning("Not enough data to create sequences for Temporal Attention.")
            return
        
        all_features = np.concatenate([hist.reshape(-1, hist.shape[-1]), ctx], axis=0)
        self.feat_mean = all_features.mean(axis=0)
        self.feat_std = all_features.std(axis=0) + 1e-6
        
        hist = (hist - self.feat_mean) / self.feat_std
        ctx = (ctx - self.feat_mean) / self.feat_std
        
        dataset = TemporalAttentionDataset(hist, ctx, y)
        num_workers = 4 if self.device.type == 'cuda' else 2
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                            num_workers=num_workers, persistent_workers=(num_workers > 0))
        
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        total_steps = epochs * len(loader)
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = WarmupScheduler(optimizer, warmup_steps, total_steps)
        criterion = nn.MSELoss()
        
        device_str = self.device.type
        grad_scaler = torch.amp.GradScaler(device_str, enabled=(device_str == 'cuda'))
        
        best_loss = float('inf')
        patience_counter = 0
        early_stop_patience = 10
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for b_hist, b_ctx, b_y in loader:
                b_hist = b_hist.to(self.device, non_blocking=True)
                b_ctx = b_ctx.to(self.device, non_blocking=True)
                b_y = b_y.to(self.device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_str, enabled=(device_str == 'cuda')):
                    preds, _ = self.model(b_hist, b_ctx)
                    loss = criterion(preds, b_y)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                scheduler.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"TemporalAttention Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")
            
            if patience_counter >= early_stop_patience:
                logger.info(f"TemporalAttention early stopping at epoch {epoch+1} (best loss: {best_loss:.4f})")
                break
        
        self.is_trained = True
        logger.info(f"TemporalAttention training complete. Final loss: {best_loss:.4f}")

    def predict(self, history: np.ndarray, next_context: np.ndarray) -> np.ndarray:
        """Predicts using history and upcoming game context."""
        if not self.is_trained:
            return None
        
        self.model.eval()
        history = (history - self.feat_mean) / self.feat_std
        next_context = (next_context - self.feat_mean) / self.feat_std
        
        hist_tensor = torch.tensor(history, dtype=torch.float32).unsqueeze(0).to(self.device)
        ctx_tensor = torch.tensor(next_context, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            preds, attn = self.model(hist_tensor, ctx_tensor)
            preds = preds.cpu().numpy()
        return preds

    def save(self, path: str):
        state = {
            'model_state': self.model.state_dict(),
            'feat_mean': self.feat_mean,
            'feat_std': self.feat_std,
            'input_dim': self.input_dim,
            'seq_len': self.seq_len,
            'config': self.config,
        }
        joblib.dump(state, path)
        logger.info(f"TemporalAttention model saved to {path}")

    @classmethod
    def load(cls, path: str):
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