import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Tuple, Optional, Any
import math

logger = logging.getLogger(__name__)


class GraphConv(nn.Module):
    """Graph Convolution layer for player interdependencies."""
    def __init__(self, in_features, out_features, dropout: float = 0.0):
        super(GraphConv, self).__init__()
        self.projection = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        support = torch.mm(adj, x)
        return F.relu(self.projection(self.dropout(support)))


class GraphAttentionConv(nn.Module):
    """Graph Attention layer for learning player relationships."""
    def __init__(self, in_features, out_features, dropout: float = 0.0):
        super(GraphAttentionConv, self).__init__()
        self.projection = nn.Linear(in_features, out_features)
        self.attention = nn.Linear(out_features * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, adj):
        h = self.projection(x)
        n = h.size(0)
        
        a_input = torch.cat([h.repeat(1, n).view(n * n, -1), h.repeat(n, 1)], dim=1)
        a_input = a_input.view(n, n, -1)
        e = self.leaky_relu(self.attention(a_input)).squeeze(-1)
        
        attention = F.softmax(e.masked_fill(adj == 0, -1e9), dim=1)
        attention = self.dropout(attention)
        
        out = torch.mm(attention, h)
        return F.elu(out)


class TeamChemistryGNN(nn.Module):
    """GNN to predict synergy-adjusted player ratings."""
    def __init__(self, input_dim, hidden_dim=64, output_dim=3, num_layers=2, 
                 dropout=0.2, use_attention=False):
        super(TeamChemistryGNN, self).__init__()
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        ConvLayer = GraphAttentionConv if use_attention else GraphConv
        
        self.convs = nn.ModuleList()
        self.convs.append(ConvLayer(input_dim, hidden_dim, dropout))
        for _ in range(num_layers - 1):
            self.convs.append(ConvLayer(hidden_dim, hidden_dim, dropout))
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        for conv in self.convs:
            x = conv(x, adj)
        x = self.dropout(x)
        return self.fc(x)


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


class GNNWrapper:
    """Wrapper for handling player graph construction and GNN inference."""
    
    DEFAULT_CONFIG = {
        'hidden_dim': 64,
        'num_layers': 2,
        'dropout': 0.2,
        'batch_size': 64,
        'epochs': 50,
        'lr': 1e-2,
        'warmup_ratio': 0.1,
        'use_attention': False,
        'top_players': 1000,
    }
    
    def __init__(self, input_dim: int, target_names: List[str] = None, 
                 config: Optional[Dict[str, Any]] = None):
        if target_names is None:
            target_names = ['PTS', 'REB', 'AST']
            
        from src.models.gpu_utils import get_device
        
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.target_names = target_names
        self.device = get_device()
        self.input_dim = input_dim
        
        self.model = TeamChemistryGNN(
            input_dim=input_dim,
            hidden_dim=self.config['hidden_dim'],
            output_dim=len(target_names),
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout'],
            use_attention=self.config['use_attention']
        ).to(self.device)
        
        self.is_trained = False
        self.player_map = {}
        self.adj = None
        self.feat_mean = None
        self.feat_std = None
        self.trained_embeddings = None
        
        logger.info(f"GNN initialized: hidden={self.config['hidden_dim']}, "
                   f"layers={self.config['num_layers']}, attention={self.config['use_attention']}, "
                   f"device={self.device}")

    def build_graph(self, df: pd.DataFrame):
        """Builds a global adjacency matrix from historical co-occurrence (vectorized)."""
        top_n = self.config['top_players']
        logger.info(f"Constructing player co-occurrence graph (top {top_n} players)...")
        top_players = df['PLAYER_ID'].value_counts().head(top_n).index.tolist()
        self.player_map = {pid: i for i, pid in enumerate(top_players)}
        n = len(top_players)
        
        adj = np.eye(n, dtype=np.float32)
        
        top_set = set(top_players)
        filtered = df[df['PLAYER_ID'].isin(top_set)][['GAME_ID', 'PLAYER_ID']].copy()
        filtered['NODE_IDX'] = filtered['PLAYER_ID'].map(self.player_map)
        
        for _, game_group in filtered.groupby('GAME_ID'):
            node_ids = game_group['NODE_IDX'].values
            if len(node_ids) < 2:
                continue
            ii, jj = np.triu_indices(len(node_ids), k=1)
            rows = node_ids[ii]
            cols = node_ids[jj]
            np.add.at(adj, (rows, cols), 1)
            np.add.at(adj, (cols, rows), 1)
        
        rowsum = adj.sum(1)
        d_inv = np.where(rowsum > 0, 1.0 / rowsum, 0.0)
        self.adj = torch.tensor(d_inv[:, None] * adj, dtype=torch.float32).to(self.device)
        logger.info(f"Graph built with {n} nodes.")

    def fit(self, df: pd.DataFrame, feature_cols: List[str], target_cols: List[str], 
            epochs: Optional[int] = None):
        """Train GNN with warmup + cosine decay scheduler."""
        epochs = epochs or self.config['epochs']
        lr = self.config['lr']
        warmup_ratio = self.config['warmup_ratio']
        
        if self.adj is None:
            self.build_graph(df)
        
        valid_pids = sorted(self.player_map.keys())
        player_data = df[df['PLAYER_ID'].isin(valid_pids)]
        grouped = player_data.groupby('PLAYER_ID')
        X_means = grouped[feature_cols].mean()
        Y_means = grouped[target_cols].mean()
        X = X_means.reindex(valid_pids).fillna(0).values
        Y = Y_means.reindex(valid_pids).fillna(0).values
        
        self.feat_mean = X.mean(axis=0)
        self.feat_std = X.std(axis=0) + 1e-6
        X_norm = (X - self.feat_mean) / self.feat_std
        
        X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(self.device)
        Y_tensor = torch.tensor(Y, dtype=torch.float32).to(self.device)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        total_steps = epochs * 1  # Single batch per epoch
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = WarmupScheduler(optimizer, warmup_steps, total_steps)
        criterion = nn.MSELoss()
        
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.model(X_tensor, self.adj)
            loss = criterion(output, Y_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            if (epoch + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"GNN Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, LR: {current_lr:.2e}")
        
        with torch.no_grad():
            self.trained_embeddings = self.model(X_tensor, self.adj).cpu().numpy()
        self.is_trained = True
        logger.info(f"GNN training complete. Final loss: {loss.item():.4f}")

    def predict(self, player_context_df: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            return np.zeros((len(player_context_df), len(self.target_names)))
        
        preds = np.zeros((len(player_context_df), len(self.target_names)))
        for i, pid in enumerate(player_context_df['PLAYER_ID']):
            if pid in self.player_map:
                idx = self.player_map[pid]
                preds[i] = self.trained_embeddings[idx]
            else:
                preds[i] = 0.0
        return preds

    def save(self, path: str):
        import joblib
        state = {
            'model_state': self.model.state_dict(),
            'player_map': self.player_map,
            'feat_mean': self.feat_mean,
            'feat_std': self.feat_std,
            'trained_embeddings': self.trained_embeddings,
            'adj': self.adj.cpu().numpy() if self.adj is not None else None,
            'target_names': self.target_names,
            'input_dim': self.input_dim,
            'config': self.config,
        }
        joblib.dump(state, path)
        logger.info(f"GNN model saved to {path}")

    @classmethod
    def load(cls, path: str):
        import joblib
        state = joblib.load(path)
        config = state.get('config', {})
        input_dim = state.get('input_dim')
        if input_dim is None:
            try:
                input_dim = state['model_state']['conv1.projection.weight'].shape[1]
            except KeyError:
                input_dim = 64  # safe default
        
        instance = cls(input_dim, target_names=state['target_names'], config=config)
        instance.model.load_state_dict(state['model_state'])
        instance.player_map = state['player_map']
        instance.feat_mean = state['feat_mean']
        instance.feat_std = state['feat_std']
        instance.trained_embeddings = state['trained_embeddings']
        if state['adj'] is not None:
            instance.adj = torch.tensor(state['adj'], dtype=torch.float32).to(instance.device)
        instance.is_trained = True
        return instance