from dataclasses import dataclass
from typing import List, Dict
import random
import os
import numpy as np

@dataclass
class ModelConfig:
    # Feature engineering
    rolling_windows: List[int] = (3, 5, 10, 20, 50)
    min_games_for_player: int = 5
    
    # GNN
    gnn_hidden_dim: int = 128
    gnn_epochs: int = 100
    
    # MultiOutputNN
    nn_hidden_dim: int = 768
    nn_num_blocks: int = 8
    nn_dropout: float = 0.3
    
    # Ensemble
    use_gpu: bool = True
    ensemble_n_estimators: int = 400
    
    # Training
    random_seed: int = 42
    early_stopping_patience: int = 20

def set_global_seed(seed: int = 42):
    """Ensure reproducible results across runs."""
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable for reproducibility
