"""Unified neural network trainer for all PyTorch-based models."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.trainer import BaseTrainer, TrainResult
from src.models.gpu_utils import get_device, clear_gpu_memory, get_gpu_memory_usage
from src.training.training_logger import get_training_logger

logger = logging.getLogger(__name__)

# Rich imports for progress bars
try:
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
    from rich.console import Console
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class NeuralNetworkTrainer(BaseTrainer):
    """Unified trainer for PyTorch neural networks.
    
    Supports LSTM, Transformer, GNN, and Joint NN models with
    automatic mixed precision, gradient checkpointing, and early stopping.
    """
    
    def __init__(
        self,
        model_name: str,
        config: Dict[str, Any],
        model_class: type,
        model_kwargs: Dict[str, Any],
        use_gpu: bool = False,
        device: Optional[str] = None,
        random_state: int = 42,
        use_amp: bool = True,
        use_compile: bool = False,
    ):
        """Initialize NN trainer."""
        super().__init__(model_name, config, use_gpu, device, random_state)
        
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.use_amp = use_amp and use_gpu  # AMP only on GPU
        self.use_compile = use_compile
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Training state
        self.history: List[Dict[str, float]] = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs
    ) -> TrainResult:
        """Train the neural network with detailed progress tracking."""
        start_time = time.time()
        training_logger = get_training_logger()
        
        # Prepare data
        X_train_t, y_train_t = self._to_tensor(X_train, y_train)
        X_val_t, y_val_t = None, None
        if X_val is not None and y_val is not None:
            X_val_t, y_val_t = self._to_tensor(X_val, y_val)
        
        # Build model
        self._build_model(X_train_t.shape)
        
        # Training config
        epochs = self.config.get('epochs', 100)
        batch_size = self.config.get('batch_size', 64)
        lr = self.config.get('lr', 1e-3)
        patience = self.config.get('early_stop_patience', 15)
        
        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Data loaders
        train_loader = self._create_loader(X_train_t, y_train_t, batch_size, shuffle=True)
        val_loader = self._create_loader(X_val_t, y_val_t, batch_size, shuffle=False) if X_val_t is not None else None
        
        # Training loop with progress bar
        best_state = None
        
        if RICH_AVAILABLE and training_logger.use_rich:
            # Use Rich progress bar
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(complete_style="green", finished_style="green"),
                MofNCompleteColumn(),
                TextColumn("[yellow]loss: {task.fields[train_loss]:.4f}"),
                TextColumn("[cyan]val: {task.fields[val_loss]:.4f}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=training_logger.console if training_logger.console else None,
            ) as progress:
                task = progress.add_task(
                    f"Training {self.model_name}",
                    total=epochs,
                    train_loss=0.0,
                    val_loss=0.0,
                )
                
                for epoch in range(epochs):
                    train_loss = self._train_epoch(train_loader, optimizer)
                    val_loss = self._validate_epoch(val_loader) if val_loader else train_loss
                    
                    self.history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
                    scheduler.step(val_loss)
                    
                    # Get current learning rate
                    current_lr = optimizer.param_groups[0]['lr']
                    
                    # Get GPU memory if available
                    gpu_mem = 0.0
                    if self.use_gpu:
                        try:
                            allocated, _ = get_gpu_memory_usage()
                            gpu_mem = allocated
                        except:
                            pass
                    
                    # Update progress bar
                    progress.update(
                        task,
                        advance=1,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        description=f"Epoch {epoch+1}/{epochs} | LR={current_lr:.6f} | GPU={gpu_mem:.1f}GB"
                    )
                    
                    # Early stopping
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.patience_counter = 0
                        best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    else:
                        self.patience_counter += 1
                        if self.patience_counter >= patience:
                            progress.console.print(f"[yellow]Early stopping at epoch {epoch}[/yellow]")
                            break
        else:
            # Fallback to simple logging without Rich
            for epoch in range(epochs):
                train_loss = self._train_epoch(train_loader, optimizer)
                val_loss = self._validate_epoch(val_loader) if val_loader else train_loss
                
                self.history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        self.is_trained = True
        training_time = time.time() - start_time
        
        # Compute metrics
        metrics = {}
        if X_val is not None and y_val is not None:
            y_pred = self.predict(X_val)
            metrics = self.compute_metrics(y_val_t.cpu().numpy(), y_pred)
        
        clear_gpu_memory()
        
        return TrainResult(model=self, metrics=metrics, training_time=training_time)
    
    def _to_tensor(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert data to tensors."""
        X_clean, y_clean = self.validate_data(X, y)
        X_t = torch.FloatTensor(X_clean).to(self.device)
        y_t = torch.FloatTensor(y_clean).to(self.device)
        if y_t.ndim == 1:
            y_t = y_t.unsqueeze(1)
        return X_t, y_t
    
    def _build_model(self, input_shape: Tuple[int, ...]) -> None:
        """Build the PyTorch model."""
        self.model = self.model_class(**self.model_kwargs).to(self.device)
        
        if self.use_compile and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                logger.info(f"Compiled {self.model_name} with torch.compile")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")
        
        logger.info(f"Built {self.model_name}: {sum(p.numel() for p in self.model.parameters())} params")
    
    def _create_loader(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        """Create DataLoader."""
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)
    
    def _train_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    y_pred = self.model(X_batch)
                    loss = nn.MSELoss()(y_pred, y_batch)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                y_pred = self.model(X_batch)
                loss = nn.MSELoss()(y_pred, y_batch)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def _validate_epoch(self, loader: DataLoader) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in loader:
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        y_pred = self.model(X_batch)
                        loss = nn.MSELoss()(y_pred, y_batch)
                else:
                    y_pred = self.model(X_batch)
                    loss = nn.MSELoss()(y_pred, y_batch)
                
                total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray], **kwargs) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise RuntimeError("Model not trained")
        
        self.model.eval()
        X_clean, _ = self.validate_data(X, None)
        X_t = torch.FloatTensor(X_clean).to(self.device)
        
        with torch.no_grad():
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    y_pred = self.model(X_t)
            else:
                y_pred = self.model(X_t)
        
        return y_pred.cpu().numpy().squeeze()
    
    def save(self, path: Union[str, Path]) -> None:
        """Save model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'model_state': self.model.state_dict() if self.model else None,
            'config': self.config,
            'model_kwargs': self.model_kwargs,
            'history': self.history,
        }
        joblib.dump(state, path)
        logger.info(f"Saved {self.model_name} to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> 'NeuralNetworkTrainer':
        """Load model."""
        path = Path(path)
        state = joblib.load(path)
        
        # Reconstruct trainer
        trainer = cls(
            model_name=kwargs.get('model_name', 'nn_model'),
            config=state['config'],
            model_class=kwargs['model_class'],
            model_kwargs=state['model_kwargs'],
            use_gpu=kwargs.get('use_gpu', False),
        )
        
        # Rebuild and load model
        if trainer.model is None:
            trainer._build_model(state['model_kwargs'].get('input_dim', 1))
        
        if state['model_state'] is not None:
            trainer.model.load_state_dict(state['model_state'])
            trainer.is_trained = True
        
        trainer.history = state.get('history', [])
        
        return trainer