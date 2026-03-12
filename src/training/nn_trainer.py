"""Unified neural network trainer for all PyTorch-based models.

This trainer provides a unified interface for training PyTorch models with:
- Automatic mixed precision (AMP) with BF16/FP16 auto-selection
- GPU optimizations (TF32, cuDNN benchmark)
- Gradient accumulation for large batch training
- Optimal DataLoader worker configuration
- torch.compile support with max-autotune
- Gradient checkpointing support
- Memory-efficient training
"""

import logging
import time
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.trainer import BaseTrainer, TrainResult
from src.models.gpu_utils import (
    get_device, 
    clear_gpu_memory, 
    get_gpu_memory_usage,
    get_optimal_dataloader_workers,
    initialize_gpu_optimizations,
    is_bf16_supported,
    get_autocast_dtype,
    create_grad_scaler,
    autocast_context,
    GPUMemoryContext,
)
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
    
    Supports LSTM, Transformer, GNN, and Joint NN models with:
    - Automatic mixed precision (AMP) with BF16/FP16 auto-selection
    - Gradient accumulation for large effective batch sizes
    - Optimal DataLoader worker configuration
    - torch.compile support with mode selection
    - Gradient checkpointing support
    - TF32 optimization on Ampere+ GPUs
    - Memory-efficient training
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
        compile_mode: str = 'reduce-overhead',
        gradient_accumulation_steps: int = 1,
    ):
        """Initialize NN trainer with performance optimizations.
        
        Args:
            model_name: Name identifier for this model
            config: Model configuration dictionary
            model_class: PyTorch model class to instantiate
            model_kwargs: Arguments to pass to model constructor
            use_gpu: Whether to use GPU acceleration
            device: Specific device to use (None for auto)
            random_state: Random seed for reproducibility
            use_amp: Whether to use automatic mixed precision
            use_compile: Whether to use torch.compile
            compile_mode: torch.compile mode ('reduce-overhead' or 'max-autotune')
            gradient_accumulation_steps: Steps to accumulate gradients (for large batches)
        """
        super().__init__(model_name, config, use_gpu, device, random_state)
        
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.use_amp = use_amp and use_gpu  # AMP only on GPU
        self.use_compile = use_compile
        self.compile_mode = compile_mode
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Determine optimal dtype for mixed precision
        self._amp_dtype = None
        self._grad_scaler = None
        
        if self.use_gpu:
            self._setup_amp()
        
        # Training state
        self.history: List[Dict[str, float]] = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self._optimal_workers = get_optimal_dataloader_workers()
    
    def _setup_amp(self):
        """Set up automatic mixed precision with optimal dtype."""
        # Use BF16 on Ampere+ GPUs, FP16 otherwise
        if is_bf16_supported():
            self._amp_dtype = torch.bfloat16
            # BF16 doesn't need gradient scaling
            self._grad_scaler = None
            logger.info(f"Using BF16 mixed precision for {self.model_name}")
        else:
            self._amp_dtype = torch.float16
            # FP16 needs gradient scaling
            self._grad_scaler = torch.amp.GradScaler('cuda', enabled=True)
            logger.info(f"Using FP16 mixed precision for {self.model_name}")
    
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
        """Build the PyTorch model with optional torch.compile.
        
        Uses the configured compile_mode for optimization level:
        - 'reduce-overhead': Good for small models, reduces Python overhead
        - 'max-autotune': Best performance but longer compile time
        """
        self.model = self.model_class(**self.model_kwargs).to(self.device)
        
        if self.use_compile and hasattr(torch, 'compile'):
            try:
                # Use the configured compile mode
                self.model = torch.compile(self.model, mode=self.compile_mode)
                logger.info(f"Compiled {self.model_name} with torch.compile(mode='{self.compile_mode}')")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")
        
        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Built {self.model_name}: {param_count:,} params")
    
    def _create_loader(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        """Create DataLoader with optimal configuration.
        
        Uses:
        - Optimal number of workers (8 for GPU, CPU count for CPU)
        - Pinned memory for faster GPU transfers
        - Non-blocking data transfer
        - Persistent workers to reduce overhead
        """
        dataset = TensorDataset(X, y)
        
        # Use optimal workers determined at initialization
        num_workers = self._optimal_workers
        
        # Pin memory for faster GPU transfers (only on CUDA)
        pin_memory = self.use_gpu and self.device.type == 'cuda'
        
        # Persistent workers reduce overhead between epochs
        persistent_workers = num_workers > 0
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if num_workers > 0 else False,
        )
    
    def _train_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
        """Train for one epoch with gradient accumulation support.
        
        Implements:
        - Proper AMP with BF16/FP16 auto-selection
        - Gradient accumulation for large effective batch sizes
        - Non-blocking tensor transfers
        """
        self.model.train()
        total_loss = 0.0
        accumulation_steps = self.gradient_accumulation_steps
        
        # Clear gradients at the start
        optimizer.zero_grad()
        
        for batch_idx, (X_batch, y_batch) in enumerate(loader):
            # Move to device (non-blocking for GPU)
            if self.use_gpu:
                X_batch = X_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)
            
            # Forward pass with proper AMP context
            if self.use_amp and self._amp_dtype is not None:
                with torch.amp.autocast(device_type='cuda', dtype=self._amp_dtype):
                    y_pred = self.model(X_batch)
                    loss = nn.MSELoss()(y_pred, y_batch)
                    # Scale loss for gradient accumulation
                    loss = loss / accumulation_steps
                
                # Backward pass with gradient scaling for FP16
                if self._grad_scaler is not None:
                    self._grad_scaler.scale(loss).backward()
                else:
                    # BF16 doesn't need scaling
                    loss.backward()
            else:
                y_pred = self.model(X_batch)
                loss = nn.MSELoss()(y_pred, y_batch)
                loss = loss / accumulation_steps
                loss.backward()
            
            # Accumulate gradients and update weights
            if (batch_idx + 1) % accumulation_steps == 0:
                if self._grad_scaler is not None:
                    self._grad_scaler.step(optimizer)
                    self._grad_scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps  # Unscale for logging
        
        # Handle any remaining gradients
        if (batch_idx + 1) % accumulation_steps != 0:
            if self._grad_scaler is not None:
                self._grad_scaler.step(optimizer)
                self._grad_scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        return total_loss / len(loader)
    
    def _validate_epoch(self, loader: DataLoader) -> float:
        """Validate for one epoch with proper AMP context."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in loader:
                # Move to device (non-blocking for GPU)
                if self.use_gpu:
                    X_batch = X_batch.to(self.device, non_blocking=True)
                    y_batch = y_batch.to(self.device, non_blocking=True)
                
                if self.use_amp and self._amp_dtype is not None:
                    with torch.amp.autocast(device_type='cuda', dtype=self._amp_dtype):
                        y_pred = self.model(X_batch)
                        loss = nn.MSELoss()(y_pred, y_batch)
                else:
                    y_pred = self.model(X_batch)
                    loss = nn.MSELoss()(y_pred, y_batch)
                
                total_loss += loss.item()
        
        return total_loss / len(loader) if len(loader) > 0 else 0.0
    
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