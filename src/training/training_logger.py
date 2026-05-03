"""Rich training logger with detailed GPU monitoring and progress tracking."""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np

# Rich imports for beautiful console output
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.style import Style
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    logging.warning("rich library not available. Install with: pip install rich")

# GPU utilities
try:
    from src.models.gpu_utils import get_gpu_memory_usage, check_gpu_compatibility
    GPU_UTILS_AVAILABLE = True
except ImportError:
    GPU_UTILS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    target: str
    model_type: str
    iteration: int
    total_iterations: int
    train_loss: float = 0.0
    val_loss: float = 0.0
    train_mae: float = 0.0
    val_mae: float = 0.0
    learning_rate: float = 0.0
    best_iteration: int = 0
    best_val_loss: float = float('inf')
    gpu_memory_gb: float = 0.0
    time_per_iter: float = 0.0
    eta_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'target': self.target,
            'model_type': self.model_type,
            'iteration': self.iteration,
            'total_iterations': self.total_iterations,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'train_mae': self.train_mae,
            'val_mae': self.val_mae,
            'learning_rate': self.learning_rate,
            'best_iteration': self.best_iteration,
            'best_val_loss': self.best_val_loss,
            'gpu_memory_gb': self.gpu_memory_gb,
            'time_per_iter': self.time_per_iter,
            'eta_seconds': self.eta_seconds,
        }


class RichTrainingLogger:
    """Rich console logger for detailed training progress."""
    
    def __init__(self, use_rich: bool = True, log_gpu: bool = True):
        """Initialize the logger.
        
        Args:
            use_rich: Whether to use rich console output
            log_gpu: Whether to log GPU memory usage
        """
        self.use_rich = use_rich and RICH_AVAILABLE
        self.log_gpu = log_gpu and GPU_UTILS_AVAILABLE
        self.console = Console() if self.use_rich else None
        self._start_times: Dict[str, float] = {}
        self._iter_times: List[float] = []
        self._current_metrics: Optional[TrainingMetrics] = None
        
    def log_header(self, title: str, subtitle: str = "") -> None:
        """Log a header section."""
        if self.use_rich:
            self.console.print()
            self.console.print(Panel(
                f"[bold cyan]{title}[/bold cyan]\n[dim]{subtitle}[/dim]" if subtitle else f"[bold cyan]{title}[/bold cyan]",
                border_style="cyan"
            ))
        else:
            print(f"\n{'='*70}")
            print(f"{title}")
            if subtitle:
                print(f"{subtitle}")
            print(f"{'='*70}")
    
    def log_hardware_info(self, hw_info: Dict[str, Any]) -> None:
        """Log hardware detection information."""
        if self.use_rich:
            table = Table(title="Hardware Detected", border_style="green")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="white")
            
            table.add_row("Type", hw_info.get('type', 'unknown').upper())
            table.add_row("Name", str(hw_info.get('name', 'unknown')))
            table.add_row("Compute Score", f"{hw_info.get('score', 0):.1f}")
            
            if hw_info.get('vram', 0) > 0:
                table.add_row("VRAM", f"{hw_info['vram']:.1f} GB")
            if hw_info.get('cores', 0) > 0:
                table.add_row("CPU Cores", str(hw_info['cores']))
            if hw_info.get('ram', 0) > 0:
                table.add_row("System RAM", f"{hw_info['ram']:.1f} GB")
            
            self.console.print(table)
        else:
            print(f"\nHardware: {hw_info.get('name', 'unknown')}")
            print(f"  Score: {hw_info.get('score', 0):.1f}")
            if hw_info.get('vram', 0) > 0:
                print(f"  VRAM: {hw_info['vram']:.1f} GB")
    
    def start_target_training(self, target: str, model_type: str, total_iterations: int) -> None:
        """Start tracking training for a target."""
        key = f"{target}_{model_type}"
        self._start_times[key] = time.time()
        self._iter_times = []
        
        if self.use_rich:
            self.console.print(f"\n[bold yellow]Training {model_type} for {target}[/bold yellow]")
    
    def log_iteration(self, metrics: TrainingMetrics) -> None:
        """Log a training iteration with rich display."""
        self._current_metrics = metrics
        
        # Track iteration times for ETA
        key = f"{metrics.target}_{metrics.model_type}"
        if key in self._start_times:
            elapsed = time.time() - self._start_times[key]
            iters_done = metrics.iteration
            if iters_done > 0:
                time_per_iter = elapsed / iters_done
                self._iter_times.append(time_per_iter)
                metrics.time_per_iter = time_per_iter
                
                # Calculate ETA
                remaining_iters = metrics.total_iterations - metrics.iteration
                metrics.eta_seconds = remaining_iters * time_per_iter
        
        # Get GPU memory if enabled
        if self.log_gpu:
            try:
                allocated, _ = get_gpu_memory_usage()
                metrics.gpu_memory_gb = allocated
            except Exception:
                pass
        
        # Rich display
        if self.use_rich and metrics.iteration % 50 == 0:  # Update every 50 iterations
            self._display_progress(metrics)
    
    def _display_progress(self, metrics: TrainingMetrics) -> None:
        """Display rich progress information."""
        progress_pct = metrics.iteration / metrics.total_iterations
        bar_width = 40
        filled = int(bar_width * progress_pct)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        eta_str = f"{metrics.eta_seconds:.0f}s" if metrics.eta_seconds > 0 else "calculating..."
        
        lines = [
            f"  {bar} {progress_pct*100:.1f}% | Iter {metrics.iteration}/{metrics.total_iterations}",
            f"  Train: RMSE={metrics.train_loss:.3f} | MAE={metrics.train_mae:.3f}",
            f"  Val:   RMSE={metrics.val_loss:.3f} | MAE={metrics.val_mae:.3f} | Best={metrics.best_val_loss:.3f}@{metrics.best_iteration}",
            f"  LR: {metrics.learning_rate:.6f} | ETA: {eta_str}",
        ]
        
        if metrics.gpu_memory_gb > 0:
            lines.append(f"  GPU Memory: {metrics.gpu_memory_gb:.2f}GB")
        
        self.console.print("\n".join(lines))
    
    def end_target_training(self, target: str, model_type: str, metrics: Dict[str, float], training_time: float) -> None:
        """Log completion of target training."""
        if self.use_rich:
            mae = metrics.get('mae', 0)
            rmse = metrics.get('rmse', 0)
            self.console.print(f"  [green]✓[/green] {target} complete: MAE={mae:.3f}, RMSE={rmse:.3f}, Time={training_time:.1f}s")
        else:
            print(f"  {target} complete: {metrics}, Time={training_time:.1f}s")
    
    def log_final_results(self, results: Dict[str, Dict[str, Any]], total_time: float) -> None:
        """Log final results table."""
        if not self.use_rich:
            print(f"\nTraining complete in {total_time:.1f}s")
            return
        
        self.console.print()
        table = Table(title="Final Training Results", border_style="green")
        table.add_column("Target", style="cyan")
        table.add_column("MAE", justify="right")
        table.add_column("RMSE", justify="right")
        table.add_column("Time", justify="right")
        table.add_column("Best Iter", justify="right")
        
        for target, data in results.items():
            metrics = data.get('metrics', {})
            table.add_row(
                target,
                f"{metrics.get('mae', 0):.3f}",
                f"{metrics.get('rmse', 0):.3f}",
                f"{data.get('training_time', 0):.1f}s",
                str(data.get('best_iteration', 'N/A'))
            )
        
        self.console.print(table)
        self.console.print(f"\n[bold green]Total Training Time: {total_time:.1f}s[/bold green]")


# Global logger instance
_training_logger: Optional[RichTrainingLogger] = None


def get_training_logger(use_rich: bool = True, log_gpu: bool = True) -> RichTrainingLogger:
    """Get or create the global training logger."""
    global _training_logger
    if _training_logger is None:
        _training_logger = RichTrainingLogger(use_rich=use_rich, log_gpu=log_gpu)
    return _training_logger


def reset_training_logger() -> None:
    """Reset the global training logger."""
    global _training_logger
    _training_logger = None