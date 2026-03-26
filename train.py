#!/usr/bin/env python3
"""New NBA Model Training Script - Efficient Training Pipeline.

This script provides an efficient, modular training pipeline with:
- Parallel training across targets
- Smart caching of features
- Experiment tracking
- Multiple training modes (quick/standard/full)
- Detailed GPU-accelerated training logs
- TF32/BF16 optimizations for Ampere+ GPUs
- torch.compile support for PyTorch 2.0+
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Add local directory to path so standalone uploaded files can import each other.
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.training.pipeline import TrainingPipeline, create_pipeline
    from src.preprocessing.data_loader import DataLoader
    from src.preprocessing.feature_engineer import FeatureEngineer
    from src.training.training_logger import get_training_logger, RichTrainingLogger
    from src.models.gpu_utils import (
        check_gpu_compatibility,
        get_gpu_memory_usage,
        initialize_gpu_optimizations,
        print_gpu_summary,
    )
    from src.utils.logging_config import setup_logging
except ModuleNotFoundError:
    from pipeline import TrainingPipeline, create_pipeline
    from data_loader import DataLoader
    from feature_engineer import FeatureEngineer
    from gpu_utils import (
        check_gpu_compatibility,
        get_gpu_memory_usage,
        initialize_gpu_optimizations,
        print_gpu_summary,
    )

    class RichTrainingLogger:
        def __init__(self, use_rich: bool = False, log_gpu: bool = False):
            self.use_rich = use_rich
            self.log_gpu = log_gpu

    _TRAINING_LOGGER = None
    def get_training_logger(use_rich: bool = False, log_gpu: bool = False):
        global _TRAINING_LOGGER
        if _TRAINING_LOGGER is None:
            _TRAINING_LOGGER = RichTrainingLogger(use_rich=use_rich, log_gpu=log_gpu)
        return _TRAINING_LOGGER

    def setup_logging():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

setup_logging()

# Get logger after setup
logger = logging.getLogger(__name__)

# Rich imports for beautiful output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_runtime_path(path_value: str, default_name: str) -> Path:
    """Resolve default relative paths against the project root.

    This keeps notebook and `python /abs/path/train.py` execution aligned with
    the repository layout while still honoring explicit custom paths.
    """
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    if path_value == default_name:
        return PROJECT_ROOT / path
    return path


def print_banner(console=None):
    """Print the training banner with rich formatting."""
    banner_text = """
╭──────────────────────────────────────────────────────────────────────────────╮
│                    🏀 NBA PREDICTION MODEL TRAINER v2.0                       │
╰──────────────────────────────────────────────────────────────────────────────╯
    """
    if console and RICH_AVAILABLE:
        console.print(banner_text)
    else:
        print("\n" + "=" * 70)
        print("NBA PREDICTION MODEL TRAINER - v2.0")
        print("=" * 70)


def print_hardware_info(hw_info, console=None):
    """Print hardware detection information."""
    if console and RICH_AVAILABLE:
        from rich.table import Table
        from rich.panel import Panel
        
        table = Table(show_header=False, box=None)
        table.add_column("Property", style="cyan", width=20)
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
        
        console.print(Panel(table, title="[bold green]Hardware Detected[/bold green]", border_style="green"))
    else:
        print(f"\nHardware: {hw_info.get('name', 'unknown')}")
        print(f"  Score: {hw_info.get('score', 0):.1f}")
        if hw_info.get('vram', 0) > 0:
            print(f"  VRAM: {hw_info['vram']:.1f} GB")


def print_data_info(merged_df, full_df, fit_df, val_df, test_df, console=None):
    """Print data loading information."""
    if console and RICH_AVAILABLE:
        from rich.table import Table
        from rich.panel import Panel
        
        table = Table(show_header=False, box=None)
        table.add_column("Property", style="cyan", width=25)
        table.add_column("Value", style="white")
        
        table.add_row("Player-game records", f"{len(merged_df):,}")
        table.add_row("Total features", f"{len(full_df.columns):,}")
        table.add_row("Training samples", f"{len(fit_df):,} ({len(fit_df)/len(full_df)*100:.1f}%)")
        table.add_row("Validation samples", f"{len(val_df):,} ({len(val_df)/len(full_df)*100:.1f}%)")
        table.add_row("Test samples", f"{len(test_df):,} ({len(test_df)/len(full_df)*100:.1f}%)")
        
        console.print(Panel(table, title="[bold blue]Data Loaded[/bold blue]", border_style="blue"))
    else:
        print(f"\nData loaded:")
        print(f"  Player-game records: {len(merged_df):,}")
        print(f"  Features: {len(full_df.columns):,}")
        print(f"  Train: {len(fit_df):,} ({len(fit_df)/len(full_df)*100:.1f}%)")
        print(f"  Val: {len(val_df):,} ({len(val_df)/len(full_df)*100:.1f}%)")
        print(f"  Test: {len(test_df):,} ({len(test_df)/len(full_df)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Train NBA prediction models with efficient pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Training Modes:
  quick    - Fast training for development/testing (500 iters, no NN models)
  standard - Full training with all optimizations (3000 iters, all models)
  full     - Extended training for max accuracy (5000 iters, all models)

Examples:
  python train.py --mode quick                    # Quick test run
  python train.py --mode standard --parallel      # Full parallel training
  python train.py --mode full --model-size large  # Extended large model training
        """
    )
    
    parser.add_argument(
        '--data-dir', type=str, default='data',
        help='Directory containing NBA data files (default: data)'
    )
    parser.add_argument(
        '--models-dir', type=str, default='models',
        help='Directory to save trained models (default: models)'
    )
    parser.add_argument(
        '--mode', type=str, default='standard',
        choices=['quick', 'standard', 'full'],
        help='Training mode (default: standard)'
    )
    parser.add_argument(
        '--model-size', type=str, default='auto',
        choices=['auto', 'small', 'medium', 'large', 'pro', 'ultra'],
        help='Model size preset (default: auto-detect)'
    )
    parser.add_argument(
        '--parallel', action='store_true',
        help='Enable parallel training across targets'
    )
    parser.add_argument(
        '--max-workers', type=int, default=None,
        help='Maximum parallel workers (default: auto)'
    )
    parser.add_argument(
        '--no-gpu', action='store_true',
        help='Disable GPU even if available'
    )
    parser.add_argument(
        '--experiment-name', type=str, default=None,
        help='Name for experiment tracking'
    )
    parser.add_argument(
        '--cache-dir', type=str, default='cache/training',
        help='Directory for feature caching'
    )
    
    args = parser.parse_args()

    data_dir = resolve_runtime_path(args.data_dir, 'data')
    models_dir = resolve_runtime_path(args.models_dir, 'models')
    cache_dir = resolve_runtime_path(args.cache_dir, 'cache/training')
    
    # Initialize rich console and training logger
    console = Console() if RICH_AVAILABLE else None
    training_logger = get_training_logger(use_rich=RICH_AVAILABLE, log_gpu=True)
    
    # Print banner
    print_banner(console)
    
    # Check GPU availability
    gpu_available = check_gpu_compatibility() if not args.no_gpu else False
    if gpu_available:
        try:
            import torch
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU detected: {gpu_name} ({vram:.1f}GB VRAM)")
        except:
            pass
    elif not args.no_gpu:
        logger.info("CUDA is unavailable or incompatible. Training will run on CPU.")
        if console and RICH_AVAILABLE:
            console.print("[yellow]GPU unavailable or incompatible; falling back to CPU training.[/yellow]")
    
    # Check for data files
    players_file = data_dir / 'nba_players.csv'
    games_file = data_dir / 'nba_games.csv'
    
    if not players_file.exists() or not games_file.exists():
        if console and RICH_AVAILABLE:
            console.print("[bold red]ERROR: Data files not found![/bold red]")
            console.print(f"Please run 'python update_data.py' first to fetch NBA data.")
        else:
            print(f"\nERROR: Data files not found in {data_dir}")
            print("Please run 'python update_data.py' first to fetch NBA data.")
        sys.exit(1)
    
    try:
        # Load data
        if console and RICH_AVAILABLE:
            console.print("\n[bold cyan]Step 1: Loading and merging datasets...[/bold cyan]")
        else:
            print("Step 1: Loading and merging datasets...")
        
        loader = DataLoader(str(players_file), str(games_file))
        merged_df = loader.merge_datasets()
        
        if console and RICH_AVAILABLE:
            console.print(f"  [green]✓[/green] Loaded {len(merged_df):,} player-game records")
        else:
            print(f"  Loaded {len(merged_df)} player-game records")
        
        # Engineer features
        if console and RICH_AVAILABLE:
            console.print("\n[bold cyan]Step 2: Engineering features...[/bold cyan]")
        else:
            print("\nStep 2: Engineering features...")
        
        feature_engineer = FeatureEngineer()
        full_df = feature_engineer.create_features(merged_df)
        
        if console and RICH_AVAILABLE:
            console.print(f"  [green]✓[/green] Created {len(full_df.columns):,} features")
        else:
            print(f"  Created {len(full_df.columns)} features")
        
        # Create and run pipeline
        if console and RICH_AVAILABLE:
            console.print(f"\n[bold cyan]Step 3: Initializing training pipeline ({args.mode} mode)...[/bold cyan]")
        else:
            print(f"\nStep 3: Initializing training pipeline ({args.mode} mode)...")
        
        pipeline = create_pipeline(
            mode=args.mode,
            data_dir=data_dir,
            models_dir=models_dir,
            cache_dir=cache_dir,
            model_size=args.model_size,
            parallel=args.parallel,
            max_workers=args.max_workers,
            use_gpu=gpu_available,
            experiment_name=args.experiment_name,
        )
        
        # Print hardware info
        print_hardware_info(pipeline.hw_info, console)
        
        # Prepare data
        if console and RICH_AVAILABLE:
            console.print("\n[bold cyan]Step 4: Preparing data splits...[/bold cyan]")
        else:
            print("\nStep 4: Preparing data splits...")
        
        fit_df, val_df, test_df = pipeline.prepare_data(full_df)
        
        # Print data info
        print_data_info(merged_df, full_df, fit_df, val_df, test_df, console)
        
        # Train models
        if console and RICH_AVAILABLE:
            console.print(f"\n[bold cyan]Step 5: Training models (this may take a while)...[/bold cyan]")
        else:
            print(f"\nStep 5: Training models (this may take a while)...")
        
        results = pipeline.train(fit_df, val_df)
        
        # Print summary
        if console and RICH_AVAILABLE:
            console.print()
            console.print(Panel("[bold green]TRAINING COMPLETE[/bold green]", border_style="green"))
            
            summary = pipeline.get_summary()
            
            # Summary table
            summary_table = Table(show_header=False, box=None)
            summary_table.add_column("Property", style="cyan", width=20)
            summary_table.add_column("Value", style="white")
            
            summary_table.add_row("Experiment", summary['experiment_name'])
            summary_table.add_row("Models trained", str(len(summary['models_trained'])))
            summary_table.add_row("Features used", str(summary['feature_count']))
            
            console.print(summary_table)
            
            # CatBoost results table
            if 'catboost' in results:
                console.print("\n[bold cyan]CatBoost Model Performance:[/bold cyan]")
                
                cb_table = Table(box=box.ROUNDED)
                cb_table.add_column("Target", style="cyan")
                cb_table.add_column("MAE", justify="right")
                cb_table.add_column("RMSE", justify="right")
                cb_table.add_column("Time", justify="right")
                
                for target, result in results['catboost'].items():
                    if result.metrics:
                        mae = result.metrics.get('mae', 0)
                        rmse = result.metrics.get('rmse', 0)
                        cb_table.add_row(
                            target,
                            f"{mae:.3f}",
                            f"{rmse:.3f}",
                            f"{result.training_time:.1f}s"
                        )
                
                console.print(cb_table)
            
            console.print(f"\n[bold]Models saved to:[/bold] {models_dir}")
            console.print(f"[bold]Experiment logs:[/bold] experiments/{summary['experiment_name']}")
            console.print()
        else:
            # Fallback to simple output
            print("\n" + "=" * 70)
            print("TRAINING COMPLETE")
            print("=" * 70)
            
            summary = pipeline.get_summary()
            print(f"\nExperiment: {summary['experiment_name']}")
            print(f"Models trained: {len(summary['models_trained'])}")
            print(f"Features used: {summary['feature_count']}")
            
            # Print CatBoost metrics
            if 'catboost' in results:
                print("\nCatBoost Model Performance:")
                for target, result in results['catboost'].items():
                    if result.metrics:
                        mae = result.metrics.get('mae', 0)
                        rmse = result.metrics.get('rmse', 0)
                        print(f"  {target}: MAE={mae:.3f}, RMSE={rmse:.3f} ({result.training_time:.1f}s)")
            
            print(f"\nModels saved to: {models_dir}")
            print(f"Experiment logs: experiments/{summary['experiment_name']}")
            print("=" * 70 + "\n")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        return 130
    except Exception as e:
        print(f"\n\nERROR: Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
