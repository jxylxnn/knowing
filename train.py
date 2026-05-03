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

from src.config import load_config
from src.training.pipeline import TrainingPipeline, create_pipeline
from src.preprocessing.data_loader import DataLoader
from src.preprocessing.feature_engineer import FeatureEngineer, build_feature_engineer
from src.training.presets import apply_recent_history_window, resolve_training_preset
from src.training.training_logger import get_training_logger, RichTrainingLogger
from src.models.gpu_utils import (
    check_gpu_compatibility,
    get_gpu_memory_usage,
    initialize_gpu_optimizations,
    print_gpu_summary,
)
from src.utils.logging_config import setup_logging

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


from src.config.model_config import normalize_model_size


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


def ensure_directory_writable(path: Path, label: str) -> None:
    """Create a directory if needed and verify that it is writable."""
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".codex_write_test"
    try:
        probe.write_text("ok", encoding="utf-8")
    except Exception as exc:
        raise RuntimeError(f"{label} directory is not writable: {path}") from exc
    finally:
        try:
            if probe.exists():
                probe.unlink()
        except Exception:
            pass


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
        '--config', type=str, default='config/default.yaml',
        help='Path to the YAML config file used for preset resolution (default: config/default.yaml)'
    )
    parser.add_argument(
        '--preset', type=str, default='full',
        choices=['small', 'full'],
        help='Training preset controlling feature groups, Transformer use, and recent-history trimming (default: full)'
    )
    parser.add_argument(
        '--mode', type=str, default=None,
        choices=['quick', 'standard', 'full'],
        help='Training mode override; defaults come from the selected preset'
    )
    parser.add_argument(
        '--model-size', type=normalize_model_size, default=None,
        choices=['auto', 'S', 'M', 'L', 'XL'],
        help='Model size override; defaults come from the selected preset'
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
        '--feature-ablation', action='store_true',
        help='Benchmark and prune weak feature groups/formulas before training'
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

    config_path = Path(args.config).expanduser()
    runtime_config = load_config(config_path)
    preset = resolve_training_preset(args.preset, getattr(runtime_config, 'training_presets', {}))

    resolved_mode = args.mode or preset.default_mode
    resolved_model_size = args.model_size or preset.default_model_size

    data_dir = resolve_runtime_path(args.data_dir, 'data')
    models_dir = resolve_runtime_path(args.models_dir, 'models')
    cache_dir = resolve_runtime_path(args.cache_dir, 'cache/training')

    logger.info(
        "Training CLI starting: data_dir=%s models_dir=%s cache_dir=%s config=%s preset=%s mode=%s model_size=%s parallel=%s max_workers=%s no_gpu=%s",
        data_dir,
        models_dir,
        cache_dir,
        config_path,
        preset.name,
        resolved_mode,
        resolved_model_size,
        args.parallel,
        args.max_workers,
        args.no_gpu,
    )

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
    
    current_stage = "startup"

    try:
        # Preflight the runtime directories before doing expensive work.
        current_stage = "preflight checks"
        logger.info("Step 0/5: %s", current_stage.title())
        ensure_directory_writable(models_dir, "models")
        ensure_directory_writable(cache_dir, "cache")

        # Check for data files
        players_file = data_dir / 'nba_players.csv'
        games_file = data_dir / 'nba_games.csv'

        if not players_file.exists() or not games_file.exists():
            if console and RICH_AVAILABLE:
                console.print("[bold red]ERROR: Data files not found![/bold red]")
                console.print("Please run 'python update_data.py' first to fetch NBA data.")
            else:
                print(f"\nERROR: Data files not found in {data_dir}")
                print("Please run 'python update_data.py' first to fetch NBA data.")
            raise FileNotFoundError(
                f"Missing required data files in {data_dir}: "
                f"{'nba_players.csv' if not players_file.exists() else ''}"
                f"{', ' if (not players_file.exists() and not games_file.exists()) else ''}"
                f"{'nba_games.csv' if not games_file.exists() else ''}"
            )

        # Load data
        current_stage = "loading and merging datasets"
        logger.info("Step 1/5: %s", current_stage.title())
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
        current_stage = "feature engineering"
        logger.info("Step 2/5: %s", current_stage.title())
        if console and RICH_AVAILABLE:
            console.print("\n[bold cyan]Step 2: Engineering features...[/bold cyan]")
        else:
            print("\nStep 2: Engineering features...")
        
        feature_engineer_kwargs = dict(preset.feature_engineer_kwargs())
        disable_columns = []
        if args.feature_ablation:
            ablation_probe = build_feature_engineer(**preset.feature_engineer_kwargs())
            ablation_report = ablation_probe.benchmark_feature_variants(merged_df, target='PTS')
            logger.info("Feature ablation report: %s", ablation_report)
            best_variant = ablation_report.get('best', {}).get('variant')
            if best_variant == 'no_matchup':
                feature_engineer_kwargs['disable_groups'] = ['matchup', 'opponent_strength']
            elif best_variant == 'no_context':
                feature_engineer_kwargs['disable_groups'] = ['context', 'fatigue']
            elif best_variant == 'no_target_encoding':
                feature_engineer_kwargs['disable_groups'] = ['target_encoding', 'league_rank']
            elif best_variant == 'formula_raw_only':
                disable_columns = ablation_probe._formula_columns_hint()

        if disable_columns:
            feature_engineer_kwargs['disable_columns'] = disable_columns

        if preset.recent_seasons is not None:
            current_stage = "preset history trimming"
            logger.info("Step 1.5/5: %s", current_stage.title())
            if console and RICH_AVAILABLE:
                console.print(
                    f"\n[bold cyan]Applying preset recent-history window: last {preset.recent_seasons} seasons...[/bold cyan]"
                )
            trimmed_df = apply_recent_history_window(merged_df, preset.recent_seasons)
            if len(trimmed_df) != len(merged_df):
                logger.info(
                    "Recent-history preset trimmed merged data from %s to %s rows",
                    len(merged_df),
                    len(trimmed_df),
                )
                merged_df = trimmed_df
            elif 'SEASON_ID' not in merged_df.columns:
                logger.warning(
                    "Preset %s requested a recent-history window, but SEASON_ID is unavailable; training on full history.",
                    preset.name,
                )

        feature_engineer = build_feature_engineer(
            rolling_windows=feature_engineer_kwargs.get('rolling_windows'),
            enable_groups=feature_engineer_kwargs.get('enable_groups'),
            disable_groups=feature_engineer_kwargs.get('disable_groups'),
            disable_columns=feature_engineer_kwargs.get('disable_columns'),
            cache_dir=cache_dir,
        )
        full_df = feature_engineer.create_features(merged_df)
        
        if console and RICH_AVAILABLE:
            console.print(f"  [green]✓[/green] Created {len(full_df.columns):,} features")
        else:
            print(f"  Created {len(full_df.columns)} features")
        
        # Create and run pipeline
        current_stage = "pipeline initialization"
        logger.info("Step 3/5: %s", current_stage.title())
        if console and RICH_AVAILABLE:
            console.print(
                f"\n[bold cyan]Step 3: Initializing training pipeline ({resolved_mode} mode, {preset.name} preset)...[/bold cyan]"
            )
        else:
            print(f"\nStep 3: Initializing training pipeline ({resolved_mode} mode, {preset.name} preset)...")
        
        pipeline = create_pipeline(
            mode=resolved_mode,
            data_dir=data_dir,
            models_dir=models_dir,
            cache_dir=cache_dir,
            model_size=resolved_model_size,
            parallel=args.parallel,
            max_workers=args.max_workers,
            use_gpu=gpu_available,
            experiment_name=args.experiment_name,
        )
        pipeline.training_preset = preset.name
        pipeline.feature_group_selection = list(preset.enable_groups)
        pipeline.model_config["transformer"]["enabled"] = bool(preset.transformer_enabled)
        pipeline.model_config.setdefault("metadata", {})
        pipeline.model_config["metadata"]["training_preset"] = preset.name
        pipeline.model_config["metadata"]["recent_seasons"] = preset.recent_seasons
        
        # Print hardware info
        print_hardware_info(pipeline.hw_info, console)
        
        # Prepare data
        current_stage = "data preparation and splitting"
        logger.info("Step 4/5: %s", current_stage.title())
        if console and RICH_AVAILABLE:
            console.print("\n[bold cyan]Step 4: Preparing data splits...[/bold cyan]")
        else:
            print("\nStep 4: Preparing data splits...")
        
        fit_df, val_df, test_df = pipeline.prepare_data(full_df)
        
        # Print data info
        print_data_info(merged_df, full_df, fit_df, val_df, test_df, console)
        
        # Train models
        current_stage = "model training"
        logger.info("Step 5/5: %s", current_stage.title())
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
            summary_table.add_row("Preset", summary.get('training_preset') or preset.name)
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
            print(f"Preset: {summary.get('training_preset') or preset.name}")
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
        logger.exception("Training failed during %s", current_stage)
        print(f"\n\nERROR: Training failed during {current_stage}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
