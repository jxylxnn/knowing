#!/usr/bin/env python3
"""New NBA Model Training Script - Efficient Training Pipeline.

This script provides an efficient, modular training pipeline with:
- Parallel training across targets
- Smart caching of features
- Experiment tracking
- Multiple training modes (quick/standard/full)
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.training.pipeline import TrainingPipeline, create_pipeline
from src.preprocessing.data_loader import DataLoader
from src.preprocessing.feature_engineer import FeatureEngineer
from src.utils.logging_config import setup_logging

setup_logging()


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
    
    print("\n" + "=" * 70)
    print("NBA PREDICTION MODEL TRAINER - v2.0")
    print("=" * 70)
    print(f"Mode: {args.mode.upper()}")
    print(f"Model Size: {args.model_size}")
    print(f"Parallel: {args.parallel}")
    print(f"GPU: {'Disabled' if args.no_gpu else 'Auto-detect'}")
    print("=" * 70 + "\n")
    
    # Check for data files
    data_dir = Path(args.data_dir)
    players_file = data_dir / 'nba_players.csv'
    games_file = data_dir / 'nba_games.csv'
    
    if not players_file.exists() or not games_file.exists():
        print(f"\nERROR: Data files not found in {data_dir}")
        print("Please run 'python update_data.py' first to fetch NBA data.")
        sys.exit(1)
    
    try:
        # Load data
        print("Step 1: Loading and merging datasets...")
        loader = DataLoader(str(players_file), str(games_file))
        merged_df = loader.merge_datasets()
        print(f"  Loaded {len(merged_df)} player-game records")
        
        # Engineer features
        print("\nStep 2: Engineering features...")
        feature_engineer = FeatureEngineer()
        full_df = feature_engineer.create_features(merged_df)
        print(f"  Created {len(full_df.columns)} features")
        
        # Create and run pipeline
        print(f"\nStep 3: Initializing training pipeline ({args.mode} mode)...")
        pipeline = create_pipeline(
            mode=args.mode,
            data_dir=args.data_dir,
            models_dir=args.models_dir,
            cache_dir=args.cache_dir,
            model_size=args.model_size,
            parallel=args.parallel,
            max_workers=args.max_workers,
            use_gpu=not args.no_gpu,
            experiment_name=args.experiment_name,
        )
        
        # Prepare data
        print("\nStep 4: Preparing data splits...")
        fit_df, val_df, test_df = pipeline.prepare_data(full_df)
        
        # Train models
        print(f"\nStep 5: Training models (this may take a while)...")
        results = pipeline.train(fit_df, val_df)
        
        # Print summary
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
        
        print(f"\nModels saved to: {args.models_dir}")
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