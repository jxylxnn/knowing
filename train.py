import argparse
import logging
import pandas as pd
from src.models.model_manager import ModelManager
from src.utils.logging_config import setup_logging
from src.config.model_config import get_model_config, print_config_summary

setup_logging()


def main():
    parser = argparse.ArgumentParser(description='Train NBA prediction models')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing NBA data files (default: data)')
    parser.add_argument('--models-dir', type=str, default='models',
                        help='Directory to save trained models (default: models)')
    parser.add_argument('--model-size', type=str, default='auto',
                        choices=['auto', 'small', 'medium', 'large', 'ultra'],
                        help='Model size preset: auto (detect), small, medium, large, ultra (default: auto)')
    args = parser.parse_args()
    
    model_config, hw_info = get_model_config(force_size=None if args.model_size == 'auto' else args.model_size)
    
    print("\n" + "=" * 60)
    print("NBA MODEL TRAINER - Auto Configuration")
    print("=" * 60)
    print(f"\nDetected Hardware:")
    print(f"  Type: {hw_info.get('type', 'unknown').upper()}")
    print(f"  Name: {hw_info.get('name', 'unknown')}")
    print(f"  Compute Score: {hw_info.get('score', 0):.1f}")
    if hw_info.get('vram', 0) > 0:
        print(f"  VRAM: {hw_info.get('vram', 0):.1f} GB")
    if hw_info.get('cores', 0) > 0:
        print(f"  CPU Cores: {hw_info.get('cores', 0)}")
    if hw_info.get('ram', 0) > 0:
        print(f"  RAM: {hw_info.get('ram', 0):.1f} GB")
    
    scale = model_config.get('metadata', {}).get('scale_factor', 1.0)
    print(f"\nGenerated Config (scale={scale:.2f}):")
    print(f"  LSTM:")
    print(f"    hidden={model_config['lstm']['hidden_dim']}, layers={model_config['lstm']['num_layers']}, "
          f"bidirectional={model_config['lstm']['bidirectional']}")
    print(f"  Transformer:")
    print(f"    d_model={model_config['transformer']['d_model']}, heads={model_config['transformer']['nhead']}, "
          f"layers={model_config['transformer']['num_layers']}")
    print(f"  MultiOutputNN:")
    print(f"    hidden={model_config['nn']['hidden_dim']}, blocks={model_config['nn']['num_blocks']}")
    print(f"  GNN:")
    print(f"    hidden={model_config['gnn']['hidden_dim']}, layers={model_config['gnn']['num_layers']}")
    print(f"  TemporalAttention:")
    print(f"    hidden={model_config['temporal']['hidden_dim']}, heads={model_config['temporal']['num_heads']}")
    print(f"  CatBoost:")
    print(f"    iterations={model_config['catboost']['iterations']}, depth={model_config['catboost']['depth']}")
    print(f"\nTraining Settings:")
    print(f"  Warmup Steps: {model_config['training']['warmup_steps']}")
    print(f"  Early Stop Patience: {model_config['training']['early_stop_patience']}")
    print(f"  torch.compile: {'Yes' if model_config['training']['use_compile'] else 'No'}")
    print(f"  Gradient Checkpointing: {'Yes' if model_config['transformer']['grad_checkpoint'] else 'No'}")
    print(f"  Mixed Precision (AMP): {'Yes' if model_config['training']['amp'] else 'No'}")
    print("=" * 60 + "\n")
    
    manager = ModelManager(
        data_dir=args.data_dir, 
        models_dir=args.models_dir,
        model_config=model_config
    )
    
    print("\n--- NBA Game Simulator: Training Phase ---")
    print("Step 1: Preparing Data and Engineering Features...")
    train_df, test_df = manager.prepare_data()
    
    print("\nStep 2: Training Models with Auto-Sized Architecture...")
    print("This may take a while depending on your hardware.")
    manager.train_all(train_df)
    
    print("\nStep 3: Evaluating Performance on Hold-out Test Set...")
    results = manager.evaluate_all(test_df)
    
    print("\n--- Model Performance Results ---")
    for target, metrics in results.items():
        print(f"\n{target} Prediction Stats:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name.upper()}: {value:.4f}")
            
    print(f"\nTraining and Evaluation Complete. Models are saved in '{args.models_dir}' directory.")


if __name__ == "__main__":
    main()