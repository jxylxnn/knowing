import argparse
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any

from src.models.model_manager import ModelManager
from src.simulation.game_simulator import GameSimulator
from src.data.schedule_scraper import ScheduleScraper
from src.simulation.season_simulator import SeasonSimulator
from src.simulation.report_generator import ReportGenerator
from src.config.config import load_config
import pandas as pd

if sys.platform == "win32":
    # Newer Python versions (like 3.14) prefer reconfigure() over wrapping
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    else:
        import io
        if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding.lower() != 'utf-8':
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main() -> None:
    """
    Main entry point for NBA season simulation.
    
    Parses command-line arguments and executes the appropriate simulation mode.
    """
    parser = argparse.ArgumentParser(description='NBA Season & Schedule Simulator')
    
    # Mode selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--today', action='store_true', help='Simulate all games for today')
    group.add_argument('--date', type=str, help='Simulate games for a specific date (YYYY-MM-DD)')
    group.add_argument('--week', action='store_true', help='Simulate games for the upcoming 7 days')
    group.add_argument('--season', action='store_true', help='Simulate all remaining games in the season')
    
    # Configuration
    parser.add_argument('--sims', type=int, default=100, help='Number of simulations per matchup (default: 100)')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers (default: 1). Use 1 for stability with GPU.')
    parser.add_argument('--stat', type=str, default='mode', choices=['mode', 'mean', 'both'],
                       help='Statistic type to display: mode (most likely), mean (average), or both')
    parser.add_argument('--no-csv', action='store_true', help='Disable CSV export')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file (default: config/default.yaml)')
    parser.add_argument('--data-dir', type=str, default=None, help='Data directory (default: from config)')
    parser.add_argument('--models-dir', type=str, default=None, help='Models directory (default: from config)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for projections (default: <data-dir>/sim_results)')
    
    args = parser.parse_args()

    cfg = load_config(args.config)
    args.data_dir = args.data_dir or str(cfg.data.data_dir)
    args.models_dir = args.models_dir or str(cfg.data.models_dir)

    # Safety check for Windows + GPU + Parallelism
    if args.workers > 1 and sys.platform == "win32":
        try:
            import torch
            if torch.cuda.is_available():
                print("\n[WARNING] GPU detected on Windows with multiple workers.", flush=True)
                print("Parallel execution with CUDA often causes deadlocks/crashes.", flush=True)
                print("Forcing --workers 1 for stability.", flush=True)
                args.workers = 1
        except ImportError:
            pass

    # 1. Initialize Components
    try:
        output_dir = args.output_dir if args.output_dir else os.path.join(args.data_dir, 'sim_results')
        
        manager = ModelManager(data_dir=args.data_dir, models_dir=args.models_dir)
        
        model_path = os.path.join(args.models_dir, 'pts_catboost.cbm')
        if not os.path.exists(model_path):
            print(f"\nError: Models not found in '{args.models_dir}' directory.")
            print("Please run training first: python train.py --models-dir <path>")
            sys.exit(1)
        
        manager._load_models()
            
        game_sim = GameSimulator(
            manager, 
            gnn_model=manager.gnn_model, 
            transformer_model=manager.attention_model
        )
        schedule_scraper = ScheduleScraper()
        season_sim = SeasonSimulator(game_sim, schedule_scraper)
        report_gen = ReportGenerator(output_dir=output_dir)
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        sys.exit(1)

    # 2. Execute Mode
    results = []
    
    if args.today:
        print("\nFetching today's games...")
        results = season_sim.simulate_today(num_sims=args.sims, max_workers=args.workers)
        
    elif args.date:
        print(f"\nFetching games for {args.date}...")
        results = season_sim.simulate_date(args.date, num_sims=args.sims, max_workers=args.workers)
        
    elif args.week:
        print("\nFetching games for the upcoming week...")
        # Simulate next 7 days
        all_games = []
        today = datetime.now().date()
        for i in range(7):
            target_date = (today + timedelta(days=i)).strftime('%Y-%m-%d')
            day_games = schedule_scraper.get_games_by_date(target_date)
            if day_games is not None and not day_games.empty:
                all_games.append(day_games)
        
        if not all_games:
            logger.warning("No games found for the upcoming week.")
            results = []
        else:
            combined_df = pd.concat(all_games).drop_duplicates(subset=['GAME_ID'])
            results = season_sim.simulate_games(combined_df, num_sims=args.sims, max_workers=args.workers)
        
    elif args.season:
        print("\nFetching remaining season schedule (next 30 days)...")
        results = season_sim.simulate_remaining_season(num_sims=args.sims, max_workers=args.workers)

    # 3. Report Results
    if results:
        print(f"\n{'='*90}", flush=True)
        print(f"Simulation phase complete. {len(results)} games simulated.", flush=True)
        print(f"{'='*90}", flush=True)
        
        print(f"\nGenerating detailed reports...\n", flush=True)
        
        try:
            report_gen.print_quick_summary(results, stat_type=args.stat)
            
            report_gen.format_console_report(results, detailed=True, stat_type=args.stat)
            
            if not args.no_csv:
                # Export both game results and player projections
                game_csv = report_gen.export_to_csv(results)
                player_csv = report_gen.export_player_projections(results)
                
                print(f"\nGame predictions exported to: {game_csv}", flush=True)
                print(f"Player projections exported to: {player_csv}", flush=True)
                
        except Exception as e:
            print(f"\n!!! ERROR generating reports: {e}", flush=True)
            import traceback
            traceback.print_exc()
            logger.error(f"Reporting error: {e}", exc_info=True)
    else:
        print("\nNo games found or all simulations failed for the selected period.", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
