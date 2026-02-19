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
    
    args = parser.parse_args()

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
        manager = ModelManager()
        # Explicitly load models so they are available for the advanced simulator
        manager._load_models()
        
        # Check if core models exist
        if not all(os.path.exists(f'models/{target.lower()}_catboost.cbm') for target in manager.core_targets):
            print("\nError: CatBoost models not found in 'models/' directory.")
            print("Please run 'python train.py' first to train your models.")
            sys.exit(1)
            
        # Pass advanced models if available (top-tier logic)
        game_sim = GameSimulator(
            manager, 
            gnn_model=manager.gnn_model, 
            transformer_model=manager.attention_model
        )
        schedule_scraper = ScheduleScraper()
        season_sim = SeasonSimulator(game_sim, schedule_scraper)
        report_gen = ReportGenerator()
        
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
            all_games.append(schedule_scraper.get_games_by_date(target_date))
        
        combined_df = pd.concat(all_games).drop_duplicates(subset=['GAME_ID'])
        results = season_sim.simulate_games(combined_df, num_sims=args.sims, max_workers=args.workers)
        
    elif args.season:
        print("\nFetching remaining season schedule (next 30 days)...")
        results = season_sim.simulate_remaining_season(num_sims=args.sims, max_workers=args.workers)

    # 3. Report Results
    if results:
        print(f"\n{'='*90}", flush=True)
        print(f"[DEBUG] Simulation phase complete. {len(results)} games simulated.", flush=True)
        print(f"{'='*90}", flush=True)
        
        # Debug: show structure of first result
        print(f"\n[DEBUG] First result keys: {list(results[0].keys())}", flush=True)
        if 'simulations' in results[0]:
            print(f"[DEBUG] Simulations count: {len(results[0]['simulations'])}", flush=True)
        if 'player_averages' in results[0]:
            print(f"[DEBUG] Player averages count: {len(results[0]['player_averages'])}", flush=True)
        
        print(f"\nGenerating detailed reports...\n", flush=True)
        
        try:
            # Print quick summary first
            print("[DEBUG] Calling print_quick_summary...", flush=True)
            report_gen.print_quick_summary(results, stat_type=args.stat)
            print("[DEBUG] Quick summary complete.", flush=True)
            
            # Then print detailed breakdown
            print("[DEBUG] Calling format_console_report...", flush=True)
            report_gen.format_console_report(results, detailed=True, stat_type=args.stat)
            print("[DEBUG] Console report complete.", flush=True)
            
            if not args.no_csv:
                # Export both game results and player projections
                print("[DEBUG] Exporting to CSV...", flush=True)
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
    
    print("\n[DEBUG] Script finished successfully.", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
