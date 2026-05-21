import logging
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Dict, Any
from src.simulation.game_simulator import GameSimulator
from src.data.schedule_scraper import ScheduleScraper
from src.simulation.input_health import summarize_input_health

logger = logging.getLogger(__name__)

class SeasonSimulator:
    """Orchestrates batch simulations of NBA games from a schedule."""
    
    def __init__(self, game_simulator: GameSimulator, schedule_scraper: ScheduleScraper, strict_mode: bool = False):
        self.game_simulator = game_simulator
        self.schedule_scraper = schedule_scraper
        self.strict_mode = strict_mode
        self.last_run_summary: Dict[str, Any] = {
            'games_requested': 0,
            'games_simulated': 0,
            'overall_status': 'healthy',
            'input_health': summarize_input_health([]),
            'schedule_health': {},
            'hard_failures': [],
        }

    def _set_schedule_health(self, health: Dict[str, Any]) -> None:
        self.last_run_summary['schedule_health'] = dict(health or {})

    def _finalize_run_summary(self, results: List[Dict[str, Any]], requested_games: int) -> None:
        source_health = []
        for result in results:
            metadata = result.get('metadata', {})
            input_health = metadata.get('input_health', {})
            source_health.extend(input_health.get('sources', []))

        aggregated_input_health = summarize_input_health(source_health)
        hard_failures = []
        schedule_health = self.last_run_summary.get('schedule_health', {})
        if schedule_health.get('status') == 'failed':
            hard_failures.append(schedule_health['source_key'])
        hard_failures.extend(aggregated_input_health.get('hard_failures', []))

        overall_status = 'healthy'
        if hard_failures:
            overall_status = 'failed'
        elif aggregated_input_health.get('degraded_sources'):
            overall_status = 'degraded'

        self.last_run_summary.update({
            'games_requested': int(requested_games),
            'games_simulated': int(len(results)),
            'overall_status': overall_status,
            'input_health': aggregated_input_health,
            'hard_failures': hard_failures,
        })

    def simulate_games(self, games_df: pd.DataFrame, num_sims: int = 100, max_workers: int = 1) -> List[Dict[str, Any]]:
        """
        Simulates a batch of games in parallel with input validation.
        NOTE: For GPU simulations, max_workers=1 is recommended to avoid GPU context switching overhead.
        
        Args:
            games_df: DataFrame with game information (HOME_TEAM, AWAY_TEAM, GAME_ID, etc.)
            num_sims: Number of Monte Carlo simulations per game (default: 100)
            max_workers: Number of parallel workers (default: 1)
            
        Returns:
            List of simulation result dictionaries
            
        Raises:
            ValueError: If input parameters are invalid
        """
        # Validate inputs
        if games_df is None or not isinstance(games_df, pd.DataFrame):
            raise ValueError("games_df must be a pandas DataFrame")
        
        if games_df.empty:
            logger.warning("No games found to simulate.")
            self._finalize_run_summary([], 0)
            return []
        
        if not isinstance(num_sims, int) or num_sims < 1:
            raise ValueError(f"num_sims must be a positive integer, got {num_sims}")
        
        if not isinstance(max_workers, int) or max_workers < 1:
            raise ValueError(f"max_workers must be a positive integer, got {max_workers}")
        
        # Validate required columns
        required_cols = ['HOME_TEAM', 'AWAY_TEAM']
        missing_cols = [c for c in required_cols if c not in games_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in games_df: {missing_cols}")
        
        results = []
        logger.info(f"Simulating {len(games_df)} games with {num_sims} simulations each (GPU Mode)...")

        # Prepare context once
        self.game_simulator.prepare_simulation_context()

        # If GPU is used, ThreadPool often hurts performance due to CUDA context locking.
        # We default to 1 worker but allow override if user knows what they are doing.
        if self.game_simulator.device.type == 'cuda' and max_workers > 1:
            logger.warning(f"GPU detected. Reducing max_workers from {max_workers} to 1 to prevent context contention.")
            max_workers = 1

        if max_workers > 1:
            logger.info(f"Using {max_workers} parallel workers...")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_game = {
                    executor.submit(
                        self.game_simulator.simulate_matchup, 
                        row['HOME_TEAM'], 
                        row['AWAY_TEAM'], 
                        num_sims,
                        row.get('GAME_DATE')
                    ): row for _, row in games_df.iterrows()
                }

                completed = 0
                for future in as_completed(future_to_game):
                    game_row = future_to_game[future]
                    try:
                        sim_result = future.result()
                        if sim_result and 'error' not in sim_result:
                            sim_result['game_id'] = game_row.get('GAME_ID', 'UNK')
                            sim_result['status'] = game_row.get('STATUS', 'Scheduled')
                            sim_result['date'] = game_row.get('GAME_DATE', '')
                            results.append(sim_result)
                            completed += 1
                            print(f"Simulation Complete: {game_row['AWAY_TEAM']} @ {game_row['HOME_TEAM']} ({completed}/{len(future_to_game)})", flush=True)
                        else:
                            logger.warning(f"Simulation failed for {game_row['AWAY_TEAM']} @ {game_row['HOME_TEAM']}: {sim_result.get('error') if sim_result else 'Unknown'}")
                    except Exception as e:
                        logger.error(f"Error simulating {game_row['AWAY_TEAM']} @ {game_row['HOME_TEAM']}: {e}", exc_info=True)
        else:
            # Sequential execution (Best for GPU)
            logger.info("Running simulations sequentially (Single-threaded / Optimized for GPU)...")
            total_games = len(games_df)
            for idx, (_, row) in enumerate(games_df.iterrows(), 1):
                try:
                    sim_result = self.game_simulator.simulate_matchup(
                        row['HOME_TEAM'], 
                        row['AWAY_TEAM'], 
                        num_sims,
                        row.get('GAME_DATE')
                    )
                    
                    if sim_result and 'error' not in sim_result:
                        sim_result['game_id'] = row.get('GAME_ID', 'UNK')
                        sim_result['status'] = row.get('STATUS', 'Scheduled')
                        sim_result['date'] = row.get('GAME_DATE', '')
                        results.append(sim_result)
                        print(f"Simulation Complete: {row['AWAY_TEAM']} @ {row['HOME_TEAM']} ({idx}/{total_games})", flush=True)
                    else:
                        logger.warning(f"Simulation failed for {row['AWAY_TEAM']} @ {row['HOME_TEAM']}: {sim_result.get('error') if sim_result else 'Unknown'}")
                except Exception as e:
                    logger.error(f"Error simulating {row['AWAY_TEAM']} @ {row['HOME_TEAM']}: {e}", exc_info=True)

        self._finalize_run_summary(results, len(games_df))
        return results

    def simulate_today(self, num_sims: int = 100, max_workers: int = 1) -> List[Dict[str, Any]]:
        df = self.schedule_scraper.get_todays_games()
        self._set_schedule_health(self.schedule_scraper.get_last_fetch_status())
        if df is None:
            self._finalize_run_summary([], 0)
            return []
        if not df.empty:
            logger.info(f"Retrieved {len(df)} games from schedule.")
        return self.simulate_games(df, num_sims, max_workers)

    def simulate_date(self, game_date: str, num_sims: int = 100, max_workers: int = 1) -> List[Dict[str, Any]]:
        df = self.schedule_scraper.get_games_by_date(game_date)
        self._set_schedule_health(self.schedule_scraper.get_last_fetch_status())
        if df is None:
            self._finalize_run_summary([], 0)
            return []
        return self.simulate_games(df, num_sims, max_workers)

    def simulate_remaining_season(self, num_sims: int = 50, max_workers: int = 1) -> List[Dict[str, Any]]:
        df = self.schedule_scraper.get_remaining_season()
        self._set_schedule_health(self.schedule_scraper.get_last_fetch_status())
        if df is None:
            self._finalize_run_summary([], 0)
            return []
        return self.simulate_games(df, num_sims, max_workers)
