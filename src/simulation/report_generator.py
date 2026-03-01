import pandas as pd
import os
import logging
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates formatted reports from simulation results."""
    
    def __init__(self, output_dir: str = 'data/sim_results'):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def _compute_mode(self, values: np.ndarray) -> float:
        """
        Compute the mode of a continuous distribution using KDE.
        For discrete-like distributions (integers), uses histogram binning.
        """
        values = np.array(values)
        if len(values) < 3:
            return float(np.mean(values))
        
        # Remove any NaN/inf values
        values = values[np.isfinite(values)]
        if len(values) < 3:
            return float(np.mean(values))
        
        # Check if values are essentially discrete (integers or half-integers)
        rounded = np.round(values * 2) / 2  # Round to nearest 0.5
        if np.allclose(values, rounded, atol=0.1):
            # Use histogram-based mode for discrete-like data
            bins = np.arange(values.min() - 0.25, values.max() + 0.75, 0.5)
            if len(bins) < 2:
                return float(np.mean(values))
            hist, bin_edges = np.histogram(values, bins=bins)
            mode_idx = np.argmax(hist)
            return float((bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2)
        
        try:
            kde = scipy_stats.gaussian_kde(values)
            x_grid = np.linspace(values.min(), values.max(), 200)
            densities = kde(x_grid)
            mode_idx = np.argmax(densities)
            return float(x_grid[mode_idx])
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.debug("KDE mode computation failed, using median: %s", e)
            return float(np.median(values))

    def _compute_stats_summary(self, values: List[float]) -> Dict[str, float]:
        """Compute mean, mode, median, and std for a list of values."""
        arr = np.array(values)
        return {
            'mean': float(np.mean(arr)),
            'mode': self._compute_mode(arr),
            'median': float(np.median(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'p5': float(np.percentile(arr, 5)),
            'p95': float(np.percentile(arr, 95)),
            'p0.5': float(np.percentile(arr, 0.5)),
            'p99.5': float(np.percentile(arr, 99.5)),
        }

    def format_console_report(
        self, 
        results: List[Dict[str, Any]], 
        detailed: bool = True,
        stat_type: str = 'mode'  # 'mode', 'mean', or 'both'
    ):
        """
        Prints a beautified report to the console.
        
        Args:
            results: List of simulation results
            detailed: Whether to show detailed player breakdowns
            stat_type: Which statistic to display - 'mode' (most likely), 'mean' (average), or 'both'
        """
        if not results:
            print("\nNo results to report.", flush=True)
            return

        print(f"\n{'='*120}", flush=True)
        print(f" NBA SIMULATION REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M')}", flush=True)
        print(f" Showing: {stat_type.upper()} projections", flush=True)
        print(f"{'='*120}", flush=True)

        for res in results:
            if 'error' in res:
                print(f"\n ERROR: {res.get('error', 'Unknown error')}", flush=True)
                continue
                
            home = res['team_a']
            away = res['team_b']
            status = res.get('status', 'Scheduled')
            
            # Get pre-calculated probabilities if available (Massive Sims Support)
            if 'win_prob_a' in res:
                home_win_pct = res['win_prob_a']
                away_win_pct = 100 - home_win_pct
            else:
                home_wins = sum(1 for s in res['simulations'] if s[home]['pts'] > s[away]['pts'])
                total_sims = len(res['simulations'])
                home_win_pct = (home_wins / total_sims) * 100
                away_win_pct = 100 - home_win_pct
            
            # Get pre-calculated team stats or compute them
            if 'team_summaries' in res:
                team_stats = res['team_summaries']
            else:
                team_stats = {}
                for team in [home, away]:
                    pts_values = [s[team]['pts'] for s in res['simulations']]
                    reb_values = [s[team]['reb'] for s in res['simulations']]
                    ast_values = [s[team]['ast'] for s in res['simulations']]
                    team_stats[team] = {
                        'pts': self._compute_stats_summary(pts_values),
                        'reb': self._compute_stats_summary(reb_values),
                        'ast': self._compute_stats_summary(ast_values),
                    }

            # Header
            print(f"\n{'='*120}", flush=True)
            print(f" {away} @ {home}", flush=True)
            print(f" {status}", flush=True)
            print(f"{'='*120}", flush=True)
            
            # Win Probability Bar
            away_bar = '█' * int(away_win_pct / 5)
            home_bar = '█' * int(home_win_pct / 5)
            print(f"\n WIN PROBABILITY", flush=True)
            print(f" {away}: {away_win_pct:5.1f}% {away_bar}", flush=True)
            print(f" {home}: {home_win_pct:5.1f}% {home_bar}", flush=True)
            
            # Projected Score
            print(f"\n PROJECTED SCORE", flush=True)
            print(f" ┌{'─'*70}┐", flush=True)
            print(f" │ {'':^10} │ {away:^26} │ {home:^26} │", flush=True)
            print(f" ├{'─'*70}┤", flush=True)
            
            if stat_type == 'both':
                # Show both mode and mean
                away_mode = team_stats[away]['pts']['mode']
                home_mode = team_stats[home]['pts']['mode']
                away_mean = team_stats[away]['pts']['mean']
                home_mean = team_stats[home]['pts']['mean']
                
                print(f" │ {'Mode':<10} │ {away_mode:^26.1f} │ {home_mode:^26.1f} │", flush=True)
                print(f" │ {'Mean':<10} │ {away_mean:^26.1f} │ {home_mean:^26.1f} │", flush=True)
            else:
                # Show selected stat type
                key = stat_type  # 'mode' or 'mean'
                away_pts = team_stats[away]['pts'][key]
                home_pts = team_stats[home]['pts'][key]
                away_std = team_stats[away]['pts']['std']
                home_std = team_stats[home]['pts']['std']
                
                away_label = f"(±{away_std:.1f})"
                home_label = f"(±{home_std:.1f})"
                
                print(f" │ {'Score':<10} │ {away_pts:^26.1f} │ {home_pts:^26.1f} │", flush=True)
                print(f" │ {'Std Dev':<10} │ {away_label:^26} │ {home_label:^26} │", flush=True)
            
            print(f" └{'─'*70}┘", flush=True)

            if detailed:
                # Detailed Team Breakdowns
                for team in [away, home]:
                    self._print_team_breakdown(team, res, team_stats[team], stat_type)
            
            print(f"\n{'─'*120}", flush=True)

    def _print_team_breakdown(
        self, 
        team: str, 
        res: Dict[str, Any], 
        team_totals: Dict,
        stat_type: str = 'mode'
    ):
        """Prints detailed breakdown for a single team including injury status and 99% Ranges."""
        
        team_players = [p for p in res['player_averages'] if p['team'] == team]
        
        if not team_players:
            print(f"\n No player data available for {team}", flush=True)
            return
        
        # Sort by the selected stat type for points
        sort_key = 'pts_mode' if stat_type == 'mode' else 'pts'
        team_players = sorted(team_players, key=lambda x: x.get(sort_key, x['pts']), reverse=True)
        
        print(f"\n {'─'*120}", flush=True)
        print(f" {team} - FULL ROSTER BREAKDOWN ({stat_type.upper()} projections)", flush=True)
        print(f" {'─'*120}", flush=True)
        
        # Team totals header
        print(f"\n TEAM TOTALS", flush=True)
        pts_key = stat_type if stat_type in ['mode', 'mean'] else 'mode'
        range_str = f"{team_totals['pts']['p0.5']:.0f} - {team_totals['pts']['p99.5']:.0f}"
        
        print(f" ┌{'─'*88}┐", flush=True)
        print(f" │ {'STAT':<12} │ {'MODE':^12} │ {'MEAN':^12} │ {'STD DEV':^12} │ {'RANGE (99%)':^20} │", flush=True)
        print(f" ├{'─'*88}┤", flush=True)
        print(f" │ {'Points':<12} │ {team_totals['pts']['mode']:^12.1f} │ {team_totals['pts']['mean']:^12.1f} │ {team_totals['pts']['std']:^12.1f} │ {range_str:^20} │", flush=True)
        print(f" │ {'Rebounds':<12} │ {team_totals['reb']['mode']:^12.1f} │ {team_totals['reb']['mean']:^12.1f} │ {team_totals['reb']['std']:^12.1f} │ {'-':^20} │", flush=True)
        print(f" │ {'Assists':<12} │ {team_totals['ast']['mode']:^12.1f} │ {team_totals['ast']['mean']:^12.1f} │ {team_totals['ast']['std']:^12.1f} │ {'-':^20} │", flush=True)
        print(f" └{'─'*88}┘", flush=True)
        
        # Player breakdown with 99% Confidence Ranges
        print(f"\n PLAYER PROJECTIONS (99% Confidence Range)", flush=True)
        
        # Wide table to accommodate PTS, REB, AST Ranges
        # Header: Player | Status | PTS | REB | AST | PTS Range | REB Range | AST Range
        print(f" ┌{'─'*134}┐", flush=True)
        print(f" │ {'PLAYER':<20} │ {'STATUS':^8} │ {'PTS':^6} │ {'REB':^6} │ {'AST':^6} │ {'PTS 99% RNG':^13} │ {'REB 99% RNG':^13} │ {'AST 99% RNG':^13} │", flush=True)
        print(f" ├{'─'*134}┤", flush=True)
        
        # Get total points for share calculation
        total_pts = team_totals['pts'][stat_type if stat_type in ['mode', 'mean'] else 'mode']
        
        for p in team_players:
            # Use mode or mean based on stat_type
            if stat_type == 'mode':
                pts_val = p.get('pts_mode', p['pts'])
                reb_val = p.get('reb_mode', p['reb'])
                ast_val = p.get('ast_mode', p['ast'])
            else:
                pts_val = p['pts']
                reb_val = p['reb']
                ast_val = p['ast']
            
            # --- 99% RANGES ---
            pts_rng = f"{p.get('pts_99_ci', [0, 0])[0]:.0f}-{p.get('pts_99_ci', [0, 0])[1]:.0f}"
            reb_rng = f"{p.get('reb_99_ci', [0, 0])[0]:.0f}-{p.get('reb_99_ci', [0, 0])[1]:.0f}"
            ast_rng = f"{p.get('ast_99_ci', [0, 0])[0]:.0f}-{p.get('ast_99_ci', [0, 0])[1]:.0f}"
            
            name = p['name'][:18] + '..' if len(p['name']) > 20 else p['name']
            
            play_prob = p.get('play_probability', 1.0)
            status = self._get_status_label(play_prob)
            
            print(f" │ {name:<20} │ {status:^8} │ {pts_val:^6.1f} │ {reb_val:^6.1f} │ {ast_val:^6.1f} │ {pts_rng:^13} │ {reb_rng:^13} │ {ast_rng:^13} │", flush=True)
        
        print(f" └{'─'*134}┘", flush=True)
        
        # Injury summary
        injured_players = [p for p in team_players if p.get('play_probability', 1.0) < 1.0]
        if injured_players:
            print(f"\n INJURY REPORT", flush=True)
            for p in injured_players:
                prob = p.get('play_probability', 1.0)
                status_text = f"{prob*100:.0f}% to play"
                print(f" • {p['name']}: {status_text}", flush=True)
        
        # Key players (excluding those who are likely out)
        active_players = [p for p in team_players if p.get('play_probability', 1.0) >= 0.5]
        if active_players:
            pts_key = 'pts_mode' if stat_type == 'mode' else 'pts'
            reb_key = 'reb_mode' if stat_type == 'mode' else 'reb'
            ast_key = 'ast_mode' if stat_type == 'mode' else 'ast'
            
            top_scorer = max(active_players, key=lambda x: x.get(pts_key, x['pts']))
            top_rebounder = max(active_players, key=lambda x: x.get(reb_key, x['reb']))
            top_playmaker = max(active_players, key=lambda x: x.get(ast_key, x['ast']))
            
            print(f"\n KEY PLAYERS (Expected to Play)", flush=True)
            print(f" • Leading Scorer:    {top_scorer['name']} ({top_scorer.get(pts_key, top_scorer['pts']):.1f} PTS)", flush=True)
            print(f" • Leading Rebounder: {top_rebounder['name']} ({top_rebounder.get(reb_key, top_rebounder['reb']):.1f} REB)", flush=True)
            print(f" • Leading Playmaker: {top_playmaker['name']} ({top_playmaker.get(ast_key, top_playmaker['ast']):.1f} AST)", flush=True)

    def _get_status_label(self, play_prob: float) -> str:
        """Convert play probability to status label."""
        if play_prob >= 1.0:
            return "✓"
        elif play_prob >= 0.85:
            return "PROB"
        elif play_prob >= 0.50:
            return "GTD"
        elif play_prob >= 0.25:
            return "DBT"
        else:
            return "OUT"

    def format_matchup_summary(self, res: Dict[str, Any], stat_type: str = 'mode') -> str:
        """Returns a compact one-line summary of a matchup."""
        if 'error' in res:
            return f"ERROR: {res.get('error')}"
            
        home = res['team_a']
        away = res['team_b']
        
        home_wins = sum(1 for s in res['simulations'] if s[home]['pts'] > s[away]['pts'])
        total_sims = len(res['simulations'])
        home_win_pct = (home_wins / total_sims) * 100
        
        home_pts_list = [s[home]['pts'] for s in res['simulations']]
        away_pts_list = [s[away]['pts'] for s in res['simulations']]
        
        if stat_type == 'mode':
            avg_home_pts = self._compute_mode(np.array(home_pts_list))
            avg_away_pts = self._compute_mode(np.array(away_pts_list))
        else:
            avg_home_pts = np.mean(home_pts_list)
            avg_away_pts = np.mean(away_pts_list)
        
        winner = home if home_win_pct > 50 else away
        win_pct = home_win_pct if home_win_pct > 50 else (100 - home_win_pct)
        
        return f"{away} @ {home}: {winner} wins ({win_pct:.0f}%) | {avg_away_pts:.0f}-{avg_home_pts:.0f}"

    def export_to_csv(self, results: List[Dict[str, Any]], filename: str = None) -> str:
        """Exports key predictions to a CSV file for backtesting."""
        if not results:
            return None

        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"sim_results_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        rows = []
        for res in results:
            if 'error' in res:
                continue
                
            home = res['team_a']
            away = res['team_b']
            
            if 'win_prob_a' in res:
                home_win_prob = res['win_prob_a'] / 100
                away_win_prob = 1 - home_win_prob
            else:
                home_wins = sum(1 for s in res['simulations'] if s[home]['pts'] > s[away]['pts'])
                total_sims = len(res['simulations'])
                home_win_prob = home_wins / total_sims
                away_win_prob = 1 - home_win_prob
            
            if 'team_summaries' in res:
                ts = res['team_summaries']
                rows.append({
                    'GAME_ID': res.get('game_id'),
                    'DATE': res.get('date'),
                    'HOME_TEAM': home,
                    'AWAY_TEAM': away,
                    'HOME_WIN_PROB': home_win_prob,
                    'AWAY_WIN_PROB': away_win_prob,
                    'MODE_HOME_PTS': ts[home]['pts']['mode'],
                    'MODE_AWAY_PTS': ts[away]['pts']['mode'],
                    'AVG_HOME_PTS': ts[home]['pts']['mean'],
                    'AVG_AWAY_PTS': ts[away]['pts']['mean'],
                    'AVG_HOME_REB': ts[home]['reb']['mean'],
                    'AVG_AWAY_REB': ts[away]['reb']['mean'],
                    'AVG_HOME_AST': ts[home]['ast']['mean'],
                    'AVG_AWAY_AST': ts[away]['ast']['mean'],
                    'HOME_PTS_STD': ts[home]['pts']['std'],
                    'AWAY_PTS_STD': ts[away]['pts']['std'],
                    'SIM_TIMESTAMP': datetime.now().isoformat()
                })
            else:
                # Fallback to slower recalculation
                home_pts_list = [s[home]['pts'] for s in res['simulations']]
                away_pts_list = [s[away]['pts'] for s in res['simulations']]
                rows.append({
                    'GAME_ID': res.get('game_id'),
                    'DATE': res.get('date'),
                    'HOME_TEAM': home,
                    'AWAY_TEAM': away,
                    'HOME_WIN_PROB': home_win_prob,
                    'AWAY_WIN_PROB': away_win_prob,
                    'MODE_HOME_PTS': self._compute_mode(np.array(home_pts_list)),
                    'MODE_AWAY_PTS': self._compute_mode(np.array(away_pts_list)),
                    'AVG_HOME_PTS': np.mean(home_pts_list),
                    'AVG_AWAY_PTS': np.mean(away_pts_list),
                    'AVG_HOME_REB': np.mean([s[home]['reb'] for s in res['simulations']]),
                    'AVG_AWAY_REB': np.mean([s[away]['reb'] for s in res['simulations']]),
                    'AVG_HOME_AST': np.mean([s[home]['ast'] for s in res['simulations']]),
                    'AVG_AWAY_AST': np.mean([s[away]['ast'] for s in res['simulations']]),
                    'HOME_PTS_STD': np.std(home_pts_list),
                    'AWAY_PTS_STD': np.std(away_pts_list),
                    'SIM_TIMESTAMP': datetime.now().isoformat()
                })
            
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        logger.info(f"Results exported to {filepath}")
        return filepath

    def export_player_projections(self, results: List[Dict[str, Any]], filename: str = None) -> str:
        """Exports detailed player projections to CSV with 99% CI."""
        if not results:
            return None

        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"player_projections_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        rows = []
        for res in results:
            if 'error' in res:
                continue
                
            game_id = res.get('game_id', 'UNKNOWN')
            date = res.get('date', '')
            home = res['team_a']
            away = res['team_b']
            
            for p in res.get('player_averages', []):
                opponent = away if p['team'] == home else home
                is_home = 1 if p['team'] == home else 0
                
                rows.append({
                    'GAME_ID': game_id,
                    'DATE': date,
                    'PLAYER_NAME': p['name'],
                    'TEAM': p['team'],
                    'OPPONENT': opponent,
                    'IS_HOME': is_home,
                    'PROJ_PTS_MODE': p.get('pts_mode', p['pts']),
                    'PROJ_PTS_MEAN': p['pts'],
                    'PROJ_REB_MODE': p.get('reb_mode', p['reb']),
                    'PROJ_REB_MEAN': p['reb'],
                    'PROJ_AST_MODE': p.get('ast_mode', p['ast']),
                    'PROJ_AST_MEAN': p['ast'],
                    'PTS_CI_LOW': p.get('pts_95_ci', [0, 0])[0],
                    'PTS_CI_HIGH': p.get('pts_95_ci', [0, 0])[1],
                    'REB_CI_LOW': p.get('reb_95_ci', [0, 0])[0],
                    'REB_CI_HIGH': p.get('reb_95_ci', [0, 0])[1],
                    'AST_CI_LOW': p.get('ast_95_ci', [0, 0])[0],
                    'AST_CI_HIGH': p.get('ast_95_ci', [0, 0])[1],
                    # --- NEW 99% CI COLUMNS ---
                    'PTS_99_CI_LOW': p.get('pts_99_ci', [0, 0])[0],
                    'PTS_99_CI_HIGH': p.get('pts_99_ci', [0, 0])[1],
                    'REB_99_CI_LOW': p.get('reb_99_ci', [0, 0])[0],
                    'REB_99_CI_HIGH': p.get('reb_99_ci', [0, 0])[1],
                    'AST_99_CI_LOW': p.get('ast_99_ci', [0, 0])[0],
                    'AST_99_CI_HIGH': p.get('ast_99_ci', [0, 0])[1],
                    # --------------------------
                    'PLAY_PROBABILITY': p.get('play_probability', 1.0),
                    'SIM_TIMESTAMP': datetime.now().isoformat()
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        logger.info(f"Player projections exported to {filepath}")
        return filepath

    def print_quick_summary(self, results: List[Dict[str, Any]], stat_type: str = 'mode'):
        """Prints a quick one-line summary for each game."""
        print(f"\n{'='*90}", flush=True)
        print(f" QUICK SUMMARY ({stat_type.upper()} projections)", flush=True)
        print(f"{'='*90}", flush=True)
        
        for res in results:
            print(f" {self.format_matchup_summary(res, stat_type)}", flush=True)
        
        print(f"{'='*90}\n", flush=True)
