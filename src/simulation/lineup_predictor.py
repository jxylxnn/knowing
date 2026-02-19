"""
Lineup Predictor - Predicts starting lineups when not confirmed.
Uses historical data, coach tendencies, and matchup analysis.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import os
import json

logger = logging.getLogger(__name__)


class LineupPredictor:
    """
    Predicts starting lineups based on historical patterns, 
    coach tendencies, injuries, and matchup context.
    """
    
    POSITION_ORDER = ['PG', 'SG', 'SF', 'PF', 'C']
    
    def __init__(self, data_dir: str = 'data', cache_dir: str = 'data/cache'):
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        self.players_df = None
        self.games_df = None
        self._lineup_history: Dict[str, List[dict]] = {}
        self._coach_patterns: Dict[str, dict] = {}
        
    def load_data(self):
        """Load player and game data."""
        if self.players_df is None:
            players_path = os.path.join(self.data_dir, 'nba_players.csv')
            games_path = os.path.join(self.data_dir, 'nba_games.csv')
            
            if os.path.exists(players_path):
                self.players_df = pd.read_csv(players_path)
                self.players_df['GAME_DATE'] = pd.to_datetime(self.players_df['GAME_DATE'])
                
            if os.path.exists(games_path):
                self.games_df = pd.read_csv(games_path)
                self.games_df['GAME_DATE'] = pd.to_datetime(self.games_df['GAME_DATE'])
    
    def predict_starting_lineup(
        self,
        team_abbr: str,
        game_date: str,
        opponent: str = None,
        is_home: bool = True,
        unavailable_players: List[str] = None,
        injury_probs: Dict[str, float] = None
    ) -> dict:
        """
        Predict starting lineup for a team.
        
        Args:
            team_abbr: 3-letter team abbreviation
            game_date: Date of game (YYYY-MM-DD)
            opponent: Opponent team abbreviation
            is_home: Whether team is home
            unavailable_players: List of players definitely out
            injury_probs: Dict of player_name -> probability of playing
            
        Returns:
            Dictionary with predicted starters, confidence, and reasoning
        """
        self.load_data()
        
        team_abbr = team_abbr.upper()
        if unavailable_players is None:
            unavailable_players = []
        if injury_probs is None:
            injury_probs = {}
            
        game_date_dt = pd.to_datetime(game_date)
        
        recent_lineups = self._get_recent_starting_lineups(team_abbr, game_date_dt, n_games=10)
        
        starter_candidates = self._identify_starter_candidates(
            team_abbr, game_date_dt, recent_lineups, injury_probs
        )
        
        predicted_starters = self._select_starters(
            starter_candidates, 
            unavailable_players, 
            injury_probs,
            recent_lineups,
            opponent,
            is_home
        )
        
        confidence = self._calculate_confidence(
            predicted_starters, 
            recent_lineups, 
            injury_probs
        )
        
        reasoning = self._generate_reasoning(
            predicted_starters,
            recent_lineups,
            injury_probs,
            unavailable_players
        )
        
        return {
            'team': team_abbr,
            'game_date': game_date,
            'predicted_starters': [p['name'] for p in predicted_starters],
            'predicted_starter_ids': [p['id'] for p in predicted_starters],
            'positions': [p['position'] for p in predicted_starters],
            'confidence': confidence,
            'reasoning': reasoning,
            'substitution_likelihood': self._estimate_substitution_likelihood(team_abbr, game_date_dt)
        }
    
    def _get_recent_starting_lineups(
        self, 
        team_abbr: str, 
        game_date: datetime,
        n_games: int = 10
    ) -> List[dict]:
        """Get the most recent starting lineups for a team."""
        if self.players_df is None:
            return []
            
        team_games = self.players_df[
            (self.players_df['TEAM_ABBREVIATION'] == team_abbr) &
            (self.players_df['GAME_DATE'] < game_date)
        ].sort_values('GAME_DATE', ascending=False)
        
        if team_games.empty:
            return []
        
        unique_dates = team_games['GAME_DATE'].unique()[:n_games]
        
        lineups = []
        for date in unique_dates:
            date_games = team_games[team_games['GAME_DATE'] == date]
            
            starters = []
            game_id = None
            for _, row in date_games.iterrows():
                if row.get('START_POSITION') or row.get('MIN', 0) >= 20:
                    starters.append({
                        'name': row.get('PLAYER_NAME', ''),
                        'id': row.get('PLAYER_ID'),
                        'position': self._infer_position(row),
                        'minutes': row.get('MIN', 0),
                        'game_date': date
                    })
                    if game_id is None:
                        game_id = row.get('GAME_ID')
            
            starters.sort(key=lambda x: x.get('minutes', 0), reverse=True)
            
            if len(starters) >= 5:
                lineups.append({
                    'game_date': date,
                    'game_id': game_id,
                    'starters': starters[:5]
                })
        
        return lineups
    
    def _infer_position(self, player_row: pd.Series) -> str:
        """Infer player position from available data."""
        if pd.notna(player_row.get('START_POSITION')):
            return player_row['START_POSITION']
        
        minutes = player_row.get('MIN', 0)
        pts = player_row.get('PTS', 0)
        reb = player_row.get('REB', 0)
        ast = player_row.get('AST', 0)
        
        if reb > 7 and pts > 12:
            return 'C' if reb > 9 else 'PF'
        elif ast > 5:
            return 'PG' if ast > 7 else 'SG'
        elif pts > 15 and reb < 5:
            return 'SG' if ast > 3 else 'SF'
        else:
            return 'SF'
    
    def _identify_starter_candidates(
        self,
        team_abbr: str,
        game_date: datetime,
        recent_lineups: List[dict],
        injury_probs: Dict[str, float]
    ) -> List[dict]:
        """Identify candidates for starting lineup."""
        if self.players_df is None:
            return []
            
        cutoff_date = game_date - timedelta(days=30)
        
        recent_players = self.players_df[
            (self.players_df['TEAM_ABBREVIATION'] == team_abbr) &
            (self.players_df['GAME_DATE'] >= cutoff_date) &
            (self.players_df['GAME_DATE'] < game_date)
        ]
        
        if recent_players.empty:
            return []
        
        player_aggregates = recent_players.groupby(['PLAYER_ID', 'PLAYER_NAME']).agg({
            'MIN': ['mean', 'std', 'count'],
            'PTS': 'mean',
            'REB': 'mean',
            'AST': 'mean',
            'GAME_ID': 'nunique'
        }).reset_index()
        
        player_aggregates.columns = ['PLAYER_ID', 'PLAYER_NAME', 'AVG_MIN', 'STD_MIN', 
                                      'GAME_COUNT', 'AVG_PTS', 'AVG_REB', 'AVG_AST', 'GAMES_PLAYED']
        
        starter_counts = {}
        for lineup in recent_lineups:
            for starter in lineup.get('starters', []):
                name = starter.get('name')
                if name:
                    starter_counts[name] = starter_counts.get(name, 0) + 1
        
        candidates = []
        for _, row in player_aggregates.iterrows():
            player_name = row['PLAYER_NAME']
            
            if row['AVG_MIN'] < 8:
                continue
                
            starter_rate = starter_counts.get(player_name, 0) / max(len(recent_lineups), 1)
            
            play_prob = injury_probs.get(player_name, 1.0)
            
            position = self._infer_position_from_stats(row['AVG_PTS'], row['AVG_REB'], row['AVG_AST'])
            
            candidates.append({
                'id': row['PLAYER_ID'],
                'name': player_name,
                'avg_minutes': row['AVG_MIN'],
                'starter_rate': starter_rate,
                'play_probability': play_prob,
                'position': position,
                'games_played': row['GAMES_PLAYED'],
                'avg_pts': row['AVG_PTS'],
                'avg_reb': row['AVG_REB'],
                'avg_ast': row['AVG_AST']
            })
        
        candidates.sort(key=lambda x: (
            -x['starter_rate'],
            -x['avg_minutes'],
            -x['play_probability']
        ))
        
        return candidates
    
    def _infer_position_from_stats(self, pts: float, reb: float, ast: float) -> str:
        """Infer position from statistical profile."""
        if reb > 6:
            return 'C' if reb > 8 else 'PF'
        elif ast > 4.5:
            return 'PG'
        elif pts > 14 and reb < 5:
            return 'SG' if ast > 3 else 'SF'
        elif reb > 4 and pts < 12:
            return 'PF'
        else:
            return 'SF'
    
    def _select_starters(
        self,
        candidates: List[dict],
        unavailable_players: List[str],
        injury_probs: Dict[str, float],
        recent_lineups: List[dict],
        opponent: str,
        is_home: bool
    ) -> List[dict]:
        """Select the 5 most likely starters."""
        available = [
            c for c in candidates 
            if c['name'] not in unavailable_players 
            and c['play_probability'] > 0.3
        ]
        
        if len(available) < 5:
            available = [c for c in candidates if c['name'] not in unavailable_players]
        
        position_groups = {pos: [] for pos in self.POSITION_ORDER}
        for candidate in available:
            pos = candidate['position']
            if pos in position_groups:
                position_groups[pos].append(candidate)
            else:
                position_groups['SF'].append(candidate)
        
        for pos in position_groups:
            position_groups[pos].sort(key=lambda x: (
                -x['starter_rate'],
                -x['avg_minutes'],
                -x['play_probability']
            ))
        
        selected = []
        used_names = set()
        
        for pos in self.POSITION_ORDER:
            for candidate in position_groups[pos]:
                if candidate['name'] not in used_names:
                    selected.append(candidate)
                    used_names.add(candidate['name'])
                    break
        
        if len(selected) < 5:
            remaining = [c for c in available if c['name'] not in used_names]
            remaining.sort(key=lambda x: -x['avg_minutes'])
            
            for candidate in remaining:
                if len(selected) >= 5:
                    break
                selected.append(candidate)
                used_names.add(candidate['name'])
        
        return selected[:5]
    
    def _calculate_confidence(
        self,
        predicted_starters: List[dict],
        recent_lineups: List[dict],
        injury_probs: Dict[str, float]
    ) -> float:
        """Calculate confidence score for prediction."""
        if not predicted_starters or not recent_lineups:
            return 0.3
        
        name_to_info = {s['name']: s for s in predicted_starters}
        
        match_scores = []
        for lineup in recent_lineups[:5]:
            lineup_names = {s['name'] for s in lineup.get('starters', [])}
            predicted_names = {s['name'] for s in predicted_starters}
            
            matches = len(lineup_names & predicted_names)
            match_scores.append(matches / 5.0)
        
        continuity_score = np.mean(match_scores) if match_scores else 0.5
        
        injury_penalty = 0
        for starter in predicted_starters:
            prob = starter.get('play_probability', 1.0)
            if prob < 1.0:
                injury_penalty += (1.0 - prob) * 0.1
        
        n_games = len(recent_lineups)
        recency_factor = min(n_games / 5.0, 1.0)
        
        confidence = continuity_score * 0.6 + recency_factor * 0.3 + 0.1 - injury_penalty
        
        return float(np.clip(confidence, 0.1, 0.95))
    
    def _generate_reasoning(
        self,
        predicted_starters: List[dict],
        recent_lineups: List[dict],
        injury_probs: Dict[str, float],
        unavailable_players: List[str]
    ) -> List[str]:
        """Generate human-readable reasoning for the prediction."""
        reasoning = []
        
        if not recent_lineups:
            reasoning.append("No recent lineup history available - using minutes-based prediction")
        else:
            last_lineup_names = [s['name'] for s in recent_lineups[0].get('starters', [])]
            predicted_names = [s['name'] for s in predicted_starters]
            
            matches = set(last_lineup_names) & set(predicted_names)
            if len(matches) >= 4:
                reasoning.append(f"High lineup continuity: {len(matches)} starters from last game")
            elif len(matches) >= 3:
                reasoning.append(f"Moderate lineup continuity: {len(matches)} starters from last game")
            else:
                reasoning.append(f"Significant lineup changes expected: only {len(matches)} returning starters")
        
        for player in unavailable_players:
            reasoning.append(f"{player} is OUT - rotation adjustment required")
        
        uncertain_players = [
            s['name'] for s in predicted_starters 
            if s.get('play_probability', 1.0) < 0.9
        ]
        if uncertain_players:
            reasoning.append(f"Injury uncertainty for: {', '.join(uncertain_players)}")
        
        return reasoning
    
    def _estimate_substitution_likelihood(
        self, 
        team_abbr: str, 
        game_date: datetime
    ) -> float:
        """Estimate how likely the team is to change their lineup."""
        recent_lineups = self._get_recent_starting_lineups(team_abbr, game_date, n_games=5)
        
        if len(recent_lineups) < 2:
            return 0.5
        
        changes = 0
        for i in range(len(recent_lineups) - 1):
            current = {s['name'] for s in recent_lineups[i].get('starters', [])}
            previous = {s['name'] for s in recent_lineups[i + 1].get('starters', [])}
            
            if current != previous:
                changes += 1
        
        return changes / (len(recent_lineups) - 1)
    
    def predict_minutes_distribution(
        self,
        team_abbr: str,
        game_date: str,
        opponent: str = None,
        is_home: bool = True,
        unavailable_players: List[str] = None,
        injury_probs: Dict[str, float] = None
    ) -> Dict[str, float]:
        """
        Predict expected minutes for all rotation players.
        
        Returns:
            Dict mapping player_name -> expected_minutes
        """
        self.load_data()
        
        if unavailable_players is None:
            unavailable_players = []
        if injury_probs is None:
            injury_probs = {}
            
        game_date_dt = pd.to_datetime(game_date)
        
        lineup_prediction = self.predict_starting_lineup(
            team_abbr, game_date, opponent, is_home, unavailable_players, injury_probs
        )
        
        if self.players_df is None:
            return {}
        
        cutoff_date = game_date_dt - timedelta(days=30)
        
        recent_players = self.players_df[
            (self.players_df['TEAM_ABBREVIATION'] == team_abbr) &
            (self.players_df['GAME_DATE'] >= cutoff_date) &
            (self.players_df['GAME_DATE'] < game_date)
        ]
        
        if recent_players.empty:
            return {}
        
        player_minutes = recent_players.groupby('PLAYER_NAME').agg({
            'MIN': ['mean', 'std', 'max', 'min', 'count']
        }).reset_index()
        
        player_minutes.columns = ['PLAYER_NAME', 'AVG_MIN', 'STD_MIN', 'MAX_MIN', 'MIN_MIN', 'GAMES']
        
        minutes_dist = {}
        total_minutes = 240.0
        
        starters = set(lineup_prediction['predicted_starters'])
        
        for _, row in player_minutes.iterrows():
            name = row['PLAYER_NAME']
            
            if name in unavailable_players:
                continue
            
            base_mins = row['AVG_MIN']
            play_prob = injury_probs.get(name, 1.0)
            
            if name in starters:
                if base_mins < 25:
                    base_mins = min(base_mins + 3, 35)
            else:
                if base_mins > 25:
                    base_mins = max(base_mins - 3, 15)
            
            expected = base_mins * play_prob
            minutes_dist[name] = expected
        
        minutes_dist = self._normalize_minutes(minutes_dist, total_minutes)
        
        return minutes_dist
    
    def _normalize_minutes(
        self, 
        minutes_dist: Dict[str, float], 
        target_total: float = 240.0
    ) -> Dict[str, float]:
        """Normalize minutes to sum to target total."""
        current_total = sum(minutes_dist.values())
        
        if current_total <= 0:
            return minutes_dist
        
        if abs(current_total - target_total) > 1:
            scale = target_total / current_total
            return {k: min(v * scale, 42.0) for k, v in minutes_dist.items()}
        
        return minutes_dist
    
    def get_lineup_matchup_advantage(
        self,
        home_lineup: List[dict],
        away_lineup: List[dict]
    ) -> dict:
        """
        Analyze positional matchups between two lineups.
        
        Returns:
            Dictionary with matchup analysis and advantages
        """
        home_by_pos = {p['position']: p for p in home_lineup}
        away_by_pos = {p['position']: p for p in away_lineup}
        
        advantages = []
        
        for pos in self.POSITION_ORDER:
            home_player = home_by_pos.get(pos)
            away_player = away_by_pos.get(pos)
            
            if home_player and away_player:
                home_off = home_player.get('avg_pts', 10)
                away_off = away_player.get('avg_pts', 10)
                
                if home_off > away_off + 3:
                    advantage = 'home'
                    magnitude = (home_off - away_off) / 10
                elif away_off > home_off + 3:
                    advantage = 'away'
                    magnitude = (away_off - home_off) / 10
                else:
                    advantage = 'neutral'
                    magnitude = 0
                
                advantages.append({
                    'position': pos,
                    'home_player': home_player['name'],
                    'away_player': away_player['name'],
                    'advantage': advantage,
                    'magnitude': magnitude
                })
        
        home_advantages = sum(1 for a in advantages if a['advantage'] == 'home')
        away_advantages = sum(1 for a in advantages if a['advantage'] == 'away')
        
        return {
            'positional_matchups': advantages,
            'home_position_wins': home_advantages,
            'away_position_wins': away_advantages,
            'overall_edge': 'home' if home_advantages > away_advantages else (
                'away' if away_advantages > home_advantages else 'neutral'
            )
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    predictor = LineupPredictor()
    
    print("Testing Lineup Predictor...")
    
    prediction = predictor.predict_starting_lineup(
        'BOS',
        datetime.now().strftime('%Y-%m-%d'),
        opponent='MIA',
        is_home=True
    )
    
    print(f"\nPredicted Celtics Lineup vs MIA:")
    print(f"  Confidence: {prediction['confidence']:.0%}")
    print(f"  Starters: {prediction['predicted_starters']}")
    print(f"  Positions: {prediction['positions']}")
    print(f"  Reasoning: {prediction['reasoning']}")