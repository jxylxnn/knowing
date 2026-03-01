"""
Error Calibration System - Self-correcting predictions based on historical residuals.
Tracks systematic biases and applies adaptive corrections to improve accuracy.
"""
import pandas as pd
import numpy as np
import logging
import os
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import pickle

logger = logging.getLogger(__name__)


class ErrorCalibrator:
    """
    Tracks prediction errors and applies calibration corrections.
    
    Key Features:
    - Rolling residual tracking per player/stat
    - Adaptive bias correction
    - Confidence-weighted blending
    - Weekly model retraining triggers
    """
    
    def __init__(self, cache_dir: str = 'data/cache', calibration_window_days: int = 30):
        self.cache_dir = cache_dir
        self.calibration_window_days = calibration_window_days
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        self.prediction_history: List[dict] = []
        self.player_biases: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.stat_biases: Dict[str, float] = defaultdict(float)
        self.recent_predictions: List[dict] = []
        
        self._load_calibration_data()
    
    def _load_calibration_data(self):
        """Load saved calibration data from disk."""
        calib_file = os.path.join(self.cache_dir, 'calibration_data.json')
        
        if os.path.exists(calib_file):
            try:
                with open(calib_file, 'r') as f:
                    data = json.load(f)
                    self.player_biases = defaultdict(lambda: defaultdict(float), data.get('player_biases', {}))
                    self.stat_biases = defaultdict(float, data.get('stat_biases', {}))
                logger.info("Loaded calibration data")
            except Exception as e:
                logger.debug(f"Failed to load calibration data: {e}")
    
    def _save_calibration_data(self):
        """Save calibration data to disk."""
        calib_file = os.path.join(self.cache_dir, 'calibration_data.json')
        
        try:
            with open(calib_file, 'w') as f:
                json.dump({
                    'player_biases': dict(self.player_biases),
                    'stat_biases': dict(self.stat_biases),
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.debug(f"Failed to save calibration data: {e}")
    
    def record_prediction(
        self,
        player_name: str,
        player_id: int,
        stat: str,
        predicted: float,
        actual: float,
        game_date: str = None,
        confidence: float = 0.5
    ):
        """
        Record a prediction vs actual for future calibration.
        
        Args:
            player_name: Player's name
            player_id: Player's ID
            stat: Stat type (PTS, REB, AST, etc.)
            predicted: Predicted value
            actual: Actual value from game
            game_date: Date of game
            confidence: Model confidence (0-1)
        """
        if game_date is None:
            game_date = datetime.now().strftime('%Y-%m-%d')
        
        record = {
            'player_name': player_name,
            'player_id': player_id,
            'stat': stat,
            'predicted': predicted,
            'actual': actual,
            'residual': actual - predicted,
            'abs_error': abs(actual - predicted),
            'game_date': game_date,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
        
        self.recent_predictions.append(record)
        
        cutoff = datetime.now() - timedelta(days=self.calibration_window_days)
        self.recent_predictions = [
            p for p in self.recent_predictions 
            if datetime.strptime(p['game_date'], '%Y-%m-%d') >= cutoff
        ]
        
        if len(self.recent_predictions) >= 100:
            self._update_biases()
    
    def _update_biases(self):
        """Update bias calculations from recent predictions."""
        if not self.recent_predictions:
            return
        
        df = pd.DataFrame(self.recent_predictions)
        
        for stat in df['stat'].unique():
            stat_df = df[df['stat'] == stat]
            
            bias = stat_df['residual'].mean()
            self.stat_biases[stat] = self.stat_biases[stat] * 0.7 + bias * 0.3
        
        for player_id in df['player_id'].unique():
            player_df = df[df['player_id'] == player_id]
            
            if len(player_df) >= 3:
                player_biases = player_df.groupby('stat')['residual'].mean().to_dict()
                
                for stat, bias in player_biases.items():
                    old_bias = self.player_biases[player_id].get(stat, 0)
                    self.player_biases[player_id][stat] = old_bias * 0.6 + bias * 0.4
        
        self._save_calibration_data()
        
        logger.info(f"Updated calibration biases from {len(self.recent_predictions)} predictions")
    
    def get_calibrated_prediction(
        self,
        player_id: int,
        stat: str,
        raw_prediction: float,
        player_name: str = None
    ) -> Tuple[float, float]:
        """
        Apply calibration corrections to a raw prediction.
        
        Args:
            player_id: Player's ID
            stat: Stat type (PTS, REB, AST, etc.)
            raw_prediction: The model's raw prediction
            player_name: Optional player name for additional lookup
            
        Returns:
            Tuple of (calibrated_prediction, confidence_adjusted_std)
        """
        player_bias = self.player_biases.get(player_id, {}).get(stat, 0)
        
        stat_bias = self.stat_biases.get(stat, 0)
        
        if player_bias != 0:
            calibrated = raw_prediction + player_bias
        else:
            calibrated = raw_prediction + stat_bias * 0.5
        
        n_observations = 0
        for pred in self.recent_predictions:
            if pred.get('player_id') == player_id and pred.get('stat') == stat:
                n_observations += 1
        
        confidence_weight = min(n_observations / 20.0, 1.0)
        
        adjusted_std = abs(player_bias) * confidence_weight
        
        calibrated = float(np.clip(calibrated, 0, 100))
        
        return calibrated, float(adjusted_std)
    
    def apply_calibration_to_predictions(
        self,
        predictions: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Apply calibration to a dict of player predictions.
        
        Args:
            predictions: {player_name: {stat: value, ...}, ...}
            
        Returns:
            Calibrated predictions in same format
        """
        calibrated = {}
        
        for player_name, stats in predictions.items():
            calibrated[player_name] = {}
            
            for stat, value in stats.items():
                if isinstance(value, (int, float)):
                    player_id = self._get_player_id(player_name)
                    
                    calibrated_val, _ = self.get_calibrated_prediction(
                        player_id, stat, value, player_name
                    )
                    calibrated[player_name][stat] = calibrated_val
                else:
                    calibrated[player_name][stat] = value
        
        return calibrated
    
    def _get_player_id(self, player_name: str) -> int:
        """Get player ID from name (reverse lookup)."""
        for pred in reversed(self.recent_predictions):
            if pred.get('player_name') == player_name:
                return pred.get('player_id', 0)
        return 0
    
    def get_calibration_report(self) -> dict:
        """
        Generate a report on current calibration status.
        
        Returns:
            Dictionary with calibration metrics
        """
        if not self.recent_predictions:
            return {
                'status': 'no_data',
                'message': 'No prediction history for calibration'
            }
        
        df = pd.DataFrame(self.recent_predictions)
        
        report = {
            'status': 'active',
            'total_predictions': len(df),
            'unique_players': df['player_id'].nunique(),
            'unique_stats': df['stat'].unique().tolist(),
            'overall_mae': df['abs_error'].mean(),
            'overall_mse': (df['residual'] ** 2).mean(),
            'stat_biases': dict(self.stat_biases),
            'player_bias_count': len(self.player_biases),
            'days_of_data': self.calibration_window_days
        }
        
        player_maes = df.groupby('player_id')['abs_error'].mean().sort_values(ascending=False)
        
        if len(player_maes) > 0:
            report['worst_players'] = player_maes.head(5).to_dict()
        
        return report
    
    def diagnose_player(
        self, 
        player_id: int, 
        stat: str = None
    ) -> dict:
        """
        Get detailed diagnostics for a specific player.
        
        Args:
            player_id: Player's ID
            stat: Optional stat to filter by
            
        Returns:
            Dictionary with player-specific calibration info
        """
        player_preds = [
            p for p in self.recent_predictions 
            if p.get('player_id') == player_id
        ]
        
        if stat:
            player_preds = [p for p in player_preds if p.get('stat') == stat]
        
        if not player_preds:
            return {
                'player_id': player_id,
                'n_predictions': 0,
                'status': 'no_data'
            }
        
        df = pd.DataFrame(player_preds)
        
        return {
            'player_id': player_id,
            'n_predictions': len(df),
            'mean_bias': df['residual'].mean(),
            'mean_abs_error': df['abs_error'].mean(),
            'std_residual': df['residual'].std(),
            'current_bias': self.player_biases.get(player_id, {}).get(stat, 0) if stat else 0,
            'recent_predictions': df.tail(5)[['game_date', 'stat', 'predicted', 'actual', 'residual']].to_dict('records')
        }
    
    def get_confidence_adjustment(
        self,
        player_id: int,
        stat: str
    ) -> float:
        """
        Get confidence multiplier based on calibration data availability.
        
        Returns:
            Confidence multiplier (0.5 to 1.5)
        """
        n_observations = sum(
            1 for p in self.recent_predictions 
            if p.get('player_id') == player_id and p.get('stat') == stat
        )
        
        if n_observations >= 20:
            return 1.0
        elif n_observations >= 10:
            return 0.95
        elif n_observations >= 5:
            return 0.85
        elif n_observations >= 1:
            return 0.7
        else:
            return 0.5
    
    def reset_player_calibration(self, player_id: int):
        """Reset calibration for a specific player."""
        if player_id in self.player_biases:
            del self.player_biases[player_id]
            self._save_calibration_data()
            logger.info(f"Reset calibration for player {player_id}")
    
    def export_predictions_csv(self, filepath: str = None):
        """Export prediction history to CSV for analysis."""
        if filepath is None:
            filepath = os.path.join(self.cache_dir, 'prediction_history.csv')
        
        if self.recent_predictions:
            df = pd.DataFrame(self.recent_predictions)
            df.to_csv(filepath, index=False)
            logger.info(f"Exported {len(df)} predictions to {filepath}")
            return filepath
        return None


class PredictionConfidenceEstimator:
    """
    Estimates prediction confidence based on various factors.
    """
    
    def __init__(self, calibrator: ErrorCalibrator = None):
        self.calibrator = calibrator or ErrorCalibrator()
    
    def estimate_confidence(
        self,
        player_id: int,
        stat: str,
        player_history: pd.DataFrame = None,
        model_confidence: float = 0.5
    ) -> Tuple[float, str]:
        """
        Estimate confidence in a prediction.
        
        Args:
            player_id: Player ID
            stat: Stat type
            player_history: Historical data for player
            model_confidence: Base model confidence
            
        Returns:
            Tuple of (confidence_score, reasoning)
        """
        factors = []
        confidence = model_confidence
        
        calibration_conf = self.calibrator.get_confidence_adjustment(player_id, stat)
        confidence *= calibration_conf
        factors.append(f"calibration_data:{calibration_conf:.2f}")
        
        if player_history is not None and not player_history.empty:
            n_games = len(player_history)
            
            if n_games >= 20:
                confidence *= 1.1
                factors.append(f"history_games:{n_games}")
            elif n_games >= 10:
                confidence *= 1.0
            elif n_games >= 5:
                confidence *= 0.9
            else:
                confidence *= 0.8
                factors.append(f"limited_history:{n_games}")
            
            recent_std = player_history.tail(10)[stat].std() if stat in player_history.columns else 10
            cv = recent_std / player_history.tail(10)[stat].mean() if stat in player_history.columns and player_history.tail(10)[stat].mean() > 0 else 0.5
            
            if cv < 0.3:
                confidence *= 1.1
                factors.append(f"low_variance:{cv:.2f}")
            elif cv > 0.6:
                confidence *= 0.9
                factors.append(f"high_variance:{cv:.2f}")
        
        confidence = np.clip(confidence, 0.1, 1.0)
        
        reason = "; ".join(factors[:3])
        
        return float(confidence), reason
    
    def get_prediction_interval(
        self,
        prediction: float,
        player_id: int,
        stat: str,
        confidence: float = 0.9
    ) -> Tuple[float, float]:
        """
        Get prediction interval (lower, upper) for a given confidence level.
        
        Args:
            prediction: The predicted value
            player_id: Player ID
            stat: Stat type
            confidence: Desired confidence level (e.g., 0.9 for 90%)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        calib_conf = self.calibrator.get_confidence_adjustment(player_id, stat)
        
        recent_preds = [
            p for p in self.calibrator.recent_predictions
            if p.get('player_id') == player_id and p.get('stat') == stat
        ]
        
        if len(recent_preds) >= 5:
            residuals = [p['abs_error'] for p in recent_preds]
            base_std = np.std(residuals)
        else:
            base_std = prediction * 0.25
        
        adjusted_std = base_std / calib_conf
        
        z_score = 1.645 if confidence == 0.9 else 1.96
        
        margin = z_score * adjusted_std
        
        lower = max(0, prediction - margin)
        upper = prediction + margin
        
        return float(lower), float(upper)


class AdaptiveBlender:
    """
    Adaptively blends predictions based on recent performance.
    
    Tracks which model/signal has been performing best recently
    and weights accordingly.
    """
    
    def __init__(self, cache_dir: str = 'data/cache'):
        self.cache_dir = cache_dir
        self.model_performance: Dict[str, List[float]] = defaultdict(list)
        self.model_weights: Dict[str, float] = {}
        
        self._load_weights()
    
    def _load_weights(self):
        """Load saved model weights."""
        weights_file = os.path.join(self.cache_dir, 'model_weights.json')
        
        if os.path.exists(weights_file):
            try:
                with open(weights_file, 'r') as f:
                    data = json.load(f)
                    self.model_weights = data.get('weights', {})
            except (OSError, json.JSONDecodeError, ValueError) as e:
                logger.debug("Failed to load model weights from %s: %s", weights_file, e)
    
    def _save_weights(self):
        """Save model weights."""
        weights_file = os.path.join(self.cache_dir, 'model_weights.json')
        
        try:
            with open(weights_file, 'w') as f:
                json.dump({'weights': self.model_weights}, f, indent=2)
        except (OSError, TypeError, ValueError) as e:
            logger.debug("Failed to save model weights to %s: %s", weights_file, e)
    
    def record_model_error(
        self, 
        model_name: str, 
        error: float
    ):
        """Record a model's prediction error."""
        self.model_performance[model_name].append(error)
        
        self.model_performance[model_name] = self.model_performance[model_name][-50:]
    
    def get_blended_prediction(
        self,
        predictions: Dict[str, float],
        method: str = 'inverse_error'
    ) -> float:
        """
        Blend multiple model predictions.
        
        Args:
            predictions: {model_name: prediction, ...}
            method: 'inverse_error', 'softmax', or 'equal'
            
        Returns:
            Blended prediction
        """
        if not predictions:
            return 0
        
        if len(predictions) == 1:
            return list(predictions.values())[0]
        
        if method == 'equal':
            return np.mean(list(predictions.values()))
        
        elif method == 'inverse_error':
            weights = {}
            total_weight = 0
            
            for model_name, pred in predictions.items():
                errors = self.model_performance.get(model_name, [])
                
                if errors:
                    avg_error = np.mean(errors)
                    weight = 1.0 / (avg_error + 1e-6)
                else:
                    weight = 1.0
                
                weights[model_name] = weight
                total_weight += weight
            
            blended = 0
            for model_name, pred in predictions.items():
                blended += pred * (weights[model_name] / total_weight)
            
            return blended
        
        return np.mean(list(predictions.values()))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    calibrator = ErrorCalibrator()
    
    calibrator.record_prediction(
        player_name="LeBron James",
        player_id=2544,
        stat="PTS",
        predicted=25.0,
        actual=28.0,
        game_date=datetime.now().strftime('%Y-%m-%d')
    )
    
    calibrator.record_prediction(
        player_name="LeBron James",
        player_id=2544,
        stat="PTS",
        predicted=24.0,
        actual=22.0,
        game_date=datetime.now().strftime('%Y-%m-%d')
    )
    
    calibrated, std = calibrator.get_calibrated_prediction(
        player_id=2544,
        stat="PTS",
        raw_prediction=25.0,
        player_name="LeBron James"
    )
    
    print(f"\nCalibrated prediction: {calibrated:.1f} (std: {std:.1f})")
    print(f"\nCalibration report:")
    print(calibrator.get_calibration_report())