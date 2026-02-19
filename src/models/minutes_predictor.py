"""
Minutes Predictor - ML model for predicting player minutes.
Uses gradient boosting to predict minutes more accurately than simple rolling averages.
"""
import pandas as pd
import numpy as np
import logging
import os
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pickle

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("LightGBM not available, using fallback minutes prediction")


class MinutesPredictor:
    """
    ML-based minutes prediction model.
    
    Features:
    - Player-specific factors (form, fatigue, injury risk)
    - Game context (rest days, B2B, importance)
    - Matchup factors (opponent pace, size)
    - Coach tendencies
    
    Outputs:
    - Expected minutes with uncertainty (std)
    - Confidence score
    """
    
    FEATURE_COLS = [
        'ROLL_MIN_AVG_5', 'ROLL_MIN_AVG_10', 'ROLL_MIN_AVG_20',
        'ROLL_PTS_AVG_5', 'ROLL_PTS_AVG_10',
        'IS_HOME', 'IS_B2B', 'REST_DAYS', 'FATIGUE_SCORE',
        'OPP_PACE', 'OPP_DEF_RATING', 'GAME_IMPORTANCE',
        'COACH_ROTATION_TIGHTNESS', 'PLAYER_INJURY_RISK',
        'MINUTES_VARIANCE', 'STARTER_PROBABILITY'
    ]
    
    def __init__(self, models_dir: str = 'models', cache_dir: str = 'data/cache'):
        self.models_dir = models_dir
        self.cache_dir = cache_dir
        self.model = None
        self.feature_cols = self.FEATURE_COLS
        self.is_trained = False
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        self._load_model()
    
    def _load_model(self):
        """Load trained model if available."""
        model_path = os.path.join(self.models_dir, 'minutes_model.pkl')
        
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data.get('model')
                    self.feature_cols = data.get('feature_cols', self.FEATURE_COLS)
                    self.is_trained = self.model is not None
                logger.info("Loaded minutes prediction model")
            except Exception as e:
                logger.debug(f"Failed to load minutes model: {e}")
    
    def train(self, players_df: pd.DataFrame, games_df: pd.DataFrame):
        """
        Train the minutes prediction model.
        
        Args:
            players_df: Historical player game logs
            games_df: Historical game data
        """
        if not HAS_LIGHTGBM:
            logger.warning("Cannot train - LightGBM not available")
            return
        
        logger.info("Training minutes prediction model...")
        
        features_df = self._prepare_training_data(players_df, games_df)
        
        if features_df.empty or len(features_df) < 5000:
            logger.warning("Insufficient data for training minutes model")
            return
        
        train_mask = features_df['GAME_DATE'] < features_df['GAME_DATE'].max() - timedelta(days=30)
        
        X_train = features_df.loc[train_mask, self.feature_cols].fillna(0)
        y_train = features_df.loc[train_mask, 'ACTUAL_MINUTES']
        
        X_val = features_df.loc[~train_mask, self.feature_cols].fillna(0)
        y_val = features_df.loc[~train_mask, 'ACTUAL_MINUTES']
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 42
        }
        
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        self.feature_cols = self.feature_cols
        self.is_trained = True
        
        self._save_model()
        
        val_pred = self.model.predict(X_val)
        val_mae = np.mean(np.abs(val_pred - y_val))
        logger.info(f"Minutes model validation MAE: {val_mae:.2f} minutes")
    
    def _prepare_training_data(
        self, 
        players_df: pd.DataFrame, 
        games_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Prepare features for training."""
        
        df = players_df.copy()
        
        if 'GAME_DATE' in df.columns:
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        
        for col in ['ROLL_MIN_AVG_5', 'ROLL_MIN_AVG_10', 'ROLL_MIN_AVG_20']:
            if col not in df.columns:
                df[col] = df.groupby('PLAYER_ID')['MIN'].transform(
                    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
                )
        
        for col in ['ROLL_PTS_AVG_5', 'ROLL_PTS_AVG_10']:
            if col not in df.columns:
                df[col] = df.groupby('PLAYER_ID')['PTS'].transform(
                    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
                )
        
        if 'IS_HOME' not in df.columns and 'MATCHUP' in df.columns:
            df['IS_HOME'] = df['MATCHUP'].str.contains('vs.').astype(int)
        
        if 'IS_B2B' not in df.columns:
            df['DAYS_SINCE_LAST'] = df.groupby('PLAYER_ID')['GAME_DATE'].diff().dt.days
            df['IS_B2B'] = (df['DAYS_SINCE_LAST'] == 1).astype(int)
        
        if 'REST_DAYS' not in df.columns:
            df['REST_DAYS'] = df['DAYS_SINCE_LAST'].clip(1, 7).fillna(4)
        
        if 'FATIGUE_SCORE' not in df.columns:
            mins_lag = df.groupby('PLAYER_ID')['MIN'].shift(1).fillna(0)
            df['MINS_LAST_3'] = mins_lag.groupby(df['PLAYER_ID']).rolling(3, min_periods=1).sum()
            df['FATIGUE_SCORE'] = (df['MINS_LAST_3'] / 100).clip(0, 1)
        
        df['OPP_PACE'] = 100.0
        df['OPP_DEF_RATING'] = 114.0
        df['GAME_IMPORTANCE'] = 0.5
        df['COACH_ROTATION_TIGHTNESS'] = 0.5
        df['PLAYER_INJURY_RISK'] = 0.1
        df['MINUTES_VARIANCE'] = 15.0
        df['STARTER_PROBABILITY'] = 0.5
        
        df['ACTUAL_MINUTES'] = df['MIN'].fillna(0)
        
        return df
    
    def _save_model(self):
        """Save trained model to disk."""
        model_path = os.path.join(self.models_dir, 'minutes_model.pkl')
        
        try:
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'feature_cols': self.feature_cols
                }, f)
            logger.info(f"Saved minutes model to {model_path}")
        except Exception as e:
            logger.error(f"Failed to save minutes model: {e}")
    
    def predict_minutes(
        self,
        player_context: pd.Series,
        game_context: dict
    ) -> Tuple[float, float]:
        """
        Predict minutes for a player in a specific game.
        
        Args:
            player_context: Player's recent stats (Series with feature values)
            game_context: Game situation (home/away, opponent, rest, etc.)
            
        Returns:
            Tuple of (expected_minutes, std_uncertainty)
        """
        if self.model is None or not self.is_trained:
            return self._fallback_minutes_prediction(player_context, game_context)
        
        features = self._build_feature_vector(player_context, game_context)
        
        features_df = pd.DataFrame([features], columns=self.feature_cols).fillna(0)
        
        pred = self.model.predict(features_df)[0]
        
        std = self._estimate_uncertainty(features)
        
        pred = np.clip(pred, 0, 48)
        
        return float(pred), float(std)
    
    def _build_feature_vector(
        self, 
        player_context: pd.Series, 
        game_context: dict
    ) -> List[float]:
        """Build feature vector for prediction."""
        
        features = []
        
        for col in self.feature_cols:
            if col == 'ROLL_MIN_AVG_5':
                val = player_context.get('ROLL_MIN_AVG_5', player_context.get('MIN', 20))
            elif col == 'ROLL_MIN_AVG_10':
                val = player_context.get('ROLL_MIN_AVG_10', player_context.get('MIN', 20))
            elif col == 'ROLL_MIN_AVG_20':
                val = player_context.get('ROLL_MIN_AVG_20', player_context.get('MIN', 20))
            elif col == 'ROLL_PTS_AVG_5':
                val = player_context.get('ROLL_PTS_AVG_5', 10)
            elif col == 'ROLL_PTS_AVG_10':
                val = player_context.get('ROLL_PTS_AVG_10', 10)
            elif col == 'IS_HOME':
                val = game_context.get('is_home', 0.5)
            elif col == 'IS_B2B':
                val = game_context.get('is_b2b', 0)
            elif col == 'REST_DAYS':
                val = game_context.get('rest_days', 3)
            elif col == 'FATIGUE_SCORE':
                val = game_context.get('fatigue_score', 0.3)
            elif col == 'OPP_PACE':
                val = game_context.get('opp_pace', 100)
            elif col == 'OPP_DEF_RATING':
                val = game_context.get('opp_def_rating', 114)
            elif col == 'GAME_IMPORTANCE':
                val = game_context.get('game_importance', 0.5)
            elif col == 'COACH_ROTATION_TIGHTNESS':
                val = game_context.get('coach_rotation_tightness', 0.5)
            elif col == 'PLAYER_INJURY_RISK':
                val = game_context.get('injury_risk', 0.1)
            elif col == 'MINUTES_VARIANCE':
                val = player_context.get('ROLL_MIN_STD_10', 5)
            elif col == 'STARTER_PROBABILITY':
                val = game_context.get('starter_probability', 0.5)
            else:
                val = 0
            
            features.append(float(val) if val is not None else 0)
        
        return features
    
    def _estimate_uncertainty(self, features: List[float]) -> float:
        """Estimate prediction uncertainty based on feature values."""
        
        base_std = 4.0
        
        starter_prob_idx = self.feature_cols.index('STARTER_PROBABILITY')
        starter_prob = features[starter_prob_idx]
        
        if starter_prob > 0.8:
            base_std = 2.5
        elif starter_prob < 0.3:
            base_std = 8.0
        
        b2b_idx = self.feature_cols.index('IS_B2B')
        if features[b2b_idx] == 1:
            base_std += 1.5
        
        rest_idx = self.feature_cols.index('REST_DAYS')
        rest_days = features[rest_idx]
        if rest_days == 0 or rest_days > 5:
            base_std += 2.0
        
        return base_std
    
    def _fallback_minutes_prediction(
        self, 
        player_context: pd.Series, 
        game_context: dict
    ) -> Tuple[float, float]:
        """Fallback prediction when model not available."""
        
        base_mins = player_context.get('ROLL_MIN_AVG_10', player_context.get('MIN', 20))
        
        starter_prob = game_context.get('starter_probability', 0.5)
        
        if starter_prob > 0.7:
            base_mins = min(base_mins + 2, 38)
        elif starter_prob < 0.3:
            base_mins = max(base_mins - 5, 10)
        
        is_b2b = game_context.get('is_b2b', False)
        if is_b2b:
            base_mins = max(base_mins - 2, 15)
        
        rest_days = game_context.get('rest_days', 3)
        if rest_days == 0:
            base_mins = max(base_mins - 3, 12)
        elif rest_days >= 4:
            base_mins = min(base_mins + 2, 40)
        
        is_home = game_context.get('is_home', True)
        base_mins += 0.5 if is_home else -0.5
        
        std = 4.0 if starter_prob > 0.7 else 8.0
        
        return float(base_mins), float(std)
    
    def predict_rotation_minutes(
        self,
        roster: List[dict],
        game_context: dict
    ) -> Dict[str, Tuple[float, float]]:
        """
        Predict minutes for entire rotation.
        
        Args:
            roster: List of player dicts with context
            game_context: Game situation
            
        Returns:
            Dict mapping player_name -> (expected_mins, std)
        """
        results = {}
        
        for player in roster:
            context = pd.Series(player)
            
            mins, std = self.predict_minutes(context, game_context)
            
            results[player.get('name', player.get('id', 'unknown'))] = (mins, std)
        
        results = self._enforce_minutes_constraint(results)
        
        return results
    
    def _enforce_minutes_constraint(
        self, 
        predictions: Dict[str, Tuple[float, float]]
    ) -> Dict[str, Tuple[float, float]]:
        """Ensure total minutes sum to ~240."""
        
        total = sum(m for m, _ in predictions.values())
        
        if total <= 0:
            return predictions
        
        target = 240.0
        scale = target / total
        
        if abs(scale - 1.0) > 0.05:
            adjusted = {}
            for name, (mins, std) in predictions.items():
                new_mins = min(mins * scale, 42)
                adjusted[name] = (new_mins, std)
            return adjusted
        
        return predictions
    
    def get_coach_tendency(self, team_abbr: str) -> float:
        """
        Get coach's rotation tightness tendency (0-1).
        
        0 = plays 11+ players (loose rotation)
        1 = plays 7 players (tight rotation)
        """
        cache_file = os.path.join(self.cache_dir, 'coach_tendencies.json')
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    tendencies = json.load(f)
                    return tendencies.get(team_abbr.upper(), {}).get('rotation_tightness', 0.5)
            except Exception:
                pass
        
        coach_profiles = {
            'MIA': 0.75, 'BOS': 0.70, 'GSW': 0.65, 'DEN': 0.65,
            'LAL': 0.60, 'PHI': 0.55, 'PHX': 0.55, 'MIL': 0.60,
            'MEM': 0.50, 'NOP': 0.50, 'CLE': 0.55, 'DAL': 0.45,
            'LAC': 0.50, 'SAC': 0.45, 'IND': 0.45, 'MIN': 0.40,
            'CHI': 0.50, 'ATL': 0.40, 'BKN': 0.45, 'TOR': 0.50,
            'HOU': 0.35, 'OKC': 0.40, 'ORL': 0.40, 'WAS': 0.35,
            'DET': 0.35, 'CHA': 0.40, 'SAS': 0.30, 'UTA': 0.45,
            'POR': 0.35, 'NYK': 0.50
        }
        
        return coach_profiles.get(team_abbr.upper(), 0.5)
    
    def get_injury_risk_factor(
        self,
        player_id: int,
        games_this_month: int,
        avg_minutes: float,
        age: int = None
    ) -> float:
        """
        Calculate injury risk based on workload.
        
        Returns:
            Risk factor 0-1 (higher = more risk of reduced minutes)
        """
        risk = 0.1
        
        if games_this_month >= 12:
            risk += 0.2
        
        if avg_minutes >= 35:
            risk += 0.15
        elif avg_minutes >= 30:
            risk += 0.1
        
        if age and age >= 32:
            risk += 0.1
        elif age and age >= 28:
            risk += 0.05
        
        return min(risk, 1.0)


class MinutesPredictorSimple:
    """
    Simple fallback minutes predictor when ML model unavailable.
    Uses rule-based heuristics.
    """
    
    @staticmethod
    def predict(
        player_history: pd.DataFrame,
        game_context: dict
    ) -> Tuple[float, float]:
        """
        Predict minutes using simple heuristics.
        """
        if player_history.empty:
            return 20.0, 8.0
        
        recent = player_history.tail(10)
        
        base_mins = recent['MIN'].mean() if 'MIN' in recent.columns else 20
        
        if game_context.get('is_b2b', False):
            base_mins -= 2
        
        rest = game_context.get('rest_days', 3)
        if rest == 0:
            base_mins -= 3
        elif rest >= 4:
            base_mins += 1
        
        if game_context.get('is_home', True):
            base_mins += 0.5
        
        base_mins = np.clip(base_mins, 5, 42)
        
        std = recent['MIN'].std() if len(recent) > 1 and 'MIN' in recent.columns else 5
        
        return float(base_mins), float(std)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    predictor = MinutesPredictor()
    
    print("Minutes Predictor initialized")
    print(f"Model trained: {predictor.is_trained}")
    
    sample_context = pd.Series({
        'ROLL_MIN_AVG_5': 28,
        'ROLL_MIN_AVG_10': 27,
        'ROLL_MIN_AVG_20': 26,
        'ROLL_PTS_AVG_5': 18,
        'ROLL_PTS_AVG_10': 17,
    })
    
    game_ctx = {
        'is_home': True,
        'is_b2b': False,
        'rest_days': 3,
        'fatigue_score': 0.3,
        'opp_pace': 102,
        'opp_def_rating': 112,
        'starter_probability': 0.85
    }
    
    mins, std = predictor.predict_minutes(sample_context, game_ctx)
    print(f"\nPredicted minutes: {mins:.1f} ± {std:.1f}")