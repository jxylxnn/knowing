import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from scipy import stats

logger = logging.getLogger(__name__)


class GPURollingFeatures:
    """GPU-accelerated rolling feature calculations using PyTorch."""
    
    def __init__(self, windows: List[int] = None):
        self.windows = windows or [3, 5, 10, 20, 50]
        self.device = None
        self._initialized = False
    
    def _init_device(self):
        """Lazy initialization of GPU device."""
        if self._initialized:
            return self.device is not None
        
        self._initialized = True
        try:
            from src.models.gpu_utils import check_gpu_compatibility, get_device
            if check_gpu_compatibility():
                self.device = get_device()
                import torch
                logger.info(f"GPURollingFeatures initialized on {self.device}")
                return True
        except ImportError:
            pass
        
        self.device = None
        return False
    
    def compute_rolling_stats_per_player(
        self, 
        tensor_data: 'torch.Tensor', 
        player_indices: np.ndarray,
        window: int,
        stats: List[str] = None
    ) -> 'torch.Tensor':
        """
        Compute rolling statistics per player on GPU.
        
        Args:
            tensor_data: 1D tensor of values for all players
            player_indices: Array mapping each row to a player
            window: Rolling window size
            stats: Statistics to compute
        
        Returns:
            2D tensor of shape (n_samples, n_stats)
        """
        import torch
        
        if stats is None:
            stats = ['mean', 'std', 'min', 'max']
        
        n_samples = len(tensor_data)
        n_stats = len(stats)
        result = torch.zeros((n_samples, n_stats), dtype=torch.float32, device=self.device)
        
        # Get unique players and their boundaries
        unique_players, player_starts = np.unique(player_indices, return_index=True)
        player_order = np.argsort(player_starts)
        sorted_players = unique_players[player_order]
        sorted_starts = player_starts[player_order]
        
        # Add end boundary
        player_ends = np.concatenate([sorted_starts[1:], [n_samples]])
        
        for i, player_id in enumerate(sorted_players):
            start = sorted_starts[i]
            end = player_ends[i]
            player_data = tensor_data[start:end]
            
            if len(player_data) < 1:
                continue
            
            # Create rolling windows using unfold
            # Pad with NaN at start
            padded = torch.nn.functional.pad(player_data, (window - 1, 0), value=float('nan'))
            windows = padded.unfold(0, window, 1)
            
            col_idx = 0
            for stat in stats:
                if stat == 'mean':
                    result[start:end, col_idx] = windows.nanmean(dim=1)
                elif stat == 'std':
                    mean = windows.nanmean(dim=1, keepdim=True)
                    variance = ((windows - mean) ** 2).nanmean(dim=1)
                    result[start:end, col_idx] = torch.sqrt(variance)
                elif stat == 'min':
                    mins, _ = windows.nanmin(dim=1)
                    result[start:end, col_idx] = mins
                elif stat == 'max':
                    maxs, _ = windows.nanmax(dim=1)
                    result[start:end, col_idx] = maxs
                col_idx += 1
        
        return result
    
    def process_dataframe(
        self, 
        df: pd.DataFrame, 
        group_col: str,
        value_cols: List[str],
        windows: List[int] = None,
        stats: List[str] = None,
        shift: int = 1
    ) -> pd.DataFrame:
        """
        Process DataFrame to add GPU-accelerated rolling features.
        
        Args:
            df: Input DataFrame (must be sorted by group_col and date)
            group_col: Column to group by (e.g., 'PLAYER_ID')
            value_cols: Columns to compute rolling stats on
            windows: Window sizes (uses self.windows if None)
            stats: Statistics to compute ['mean', 'std', 'min', 'max']
            shift: Number of rows to shift before computing (for leakage prevention)
        
        Returns:
            DataFrame with added rolling feature columns
        """
        if not self._init_device():
            return None  # Signal that CPU should be used
        
        import torch
        
        windows = windows or self.windows
        stats = stats or ['mean', 'std']
        
        result_cols = {}
        player_indices = df[group_col].values
        
        logger.info(f"GPU Rolling: Computing stats for {len(value_cols)} columns, {len(windows)} windows")
        
        for col in value_cols:
            if col not in df.columns:
                continue
            
            # Shift to prevent leakage
            shifted_values = df[col].shift(shift).values
            
            # Convert to tensor
            col_tensor = torch.tensor(shifted_values, dtype=torch.float32, device=self.device)
            
            for window in windows:
                # Determine which stats to compute based on window size
                window_stats = ['mean']
                if window >= 5:
                    window_stats.append('std')
                if window >= 10:
                    window_stats.extend(['min', 'max'])
                
                # Compute on GPU
                stats_tensor = self.compute_rolling_stats_per_player(
                    col_tensor, player_indices, window, window_stats
                )
                
                # Move back to CPU and create columns
                stats_np = stats_tensor.cpu().numpy()
                
                for i, stat in enumerate(window_stats):
                    feat_name = f'ROLL_{col}_{stat.upper()}_{window}'
                    result_cols[feat_name] = stats_np[:, i]
        
        # Clear GPU memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Create result DataFrame
        result_df = pd.DataFrame(result_cols, index=df.index)
        
        logger.info(f"GPU Rolling: Added {len(result_cols)} columns")
        return result_df


class FeatureEngineer:
    """High-Performance Feature Engineering with GPU Acceleration Support."""
    
    def __init__(self, rolling_windows: List[int] = [3, 5, 10, 20, 50], use_gpu: bool = True):
        self.rolling_windows = rolling_windows
        self.target_cols = ['PTS', 'REB', 'AST']
        self.efficiency_cols = ['FGA', 'FGM', 'FTA', 'FTM', 'FG3M', 'FG3A', 'TOV', 'MIN']
        self.use_gpu = use_gpu
        self._gpu_roller = None
        
        if use_gpu:
            self._init_gpu()
    
    def _init_gpu(self):
        """Initialize GPU rolling feature calculator."""
        try:
            self._gpu_roller = GPURollingFeatures(windows=self.rolling_windows)
            if self._gpu_roller._init_device():
                logger.info("FeatureEngineer initialized with GPU acceleration")
            else:
                logger.info("GPU not available, using CPU for feature engineering")
                self._gpu_roller = None
        except Exception as e:
            logger.warning(f"GPU initialization failed: {e}. Using CPU.")
            self._gpu_roller = None

    def create_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Main entry point for feature engineering pipeline.
        Includes strict data leakage prevention by shifting all stats before use.
        """
        logger.info("--- Starting Smart Feature Engineering Pipeline ---")
        
        # Validate input
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to feature engineering")
            return pd.DataFrame()
        
        df = df.copy()
        
        # Validate required columns
        required_cols = ['PLAYER_ID', 'GAME_DATE']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return pd.DataFrame()
        
        # Critical: Sort and reset index to prevent leakage
        df = df.sort_values(['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)
        
        # Ensure proper datetime type
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')
        
        # Remove rows with invalid dates before proceeding
        df = df.dropna(subset=['GAME_DATE'])
        
        # Phase 1: Base Rolling Features (GPU-accelerated if available)
        logger.info("Phase 1/15: Creating Rolling Features...")
        if self._gpu_roller is not None and len(df) > 10000:
            # Use GPU for large datasets
            try:
                rolling_df = self._create_rolling_features_gpu(df)
                if rolling_df is not None:
                    df = pd.concat([df, rolling_df], axis=1)
                    logger.info("Used GPU-accelerated rolling features")
                else:
                    df = self._create_rolling_features_vectorized(df)
            except Exception as e:
                logger.warning(f"GPU rolling failed: {e}. Falling back to CPU.")
                df = self._create_rolling_features_vectorized(df)
        else:
            df = self._create_rolling_features_vectorized(df)
        
        # Phase 2: Efficiency
        logger.info("Phase 2/15: Creating Efficiency Features...")
        df = self._create_efficiency_features(df)
        
        # Phase 3: Momentum
        logger.info("Phase 3/15: Creating Momentum Features...")
        df = self._create_momentum_features(df)
        
        # Phase 4: Contextual
        logger.info("Phase 4/15: Creating Contextual Features...")
        df = self._create_contextual_features(df)
        
        # Phase 5: Matchup History
        logger.info("Phase 5/15: Creating Matchup History...")
        df = self._add_vs_team_history(df)
        
        # Phase 6: Advanced Scoring Features (NEW)
        logger.info("Phase 6/15: Adding Advanced Scoring & Rebound Features...")
        df = self._add_advanced_scoring_features(df)
        
        # Phase 7: Pace Adjusted
        logger.info("Phase 7/15: Creating Pace Adjusted Stats...")
        df = self._add_pace_adjusted_stats(df)
        
        # Phase 8: Teammate
        logger.info("Phase 8/15: Creating Teammate Features...")
        df = self._add_teammate_features(df)
        
        # Phase 9: Opponent Strength
        logger.info("Phase 9/15: Adding Opponent Strength...")
        df = self._add_opponent_strength(df)
        
        # Phase 9b: Defensive Matchup Features (NEW)
        logger.info("Phase 9b/15: Adding Defensive Matchup Features...")
        df = self._add_defensive_matchup_features(df)
        
        # Phase 10: Interactions
        logger.info("Phase 10/15: Adding Interaction Features...")
        df = self._add_interaction_features(df)
        
        # Phase 11: Bayesian
        logger.info("Phase 11/15: Adding Bayesian Estimates...")
        df = self._add_bayesian_estimates(df)
        
        # Phase 12: Seasonality
        logger.info("Phase 12/15: Adding Seasonality Features...")
        df = self._add_seasonality_features(df)
        
        # Phase 13: Advanced Custom
        logger.info("Phase 13/15: Adding Advanced Custom Features...")
        df = self._add_advanced_custom_features(df)
        
        # Phase 14: League Ranking (Percentiles) - Only for training to avoid leakage
        if is_training:
            logger.info("Phase 14/15: Adding League Percentile Rankings...")
            df = self._add_league_ranking_features(df)
        
        # Phase 15: Target Encoding - Only for training to avoid leakage
        if is_training:
            logger.info("Phase 15/15: Adding Target Encodings...")
            df = self._add_target_encodings(df)
        
        # Cleanup
        # Remove duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Remove rows with insufficient historical data (prevents leakage from insufficient context)
        if 'ROLL_PTS_AVG_5' in df.columns:
            df = df.dropna(subset=['ROLL_PTS_AVG_5'])
        
        # Clean up temporary columns
        tmp_cols = [c for c in df.columns if c.startswith('_tmp_')]
        if tmp_cols:
            logger.debug(f"Removing {len(tmp_cols)} temporary columns")
            df = df.drop(columns=tmp_cols)
        
        # Handle infinite values and NaN safely
        df = df.replace([np.inf, -np.inf], 0)
        df = df.fillna(0)
        
        logger.info(f"Feature engineering complete. Final shape: {df.shape}")
        return df

    def create_features_chunked(self, df: pd.DataFrame, chunk_size: int = 500_000):
        """Process features in chunks by player to avoid OOM errors.
        
        Chunks are split by PLAYER_ID so that each player's full history
        stays within a single chunk. This ensures rolling/expanding features
        are computed correctly.
        """
        if len(df) < chunk_size:
            return self.create_features(df)
        
        # Sort by player first to keep player games together
        df = df.sort_values(['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)
        
        # Split by player groups, not by row index
        player_ids = df['PLAYER_ID'].unique()
        chunks = []
        current_chunk_ids = []
        current_chunk_size = 0
        
        # Group player IDs into chunks that respect the size limit
        player_sizes = df.groupby('PLAYER_ID').size()
        for pid in player_ids:
            pid_size = player_sizes[pid]
            if current_chunk_size + pid_size > chunk_size and current_chunk_ids:
                # Process current chunk
                chunk_df = df[df['PLAYER_ID'].isin(current_chunk_ids)].copy()
                chunk_df = self.create_features(chunk_df, is_training=True)
                chunks.append(chunk_df)
                logger.info(f"Processed chunk {len(chunks)} ({current_chunk_size} rows, {len(current_chunk_ids)} players)")
                current_chunk_ids = []
                current_chunk_size = 0
            current_chunk_ids.append(pid)
            current_chunk_size += pid_size
        
        # Process final chunk
        if current_chunk_ids:
            chunk_df = df[df['PLAYER_ID'].isin(current_chunk_ids)].copy()
            chunk_df = self.create_features(chunk_df, is_training=True)
            chunks.append(chunk_df)
            logger.info(f"Processed chunk {len(chunks)} ({current_chunk_size} rows, {len(current_chunk_ids)} players)")
        
        return pd.concat(chunks, ignore_index=True)
    
    def _create_rolling_features_gpu(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """GPU-accelerated rolling feature computation."""
        if self._gpu_roller is None:
            return None
        
        stat_cols = self.target_cols + self.efficiency_cols
        valid_cols = [c for c in stat_cols if c in df.columns]
        
        try:
            return self._gpu_roller.process_dataframe(
                df=df,
                group_col='PLAYER_ID',
                value_cols=valid_cols,
                windows=self.rolling_windows,
                shift=1  # Prevent leakage
            )
        except Exception as e:
            logger.warning(f"GPU rolling computation failed: {e}")
            return None

    def _create_rolling_features_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates rolling features using fast vectorized operations.
        Avoids slow lambda functions and processes columns efficiently.
        """
        stat_cols = self.target_cols + self.efficiency_cols
        valid_cols = [c for c in stat_cols if c in df.columns]
        
        logger.info(f"Starting rolling feature generation for {len(valid_cols)} columns, {len(self.rolling_windows)} windows...")
        
        # Pre-shift all columns at once (avoids repeated groupby.shift calls)
        shifted = df.groupby('PLAYER_ID')[valid_cols].shift(1)
        
        total_windows = len(self.rolling_windows)
        for idx, window in enumerate(self.rolling_windows):
            logger.info(f"  Processing window {window} ({idx+1}/{total_windows})...")
            
            min_periods = max(1, window // 3)
            
            # Core aggregations - use only string names for speed
            # Pandas optimizes these with C/Cython code
            core_agg_funcs = ['mean']
            if window >= 5: 
                core_agg_funcs.append('std')
            if window >= 10: 
                core_agg_funcs.extend(['min', 'max'])
            
            try:
                # Create rolling object once
                rolled = shifted.groupby(df['PLAYER_ID']).rolling(
                    window=window, min_periods=min_periods
                )
                
                # Apply only fast string-based aggregations
                window_stats = rolled.agg(core_agg_funcs)
                
                # Flatten MultiIndex columns
                new_columns = []
                for col, func in window_stats.columns:
                    new_columns.append(f"ROLL_{col}_{func.upper()}_{window}")
                window_stats.columns = new_columns
                
                # Remove grouping index to align with original df
                window_stats = window_stats.reset_index(level=0, drop=True)
                
                # Merge efficiently
                df = pd.concat([df, window_stats], axis=1)
                
            except Exception as e:
                logger.warning(f"Rolling window {window} failed: {e}")
                continue

        # Post-processing for Range which require multi-column ops
        for window in [10, 20]:
            base_col = f'ROLL_PTS_MIN_{window}'
            if base_col in df.columns:
                for col in valid_cols:
                    min_c = f'ROLL_{col}_MIN_{window}'
                    max_c = f'ROLL_{col}_MAX_{window}'
                    if min_c in df.columns and max_c in df.columns:
                        df[f'ROLL_{col}_RANGE_{window}'] = df[max_c] - df[min_c]
        
        logger.info(f"Rolling features complete. Added {len([c for c in df.columns if c.startswith('ROLL_')])} columns.")
        return df

    def _create_efficiency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized efficiency metrics using pre-shifted rolling sums."""
        new_cols = {}
        
        eff_cols = ['FGA', 'FTA', 'TOV', 'PTS', 'FGM', 'FG3M', 'FG3A', 'AST', 'MIN'] + [
            s for s in self.target_cols if s not in ['FGA', 'FTA', 'TOV', 'PTS', 'FGM', 'FG3M', 'FG3A', 'AST', 'MIN']
        ]
        valid_eff = [c for c in eff_cols if c in df.columns]
        
        shifted = df.groupby('PLAYER_ID')[valid_eff].shift(1)
        
        for window in [5, 10, 20]:
            rolled = shifted.groupby(df['PLAYER_ID']).rolling(window, min_periods=1).sum()
            rolled = rolled.reset_index(level=0, drop=True)
            
            fg_sum = rolled['FGA'].fillna(0) if 'FGA' in rolled.columns else 0
            ft_sum = rolled['FTA'].fillna(0) if 'FTA' in rolled.columns else 0
            tov_sum = rolled['TOV'].fillna(0) if 'TOV' in rolled.columns else 0
            pts_sum = rolled['PTS'].fillna(0) if 'PTS' in rolled.columns else 0
            fgm_sum = rolled['FGM'].fillna(0) if 'FGM' in rolled.columns else 0
            fg3m_sum = rolled['FG3M'].fillna(0) if 'FG3M' in rolled.columns else 0
            fg3a_sum = rolled['FG3A'].fillna(0) if 'FG3A' in rolled.columns else 0
            ast_sum = rolled['AST'].fillna(0) if 'AST' in rolled.columns else 0
            mins_sum = rolled['MIN'].fillna(1) if 'MIN' in rolled.columns else 1

            new_cols[f'ROLL_TS_PCT_{window}'] = pts_sum / (2 * (fg_sum + 0.44 * ft_sum + 1e-6))
            new_cols[f'ROLL_EFG_PCT_{window}'] = (fgm_sum + 0.5 * fg3m_sum) / (fg_sum + 1e-6)
            new_cols[f'ROLL_3PT_PCT_{window}'] = fg3m_sum / (fg3a_sum + 1e-6)
            new_cols[f'ROLL_AST_TOV_{window}'] = ast_sum / (tov_sum + 1e-6)
            
            for stat in self.target_cols:
                if stat in rolled.columns:
                    stat_sum = rolled[stat].fillna(0)
                    new_cols[f'ROLL_{stat}_PER_MIN_{window}'] = stat_sum / (mins_sum + 1e-6)

        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    def _create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momentum features using EWMA and expanding windows."""
        new_cols = {}
        
        for stat in self.target_cols:
            for span in [3, 5, 10, 20]:
                new_cols[f'{stat}_EWMA_{span}'] = df.groupby('PLAYER_ID')[stat].transform(
                    lambda x: x.shift(1).ewm(span=span, adjust=False).mean()
                )
            
            # Expanding mean (Vectorized)
            season_avg = df.groupby('PLAYER_ID')[stat].transform(
                lambda x: x.shift(1).expanding().mean()
            )
            new_cols[f'{stat}_SEASON_AVG'] = season_avg
            
            # Trend
            for short, long in [(3, 10), (5, 20)]:
                short_avg = df.groupby('PLAYER_ID')[stat].transform(lambda x: x.shift(1).rolling(short, min_periods=1).mean())
                long_avg = df.groupby('PLAYER_ID')[stat].transform(lambda x: x.shift(1).rolling(long, min_periods=long//3).mean())
                new_cols[f'{stat}_TREND_{short}_{long}'] = short_avg - long_avg

            # Streaks
            roll_3 = new_cols.get(f'{stat}_EWMA_3', df.get(f'ROLL_{stat}_AVG_3'))
            roll_10 = new_cols.get(f'{stat}_EWMA_10', df.get(f'ROLL_{stat}_AVG_10'))
            if roll_3 is not None and roll_10 is not None:
                new_cols[f'{stat}_HOT_STREAK'] = (roll_3 > roll_10 * 1.15).astype(int)
                new_cols[f'{stat}_COLD_STREAK'] = (roll_3 < roll_10 * 0.85).astype(int)
                
        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    def _create_contextual_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Context-aware features."""
        new_cols = {}
        
        if 'MATCHUP' in df.columns:
            new_cols['IS_HOME'] = df['MATCHUP'].str.contains('vs.').astype(int)
        
        new_cols['DAYS_SINCE_LAST'] = df.groupby('PLAYER_ID')['GAME_DATE'].diff().dt.days.fillna(4)
        new_cols['REST_DAYS'] = new_cols['DAYS_SINCE_LAST'].clip(0, 7)
        new_cols['IS_B2B'] = (new_cols['DAYS_SINCE_LAST'] == 1).astype(int)
        
        # Fatigue
        mins_lag = df.groupby('PLAYER_ID')['MIN'].shift(1).fillna(0)
        new_cols['MINS_LAST_3'] = mins_lag.groupby(df['PLAYER_ID']).rolling(3, min_periods=1).sum().reset_index(level=0, drop=True)
        new_cols['MINS_LAST_7'] = mins_lag.groupby(df['PLAYER_ID']).rolling(7, min_periods=1).sum().reset_index(level=0, drop=True)
        
        new_cols['FATIGUE_SCORE'] = (
            (new_cols['MINS_LAST_3'] / 100) * 0.4 +
            (new_cols['IS_B2B'] * 0.3) +
            ((4 - new_cols['REST_DAYS'].clip(0, 4)) * 0.3)
        )
        
        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    def _add_vs_team_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimized Matchup History.
        Replaces lambda x: x.shift(1).expanding().mean() with direct groupby operations.
        """
        if 'OPPONENT_ID' not in df.columns:
            return df
            
        new_cols = {}
        groups = df.groupby(['PLAYER_ID', 'OPPONENT_ID'])
        
        for stat in self.target_cols:
            # Career average vs opponent
            # Shift first to avoid leakage
            shifted = groups[stat].shift(1)
            
            # Expanding mean using direct method.
            expanded_mean = shifted.groupby([df['PLAYER_ID'], df['OPPONENT_ID']]).expanding().mean()
            expanded_mean = expanded_mean.reset_index(level=[0, 1], drop=True)
            
            new_cols[f'VS_OPP_{stat}_AVG'] = expanded_mean
            
            # Recent average vs opponent (last 5) - Rolling
            recent_mean = shifted.groupby([df['PLAYER_ID'], df['OPPONENT_ID']]).rolling(5, min_periods=1).mean()
            recent_mean = recent_mean.reset_index(level=[0, 1], drop=True)
            
            new_cols[f'VS_OPP_{stat}_RECENT'] = recent_mean

        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    def _add_pace_adjusted_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pace-adjusted statistics."""
        # Placeholder logic if team stats missing
        team_cols = ['FGA_TEAM', 'FTA_TEAM', 'OREB_TEAM', 'OPP_DREB', 'TOV_TEAM']
        for col in team_cols:
            if col not in df.columns:
                df[col] = 85 if 'FGA' in col else 20 if 'FTA' in col else 10
        
        fgm_team = df.get('FGM_TEAM', df['FGA_TEAM'] * 0.45)
        df['EST_POSS'] = 0.5 * (
            df['FGA_TEAM'] + 0.4 * df['FTA_TEAM'] - 
            1.07 * (df['OREB_TEAM'] / (df['OREB_TEAM'] + df['OPP_DREB'] + 1e-6)) * 
            (df['FGA_TEAM'] - fgm_team) + df['TOV_TEAM']
        )
        
        df['TEAM_PACE_10'] = df.groupby('TEAM_ID')['EST_POSS'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=3).mean()
        ).fillna(100)
        
        league_avg_pace = 100
        df['PACE_FACTOR'] = (df['TEAM_PACE_10'] / league_avg_pace).clip(0.8, 1.2)
        
        return df

    def _add_teammate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Teammate context features.
        
        Computes features capturing team-level context for each player:
        - How dominant is this player relative to their team (share of team production)
        - Rolling team performance (shifted to prevent leakage)
        """
        new_cols = {}
        
        # Default: no star teammate out info available
        new_cols['STAR_TEAMMATE_OUT'] = 0
        
        # Team-level rolling stats (shifted to prevent leakage)
        if 'TEAM_ID' in df.columns:
            for stat in self.target_cols:
                team_col = f'{stat}_TEAM'
                if team_col in df.columns:
                    # Rolling team average for context (shifted by 1 to prevent leakage)
                    team_avg = df.groupby('TEAM_ID')[team_col].transform(
                        lambda x: x.shift(1).rolling(10, min_periods=3).mean()
                    )
                    # Player's share of team production (shifted)
                    player_shifted = df.groupby('PLAYER_ID')[stat].shift(1)
                    team_shifted = df.groupby('TEAM_ID')[team_col].transform(
                        lambda x: x.shift(1)
                    )
                    player_share = (player_shifted / (team_shifted + 1e-6)).clip(0, 1)
                    
                    new_cols[f'TEAM_{stat}_AVG_10'] = team_avg
                    new_cols[f'PLAYER_{stat}_SHARE'] = player_share.groupby(
                        df['PLAYER_ID']
                    ).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
        
        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    def _add_opponent_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """Opponent defensive strength."""
        new_cols = {}
        league_avgs = {'PTS': 105, 'REB': 42, 'AST': 22}
        
        for stat in self.target_cols:
            opp_col = f'OPP_DEF_{stat}_ALLOWED'
            if opp_col in df.columns:
                new_cols[f'RELATIVE_OPP_DEF_{stat}'] = df[opp_col] / league_avgs[stat]
            else:
                new_cols[f'RELATIVE_OPP_DEF_{stat}'] = 1.0
                new_cols[f'ROLL_OPP_DEF_{stat}_10'] = 1.0
                
        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    def _add_defensive_matchup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        NEW: Defensive Matchup Features.
        These features capture how players perform against specific defensive opponents.
        
        Key features added:
        1. DEF_MATCHUP_IMPACT - Estimated impact of opponent defense on player production
        2. OPP_DEF_RATING - Opponent defensive rating (points allowed per 100 possessions)
        3. PLAYER_DEF_OVERAGE - How much better/worse player does against elite defenses
        4. DEF_MATCHUP_TREND - Rolling trend of performance against tough defenses
        5. HOME_AWAY_VS_DEF - Home court advantage adjustment based on opponent defense
        """
        new_cols = {}
        
        # Get league average defensive metrics
        league_defensive_ratings = {
            'PTS': 105,  # League average points allowed
            'REB': 42,   # League average rebounds allowed
            'AST': 22,   # League average assists allowed
        }
        
        # 1. DEF_MATCHUP_IMPACT (Defense vs Player Performance Interaction)
        # Measures how much the opponent's defense affects the player's expected production
        for stat in self.target_cols:
            # Get opponent's defensive stats for this game
            opp_def_col = f'OPP_DEF_{stat}_ALLOWED'
            if opp_def_col in df.columns:
                # Normalize opponent defense (1.0 = league average)
                opp_def_normalized = df[opp_def_col] / league_defensive_ratings.get(stat, 105)
                
                # Defensive Impact: lower is better (defense allows fewer points)
                # If opponent allows 120 PTS (above avg), this factor will be > 1 (bad for player)
                # If opponent allows 90 PTS (below avg), this factor will be < 1 (good for player)
                new_cols[f'DEF_MATCHUP_{stat}_IMPACT'] = 2 - opp_def_normalized
                
                # Rolling version: opponent's defense over past 5 games
                opp_def_rolling = df.groupby('OPPONENT_ID')[opp_def_col].transform(
                    lambda x: x.shift(1).rolling(5, min_periods=2).mean()
                )
                new_cols[f'ROLL_OPP_DEF_{stat}_5'] = opp_def_rolling / league_defensive_ratings.get(stat, 105)
        
        # 2. OPP_DEF_RATING - Combined defensive difficulty metric
        # Weighted average of defense against PTS, REB, AST
        def_scores = []
        for stat, default_val in league_defensive_ratings.items():
            col = f'OPP_DEF_{stat}_ALLOWED'
            if col in df.columns:
                def_scores.append(df[col] / default_val)
        
        if def_scores:
            # Lower score = harder defense (allows fewer stats)
            combined_def = pd.concat(def_scores, axis=1).mean(axis=1)
            new_cols['OPP_DEF_RATING'] = combined_def
            new_cols['DEF_DIFFICULTY'] = (2 - combined_def)  # Higher = easier opponent
        else:
            new_cols['OPP_DEF_RATING'] = 1.0
            new_cols['DEF_DIFFICULTY'] = 1.0
        
        # 3. PLAYER_DEF_OVERAGE - How player performs vs elite defenses
        # Compare player's performance against top-5 defenses vs their season average
        if 'OPP_DEF_RATING' in new_cols:
            # Identify elite defenses (top 20% defense - lowest scores)
            elite_def_mask = new_cols['OPP_DEF_RATING'] < new_cols['OPP_DEF_RATING'].quantile(0.20)
            
            for stat in self.target_cols:
                # Player's average against elite defenses (if any)
                # Using conditional logic to compute over/under expectation
                season_avg = df.groupby('PLAYER_ID')[stat].transform(
                    lambda x: x.shift(1).expanding().mean()
                )
                
                # Defensive overage: how much above/below season avg when facing elite defense
                # This is computed at game level for historical data
                if f'VS_OPP_{stat}_AVG' in df.columns:
                    # Compare vs opponent average to season average
                    vs_opp = df[f'VS_OPP_{stat}_AVG']
                    overage = (vs_opp / (season_avg + 1e-6)) - 1
                    new_cols[f'DEF_OVERAGE_{stat}'] = overage
                else:
                    new_cols[f'DEF_OVERAGE_{stat}'] = 0.0
        
        # 4. DEF_MATCHUP_TREND - Rolling trend of performance against tough defenses
        for stat in self.target_cols:
            # If we have opponent defense info, compute player's rolling trend
            if 'OPP_DEF_RATING' in new_cols:
                # Interaction: player's recent production vs opponent defense quality
                recent_prod = df.groupby('PLAYER_ID')[stat].transform(
                    lambda x: x.shift(1).rolling(10, min_periods=3).mean()
                )
                # Defensive trend: how does player perform as opponent defenses get tougher?
                # Higher value = player performs better against tough defenses
                def_trend = recent_prod / (new_cols['OPP_DEF_RATING'] + 1e-6)
                new_cols[f'DEF_MATCHUP_TREND_{stat}'] = def_trend
        
        # 5. HOME_AWAY_VS_DEF - Home court advantage adjustment based on opponent defense
        if 'IS_HOME' in df.columns and 'OPP_DEF_RATING' in new_cols:
            # Home court can offset tough defenses
            home_advantage = 1.05  # ~5% boost at home
            tough_def_penalty = new_cols['OPP_DEF_RATING'] * 0.05  # Penalty for tough defense
            
            # Combined effect: home vs away adjustment
            new_cols['DEF_MATCHUP_HOME_ADJ'] = (
                home_advantage - (1 - df['IS_HOME']) * tough_def_penalty
            )
            new_cols['DEF_MATCHUP_AWAY_ADJ'] = (
                1.0 - df['IS_HOME'] * tough_def_penalty
            )
        
        # 6. QUALITY_DEFENSE_AVOIDANCE - How well player avoids top defenses
        # Based on opponent's defensive ranking vs league
        if 'OPP_DEF_RATING' in new_cols:
            # Rank of opponent defense (lower is better)
            opponent_rank = new_cols['OPP_DEF_RATING'].groupby(df['GAME_DATE']).rank(pct=True)
            new_cols['OPP_DEF_RANK'] = opponent_rank
            # Avoidance score: how much player avoids top defenses (if schedule allows)
            new_cols['QUALITY_DEF_AVOIDANCE'] = 1 - opponent_rank
        
        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Updated with smarter interactions."""
        new_cols = {}
        
        # 1. Standard Potential
        for stat in self.target_cols:
            roll_col = f'ROLL_{stat}_AVG_10'
            opp_col = f'RELATIVE_OPP_DEF_{stat}'
            if roll_col in df.columns and opp_col in df.columns:
                new_cols[f'{stat}_POTENTIAL'] = df[roll_col] * df[opp_col]
        
        # 2. NEW: Rest-Adjusted Production
        # Interaction: How does this player perform on specific rest days?
        if 'REST_DAYS' in df.columns and 'ROLL_PTS_AVG_10' in df.columns:
            # Create a bucket for rest
            df['REST_BUCKET'] = pd.cut(df['REST_DAYS'], bins=[-1, 1, 3, 7, 100], labels=[0, 1, 2, 3])
            
            # We can't easily calculate historical average per bucket without heavy groupby,
            # but we can create an interaction feature for the current state
            # Example: Rest 0 (B2B) multiplier
            b2b_mask = (df['REST_DAYS'] <= 1).astype(int)
            new_cols['B2B_IMPACT_FACTOR'] = df['ROLL_PTS_AVG_10'] * b2b_mask * 0.9 # Assume 10% hit on B2B
            
        if 'IS_B2B' in df.columns:
            new_cols['B2B_MIN_IMPACT'] = df['IS_B2B'] * df.get('ROLL_MIN_AVG_10', 0)
            
        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    def _add_bayesian_estimates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bayesian shrinkage estimates (Optimized)."""
        new_cols = {}
        
        for stat in self.target_cols:
            # Global stats
            global_mean = df[stat].mean()
            
            # Player stats (Optimized: no lambda for expanding)
            player_means = df.groupby('PLAYER_ID')[stat].transform(lambda x: x.shift(1).expanding().mean())
            # Count games per player
            player_counts = df.groupby('PLAYER_ID').cumcount() + 1
            
            # Shrinkage
            shrinkage = player_counts / (player_counts + 10)
            new_cols[f'{stat}_BAYESIAN'] = (
                shrinkage * player_means + (1 - shrinkage) * global_mean
            )
            
        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    def _add_seasonality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fourier features."""
        new_cols = {}
        day_of_year = df['GAME_DATE'].dt.dayofyear
        for k in [1, 2]:
            new_cols[f'SEASON_SIN_{k}'] = np.sin(2 * np.pi * k * day_of_year / 365)
            new_cols[f'SEASON_COS_{k}'] = np.cos(2 * np.pi * k * day_of_year / 365)
        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    def _add_advanced_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds 5 high-signal features: Usage Rate, Pace-Adjusted Usage, 
        Strength of Schedule, Efficiency Z-Score, and Fantasy Points.
        """
        new_cols = {}
        
        # 1. & 2. Usage and Pace-Adjusted Usage
        team_cols = ['FGA_TEAM', 'FTA_TEAM', 'TOV_TEAM']
        if all(c in df.columns for c in team_cols):
            player_poss = df['FGA'] + 0.44 * df['FTA'] + df['TOV']
            team_poss = df['FGA_TEAM'] + 0.44 * df['FTA_TEAM'] + df['TOV_TEAM']
            safe_min = df['MIN'].replace(0, 1)
            mins_factor = 48 / safe_min
            raw_usage = (player_poss / (team_poss + 1e-6)) * mins_factor
            raw_usage = raw_usage.clip(0, 0.40)
            
            usg_rolled = raw_usage.groupby(df['PLAYER_ID']).transform(
                lambda x: x.shift(1).rolling(10, min_periods=3).mean()
            )
            new_cols['ROLL_USG_PCT_10'] = usg_rolled
            if 'PACE_FACTOR' in df.columns:
                new_cols['PACE_ADJ_USAGE'] = usg_rolled * df['PACE_FACTOR']
                
        # 3. SOS_OPP_DEF_5 (Strength of Schedule)
        if 'OPPONENT_ID' in df.columns and 'TEAM_DEF_OPP_PTS_ALLOWED_ROLL_10' in df.columns:
            # Map Team ID -> Their Defensive Rating for vectorized lookup
            team_def_map = df.groupby('TEAM_ID')['TEAM_DEF_OPP_PTS_ALLOWED_ROLL_10'].last().to_dict()
            temp_opp_def = df['OPPONENT_ID'].map(team_def_map).fillna(105)
            
            new_cols['SOS_OPP_DEF_5'] = temp_opp_def.groupby(df['PLAYER_ID']).transform(
                lambda x: x.shift(1).rolling(5, min_periods=2).mean()
            )
            
        # 4. EFF_Z_SCORE (Hot Hand / Slump Detector)
        if 'PTS' in df.columns:
            player_mean = df.groupby('PLAYER_ID')['PTS'].transform(lambda x: x.shift(1).expanding().mean())
            player_std = df.groupby('PLAYER_ID')['PTS'].transform(lambda x: x.shift(1).expanding().std())
            recent_avg = df.groupby('PLAYER_ID')['PTS'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
            safe_std = player_std.fillna(player_std.mean()).replace(0, 1)
            new_cols['EFF_Z_SCORE'] = (recent_avg - player_mean) / (safe_std + 1e-6)

        # 5. ROLL_FANTASY_PTS_10
        if 'FANTASY_PTS' in df.columns:
            new_cols['ROLL_FANTASY_PTS_10'] = df.groupby('PLAYER_ID')['FANTASY_PTS'].transform(
                lambda x: x.shift(1).rolling(10, min_periods=3).mean()
            )

        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    def _add_league_ranking_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        NEW: Converts raw stats to Percentile Ranks (0-1).
        This allows the model to understand that 20 PTS in 2015 is 'Elite' 
        but 20 PTS in 2023 might be 'Average'.
        """
        new_cols = {}
        
        # We calculate a rolling rank over the last 30 days globally to simulate "current league landscape"
        df_sorted = df.sort_values('GAME_DATE')
        
        # Rolling window for league context (e.g., last 100 games league-wide)
        window_size = 2000 
        
        for stat in self.target_cols:
            # Calculate percentile rank within a rolling window of games
            pct_rank = df_sorted[stat].rolling(window=window_size, min_periods=500).apply(
                lambda x: (x.iloc[-1] >= x).mean(), raw=False
            )
            
            # Align index back to df
            new_cols[f'LEAGUE_PCT_{stat}'] = pct_rank.reindex(df.index)
            
        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


    def _add_target_encodings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Target encoding (Optimized).
        Replaces lambda x: x.shift(1).expanding().mean() with direct groupby operations.
        """
        new_cols = {}
        
        for stat in self.target_cols:
            global_mean = df[stat].mean()
            smoothing = 20
            
            # Player Target Encoding
            # Direct expanding mean is faster than lambda
            player_expanding = df.groupby('PLAYER_ID')[stat].transform(lambda x: x.shift(1).expanding().mean())
            player_counts = df.groupby('PLAYER_ID').cumcount()
            
            # Shrinkage weight
            shrinkage_weight = player_counts / (player_counts + smoothing)
            new_cols[f'{stat}_PLAYER_TE'] = (
                shrinkage_weight * player_expanding + 
                (1 - shrinkage_weight) * global_mean
            ).fillna(global_mean)
            
            # Team Target Encoding
            if 'TEAM_ID' in df.columns:
                team_expanding = df.groupby('TEAM_ID')[stat].transform(lambda x: x.shift(1).expanding().mean())
                team_counts = df.groupby('TEAM_ID').cumcount()
                shrinkage_weight = team_counts / (team_counts + smoothing)
                new_cols[f'{stat}_TEAM_TE'] = (
                    shrinkage_weight * team_expanding + 
                    (1 - shrinkage_weight) * global_mean
                ).fillna(global_mean)

        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    def _add_advanced_scoring_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced features specifically tuned for PTS and REB prediction.
        Adds Usage Rate, Opportunity Metrics, and Shot Profiles.
        """
        new_cols = {}
        
        # --- 1. USAGE RATE (The Alpha Feature for PTS) ---
        # Formula: 100 * ((FGA + 0.44 * FTA + TOV) * (Team_Mins / (5 * Mins))) / (Team_FGA + 0.44 * Team_FTA + Team_TOV)
        player_poss = df['FGA'] + 0.44 * df['FTA'] + df['TOV']
        
        team_poss = df.get('FGA_TEAM', 0) + 0.44 * df.get('FTA_TEAM', 0) + df.get('TOV_TEAM', 0)
        team_poss = team_poss.replace(0, 100) # Baseline if missing
        
        mins_factor = (df['MIN'] / 48.0).replace(0, 1) # Standard 48 min game
        
        # Store temporary column for transform
        df['_tmp_usage'] = (player_poss / team_poss) * (1 / mins_factor)
        df['_tmp_usage'] = df['_tmp_usage'].clip(0, 0.50).fillna(0)
        
        for window in [5, 10]:
            new_cols[f'ROLL_USG_PCT_{window}'] = df.groupby('PLAYER_ID')['_tmp_usage'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

        # --- 2. OPPONENT MISSES (The Alpha Feature for REB) ---
        if 'OPP_FGA_ALLOWED' in df.columns and 'OPP_FGM_ALLOWED' in df.columns:
            opp_fg_pct = (df['OPP_FGM_ALLOWED'] / (df['OPP_FGA_ALLOWED'] + 1e-6)).clip(0.3, 0.7)
            df['_tmp_reb_opp'] = 1.0 - opp_fg_pct
            
            for window in [5, 10]:
                new_cols[f'ROLL_REB_OPPORTUNITY_{window}'] = df.groupby('PLAYER_ID')['_tmp_reb_opp'].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )

        # --- 3. SHOT PROFILE (For PTS Variance & Ceiling) ---
        if 'FG3A' in df.columns and 'FGA' in df.columns:
            df['_tmp_3pt_freq'] = (df['FG3A'] / (df['FGA'] + 1e-6)).clip(0, 1)
            for window in [10, 20]:
                new_cols[f'ROLL_3PT_FREQ_{window}'] = df.groupby('PLAYER_ID')['_tmp_3pt_freq'].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )
        
        if 'FTA' in df.columns and 'FGA' in df.columns:
            df['_tmp_ft_rate'] = (df['FTA'] / (df['FGA'] + 1e-6)).clip(0, 1)
            for window in [10]:
                new_cols[f'ROLL_FT_RATE_{window}'] = df.groupby('PLAYER_ID')['_tmp_ft_rate'].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )

        # --- 4. ROLE / SHARE FEATURES ---
        if 'PTS_TEAM' in df.columns:
            df['_tmp_pts_share'] = (df['PTS'] / (df['PTS_TEAM'] + 1e-6)).clip(0, 0.6)
            for window in [10]:
                new_cols[f'ROLL_PTS_SHARE_{window}'] = df.groupby('PLAYER_ID')['_tmp_pts_share'].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )

        # --- 5. EFFICIENCY MOMENTUM ---
        ts_fga = df['FGA'] + 0.44 * df['FTA']
        df['_tmp_ts_pct'] = (df['PTS'] / (2 * ts_fga + 1e-6)).clip(0.3, 0.8)
        
        for window in [5, 10]:
            new_cols[f'ROLL_TS_PCT_MOMENTUM_{window}'] = df.groupby('PLAYER_ID')['_tmp_ts_pct'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

        # Clean up temporary columns
        tmp_cols = [c for c in df.columns if c.startswith('_tmp_')]
        df.drop(columns=tmp_cols, inplace=True)

        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)