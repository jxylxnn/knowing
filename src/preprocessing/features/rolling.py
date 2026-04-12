"""Rolling, efficiency, and momentum feature groups."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from src.preprocessing.features.base import (
    FeatureContext,
    FeatureDiagnostics,
    FeatureGroup,
    fill_series_with_prior,
)


def _concat_new_columns(df: pd.DataFrame, new_columns: dict[str, pd.Series]) -> pd.DataFrame:
    """Append a batch of aligned columns in a single concat."""
    if not new_columns:
        return df
    new_df = pd.DataFrame(new_columns, index=df.index)
    return pd.concat([df, new_df], axis=1)


class RollingFeatureGroup(FeatureGroup):
    """Past-only rolling averages and volatility features."""

    def __init__(
        self,
        windows: Optional[List[int]] = None,
        target_cols: Optional[List[str]] = None,
        efficiency_cols: Optional[List[str]] = None,
    ):
        self.windows = windows or [3, 5, 10, 20, 50]
        self.target_cols = target_cols or ['PTS', 'REB', 'AST']
        self.efficiency_cols = efficiency_cols or ['FGA', 'FGM', 'FTA', 'FTM', 'FG3M', 'FG3A', 'TOV', 'MIN']

    @property
    def name(self) -> str:
        return 'rolling'

    @property
    def required_columns(self) -> List[str]:
        return ['PLAYER_ID', 'GAME_DATE']

    def create(
        self,
        df: pd.DataFrame,
        *,
        diagnostics: Optional[FeatureDiagnostics] = None,
        context: Optional[FeatureContext] = None,
    ) -> pd.DataFrame:
        self._check_columns(df, diagnostics)
        df = df.copy()
        context = context or FeatureContext()

        stat_cols = [c for c in self.target_cols + self.efficiency_cols if c in df.columns]
        if not stat_cols:
            return df

        new_columns: dict[str, pd.Series] = {}
        for col in stat_cols:
            prior = float(context.league_priors.get(col, 0.0))
            shifted = df.groupby('PLAYER_ID')[col].shift(1)
            for window in self.windows:
                rolling_mean = shifted.groupby(df['PLAYER_ID']).transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                rolling_std = shifted.groupby(df['PLAYER_ID']).transform(
                    lambda x: x.rolling(window, min_periods=2).std()
                )
                history_count = shifted.groupby(df['PLAYER_ID']).transform(
                    lambda x: x.rolling(window, min_periods=1).count()
                )

                mean_col = f'ROLL_{col}_AVG_{window}'
                std_col = f'ROLL_{col}_STD_{window}'
                hist_col = f'ROLL_{col}_HIST_{window}'
                cold_col = f'ROLL_{col}_COLD_START_{window}'

                new_columns[mean_col] = fill_series_with_prior(rolling_mean, prior, diagnostics, mean_col)
                new_columns[std_col] = fill_series_with_prior(rolling_std, 0.0, diagnostics, std_col).fillna(0.0)
                new_columns[hist_col] = history_count.fillna(0).astype(float)
                new_columns[cold_col] = (history_count < window).astype(int)

                if diagnostics is not None:
                    diagnostics.record_imputation(hist_col, int(history_count.isna().sum()))
                    diagnostics.record_imputation(cold_col, int(history_count.isna().sum()))

            if diagnostics is not None:
                diagnostics.record_imputation(col, int(shifted.isna().sum()))

        return _concat_new_columns(df, new_columns)


class EfficiencyFeatureGroup(FeatureGroup):
    """Rolling efficiency metrics derived from historical box-score components."""

    def __init__(self, windows: Optional[List[int]] = None):
        self.windows = windows or [5, 10, 20]

    @property
    def name(self) -> str:
        return 'efficiency'

    @property
    def required_columns(self) -> List[str]:
        return ['PLAYER_ID']

    def create(
        self,
        df: pd.DataFrame,
        *,
        diagnostics: Optional[FeatureDiagnostics] = None,
        context: Optional[FeatureContext] = None,
    ) -> pd.DataFrame:
        self._check_columns(df, diagnostics)
        df = df.copy()
        context = context or FeatureContext()
        eps = 1e-7

        needed = [c for c in ['FGA', 'FTA', 'TOV', 'PTS', 'FGM', 'FG3M', 'FG3A', 'AST', 'MIN'] if c in df.columns]
        if not needed:
            return df

        shifted = df.groupby('PLAYER_ID')[needed].shift(1)
        grouped = shifted.groupby(df['PLAYER_ID'])
        new_columns: dict[str, pd.Series] = {}

        for window in self.windows:
            rolled = grouped.rolling(window, min_periods=1).sum().reset_index(level=0, drop=True)

            raw_cols = {
                f'RAW_FGA_SUM_{window}': rolled['FGA'].fillna(0) if 'FGA' in rolled.columns else 0.0,
                f'RAW_FTA_SUM_{window}': rolled['FTA'].fillna(0) if 'FTA' in rolled.columns else 0.0,
                f'RAW_TOV_SUM_{window}': rolled['TOV'].fillna(0) if 'TOV' in rolled.columns else 0.0,
                f'RAW_PTS_SUM_{window}': rolled['PTS'].fillna(0) if 'PTS' in rolled.columns else 0.0,
                f'RAW_FGM_SUM_{window}': rolled['FGM'].fillna(0) if 'FGM' in rolled.columns else 0.0,
                f'RAW_FG3M_SUM_{window}': rolled['FG3M'].fillna(0) if 'FG3M' in rolled.columns else 0.0,
                f'RAW_FG3A_SUM_{window}': rolled['FG3A'].fillna(0) if 'FG3A' in rolled.columns else 0.0,
                f'RAW_AST_SUM_{window}': rolled['AST'].fillna(0) if 'AST' in rolled.columns else 0.0,
                f'RAW_MIN_SUM_{window}': rolled['MIN'].fillna(0) if 'MIN' in rolled.columns else 0.0,
            }

            for col_name, values in raw_cols.items():
                new_columns[col_name] = pd.Series(values, index=df.index).fillna(0.0)
                if diagnostics is not None:
                    diagnostics.record_imputation(col_name, int(pd.Series(values).isna().sum()))

            fg_sum = new_columns[f'RAW_FGA_SUM_{window}']
            ft_sum = new_columns[f'RAW_FTA_SUM_{window}']
            tov_sum = new_columns[f'RAW_TOV_SUM_{window}']
            pts_sum = new_columns[f'RAW_PTS_SUM_{window}']
            fgm_sum = new_columns[f'RAW_FGM_SUM_{window}']
            fg3m_sum = new_columns[f'RAW_FG3M_SUM_{window}']
            fg3a_sum = new_columns[f'RAW_FG3A_SUM_{window}']
            ast_sum = new_columns[f'RAW_AST_SUM_{window}']
            mins_sum = new_columns[f'RAW_MIN_SUM_{window}'].replace(0, 1)

            new_columns[f'ROLL_TS_PCT_{window}'] = (pts_sum / (2 * (fg_sum + 0.44 * ft_sum + eps))).fillna(context.league_priors.get('TS_PCT', 0.56))
            new_columns[f'ROLL_EFG_PCT_{window}'] = ((fgm_sum + 0.5 * fg3m_sum) / (fg_sum + eps)).fillna(context.league_priors.get('EFG_PCT', 0.52))
            new_columns[f'ROLL_3PT_PCT_{window}'] = (fg3m_sum / (fg3a_sum + eps)).fillna(context.league_priors.get('3PT_PCT', 0.36))
            new_columns[f'ROLL_AST_TOV_{window}'] = (ast_sum / (tov_sum + eps)).fillna(context.league_priors.get('AST_TOV', 1.4))

            for stat in ['PTS', 'REB', 'AST']:
                if f'RAW_{stat}_SUM_{window}' in new_columns:
                    new_columns[f'ROLL_{stat}_PER_MIN_{window}'] = (
                        new_columns[f'RAW_{stat}_SUM_{window}'] / (mins_sum + eps)
                    ).fillna(context.league_priors.get(stat, 0.0))

        return _concat_new_columns(df, new_columns)


class MomentumFeatureGroup(FeatureGroup):
    """EWMA, season averages, and trend flags."""

    def __init__(self, target_cols: Optional[List[str]] = None):
        self.target_cols = target_cols or ['PTS', 'REB', 'AST']

    @property
    def name(self) -> str:
        return 'momentum'

    @property
    def required_columns(self) -> List[str]:
        return ['PLAYER_ID']

    def create(
        self,
        df: pd.DataFrame,
        *,
        diagnostics: Optional[FeatureDiagnostics] = None,
        context: Optional[FeatureContext] = None,
    ) -> pd.DataFrame:
        self._check_columns(df, diagnostics)
        df = df.copy()
        context = context or FeatureContext()

        new_columns: dict[str, pd.Series] = {}
        for stat in [c for c in self.target_cols if c in df.columns]:
            prior = float(context.league_priors.get(stat, 0.0))
            shifted = df.groupby('PLAYER_ID')[stat].shift(1)
            for span in [3, 5, 10, 20]:
                ewma = shifted.groupby(df['PLAYER_ID']).transform(
                    lambda x: x.ewm(span=span, adjust=False).mean()
                )
                col = f'{stat}_EWMA_{span}'
                new_columns[col] = fill_series_with_prior(ewma, prior, diagnostics, col)

            season_avg = shifted.groupby(df['PLAYER_ID']).transform(lambda x: x.expanding().mean())
            new_columns[f'{stat}_SEASON_AVG'] = fill_series_with_prior(season_avg, prior, diagnostics, f'{stat}_SEASON_AVG')

            for short, long in [(3, 10), (5, 20)]:
                short_avg = shifted.groupby(df['PLAYER_ID']).transform(lambda x: x.rolling(short, min_periods=1).mean())
                long_avg = shifted.groupby(df['PLAYER_ID']).transform(lambda x: x.rolling(long, min_periods=max(1, long // 3)).mean())
                trend = short_avg - long_avg
                new_columns[f'{stat}_TREND_{short}_{long}'] = trend.fillna(0.0)

            roll_3 = new_columns[f'{stat}_EWMA_3']
            roll_10 = new_columns[f'{stat}_EWMA_10']
            new_columns[f'{stat}_HOT_STREAK'] = (roll_3 > roll_10 * 1.15).astype(int)
            new_columns[f'{stat}_COLD_STREAK'] = (roll_3 < roll_10 * 0.85).astype(int)

        return _concat_new_columns(df, new_columns)
