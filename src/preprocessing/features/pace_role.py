"""Pace, team-role, and advanced scoring features."""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from src.preprocessing.features.base import (
    FeatureContext,
    FeatureDiagnostics,
    FeatureGroup,
    fill_series_with_prior,
)


class PaceFeatureGroup(FeatureGroup):
    """Safe team pace estimate from historical team box scores."""

    @property
    def name(self) -> str:
        return 'pace'

    @property
    def required_columns(self) -> List[str]:
        return ['TEAM_ID']

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

        eps = 1e-6
        fallback_pace = context.league_priors.get('TEAM_PACE', 100.0)

        team_cols = [
            'TEAM_FGA_ROLL_10',
            'TEAM_FTA_ROLL_10',
            'TEAM_TOV_ROLL_10',
            'TEAM_OREB_ROLL_10',
            'OPP_TEAM_DREB_ROLL_10',
            'TEAM_FGM_ROLL_10',
        ]
        if any(c in df.columns for c in team_cols):
            fga = df.get('TEAM_FGA_ROLL_10', fallback_pace)
            fta = df.get('TEAM_FTA_ROLL_10', 20.0)
            tov = df.get('TEAM_TOV_ROLL_10', 12.0)
            oreb = df.get('TEAM_OREB_ROLL_10', 10.0)
            opp_dreb = df.get('OPP_TEAM_DREB_ROLL_10', 30.0)
            fgm = df.get('TEAM_FGM_ROLL_10', fga * 0.45)
            est_poss = 0.5 * (
                fga
                + 0.4 * fta
                - 1.07 * (oreb / (oreb + opp_dreb + eps)) * (fga - fgm)
                + tov
            )
            df['EST_POSS'] = pd.Series(est_poss, index=df.index).fillna(fallback_pace)
        else:
            df['EST_POSS'] = fallback_pace

        df['TEAM_PACE_10'] = df.groupby('TEAM_ID')['EST_POSS'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=3).mean()
        ).fillna(fallback_pace)
        df['PACE_FACTOR'] = (df['TEAM_PACE_10'] / fallback_pace).clip(0.8, 1.2)

        if diagnostics is not None:
            diagnostics.record_imputation('EST_POSS', int(pd.isna(df['EST_POSS']).sum()))
            diagnostics.record_imputation('TEAM_PACE_10', int(pd.isna(df['TEAM_PACE_10']).sum()))

        return df


class TeamRoleFeatureGroup(FeatureGroup):
    """Raw role signals plus their rolling formulas."""

    @property
    def name(self) -> str:
        return 'team_role'

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

        if 'MIN' not in df.columns:
            df['MIN'] = context.league_priors.get('MIN', 24.0)

        shifted = {
            'FGA': df.groupby('PLAYER_ID')['FGA'].shift(1) if 'FGA' in df.columns else None,
            'FTA': df.groupby('PLAYER_ID')['FTA'].shift(1) if 'FTA' in df.columns else None,
            'TOV': df.groupby('PLAYER_ID')['TOV'].shift(1) if 'TOV' in df.columns else None,
            'FG3A': df.groupby('PLAYER_ID')['FG3A'].shift(1) if 'FG3A' in df.columns else None,
            'PTS': df.groupby('PLAYER_ID')['PTS'].shift(1) if 'PTS' in df.columns else None,
            'MIN': df.groupby('PLAYER_ID')['MIN'].shift(1),
        }

        team_poss = None
        if {'TEAM_FGA_ROLL_10', 'TEAM_FTA_ROLL_10', 'TEAM_TOV_ROLL_10'}.issubset(df.columns):
            team_poss = df['TEAM_FGA_ROLL_10'] + 0.44 * df['TEAM_FTA_ROLL_10'] + df['TEAM_TOV_ROLL_10']
        elif {'TEAM_FGA_ROLL_10', 'TEAM_FTA_ROLL_10', 'TEAM_TOV_ROLL_10'} & set(df.columns):
            team_poss = (
                df.get('TEAM_FGA_ROLL_10', context.league_priors.get('FGA', 9.0))
                + 0.44 * df.get('TEAM_FTA_ROLL_10', context.league_priors.get('FTA', 3.0))
                + df.get('TEAM_TOV_ROLL_10', context.league_priors.get('TOV', 1.5))
            )
        else:
            team_poss = pd.Series(100.0, index=df.index)

        player_poss = shifted['FGA'].fillna(0.0) if shifted['FGA'] is not None else 0.0
        if shifted['FTA'] is not None:
            player_poss = player_poss + 0.44 * shifted['FTA'].fillna(0.0)
        if shifted['TOV'] is not None:
            player_poss = player_poss + shifted['TOV'].fillna(0.0)

        mins_factor = (shifted['MIN'] / 48.0).fillna(0.7).clip(0.1, 1.0)
        raw_usage = ((player_poss / (team_poss + eps)) * (1 / mins_factor)).clip(0, 0.5)
        raw_usage = raw_usage.fillna(context.league_priors.get('USAGE', 0.18))
        df['RAW_USAGE'] = raw_usage
        df['RAW_USAGE_NUMERATOR'] = player_poss
        df['RAW_USAGE_DENOMINATOR'] = team_poss

        if diagnostics is not None:
            diagnostics.record_imputation('RAW_USAGE', int(pd.isna(raw_usage).sum()))

        if 'OPP_TEAM_FGM_ROLL_10' in df.columns and 'OPP_TEAM_FGA_ROLL_10' in df.columns:
            opp_fg_pct = (df['OPP_TEAM_FGM_ROLL_10'] / (df['OPP_TEAM_FGA_ROLL_10'] + eps)).clip(0.3, 0.7)
            raw_reb_opp = (1.0 - opp_fg_pct).fillna(context.league_priors.get('REB_OPP', 0.48))
        else:
            raw_reb_opp = pd.Series(context.league_priors.get('REB_OPP', 0.48), index=df.index)
        df['RAW_REB_OPPORTUNITY'] = raw_reb_opp

        if 'FG3A' in df.columns and 'FGA' in df.columns:
            raw_3pt_freq = (shifted['FG3A'].fillna(0.0) / (df.groupby('PLAYER_ID')['FGA'].shift(1).fillna(0.0) + eps)).clip(0, 1)
        else:
            raw_3pt_freq = pd.Series(0.0, index=df.index)
        df['RAW_3PT_FREQ'] = raw_3pt_freq.fillna(0.0)

        if 'FTA' in df.columns and 'FGA' in df.columns:
            raw_ft_rate = (shifted['FTA'].fillna(0.0) / (df.groupby('PLAYER_ID')['FGA'].shift(1).fillna(0.0) + eps)).clip(0, 1)
        else:
            raw_ft_rate = pd.Series(0.0, index=df.index)
        df['RAW_FT_RATE'] = raw_ft_rate.fillna(0.0)

        if 'PTS' in df.columns and 'PTS_TEAM_ROLL_10' in df.columns:
            team_pts = df['PTS_TEAM_ROLL_10']
            raw_pts_share = (shifted['PTS'].fillna(0.0) / (team_pts + eps)).clip(0, 0.6)
        elif 'PTS' in df.columns and 'TEAM_PTS_ROLL_10' in df.columns:
            raw_pts_share = (shifted['PTS'].fillna(0.0) / (df['TEAM_PTS_ROLL_10'] + eps)).clip(0, 0.6)
        else:
            raw_pts_share = pd.Series(context.league_priors.get('PTS_SHARE', 0.22), index=df.index)
        df['RAW_PTS_SHARE'] = raw_pts_share.fillna(context.league_priors.get('PTS_SHARE', 0.22))

        shifted_pts = shifted['PTS'] if shifted['PTS'] is not None else pd.Series(0.0, index=df.index)
        shifted_fta = shifted['FTA'] if shifted['FTA'] is not None else pd.Series(0.0, index=df.index)
        shifted_fga = shifted['FGA'] if shifted['FGA'] is not None else pd.Series(0.0, index=df.index)
        raw_ts = (shifted_pts / (2 * (shifted_fga + 0.44 * shifted_fta + eps))).clip(0.3, 0.8)
        df['RAW_TS_PCT'] = raw_ts.fillna(context.league_priors.get('TS_PCT', 0.56))

        for window in [5, 10]:
            df[f'ROLL_USG_PCT_{window}'] = df.groupby('PLAYER_ID')['RAW_USAGE'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            ).fillna(context.league_priors.get('USAGE', 0.18))
            df[f'ROLL_REB_OPPORTUNITY_{window}'] = df.groupby('PLAYER_ID')['RAW_REB_OPPORTUNITY'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            ).fillna(context.league_priors.get('REB_OPP', 0.48))
            if window in [10, 20]:
                df[f'ROLL_3PT_FREQ_{window}'] = df.groupby('PLAYER_ID')['RAW_3PT_FREQ'].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                ).fillna(0.0)
            if window == 10:
                df[f'ROLL_FT_RATE_{window}'] = df.groupby('PLAYER_ID')['RAW_FT_RATE'].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                ).fillna(0.0)
                df[f'ROLL_PTS_SHARE_{window}'] = df.groupby('PLAYER_ID')['RAW_PTS_SHARE'].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                ).fillna(context.league_priors.get('PTS_SHARE', 0.22))
            df[f'ROLL_TS_PCT_MOMENTUM_{window}'] = df.groupby('PLAYER_ID')['RAW_TS_PCT'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            ).fillna(context.league_priors.get('TS_PCT', 0.56))

        if 'PACE_FACTOR' in df.columns:
            df['PACE_ADJ_USAGE'] = (df['ROLL_USG_PCT_10'] * df['PACE_FACTOR']).fillna(context.league_priors.get('USAGE', 0.18))

        if 'ROLL_PTS_AVG_10' in df.columns:
            pts_avg = df['ROLL_PTS_AVG_10']
            pts_std = df.get('ROLL_PTS_STD_10', pd.Series(1.0, index=df.index)).replace(0, 1)
            recent_avg = df.groupby('PLAYER_ID')['PTS'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()) if 'PTS' in df.columns else pd.Series(0.0, index=df.index)
            df['EFF_Z_SCORE'] = ((recent_avg - pts_avg) / (pts_std + 1e-6)).fillna(0.0)

        return df
