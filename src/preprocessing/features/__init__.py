"""Feature groups for modular feature engineering."""

from src.preprocessing.features.base import FeatureGroup
from src.preprocessing.features.rolling import (
    RollingFeatureGroup,
    EfficiencyFeatureGroup,
    MomentumFeatureGroup,
)
from src.preprocessing.features.context import ContextualFeatureGroup, FatigueFeatureGroup
from src.preprocessing.features.archetype import PlayerArchetypeFeatureGroup
from src.preprocessing.features.matchup import MatchupFeatureGroup, OpponentStrengthFeatureGroup
from src.preprocessing.features.pace_role import PaceFeatureGroup, TeamRoleFeatureGroup
from src.preprocessing.features.target_encoding import TargetEncodingFeatureGroup, LeagueRankingFeatureGroup
from src.preprocessing.features.minutes_confidence import MinutesConfidenceFeatureGroup
from src.preprocessing.features.recency_form import RecencyFormFeatureGroup
from src.preprocessing.features.lineup_stability import LineupStabilityFeatureGroup
from src.preprocessing.features.rest_density import RestGameDensityFeatureGroup
from src.preprocessing.features.injury_opportunity import InjuryAdjustedOpportunityFeatureGroup
from src.preprocessing.features.teammate_usage import TeammateUsageFeatureGroup
from src.preprocessing.features.defense_position import DefensePositionFeatureGroup
from src.preprocessing.features.injury_risk import InjuryRiskFeatureGroup
from src.preprocessing.features.aging_curve import AgingCurveFeatureGroup
from src.preprocessing.features.kan_aging import KANAgingFeatureGroup
from src.preprocessing.features.skill_development import SkillDevelopmentFeatureGroup
from src.preprocessing.features.season_phase import SeasonPhaseFeatureGroup
from src.preprocessing.features.team_motivation import TeamMotivationFeatureGroup
from src.preprocessing.features.postseason_context import PostseasonContextFeatureGroup

__all__ = [
    'FeatureGroup',
    'RollingFeatureGroup',
    'EfficiencyFeatureGroup',
    'MomentumFeatureGroup',
    'ContextualFeatureGroup',
    'FatigueFeatureGroup',
    'PlayerArchetypeFeatureGroup',
    'MatchupFeatureGroup',
    'OpponentStrengthFeatureGroup',
    'PaceFeatureGroup',
    'TeamRoleFeatureGroup',
    'TargetEncodingFeatureGroup',
    'LeagueRankingFeatureGroup',
    'MinutesConfidenceFeatureGroup',
    'RecencyFormFeatureGroup',
    'LineupStabilityFeatureGroup',
    'RestGameDensityFeatureGroup',
    'InjuryAdjustedOpportunityFeatureGroup',
    'TeammateUsageFeatureGroup',
    'DefensePositionFeatureGroup',
    'InjuryRiskFeatureGroup',
    'AgingCurveFeatureGroup',
    'KANAgingFeatureGroup',
    'SkillDevelopmentFeatureGroup',
    'SeasonPhaseFeatureGroup',
    'TeamMotivationFeatureGroup',
    'PostseasonContextFeatureGroup',
]
