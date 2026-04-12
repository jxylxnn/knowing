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
]
