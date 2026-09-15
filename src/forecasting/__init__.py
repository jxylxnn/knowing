"""Opportunity-first Model v2 forecasting components."""

from src.forecasting.minutes import allocate_team_minutes
from src.forecasting.rates import totals_from_minutes_and_rates

__all__ = ["allocate_team_minutes", "totals_from_minutes_and_rates"]
