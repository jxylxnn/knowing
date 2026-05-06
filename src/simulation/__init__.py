"""Simulation module for game and season simulations.

The simulator modules pull in Torch-backed dependencies. Import them lazily so
basic package imports stay safe in CPU-only or partially configured test
environments.

Sub-modules:
  - archetype         — Player archetype inference & profiles
  - role_sampler      — Role state sampling per simulation run
  - phase_simulator   — Phase-by-phase Monte Carlo game loop
  - sim_types          — Typed dataclasses (PlayerProjection, TeamContext, …)
  - stat_utils         — Shared stats (compute_mode, compute_stats_summary)
  - game_simulator     — Main GameSimulator orchestrator
  - season_simulator   — SeasonSimulator for batch simulations
"""

from importlib import import_module
from typing import Any

__all__ = ['GameSimulator', 'SeasonSimulator']


def __getattr__(name: str) -> Any:
    if name == 'GameSimulator':
        return import_module('.game_simulator', __name__).GameSimulator
    if name == 'SeasonSimulator':
        return import_module('.season_simulator', __name__).SeasonSimulator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")