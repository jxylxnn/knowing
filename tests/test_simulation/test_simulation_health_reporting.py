import pandas as pd
from unittest.mock import Mock


def test_season_simulator_treats_schedule_failure_as_hard_failure():
    from src.simulation.season_simulator import SeasonSimulator

    game_simulator = Mock()
    schedule_scraper = Mock()
    schedule_scraper.get_games_by_date.return_value = pd.DataFrame()
    schedule_scraper.get_last_fetch_status.return_value = {
        'source_key': 'schedule',
        'status': 'failed',
        'required': True,
        'message': 'schedule failed',
        'details': {},
    }

    simulator = SeasonSimulator(game_simulator, schedule_scraper)
    results = simulator.simulate_date('2024-01-01', num_sims=10, max_workers=1)

    assert results == []
    assert simulator.last_run_summary['overall_status'] == 'failed'
    assert 'schedule' in simulator.last_run_summary['hard_failures']


def test_season_simulator_aggregates_optional_degraded_inputs():
    from src.simulation.season_simulator import SeasonSimulator

    game_simulator = Mock()
    game_simulator.device.type = 'cpu'
    game_simulator.prepare_simulation_context.return_value = None
    game_simulator.simulate_matchup.return_value = {
        'team_a': 'LAL',
        'team_b': 'BOS',
        'simulations': [],
        'player_averages': [],
        'metadata': {
            'input_health': {
                'overall_status': 'degraded',
                'counts': {'success': 1, 'fallback': 1, 'failed': 1, 'disabled': 0},
                'degraded_sources': ['betting', 'lineup_lal'],
                'hard_failures': [],
                'sources': [
                    {'source_key': 'betting', 'status': 'fallback', 'required': False, 'message': 'betting fallback', 'details': {}},
                    {'source_key': 'lineup_lal', 'status': 'failed', 'required': False, 'message': 'lineup failed', 'details': {}},
                ],
            }
        },
    }
    schedule_scraper = Mock()
    simulator = SeasonSimulator(game_simulator, schedule_scraper)
    simulator._set_schedule_health({
        'source_key': 'schedule',
        'status': 'success',
        'required': True,
        'message': 'schedule ok',
        'details': {},
    })

    games_df = pd.DataFrame([{'HOME_TEAM': 'LAL', 'AWAY_TEAM': 'BOS', 'GAME_DATE': '2024-01-01'}])
    results = simulator.simulate_games(games_df, num_sims=10, max_workers=1)

    assert len(results) == 1
    assert simulator.last_run_summary['overall_status'] == 'degraded'
    assert 'lineup_lal' in simulator.last_run_summary['input_health']['degraded_sources']


def test_format_input_health_summary_includes_degraded_sources():
    from simulate_season import format_input_health_summary

    summary = format_input_health_summary({
        'overall_status': 'degraded',
        'schedule_health': {
            'source_key': 'schedule',
            'status': 'success',
            'required': True,
            'message': 'schedule ok',
            'details': {},
        },
        'input_health': {
            'counts': {'success': 2, 'fallback': 1, 'failed': 1, 'disabled': 0},
            'degraded_sources': ['betting', 'lineup_lal'],
            'sources': [
                {'source_key': 'betting', 'status': 'fallback', 'required': False, 'message': 'betting fallback', 'details': {}},
                {'source_key': 'lineup_lal', 'status': 'failed', 'required': False, 'message': 'lineup failed', 'details': {}},
            ],
        },
        'hard_failures': [],
    })

    assert 'INPUT HEALTH SUMMARY [DEGRADED]' in summary
    assert 'Degraded sources: betting, lineup_lal' in summary
