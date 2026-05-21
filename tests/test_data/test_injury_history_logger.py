"""Tests for InjuryHistoryLogger."""

import os
from datetime import datetime

import pandas as pd
import pytest

from src.data.injury_history_logger import InjuryHistoryLogger


@pytest.fixture
def history_logger(tmp_path):
    return InjuryHistoryLogger(history_dir=str(tmp_path))


class TestInjuryHistoryLoggerConstructor:
    def test_creates_directory(self, tmp_path):
        subdir = str(tmp_path / 'deep' / 'dir')
        logger = InjuryHistoryLogger(history_dir=subdir)
        assert os.path.isdir(subdir)

    def test_default_filename(self, history_logger, tmp_path):
        assert history_logger.path == os.path.join(str(tmp_path), 'injury_history.csv')


class TestLogInjuries:
    def test_log_single_event(self, history_logger):
        event = {
            'PLAYER_ID': 2544, 'PLAYER': 'LeBron James',
            'TEAM_ABBR': 'LAL', 'STATUS': 'OUT',
            'INJURY_TYPE': 'Ankle', 'DATE': '2025-03-15',
            'PLAY_PROBABILITY': 0.0,
        }
        history_logger.log_injuries([event])
        df = history_logger.load_history()
        assert len(df) == 1
        assert df.iloc[0]['PLAYER_ID'] == 2544

    def test_log_multiple_events(self, history_logger):
        events = [
            {'PLAYER_ID': 2544, 'PLAYER': 'LeBron James', 'TEAM_ABBR': 'LAL',
             'STATUS': 'OUT', 'INJURY_TYPE': 'Ankle', 'DATE': '2025-03-15',
             'PLAY_PROBABILITY': 0.0},
            {'PLAYER_ID': 201939, 'PLAYER': 'Stephen Curry', 'TEAM_ABBR': 'GSW',
             'STATUS': 'QUESTIONABLE', 'INJURY_TYPE': 'Knee', 'DATE': '2025-03-16',
             'PLAY_PROBABILITY': 0.5},
        ]
        history_logger.log_injuries(events)
        df = history_logger.load_history()
        assert len(df) == 2

    def test_deduplicates_same_event(self, history_logger):
        event = {
            'PLAYER_ID': 2544, 'PLAYER': 'LeBron James',
            'TEAM_ABBR': 'LAL', 'STATUS': 'OUT',
            'INJURY_TYPE': 'Ankle', 'DATE': '2025-03-15',
            'PLAY_PROBABILITY': 0.0,
        }
        # Log same event twice
        history_logger.log_injuries([event])
        history_logger.log_injuries([event])
        df = history_logger.load_history()
        assert len(df) == 1

    def test_allows_different_injuries_same_day(self, history_logger):
        events = [
            {'PLAYER_ID': 2544, 'PLAYER': 'LeBron James', 'TEAM_ABBR': 'LAL',
             'STATUS': 'OUT', 'INJURY_TYPE': 'Ankle', 'DATE': '2025-03-15',
             'PLAY_PROBABILITY': 0.0},
            {'PLAYER_ID': 2544, 'PLAYER': 'LeBron James', 'TEAM_ABBR': 'LAL',
             'STATUS': 'QUESTIONABLE', 'INJURY_TYPE': 'Back', 'DATE': '2025-03-15',
             'PLAY_PROBABILITY': 0.5},
        ]
        history_logger.log_injuries(events)
        df = history_logger.load_history()
        assert len(df) == 2  # different injury types on same day

    def test_log_empty_list(self, history_logger):
        history_logger.log_injuries([])
        df = history_logger.load_history()
        assert df.empty

    def test_log_events_without_player_id(self, history_logger):
        """Events with only PLAYER name should still be logged."""
        event = {
            'PLAYER': 'Unknown Player', 'TEAM_ABBR': 'LAL',
            'STATUS': 'OUT', 'INJURY_TYPE': 'Knee', 'DATE': '2025-04-01',
            'PLAY_PROBABILITY': 0.0,
        }
        history_logger.log_injuries([event])
        df = history_logger.load_history()
        assert len(df) == 1

    def test_fills_missing_columns(self, history_logger):
        """Columns not in input should be filled with None."""
        event = {
            'PLAYER_ID': 2544, 'PLAYER': 'LeBron James',
            'DATE': '2025-03-15',
        }
        history_logger.log_injuries([event])
        df = history_logger.load_history()
        assert 'STATUS' in df.columns
        assert 'TEAM_ABBR' in df.columns


class TestLoadHistory:
    def test_load_empty_when_no_file(self, history_logger):
        df = history_logger.load_history()
        assert df.empty

    def test_load_after_log(self, history_logger):
        event = {
            'PLAYER_ID': 2544, 'PLAYER': 'LeBron James',
            'TEAM_ABBR': 'LAL', 'STATUS': 'OUT',
            'INJURY_TYPE': 'Ankle', 'DATE': '2025-03-15',
            'PLAY_PROBABILITY': 0.0,
        }
        history_logger.log_injuries([event])
        # Create a fresh logger to test file persistence
        logger2 = InjuryHistoryLogger(history_dir=os.path.dirname(history_logger.path))
        df = logger2.load_history()
        assert len(df) == 1


class TestGetPlayerHistory:
    def test_get_by_id(self, history_logger):
        events = [
            {'PLAYER_ID': 2544, 'PLAYER': 'LeBron James', 'TEAM_ABBR': 'LAL',
             'STATUS': 'OUT', 'INJURY_TYPE': 'Ankle', 'DATE': '2025-03-15',
             'PLAY_PROBABILITY': 0.0},
            {'PLAYER_ID': 201939, 'PLAYER': 'Stephen Curry', 'TEAM_ABBR': 'GSW',
             'STATUS': 'OUT', 'INJURY_TYPE': 'Knee', 'DATE': '2025-03-16',
             'PLAY_PROBABILITY': 0.0},
        ]
        history_logger.log_injuries(events)
        df = history_logger.get_player_history(2544)
        assert len(df) == 1
        assert df.iloc[0]['PLAYER'] == 'LeBron James'

    def test_get_by_name(self, history_logger):
        events = [
            {'PLAYER_ID': 2544, 'PLAYER': 'LeBron James', 'TEAM_ABBR': 'LAL',
             'STATUS': 'OUT', 'INJURY_TYPE': 'Ankle', 'DATE': '2025-03-15',
             'PLAY_PROBABILITY': 0.0},
        ]
        history_logger.log_injuries(events)
        df = history_logger.get_player_history_by_name('LeBron')
        assert len(df) == 1

    def test_get_empty_for_unknown_player(self, history_logger):
        events = [
            {'PLAYER_ID': 2544, 'PLAYER': 'LeBron James', 'TEAM_ABBR': 'LAL',
             'STATUS': 'OUT', 'INJURY_TYPE': 'Ankle', 'DATE': '2025-03-15',
             'PLAY_PROBABILITY': 0.0},
        ]
        history_logger.log_injuries(events)
        df = history_logger.get_player_history(99999)
        assert df.empty


class TestCountEventsSince:
    def test_count_recent_events(self, history_logger):
        events = [
            {'PLAYER_ID': 2544, 'PLAYER': 'LeBron James', 'TEAM_ABBR': 'LAL',
             'STATUS': 'OUT', 'INJURY_TYPE': 'Ankle', 'DATE': '2025-01-15',
             'PLAY_PROBABILITY': 0.0},
            {'PLAYER_ID': 2544, 'PLAYER': 'LeBron James', 'TEAM_ABBR': 'LAL',
             'STATUS': 'OUT', 'INJURY_TYPE': 'Back', 'DATE': '2025-03-15',
             'PLAY_PROBABILITY': 0.0},
        ]
        history_logger.log_injuries(events)
        count = history_logger.count_events_since(2544, '2025-03-01')
        assert count == 1

    def test_count_all_if_all_recent(self, history_logger):
        events = [
            {'PLAYER_ID': 2544, 'PLAYER': 'LeBron James', 'TEAM_ABBR': 'LAL',
             'STATUS': 'OUT', 'INJURY_TYPE': 'Ankle', 'DATE': '2025-04-15',
             'PLAY_PROBABILITY': 0.0},
        ]
        history_logger.log_injuries(events)
        count = history_logger.count_events_since(2544, '2025-01-01')
        assert count == 1

    def test_count_zero_for_unknown(self, history_logger):
        count = history_logger.count_events_since(99999, '2025-01-01')
        assert count == 0