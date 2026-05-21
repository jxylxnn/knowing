"""Tests for PlayerBioScraper."""

import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


@pytest.fixture
def bio_scraper(tmp_path):
    from src.data.player_bio_scraper import PlayerBioScraper
    return PlayerBioScraper(cache_dir=str(tmp_path / 'cache'), rate_delay=0)


@pytest.fixture
def mock_bio_response():
    """Mock DataFrame returned by commonplayerinfo endpoint."""
    return pd.DataFrame({
        'PERSON_ID': [2544],
        'DISPLAY_FIRST_LAST': ['LeBron James'],
        'BIRTHDATE': ['1984-12-30T00:00:00'],
        'POSITION': ['Forward'],
        'HEIGHT': ['6-9'],
        'WEIGHT': ['250'],
        'COUNTRY': ['USA'],
        'DRAFT_YEAR': ['2003'],
        'DRAFT_ROUND': ['1'],
        'DRAFT_NUMBER': ['1'],
        'FROM_YEAR': ['2003'],
        'TO_YEAR': ['2025'],
        'SEASON_EXP': ['22'],
        'TEAM_ABBREVIATION': ['LAL'],
    })


class TestPlayerBioScraperConstructor:
    def test_creates_cache_dir(self, tmp_path):
        from src.data.player_bio_scraper import PlayerBioScraper
        cache_dir = str(tmp_path / 'new_cache')
        scraper = PlayerBioScraper(cache_dir=cache_dir)
        assert os.path.isdir(cache_dir)

    def test_default_rate_delay(self, bio_scraper):
        assert bio_scraper.rate_delay == 0

    def test_last_fetch_status_initial(self, bio_scraper):
        status = bio_scraper.get_last_fetch_status()
        assert isinstance(status, dict)


class TestNormalizeBio:
    def test_renames_columns_correctly(self, bio_scraper, mock_bio_response):
        result = bio_scraper._normalize_bio(mock_bio_response)
        assert 'PLAYER_ID' in result.columns
        assert 'PLAYER_NAME' in result.columns
        assert 'BIRTHDATE' in result.columns
        assert 'AGE' in result.columns
        assert 'POSITION' in result.columns
        assert 'HEIGHT' in result.columns
        assert 'WEIGHT' in result.columns

    def test_computes_age_from_birthdate(self, bio_scraper, mock_bio_response):
        result = bio_scraper._normalize_bio(mock_bio_response)
        age = result.iloc[0]['AGE']
        # LeBron born 1984-12-30, should be ~41 years old
        assert 40 <= age <= 42

    def test_weight_parsed_as_numeric(self, bio_scraper, mock_bio_response):
        result = bio_scraper._normalize_bio(mock_bio_response)
        assert result.iloc[0]['WEIGHT'] == 250.0

    def test_career_start_parsed_as_numeric(self, bio_scraper, mock_bio_response):
        result = bio_scraper._normalize_bio(mock_bio_response)
        assert result.iloc[0]['CAREER_START'] == 2003

    def test_empty_input_returns_empty(self, bio_scraper):
        result = bio_scraper._normalize_bio(pd.DataFrame())
        assert result.empty

    def test_output_columns_present_even_if_missing(self, bio_scraper):
        """All OUTPUT_COLUMNS should exist even with partial API data."""
        partial = pd.DataFrame({
            'PERSON_ID': [999],
            'BIRTHDATE': ['1995-06-15T00:00:00'],
        })
        result = bio_scraper._normalize_bio(partial)
        for col in bio_scraper.OUTPUT_COLUMNS:
            assert col in result.columns


class TestFetchPlayerBio:
    def test_fetch_single_player(self, bio_scraper, mock_bio_response):
        mock_endpoint = MagicMock()
        mock_endpoint.common_player_info.get_data_frame.return_value = mock_bio_response

        with patch('src.data.player_bio_scraper.commonplayerinfo', MagicMock(
            CommonPlayerInfo=MagicMock(return_value=mock_endpoint)
        )):
            result = bio_scraper.fetch_player_bio(2544)
            assert not result.empty
            assert 'PLAYER_ID' in result.columns
            assert result.iloc[0]['PLAYER_ID'] == 2544

    def test_fetch_failure_returns_empty(self, bio_scraper):
        with patch('src.data.player_bio_scraper.commonplayerinfo', MagicMock(
            CommonPlayerInfo=MagicMock(side_effect=Exception("API error"))
        )):
            result = bio_scraper.fetch_player_bio(999999)
            assert result.empty


class TestFetchAllBios:
    def test_fetches_multiple_players(self, bio_scraper):
        bio1 = pd.DataFrame({
            'PERSON_ID': [2544], 'DISPLAY_FIRST_LAST': ['LeBron James'],
            'BIRTHDATE': ['1984-12-30T00:00:00'], 'POSITION': ['Forward'],
            'HEIGHT': ['6-9'], 'WEIGHT': ['250'], 'COUNTRY': ['USA'],
            'DRAFT_YEAR': ['2003'], 'DRAFT_ROUND': ['1'], 'DRAFT_NUMBER': ['1'],
            'FROM_YEAR': ['2003'], 'TO_YEAR': ['2025'], 'SEASON_EXP': ['22'],
            'TEAM_ABBREVIATION': ['LAL'],
        })
        bio2 = pd.DataFrame({
            'PERSON_ID': [201939], 'DISPLAY_FIRST_LAST': ['Stephen Curry'],
            'BIRTHDATE': ['1988-03-14T00:00:00'], 'POSITION': ['Guard'],
            'HEIGHT': ['6-2'], 'WEIGHT': ['185'], 'COUNTRY': ['USA'],
            'DRAFT_YEAR': ['2009'], 'DRAFT_ROUND': ['1'], 'DRAFT_NUMBER': ['7'],
            'FROM_YEAR': ['2009'], 'TO_YEAR': ['2025'], 'SEASON_EXP': ['16'],
            'TEAM_ABBREVIATION': ['GSW'],
        })

        call_count = [0]

        def mock_fetch(pid):
            call_count[0] += 1
            if pid == 2544:
                return bio_scraper._normalize_bio(bio1)
            elif pid == 201939:
                return bio_scraper._normalize_bio(bio2)
            return pd.DataFrame()

        with patch.object(bio_scraper, 'fetch_player_bio', side_effect=mock_fetch):
            result = bio_scraper.fetch_all_bios([2544, 201939], force_refresh=True)
            assert len(result) == 2
            assert call_count[0] == 2

    def test_uses_cache_for_already_fetched(self, bio_scraper):
        """Second call should use cache and not re-fetch."""
        bio_df = pd.DataFrame({
            'PLAYER_ID': [2544], 'PLAYER_NAME': ['LeBron James'],
            'BIRTHDATE': ['1984-12-30'], 'AGE': [40.5], 'POSITION': ['Forward'],
            'HEIGHT': ['6-9'], 'WEIGHT': [250.0], 'COUNTRY': ['USA'],
            'DRAFT_YEAR': [2003.0], 'DRAFT_ROUND': ['1'], 'DRAFT_NUMBER': ['1'],
            'CAREER_START': [2003.0], 'CAREER_END': [2025.0], 'YEARS_EXPERIENCE': [22.0],
            'TEAM_ABBR': ['LAL'],
        })
        # Save a cache manually
        bio_scraper._save_cache(bio_df)

        # fetch_all_bios should return from cache without calling fetch_player_bio
        with patch.object(bio_scraper, 'fetch_player_bio') as mock_fetch:
            result = bio_scraper.fetch_all_bios([2544])
            mock_fetch.assert_not_called()
            assert len(result) == 1

    def test_only_fetches_missing_from_cache(self, bio_scraper):
        """If cache has some players, only fetch the missing ones."""
        cached = pd.DataFrame({
            'PLAYER_ID': [2544], 'PLAYER_NAME': ['LeBron James'],
            'BIRTHDATE': ['1984-12-30'], 'AGE': [40.5], 'POSITION': ['Forward'],
            'HEIGHT': ['6-9'], 'WEIGHT': [250.0], 'COUNTRY': ['USA'],
            'DRAFT_YEAR': [2003.0], 'DRAFT_ROUND': ['1'], 'DRAFT_NUMBER': ['1'],
            'CAREER_START': [2003.0], 'CAREER_END': [2025.0], 'YEARS_EXPERIENCE': [22.0],
            'TEAM_ABBR': ['LAL'],
        })
        bio_scraper._save_cache(cached)

        def mock_fetch(pid):
            if pid == 201939:
                return pd.DataFrame({
                    'PLAYER_ID': [201939], 'PLAYER_NAME': ['Stephen Curry'],
                    'BIRTHDATE': ['1988-03-14'], 'AGE': [37.2], 'POSITION': ['Guard'],
                    'HEIGHT': ['6-2'], 'WEIGHT': [185.0], 'COUNTRY': ['USA'],
                    'DRAFT_YEAR': [2009.0], 'DRAFT_ROUND': ['1'], 'DRAFT_NUMBER': ['7'],
                    'CAREER_START': [2009.0], 'CAREER_END': [2025.0], 'YEARS_EXPERIENCE': [16.0],
                    'TEAM_ABBR': ['GSW'],
                })
            return pd.DataFrame()

        with patch.object(bio_scraper, 'fetch_player_bio', side_effect=mock_fetch):
            result = bio_scraper.fetch_all_bios([2544, 201939])
            assert len(result) == 2


class TestResolveNameToId:
    def test_resolve_known_player(self, bio_scraper):
        mock_players = [
            {'id': 2544, 'full_name': 'LeBron James', 'first_name': 'LeBron',
             'last_name': 'James', 'is_active': True},
        ]
        mock_mod = MagicMock()
        mock_mod.get_players.return_value = mock_players
        with patch.dict('sys.modules', {'nba_api.stats.static.players': mock_mod}):
            result = bio_scraper.resolve_name_to_id('LeBron James')
            assert result == 2544

    def test_resolve_case_insensitive(self, bio_scraper):
        mock_players = [
            {'id': 2544, 'full_name': 'LeBron James', 'first_name': 'LeBron',
             'last_name': 'James', 'is_active': True},
        ]
        mock_mod = MagicMock()
        mock_mod.get_players.return_value = mock_players
        with patch.dict('sys.modules', {'nba_api.stats.static.players': mock_mod}):
            result = bio_scraper.resolve_name_to_id('lebron james')
            assert result == 2544

    def test_resolve_unknown_returns_none(self, bio_scraper):
        mock_mod = MagicMock()
        mock_mod.get_players.return_value = []
        with patch.dict('sys.modules', {'nba_api.stats.static.players': mock_mod}):
            result = bio_scraper.resolve_name_to_id('Nonexistent Player')
            assert result is None