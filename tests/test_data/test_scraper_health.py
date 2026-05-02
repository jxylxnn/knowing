import pandas as pd
from unittest.mock import Mock, patch


def test_schedule_scraper_fetch_path_uses_initialized_config_attrs(tmp_path):
    from src.data.schedule_scraper import ScheduleScraper

    fake_board = Mock()
    fake_board.get_dict.return_value = {
        'resultSets': [{
            'headers': ['GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'GAME_STATUS_TEXT'],
            'rowSet': [[123, 1610612738, 1610612747, 'Scheduled']],
        }]
    }

    with patch('src.data.schedule_scraper.scoreboardv2.ScoreboardV2', return_value=fake_board):
        scraper = ScheduleScraper(cache_dir=str(tmp_path))
        games = scraper.get_games_by_date('2024-01-01')

    assert not games.empty
    assert games.iloc[0]['HOME_TEAM'] == 'BOS'
    assert scraper.get_last_fetch_status()['status'] == 'success'


def test_lineup_scraper_constructor_initializes_required_state(tmp_path):
    from src.data.lineup_scraper import LineupScraper

    scraper = LineupScraper(cache_dir=str(tmp_path))

    assert scraper._lineup_cache == {}
    assert isinstance(scraper._coach_tendencies, dict)
    assert isinstance(scraper.nba_headers, dict)
    assert scraper._session.headers['Origin'] == 'https://www.nba.com'


def test_lineup_scraper_failed_source_is_explicit(tmp_path):
    from src.data.lineup_scraper import LineupScraper

    scraper = LineupScraper(cache_dir=str(tmp_path))
    with patch.object(scraper, '_fetch_from_nba_stats', return_value=None), \
         patch.object(scraper, '_fetch_projected_lineup', return_value=None), \
         patch.object(scraper, '_infer_lineup_from_history', return_value={'starters': [], 'starter_ids': [], 'source': 'historical_inference'}):
        lineup = scraper.get_starting_lineup('BOS', '2024-01-01')

    assert lineup['source'] == 'failed'
    assert lineup['health_status'] == 'failed'
    assert scraper.get_last_fetch_status()['status'] == 'failed'


def test_basketball_ref_scraper_fetch_path_uses_initialized_config_attrs(tmp_path):
    from src.data.basketball_ref_scraper import BasketballRefScraper

    html = """
    <html>
      <body>
        <table id="team_misc">
          <tr><th>Pace</th><td>99.5</td></tr>
          <tr><th>Offensive Rating</th><td>118.2</td></tr>
          <tr><th>Defensive Rating</th><td>110.1</td></tr>
        </table>
      </body>
    </html>
    """
    response = Mock()
    response.raise_for_status.return_value = None
    response.text = html

    scraper = BasketballRefScraper(cache_dir=str(tmp_path))
    scraper._session.get = Mock(return_value=response)

    result = scraper._fetch_team_stats('BOS', '2024-25')

    assert result['team'] == 'BOS'
    assert result['pace'] == 99.5
    assert result['offensive_rating'] == 118.2


def test_rotowire_lineup_scraper_constructor_and_attrs(tmp_path):
    from src.data.rotowire_lineup_scraper import RotoWireLineupScraper

    scraper = RotoWireLineupScraper(cache_dir=str(tmp_path))

    # _cache_timestamp should be initialized (was previously unreachable dead code)
    assert scraper._cache_timestamp is None
    # lowercase instance attributes must exist (previously uppercase references crashed)
    assert isinstance(scraper.max_retries, int)
    assert isinstance(scraper.retry_delay, (int, float))
    assert isinstance(scraper.cache_ttl_minutes, (int, float))


def test_rotowire_lineup_scraper_get_config_value_no_unreachable_code(tmp_path):
    from src.data.rotowire_lineup_scraper import RotoWireLineupScraper

    scraper = RotoWireLineupScraper(cache_dir=str(tmp_path))
    # _get_config_value should return without executing dead code after return
    assert scraper._get_config_value('missing.key', 'default') == 'default'
    assert scraper._cache_timestamp is None  # ensure it wasn't accidentally set


def test_nba_defense_scraper_imports_and_analyzer_attrs(tmp_path):
    from src.data.nba_defense_scraper import NBADefenseScraper, DefensiveMatchupAnalyzer

    scraper = NBADefenseScraper(cache_dir=str(tmp_path))
    # ABBR_TO_ID import must resolve so these don't raise NameError
    assert scraper.get_all_team_defense_ratings is not None

    analyzer = DefensiveMatchupAnalyzer(defense_scraper=scraper, cache_dir=str(tmp_path))
    # DefensiveMatchupAnalyzer must have retry attrs and headers
    assert hasattr(analyzer, 'max_retries')
    assert hasattr(analyzer, 'retry_delay')
    assert hasattr(analyzer, 'headers')


def test_schedule_scraper_remaining_season_horizon(tmp_path):
    from src.data.schedule_scraper import ScheduleScraper
    from unittest.mock import patch

    scraper = ScheduleScraper(cache_dir=str(tmp_path))

    # Patch get_games_by_date to return an empty frame so we exercise the horizon logic
    with patch.object(scraper, 'get_games_by_date', return_value=pd.DataFrame()):
        df = scraper.get_remaining_season('2099-00')

    # With a far-future season, max_days should be capped and we should get an empty frame
    assert df.empty
    status = scraper.get_last_fetch_status()
    assert status['status'] == 'failed'
    assert 'days_attempted' in status['details']
    # The cap is 180 days, so days_attempted should be 181 (0..180 inclusive)
    assert status['details']['days_attempted'] == 181
