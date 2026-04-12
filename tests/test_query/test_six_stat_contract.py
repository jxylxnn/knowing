"""Regression tests for the 6-stat export/load contract.

These tests verify that the simulator export path (report_generator.py) and 
query/load path (projection_loader.py) correctly handle all 6 stats:
PTS, REB, AST, STL, BLK, TOV.

Each stat exports MEAN/MODE/CI columns and the loader reads them back correctly.
"""

import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path

from src.simulation.report_generator import ReportGenerator
from src.query.projection_loader import ProjectionLoader, PlayerProjection


# Expected column names for all 6 stats (95% CI columns)
EXPECTED_STAT_COLUMNS = {
    'pts': ['PROJ_PTS_MEAN', 'PROJ_PTS_MODE', 'PTS_CI_LOW', 'PTS_CI_HIGH'],
    'reb': ['PROJ_REB_MEAN', 'PROJ_REB_MODE', 'REB_CI_LOW', 'REB_CI_HIGH'],
    'ast': ['PROJ_AST_MEAN', 'PROJ_AST_MODE', 'AST_CI_LOW', 'AST_CI_HIGH'],
    'stl': ['PROJ_STL_MEAN', 'PROJ_STL_MODE', 'STL_CI_LOW', 'STL_CI_HIGH'],
    'blk': ['PROJ_BLK_MEAN', 'PROJ_BLK_MODE', 'BLK_CI_LOW', 'BLK_CI_HIGH'],
    'tov': ['PROJ_TOV_MEAN', 'PROJ_TOV_MODE', 'TOV_CI_LOW', 'TOV_CI_HIGH'],
}


class TestExportContainsAllSixStats:
    """Test that export_player_projections() writes all 6 stat columns."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary directory for CSV output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def sample_results(self):
        """Create sample simulation results with all 6 stats populated."""
        return [{
            'game_id': 'TEST_001',
            'date': '2025-01-15',
            'team_a': 'BOS',
            'team_b': 'LAL',
            'player_averages': [
                {
                    'name': 'Test Player',
                    'team': 'BOS',
                    'pts': 25.0,
                    'pts_mode': 24.5,
                    'pts_95_ci': [18.0, 32.0],
                    'pts_99_ci': [15.0, 35.0],
                    'reb': 8.0,
                    'reb_mode': 8.0,
                    'reb_95_ci': [5.0, 11.0],
                    'reb_99_ci': [3.0, 13.0],
                    'ast': 6.0,
                    'ast_mode': 6.0,
                    'ast_95_ci': [3.0, 9.0],
                    'ast_99_ci': [2.0, 10.0],
                    'stl': 1.2,
                    'stl_mode': 1.0,
                    'stl_95_ci': [0.0, 3.0],
                    'stl_99_ci': [0.0, 4.0],
                    'blk': 0.8,
                    'blk_mode': 1.0,
                    'blk_95_ci': [0.0, 2.0],
                    'blk_99_ci': [0.0, 3.0],
                    'tov': 2.5,
                    'tov_mode': 2.0,
                    'tov_95_ci': [1.0, 4.0],
                    'tov_99_ci': [0.0, 5.0],
                    'play_probability': 1.0,
                }
            ]
        }]

    def test_export_contains_all_six_stats(self, temp_output_dir, sample_results):
        """Verify exported CSV contains all expected columns for all 6 stats."""
        generator = ReportGenerator(output_dir=temp_output_dir)
        filepath = generator.export_player_projections(sample_results, filename='test_projections.csv')

        assert filepath is not None
        assert os.path.exists(filepath)

        df = pd.read_csv(filepath)
        columns = list(df.columns)

        # Check PTS columns
        for col in EXPECTED_STAT_COLUMNS['pts']:
            assert col in columns, f"Missing PTS column: {col}"

        # Check REB columns
        for col in EXPECTED_STAT_COLUMNS['reb']:
            assert col in columns, f"Missing REB column: {col}"

        # Check AST columns
        for col in EXPECTED_STAT_COLUMNS['ast']:
            assert col in columns, f"Missing AST column: {col}"

        # Check STL columns
        for col in EXPECTED_STAT_COLUMNS['stl']:
            assert col in columns, f"Missing STL column: {col}"

        # Check BLK columns
        for col in EXPECTED_STAT_COLUMNS['blk']:
            assert col in columns, f"Missing BLK column: {col}"

        # Check TOV columns
        for col in EXPECTED_STAT_COLUMNS['tov']:
            assert col in columns, f"Missing TOV column: {col}"

    def test_export_values_match_input(self, temp_output_dir, sample_results):
        """Verify exported values match the input values."""
        generator = ReportGenerator(output_dir=temp_output_dir)
        filepath = generator.export_player_projections(sample_results, filename='test_values.csv')

        df = pd.read_csv(filepath)
        row = df.iloc[0]

        # PTS checks
        assert row['PROJ_PTS_MEAN'] == 25.0
        assert row['PROJ_PTS_MODE'] == 24.5
        assert row['PTS_CI_LOW'] == 18.0
        assert row['PTS_CI_HIGH'] == 32.0

        # REB checks
        assert row['PROJ_REB_MEAN'] == 8.0
        assert row['PROJ_REB_MODE'] == 8.0
        assert row['REB_CI_LOW'] == 5.0
        assert row['REB_CI_HIGH'] == 11.0

        # AST checks
        assert row['PROJ_AST_MEAN'] == 6.0
        assert row['PROJ_AST_MODE'] == 6.0
        assert row['AST_CI_LOW'] == 3.0
        assert row['AST_CI_HIGH'] == 9.0

        # STL checks
        assert row['PROJ_STL_MEAN'] == 1.2
        assert row['PROJ_STL_MODE'] == 1.0
        assert row['STL_CI_LOW'] == 0.0
        assert row['STL_CI_HIGH'] == 3.0

        # BLK checks
        assert row['PROJ_BLK_MEAN'] == 0.8
        assert row['PROJ_BLK_MODE'] == 1.0
        assert row['BLK_CI_LOW'] == 0.0
        assert row['BLK_CI_HIGH'] == 2.0

        # TOV checks
        assert row['PROJ_TOV_MEAN'] == 2.5
        assert row['PROJ_TOV_MODE'] == 2.0
        assert row['TOV_CI_LOW'] == 1.0
        assert row['TOV_CI_HIGH'] == 4.0


class TestLoaderReadsAllSixStats:
    """Test that ProjectionLoader correctly reads all 6 stats from CSV."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory for CSV input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def sample_csv(self, temp_data_dir):
        """Create a minimal projection CSV with all 6 stats."""
        # Single-line CSV with all required columns
        csv_content = (
            "GAME_ID,DATE,PLAYER_NAME,TEAM,OPPONENT,IS_HOME,"
            "PROJ_PTS_MEAN,PROJ_PTS_MODE,PTS_CI_LOW,PTS_CI_HIGH,"
            "PROJ_REB_MEAN,PROJ_REB_MODE,REB_CI_LOW,REB_CI_HIGH,"
            "PROJ_AST_MEAN,PROJ_AST_MODE,AST_CI_LOW,AST_CI_HIGH,"
            "PROJ_STL_MEAN,PROJ_STL_MODE,STL_CI_LOW,STL_CI_HIGH,"
            "PROJ_BLK_MEAN,PROJ_BLK_MODE,BLK_CI_LOW,BLK_CI_HIGH,"
            "PROJ_TOV_MEAN,PROJ_TOV_MODE,TOV_CI_LOW,TOV_CI_HIGH,"
            "PLAY_PROBABILITY\n"
            "TEST_001,2025-01-15,Test Player,BOS,LAL,1,"
            "25.0,24.5,18.0,32.0,"
            "8.0,8.0,5.0,11.0,"
            "6.0,6.0,3.0,9.0,"
            "1.2,1.0,0.0,3.0,"
            "0.8,1.0,0.0,2.0,"
            "2.5,2.0,1.0,4.0,"
            "1.0"
        )

        filepath = os.path.join(temp_data_dir, 'player_projections_test.csv')
        with open(filepath, 'w') as f:
            f.write(csv_content)
        return filepath

    def test_loader_reads_all_six_stats(self, temp_data_dir, sample_csv):
        """Verify loader reads all 6 stat values correctly."""
        loader = ProjectionLoader(data_dir=temp_data_dir)

        # Rename to match expected pattern
        os.rename(sample_csv, os.path.join(temp_data_dir, 'player_projections_20250115_120000.csv'))

        projection = loader.find_player('Test Player')

        assert projection is not None
        assert projection.player_name == 'Test Player'

        # PTS
        assert projection.pts_mean == 25.0
        assert projection.pts_mode == 24.5
        assert projection.pts_ci_low == 18.0
        assert projection.pts_ci_high == 32.0

        # REB
        assert projection.reb_mean == 8.0
        assert projection.reb_mode == 8.0
        assert projection.reb_ci_low == 5.0
        assert projection.reb_ci_high == 11.0

        # AST
        assert projection.ast_mean == 6.0
        assert projection.ast_mode == 6.0
        assert projection.ast_ci_low == 3.0
        assert projection.ast_ci_high == 9.0

        # STL
        assert projection.stl_mean == 1.2
        assert projection.stl_mode == 1.0
        assert projection.stl_ci_low == 0.0
        assert projection.stl_ci_high == 3.0

        # BLK
        assert projection.blk_mean == 0.8
        assert projection.blk_mode == 1.0
        assert projection.blk_ci_low == 0.0
        assert projection.blk_ci_high == 2.0

        # TOV
        assert projection.tov_mean == 2.5
        assert projection.tov_mode == 2.0
        assert projection.tov_ci_low == 1.0
        assert projection.tov_ci_high == 4.0

    def test_get_stat_mean_returns_correct_values(self, temp_data_dir, sample_csv):
        """Verify get_stat_mean() works for all 6 stats."""
        os.rename(sample_csv, os.path.join(temp_data_dir, 'player_projections_20250115_120000.csv'))

        loader = ProjectionLoader(data_dir=temp_data_dir)
        projection = loader.find_player('Test Player')

        assert projection is not None
        assert projection.get_stat_mean('pts') == 25.0
        assert projection.get_stat_mean('reb') == 8.0
        assert projection.get_stat_mean('ast') == 6.0
        assert projection.get_stat_mean('stl') == 1.2
        assert projection.get_stat_mean('blk') == 0.8
        assert projection.get_stat_mean('tov') == 2.5

    def test_get_stat_ci_returns_correct_values(self, temp_data_dir, sample_csv):
        """Verify get_stat_ci() works for all 6 stats."""
        os.rename(sample_csv, os.path.join(temp_data_dir, 'player_projections_20250115_120000.csv'))

        loader = ProjectionLoader(data_dir=temp_data_dir)
        projection = loader.find_player('Test Player')

        assert projection is not None

        # PTS CI
        ci = projection.get_stat_ci('pts')
        assert ci == (18.0, 32.0)

        # REB CI
        ci = projection.get_stat_ci('reb')
        assert ci == (5.0, 11.0)

        # AST CI
        ci = projection.get_stat_ci('ast')
        assert ci == (3.0, 9.0)

        # STL CI
        ci = projection.get_stat_ci('stl')
        assert ci == (0.0, 3.0)

        # BLK CI
        ci = projection.get_stat_ci('blk')
        assert ci == (0.0, 2.0)

        # TOV CI
        ci = projection.get_stat_ci('tov')
        assert ci == (1.0, 4.0)


class TestLoaderUsesStatColumnsMapping:
    """Test that STAT_COLUMNS mapping is correctly defined for all 6 stats."""

    def test_stat_columns_contains_all_six_stats(self):
        """Verify STAT_COLUMNS contains entries for all 6 stats."""
        expected_stats = ['pts', 'reb', 'ast', 'stl', 'blk', 'tov']

        for stat in expected_stats:
            assert stat in ProjectionLoader.STAT_COLUMNS, f"Missing stat in STAT_COLUMNS: {stat}"

    def test_stat_columns_has_correct_keys(self):
        """Verify each stat entry has all required keys."""
        required_keys = {'mean', 'mode', 'std', 'ci_low', 'ci_high'}

        for stat, cols in ProjectionLoader.STAT_COLUMNS.items():
            for key in required_keys:
                assert key in cols, f"STAT_COLUMNS['{stat}'] missing key: {key}"

    def test_stat_columns_column_names_match_export(self):
        """Verify STAT_COLUMNS values match the expected export column names."""
        # PTS
        pts_cols = ProjectionLoader.STAT_COLUMNS['pts']
        assert pts_cols['mean'] == 'PROJ_PTS_MEAN'
        assert pts_cols['mode'] == 'PROJ_PTS_MODE'
        assert pts_cols['ci_low'] == 'PTS_CI_LOW'
        assert pts_cols['ci_high'] == 'PTS_CI_HIGH'

        # REB
        reb_cols = ProjectionLoader.STAT_COLUMNS['reb']
        assert reb_cols['mean'] == 'PROJ_REB_MEAN'
        assert reb_cols['mode'] == 'PROJ_REB_MODE'
        assert reb_cols['ci_low'] == 'REB_CI_LOW'
        assert reb_cols['ci_high'] == 'REB_CI_HIGH'

        # AST
        ast_cols = ProjectionLoader.STAT_COLUMNS['ast']
        assert ast_cols['mean'] == 'PROJ_AST_MEAN'
        assert ast_cols['mode'] == 'PROJ_AST_MODE'
        assert ast_cols['ci_low'] == 'AST_CI_LOW'
        assert ast_cols['ci_high'] == 'AST_CI_HIGH'

        # STL
        stl_cols = ProjectionLoader.STAT_COLUMNS['stl']
        assert stl_cols['mean'] == 'PROJ_STL_MEAN'
        assert stl_cols['mode'] == 'PROJ_STL_MODE'
        assert stl_cols['ci_low'] == 'STL_CI_LOW'
        assert stl_cols['ci_high'] == 'STL_CI_HIGH'

        # BLK
        blk_cols = ProjectionLoader.STAT_COLUMNS['blk']
        assert blk_cols['mean'] == 'PROJ_BLK_MEAN'
        assert blk_cols['mode'] == 'PROJ_BLK_MODE'
        assert blk_cols['ci_low'] == 'BLK_CI_LOW'
        assert blk_cols['ci_high'] == 'BLK_CI_HIGH'

        # TOV
        tov_cols = ProjectionLoader.STAT_COLUMNS['tov']
        assert tov_cols['mean'] == 'PROJ_TOV_MEAN'
        assert tov_cols['mode'] == 'PROJ_TOV_MODE'
        assert tov_cols['ci_low'] == 'TOV_CI_LOW'
        assert tov_cols['ci_high'] == 'TOV_CI_HIGH'


class TestMissingStatColumnsFailsLoudly:
    """Test that missing required stat columns are handled gracefully."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory for CSV input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_missing_tov_columns_loads_with_defaults(self, temp_data_dir):
        """Verify CSV missing TOV columns loads with default values."""
        # CSV with only PTS, REB, AST (old format)
        csv_content = """GAME_ID,DATE,PLAYER_NAME,TEAM,OPPONENT,IS_HOME,
PROJ_PTS_MEAN,PROJ_PTS_MODE,PTS_CI_LOW,PTS_CI_HIGH,
PROJ_REB_MEAN,PROJ_REB_MODE,REB_CI_LOW,REB_CI_HIGH,
PROJ_AST_MEAN,PROJ_AST_MODE,AST_CI_LOW,AST_CI_HIGH,
PLAY_PROBABILITY
TEST_001,2025-01-15,Test Player,BOS,LAL,1,
25.0,24.5,18.0,32.0,
8.0,8.0,5.0,11.0,
6.0,6.0,3.0,9.0,
1.0"""

        filepath = os.path.join(temp_data_dir, 'player_projections_20250115_120000.csv')
        with open(filepath, 'w') as f:
            f.write(csv_content)

        loader = ProjectionLoader(data_dir=temp_data_dir)
        projection = loader.find_player('Test Player')

        assert projection is not None
        # TOV should default to 0.0 when column is missing
        assert projection.tov_mean == 0.0
        assert projection.tov_mode == 0.0
        assert projection.tov_ci_low == 0.0
        assert projection.tov_ci_high == 0.0


class TestInteractiveCLIHelpTextContainsTov:
    """Test that HELP_TEXT includes tov/turnovers."""

    def test_help_text_contains_tov(self):
        """Verify HELP_TEXT mentions tov/turnovers."""
        from src.query.interactive_cli import InteractiveCLI

        assert 'tov' in InteractiveCLI.HELP_TEXT.lower()
        assert 'turnover' in InteractiveCLI.HELP_TEXT.lower()

    def test_help_text_lists_all_six_stats(self):
        """Verify HELP_TEXT lists all 6 stats."""
        from src.query.interactive_cli import InteractiveCLI

        help_lower = InteractiveCLI.HELP_TEXT.lower()

        # Check all stats are mentioned
        assert 'pts' in help_lower or 'point' in help_lower
        assert 'reb' in help_lower or 'rebound' in help_lower
        assert 'ast' in help_lower or 'assist' in help_lower
        assert 'stl' in help_lower or 'steal' in help_lower
        assert 'blk' in help_lower or 'block' in help_lower
        assert 'tov' in help_lower or 'turnover' in help_lower


class TestQueryAllSixStatsReturnValues:
    """Test that querying all 6 stats returns real values, not just 0.0 defaults."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory for CSV input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def complete_csv(self, temp_data_dir):
        """Create a CSV with all 6 stats properly populated with non-zero values."""
        csv_content = """GAME_ID,DATE,PLAYER_NAME,TEAM,OPPONENT,IS_HOME,PROJ_PTS_MEAN,PROJ_PTS_MODE,PTS_CI_LOW,PTS_CI_HIGH,PROJ_REB_MEAN,PROJ_REB_MODE,REB_CI_LOW,REB_CI_HIGH,PROJ_AST_MEAN,PROJ_AST_MODE,AST_CI_LOW,AST_CI_HIGH,PROJ_STL_MEAN,PROJ_STL_MODE,STL_CI_LOW,STL_CI_HIGH,PROJ_BLK_MEAN,PROJ_BLK_MODE,BLK_CI_LOW,BLK_CI_HIGH,PROJ_TOV_MEAN,PROJ_TOV_MODE,TOV_CI_LOW,TOV_CI_HIGH,PLAY_PROBABILITY
TEST_001,2025-01-15,Test Player,BOS,LAL,1,25.0,24.5,18.0,32.0,8.0,8.0,5.0,11.0,6.0,6.0,3.0,9.0,1.2,1.0,0.0,3.0,0.8,1.0,0.0,2.0,2.5,2.0,1.0,4.0,1.0"""

        filepath = os.path.join(temp_data_dir, 'player_projections_20250115_120000.csv')
        with open(filepath, 'w') as f:
            f.write(csv_content)
        return filepath

    def test_all_six_stats_return_nonzero_values(self, temp_data_dir, complete_csv):
        """Verify all 6 stats return real loaded values (not 0.0 defaults)."""
        loader = ProjectionLoader(data_dir=temp_data_dir)
        projection = loader.find_player('Test Player')

        assert projection is not None

        # All means should be non-zero
        assert projection.pts_mean != 0.0, "PTS mean should not be 0.0"
        assert projection.reb_mean != 0.0, "REB mean should not be 0.0"
        assert projection.ast_mean != 0.0, "AST mean should not be 0.0"
        assert projection.stl_mean != 0.0, "STL mean should not be 0.0"
        assert projection.blk_mean != 0.0, "BLK mean should not be 0.0"
        assert projection.tov_mean != 0.0, "TOV mean should not be 0.0"

        # Verify specific expected values
        assert projection.pts_mean == 25.0
        assert projection.reb_mean == 8.0
        assert projection.ast_mean == 6.0
        assert projection.stl_mean == 1.2
        assert projection.blk_mean == 0.8
        assert projection.tov_mean == 2.5

    def test_stl_blk_tov_are_not_ignored(self, temp_data_dir, complete_csv):
        """Verify STL, BLK, and TOV are read correctly (not silently ignored)."""
        loader = ProjectionLoader(data_dir=temp_data_dir)
        projection = loader.find_player('Test Player')

        assert projection is not None

        # STL verification
        assert projection.stl_mean == 1.2
        assert projection.stl_mode == 1.0
        assert projection.stl_ci_low == 0.0
        assert projection.stl_ci_high == 3.0

        # BLK verification
        assert projection.blk_mean == 0.8
        assert projection.blk_mode == 1.0
        assert projection.blk_ci_low == 0.0
        assert projection.blk_ci_high == 2.0

        # TOV verification
        assert projection.tov_mean == 2.5
        assert projection.tov_mode == 2.0
        assert projection.tov_ci_low == 1.0
        assert projection.tov_ci_high == 4.0


class TestStatDisplayNamesComplete:
    """Test that STAT_DISPLAY_NAMES includes all 6 stats."""

    def test_stat_display_names_has_all_six_stats(self):
        """Verify STAT_DISPLAY_NAMES contains all 6 stats."""
        expected_stats = ['pts', 'reb', 'ast', 'stl', 'blk', 'tov']

        for stat in expected_stats:
            assert stat in ProjectionLoader.STAT_DISPLAY_NAMES, f"Missing stat in STAT_DISPLAY_NAMES: {stat}"

    def test_stat_display_names_values(self):
        """Verify STAT_DISPLAY_NAMES values are human-readable."""
        display_names = ProjectionLoader.STAT_DISPLAY_NAMES

        assert display_names['pts'] == 'Points'
        assert display_names['reb'] == 'Rebounds'
        assert display_names['ast'] == 'Assists'
        assert display_names['stl'] == 'Steals'
        assert display_names['blk'] == 'Blocks'
        assert display_names['tov'] == 'Turnovers'
