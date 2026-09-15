import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from update_data import enrich_with_player_bios, fetch_player_logs_since


def test_game_log_fetch_passes_configured_timeout():
    endpoint = MagicMock()
    endpoint.get_data_frames.return_value = [pd.DataFrame()]

    with patch(
        "update_data.playergamelogs.PlayerGameLogs",
        return_value=endpoint,
    ) as fetch:
        result = fetch_player_logs_since(
            "2025-26",
            "2026-04-13",
            request_timeout=7,
        )

    assert result is None
    fetch.assert_called_once_with(
        season_nullable="2025-26",
        season_type_nullable="Regular Season",
        date_from_nullable="2026-04-13",
        timeout=7,
    )


def test_cached_bio_mode_never_calls_player_api(tmp_path):
    players = pd.DataFrame(
        {
            "PLAYER_ID": [1, 2],
            "PLAYER_NAME": ["Cached Player", "Uncached Player"],
            "GAME_DATE": ["2026-01-15", "2026-01-15"],
            "POSITION": [None, "Guard"],
        }
    )
    cached_bios = pd.DataFrame(
        {
            "PLAYER_ID": [1],
            "BIRTHDATE": ["2000-01-15"],
            "POSITION": ["Forward"],
            "HEIGHT": ["6-8"],
            "WEIGHT": [220],
            "DRAFT_YEAR": [2020],
            "CAREER_START": [2020],
            "YEARS_EXPERIENCE": [6],
        }
    )
    cached_bios.to_csv(tmp_path / "player_bios.csv", index=False)

    with patch(
        "src.data.player_bio_scraper.PlayerBioScraper.fetch_all_bios"
    ) as fetch_all:
        enriched = enrich_with_player_bios(
            players,
            str(tmp_path),
            refresh_missing=False,
        )

    fetch_all.assert_not_called()
    cached_row = enriched.loc[enriched["PLAYER_ID"] == 1].iloc[0]
    uncached_row = enriched.loc[enriched["PLAYER_ID"] == 2].iloc[0]
    assert cached_row["POSITION"] == "Forward"
    assert cached_row["AGE"] == 26.0
    assert uncached_row["POSITION"] == "Guard"


def test_colab_quick_mode_is_bounded_and_colab_compatible():
    notebook_path = Path(__file__).resolve().parents[2] / "train_colab.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][1]["source"])
    requirements_source = (
        notebook_path.parent / "requirements.txt"
    ).read_text(encoding="utf-8")
    requirements_pins = {}
    for line in requirements_source.splitlines():
        if "==" in line and not line.lstrip().startswith("#"):
            distribution, version = line.split("==", 1)
            requirements_pins[distribution] = version

    assert 'DATA_REFRESH = "quick"' in source
    assert '["--bio-mode", "cached", "--request-timeout", "10"]' in source
    assert 'DATA_REFRESH == "current_files"' in source
    assert "create_source_snapshot(DATA_DIR)" in source
    assert 'DATA_REFRESH == "reuse_latest"' in source
    for distribution in (
        "scipy", "scikit-learn", "PyYAML", "joblib", "nba_api",
        "curl-cffi",
    ):
        assert (
            f'"{distribution}=={requirements_pins[distribution]}"' in source
        )
    assert '"numpy>=2.0,<2.3"' in source
    assert '"pandas==2.2.3"' in source
    assert '"requests==2.32.4"' in source
    assert '"psutil>=5.9.0"' in source
    assert '"numpy==2.3.5"' not in source
    assert '"pandas==2.3.3"' not in source
    assert '"requests==2.32.5"' not in source
    assert "from importlib import metadata" in source
    assert "module in sys.modules" in source
    assert "loaded_versions.items()" in source
    assert "Version('2.3')" in source
    assert "import numpy, numpy.strings, pandas" in source
    assert "kernel.do_shutdown(restart=True)" in source
    assert "allowed_returncodes=(0, 2)" in source
    assert 'canonical_report.get("invalid_games", 0)' in source
    assert 'canonical_report.get("missing_player_team_game", 0)' in source
    assert "checking whether core coverage remains sufficient" in source
    assert source.index("packages = [") < source.index(
        "from google.colab import drive"
    )
