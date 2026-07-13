import pandas as pd

from src.preprocessing.data_loader import DataLoader
from src.utils.prediction_utils import FeatureSelector


def test_current_game_team_outcomes_never_enter_feature_schema():
    frame = pd.DataFrame(
        {
            "PTS": [10.0, 12.0],
            "PTS_TEAM": [110.0, 120.0],
            "REB_TEAM": [45.0, 48.0],
            "TEAM_PACE_10": [99.0, 101.0],
            "PTS_PLAYER_TE": [9.0, 10.0],
            "UNSAFE_TELEMETRY": [1.0, 2.0],
        }
    )
    schema = FeatureSelector(targets=["PTS"]).fit(frame)
    assert "PTS_TEAM" not in schema.feature_cols
    assert "REB_TEAM" not in schema.feature_cols
    assert "TEAM_PACE_10" in schema.feature_cols
    assert "PTS_PLAYER_TE" in schema.feature_cols
    assert "UNSAFE_TELEMETRY" not in schema.feature_cols


def test_data_loader_does_not_merge_same_game_team_outcomes(tmp_path):
    players_path = tmp_path / "players.csv"
    games_path = tmp_path / "games.csv"
    players_path.write_text(
        "PLAYER_ID,PLAYER_NAME,TEAM_ID,GAME_ID,GAME_DATE,PTS,REB,AST,MIN\n"
        "1,A,10,1,2026-01-01,10,4,2,20\n"
        "1,A,10,2,2026-01-03,12,5,3,22\n"
        "1,A,10,3,2026-01-05,14,6,4,24\n",
        encoding="utf-8",
    )
    games_path.write_text(
        "GAME_ID,TEAM_ID,GAME_DATE,PTS,REB,AST,FGA,FTA,FGM,OREB,DREB,TOV\n"
        "1,10,2026-01-01,100,40,20,80,20,35,8,32,10\n"
        "1,20,2026-01-01,90,42,18,78,18,33,7,35,12\n"
        "2,10,2026-01-03,105,41,22,82,21,37,9,32,11\n"
        "2,20,2026-01-03,95,39,19,79,17,34,6,33,13\n"
        "3,10,2026-01-05,110,43,24,84,22,39,10,33,9\n"
        "3,20,2026-01-05,98,40,20,81,19,36,8,32,12\n",
        encoding="utf-8",
    )

    merged = DataLoader(str(players_path), str(games_path)).merge_datasets()

    assert "PTS_TEAM" not in merged.columns
    assert "REB_TEAM" not in merged.columns
    assert "AST_TEAM" not in merged.columns
    assert "FGA_TEAM" not in merged.columns
    assert "FTA_TEAM" not in merged.columns
    assert "OREB_TEAM" not in merged.columns
    assert "DREB_TEAM" not in merged.columns
    assert "TOV_TEAM" not in merged.columns
    assert "OPPONENT_ID" in merged.columns
    assert "TEAM_PTS_ROLL_5" in merged.columns


def _write_mutation_fixture(tmp_path, *, player_pts=None, game_pts=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    player_pts = player_pts or {1: 10, 2: 12, 3: 14, 4: 16, 5: 18}
    game_pts = game_pts or {1: (100, 90), 2: (101, 91), 3: (102, 92), 4: (103, 93), 5: (104, 94)}
    players_rows = [
        "PLAYER_ID,PLAYER_NAME,TEAM_ID,GAME_ID,GAME_DATE,PTS,REB,AST,MIN"
    ]
    for game_id, pts in player_pts.items():
        players_rows.append(
            f"1,A,10,{game_id},2026-01-{game_id:02d},{pts},4,2,20"
        )
    games_rows = [
        "GAME_ID,TEAM_ID,GAME_DATE,PTS,REB,AST,FGA,FTA,FGM,OREB,DREB,TOV"
    ]
    for game_id, (home_pts, away_pts) in game_pts.items():
        games_rows.extend(
            [
                f"{game_id},10,2026-01-{game_id:02d},{home_pts},40,20,80,20,35,8,32,10",
                f"{game_id},20,2026-01-{game_id:02d},{away_pts},42,18,78,18,33,7,35,12",
            ]
        )
    players_path = tmp_path / "players.csv"
    games_path = tmp_path / "games.csv"
    players_path.write_text("\n".join(players_rows) + "\n", encoding="utf-8")
    games_path.write_text("\n".join(games_rows) + "\n", encoding="utf-8")
    return players_path, games_path


def _safe_context(merged):
    columns = [
        column
        for column in merged.columns
        if column == "OPPONENT_ID"
        or column.startswith("TEAM_")
        or column.startswith("OPP_TEAM_")
    ]
    return merged.loc[merged["GAME_ID"] == 4, columns].sort_index(axis=1).reset_index(drop=True)


def test_same_game_player_and_team_mutations_do_not_change_pregame_context(tmp_path):
    players_path, games_path = _write_mutation_fixture(tmp_path)
    baseline = _safe_context(DataLoader(str(players_path), str(games_path)).merge_datasets())

    changed_players, unchanged_games = _write_mutation_fixture(
        tmp_path / "player_mutation", player_pts={1: 10, 2: 12, 3: 14, 4: 999, 5: 18}
    )
    changed_player_context = _safe_context(
        DataLoader(str(changed_players), str(unchanged_games)).merge_datasets()
    )

    _, changed_games = _write_mutation_fixture(
        tmp_path / "team_mutation", game_pts={1: (100, 90), 2: (101, 91), 3: (102, 92), 4: (999, 1), 5: (104, 94)}
    )
    changed_team_context = _safe_context(
        DataLoader(str(players_path), str(changed_games)).merge_datasets()
    )

    assert baseline.equals(changed_player_context)
    assert baseline.equals(changed_team_context)


def test_later_game_mutation_does_not_change_earlier_context(tmp_path):
    players_path, games_path = _write_mutation_fixture(tmp_path)
    baseline = _safe_context(DataLoader(str(players_path), str(games_path)).merge_datasets())
    _, changed_games = _write_mutation_fixture(
        tmp_path / "future_mutation", game_pts={1: (100, 90), 2: (101, 91), 3: (102, 92), 4: (103, 93), 5: (999, 1)}
    )
    changed = _safe_context(
        DataLoader(str(players_path), str(changed_games)).merge_datasets()
    )
    assert baseline.equals(changed)


def test_extension_prefix_cannot_reallow_forbidden_exact_column():
    selector = FeatureSelector(targets=["PTS"])
    selector.SAFE_PREFIXES = tuple(selector.SAFE_PREFIXES) + ("PTS_TEAM",)
    assert selector._is_safe_feature("PTS_TEAM") is False
