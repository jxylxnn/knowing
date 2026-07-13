from src.reasoning import PlayerReasoningReport, load_reasoning_sidecar, sidecar_path, write_reasoning_sidecar


def test_reasoning_sidecar_round_trip(tmp_path):
    projection = tmp_path / "player_projections_20260709_120000.csv"
    target = sidecar_path(projection)
    report = PlayerReasoningReport(player_name="Player One", model_version="v1")
    write_reasoning_sidecar([report], target)
    payload = load_reasoning_sidecar(target)
    assert target.name == "player_reasoning_20260709_120000.jsonl"
    assert payload["player one"]["model_version"] == "v1"
