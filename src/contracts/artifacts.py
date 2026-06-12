from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import time

from src.contracts.errors import ArtifactContractError

CANONICAL_TARGETS = ("PTS", "REB", "AST", "STL", "BLK", "TOV")


@dataclass(frozen=True)
class ArtifactContract:
    models_dir: Path
    transformer_required: bool = False
    max_age_hours: float | None = None
    residual_required: bool = False


def _required_files(transformer_required: bool, residual_required: bool = False) -> list[str]:
    files: list[str] = []

    for target in CANONICAL_TARGETS:
        lower = target.lower()
        files.append(f"{lower}_catboost.cbm")
        files.append(f"{lower}_metadata.joblib")

    files.extend(
        [
            "feature_schema.pkl",
            "feature_cols.pkl",
            "blend_weights.pkl",
            "model_stack_metadata.pkl",
        ]
    )

    if transformer_required:
        files.append("attention_transformer.pkl")

    if residual_required:
        for target in CANONICAL_TARGETS:
            files.append(f"residual/{target.lower()}_residual.cbm")
        files.append("residual/residual_metadata.json")
        files.append("residual/residual_feature_schema.json")

    return files


def validate_runtime_artifacts(contract: ArtifactContract) -> None:
    models_dir = Path(contract.models_dir)

    if not models_dir.exists():
        raise ArtifactContractError(f"Models directory does not exist: {models_dir}")

    missing = [
        name
        for name in _required_files(contract.transformer_required, contract.residual_required)
        if not (models_dir / name).exists()
    ]
    if missing:
        raise ArtifactContractError("Missing required runtime artifacts:\n" + "\n".join(f"- {name}" for name in missing))

    if contract.max_age_hours is not None:
        newest_allowed_age = contract.max_age_hours * 3600
        now = time.time()
        stale = []
        for name in _required_files(contract.transformer_required, contract.residual_required):
            path = models_dir / name
            age_seconds = now - path.stat().st_mtime
            if age_seconds > newest_allowed_age:
                stale.append(name)

        if stale:
            raise ArtifactContractError(
                f"Artifacts older than {contract.max_age_hours} hours:\n"
                + "\n".join(f"- {name}" for name in stale)
            )

    _validate_metadata(models_dir / "model_stack_metadata.pkl")


def _validate_metadata(path: Path) -> None:
    try:
        with path.open("rb") as f:
            metadata = pickle.load(f)
    except Exception as exc:
        raise ArtifactContractError(f"Could not load model stack metadata: {path}") from exc

    expected_targets = set(CANONICAL_TARGETS)
    actual_targets = set(metadata.get("targets", CANONICAL_TARGETS))

    if actual_targets != expected_targets:
        raise ArtifactContractError(
            f"Model metadata targets mismatch. Expected {sorted(expected_targets)}, got {sorted(actual_targets)}"
        )
