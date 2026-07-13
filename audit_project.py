"""Read-only repository, data, and model baseline audit.

The audit intentionally does not import the runtime pipeline or contact any
external service.  It is safe to run against a legacy artifact directory: an
unsafe schema is reported in JSON and does not, by itself, make the command
fail.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import pickle
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


REPORT_SCHEMA_VERSION = "audit_report_v1"
OPTIONAL_DATA_FILES = (
    "player_bios.csv",
    "injury_history.csv",
    "advanced_tracking.csv",
)
CURRENT_GAME_TEAM_OUTCOMES = frozenset(
    {
        "MIN",
        "PTS",
        "REB",
        "AST",
        "STL",
        "BLK",
        "TOV",
        "FGM",
        "FGA",
        "FG_PCT",
        "FG3M",
        "FG3A",
        "FG3_PCT",
        "FTM",
        "FTA",
        "FT_PCT",
        "OREB",
        "DREB",
        "PF",
        "PFD",
        "PLUS_MINUS",
    }
)
TEAM_OUTCOME_PATTERN = re.compile(
    r"^(?:[A-Z0-9]+_)*(?:" + "|".join(sorted(CURRENT_GAME_TEAM_OUTCOMES, key=len, reverse=True)) + r")_TEAM$"
)


class AuditInputError(RuntimeError):
    """Raised when an explicitly requested audit input cannot be read."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise AuditInputError(f"Could not read {path}: {exc}") from exc
    return digest.hexdigest()


def _file_record(root: Path, path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError as exc:
        raise AuditInputError(f"Could not stat {path}: {exc}") from exc
    return {
        "path": str(path.relative_to(root)),
        "size_bytes": stat.st_size,
        "sha256": _sha256(path),
    }


def _iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        raise AuditInputError(f"Directory does not exist: {root}")
    if not root.is_dir():
        raise AuditInputError(f"Expected a directory: {root}")
    try:
        return iter(sorted(path for path in root.rglob("*") if path.is_file()))
    except OSError as exc:
        raise AuditInputError(f"Could not enumerate {root}: {exc}") from exc


def _csv_record(path: Path, root: Path) -> dict[str, Any]:
    record = _file_record(root, path)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or [])
            row_count = 0
            dates: list[str] = []
            game_ids: set[str] = set()
            player_ids: set[str] = set()
            for row in reader:
                row_count += 1
                date_value = next(
                    (
                        row.get(name, "").strip()
                        for name in ("GAME_DATE", "GAME_DATE_EST", "DATE", "EVENT_DATE")
                        if row.get(name, "").strip()
                    ),
                    "",
                )
                if date_value:
                    dates.append(date_value)
                game_id = row.get("GAME_ID", "").strip()
                if game_id:
                    game_ids.add(game_id)
                player_id = row.get("PLAYER_ID", "").strip()
                if player_id:
                    player_ids.add(player_id)
    except (OSError, csv.Error, UnicodeError) as exc:
        raise AuditInputError(f"Could not read CSV {path}: {exc}") from exc

    record.update(
        {
            "row_count": row_count,
            "column_count": len(columns),
            "columns": columns,
            "date_min": min(dates) if dates else None,
            "date_max": max(dates) if dates else None,
            "game_count": len(game_ids),
            "player_count": len(player_ids),
        }
    )
    return record


def _data_report(data_dir: Path) -> dict[str, Any]:
    files = list(_iter_files(data_dir))
    file_records = [_file_record(data_dir, path) for path in files]
    csv_records = [
        _csv_record(path, data_dir)
        for path in files
        if path.suffix.lower() == ".csv"
    ]
    missing_optional = [
        name for name in OPTIONAL_DATA_FILES if not (data_dir / name).exists()
    ]
    return {
        "directory": str(data_dir),
        "files": file_records,
        "csv_tables": csv_records,
        "missing_optional_files": missing_optional,
    }


def _extract_feature_columns(value: Any) -> list[str] | None:
    if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
        return list(value)
    if isinstance(value, dict):
        for key in ("feature_cols", "features", "columns"):
            columns = _extract_feature_columns(value.get(key))
            if columns is not None:
                return columns
    for attribute in ("feature_cols", "features", "columns"):
        columns = _extract_feature_columns(getattr(value, attribute, None))
        if columns is not None:
            return columns
    return None


def _load_feature_columns(path: Path) -> tuple[list[str] | None, str | None]:
    if not path.exists():
        return None, "missing"
    try:
        with path.open("rb") as handle:
            value = pickle.load(handle)
    except (OSError, EOFError, pickle.PickleError, AttributeError, ImportError) as exc:
        return None, f"unreadable: {exc}"
    columns = _extract_feature_columns(value)
    if columns is None:
        return None, "does not contain a feature column list"
    return columns, None


def _forbidden_features(columns: Iterable[str]) -> list[str]:
    return sorted(
        {
            column
            for column in columns
            if column in {f"{name}_TEAM" for name in CURRENT_GAME_TEAM_OUTCOMES}
            or TEAM_OUTCOME_PATTERN.fullmatch(column)
        }
    )


def _feature_schema_report(models_dir: Path) -> dict[str, Any]:
    feature_cols, feature_cols_error = _load_feature_columns(models_dir / "feature_cols.pkl")
    schema_cols, schema_error = _load_feature_columns(models_dir / "feature_schema.pkl")
    selected = feature_cols or schema_cols or []
    return {
        "feature_cols_count": len(feature_cols) if feature_cols is not None else None,
        "feature_schema_count": len(schema_cols) if schema_cols is not None else None,
        "feature_cols_error": feature_cols_error,
        "feature_schema_error": schema_error,
        "forbidden_features": _forbidden_features(selected),
        "feature_cols_matches_schema": (
            feature_cols == schema_cols
            if feature_cols is not None and schema_cols is not None
            else None
        ),
    }


def _models_report(models_dir: Path) -> dict[str, Any]:
    files = list(_iter_files(models_dir))
    return {
        "directory": str(models_dir),
        "files": [_file_record(models_dir, path) for path in files],
        "manifests": {
            "champion_json": (models_dir / "champion.json").exists(),
            "bundle_manifest_json": (models_dir / "bundle_manifest.json").exists(),
        },
        "feature_schema": _feature_schema_report(models_dir),
    }


def _git_report(repo_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"root": str(repo_root)}
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        status = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        result["error"] = str(exc)
        return result
    result.update(
        {
            "commit": commit.stdout.strip(),
            "dirty": bool(status.stdout.strip()),
        }
    )
    return result


def _requirement_names(path: Path) -> list[str]:
    if not path.exists():
        return []
    names: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith(("-", ".")):
            continue
        name = re.split(r"[<>=!~;\[\s]", line, maxsplit=1)[0]
        if name:
            names.append(name)
    return names


def _dependency_report(repo_root: Path) -> dict[str, Any]:
    names: list[str] = []
    for filename in ("requirements.txt", "requirements-dev.txt"):
        names.extend(_requirement_names(repo_root / filename))
    versions: dict[str, str | None] = {}
    for name in sorted(set(names), key=str.lower):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def build_report(
    data_dir: Path,
    models_dir: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable report without changing any input."""

    data_dir = Path(data_dir).expanduser().resolve()
    models_dir = Path(models_dir).expanduser().resolve()
    root = Path(repo_root or Path(__file__).resolve().parent).expanduser().resolve()
    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "runtime": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "dependencies": _dependency_report(root),
        },
        "repository": _git_report(root),
        "data": _data_report(data_dir),
        "models": _models_report(models_dir),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Data directory to inspect")
    parser.add_argument("--models-dir", default="models", help="Model directory to inspect")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = build_report(Path(args.data_dir), Path(args.models_dir))
    except AuditInputError as exc:
        print(
            json.dumps(
                {
                    "report_schema_version": REPORT_SCHEMA_VERSION,
                    "status": "error",
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        print(
            json.dumps(
                {
                    "report_schema_version": REPORT_SCHEMA_VERSION,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
