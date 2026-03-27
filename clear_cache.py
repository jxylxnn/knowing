#!/usr/bin/env python3
"""Remove generated caches and artifacts from the NBA prediction project."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List, Set


PROJECT_ROOT = Path(__file__).resolve().parent
_SKIP_DIR_NAMES = {"venv", ".venv", "env", ".env"}


def _resolve_path(value: str | None, default: Path) -> Path:
    if value is None or value == "":
        return default
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _collect_targets(root: Path) -> List[Path]:
    """Collect generated directories and Python cache folders to remove."""
    targets: Set[Path] = set()
    direct_dirs = [
        root / "cache",
        root / "data" / "cache",
        root / "data" / "sim_cache",
        root / "data" / "sim_results",
        root / "models",
        root / "experiments",
        root / ".pytest_cache",
    ]

    for path in direct_dirs:
        if path.exists():
            targets.add(path)

    for path in root.rglob("__pycache__"):
        if _SKIP_DIR_NAMES.intersection(path.parts):
            continue
        targets.add(path)

    return sorted(targets, key=lambda p: len(p.parts), reverse=True)


def _remove_path(path: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] remove {path}")
        return

    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)
    print(f"removed {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Delete generated cache directories and artifacts."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Remove all generated cache and artifact directories.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without deleting anything.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Override the data directory to clean (default: data).",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Override the models directory to clean (default: models).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="cache",
        help="Override the root cache directory to clean (default: cache).",
    )
    parser.add_argument(
        "--experiments-dir",
        type=str,
        default="experiments",
        help="Override the experiments directory to clean (default: experiments).",
    )

    args = parser.parse_args()

    if not args.all:
        parser.print_help()
        return 1

    data_dir = _resolve_path(args.data_dir, PROJECT_ROOT / "data")
    models_dir = _resolve_path(args.models_dir, PROJECT_ROOT / "models")
    cache_dir = _resolve_path(args.cache_dir, PROJECT_ROOT / "cache")
    experiments_dir = _resolve_path(args.experiments_dir, PROJECT_ROOT / "experiments")

    targets = _collect_targets(PROJECT_ROOT)
    for path in [data_dir / "cache", data_dir / "sim_cache", data_dir / "sim_results", models_dir, cache_dir, experiments_dir]:
        if path.exists() and path not in targets:
            targets.append(path)

    if not targets:
        print("No generated cache or artifact directories were found.")
        return 0

    print("The following generated paths will be removed:")
    for path in targets:
        print(f"  - {path}")

    if not args.yes and not args.dry_run:
        confirm = input("\nType 'yes' to continue: ").strip().lower()
        if confirm != "yes":
            print("Aborted.")
            return 1

    for path in targets:
        _remove_path(path, args.dry_run)

    if args.dry_run:
        print("\nDry run complete. No files were deleted.")
    else:
        print("\nCleanup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
