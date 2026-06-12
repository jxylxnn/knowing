"""Regenerate project-brain/FULL_PROJECT_DUMP.md from live project files.

Run from the repo root. Writes to project-brain/FULL_PROJECT_DUMP.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DUMP = REPO / "project-brain" / "FULL_PROJECT_DUMP.md"

HEADER = (
    "# KNOWING — NBA Player Stats Prediction System\n"
    "## Complete Project Dump\n"
    "\n"
)

SEPARATOR = (
    "# =============================================================================\n"
)

DIVIDER = SEPARATOR * 2


def section(title: str) -> str:
    return f"{DIVIDER}# PART {title}\n{DIVIDER}\n\n\n"


def file_block(rel_path: str) -> str:
    full = REPO / rel_path
    body = full.read_text(encoding="utf-8")
    return (
        f"{DIVIDER}"
        f"# FILE: {rel_path}\n"
        f"{DIVIDER}\n"
        f"\n"
        f"{body}"
        f"\n\n\n"
    )


def main() -> int:
    parts: list[str] = [HEADER]

    parts.append(section("1: PROJECT-BRAIN DOCUMENTS"))
    for rel in [
        "project-brain/ARCHITECTURE.md",
        "project-brain/CODE_RULES.md",
        "project-brain/CURRENT_STATE.md",
        "project-brain/DECISIONS.md",
        "project-brain/FILE_MAP.md",
        "project-brain/KNOWN_BUGS.md",
        "project-brain/PROJECT_CONTEXT.md",
        "project-brain/TASKS.md",
        "project-brain/TRANSFORMER_SEQ_PLAN.md",
    ]:
        parts.append(file_block(rel))

    parts.append(section("2: ROOT-LEVEL SOURCE FILES"))
    for rel in [
        "AGENTS.md",
        "README.md",
        "requirements.txt",
        "update_data.py",
        "train.py",
        "simulate_season.py",
        "query_prob.py",
        "clear_cache.py",
        "backtest.py",
        "optimize_weights.py",
        "optimize_variance.py",
        "config/default.yaml",
    ]:
        parts.append(file_block(rel))

    parts.append(section("3: SRC/ FILES"))
    src_py = sorted(p for p in (REPO / "src").rglob("*.py") if p.is_file())
    for path in src_py:
        rel = path.relative_to(REPO).as_posix()
        parts.append(file_block(rel))

    parts.append(section("4: TEST FILES"))
    test_py = sorted(p for p in (REPO / "tests").rglob("*.py") if p.is_file())
    for path in test_py:
        rel = path.relative_to(REPO).as_posix()
        parts.append(file_block(rel))

    parts.append(section("5: OTHER CONFIG FILES"))
    for rel in [".gitignore"]:
        parts.append(file_block(rel))

    output = "".join(parts)
    DUMP.write_text(output, encoding="utf-8")
    line_count = output.count("\n")
    print(f"Wrote {DUMP} ({line_count} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
