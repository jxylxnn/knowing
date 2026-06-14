from __future__ import annotations

import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass

import pandas as pd


DIAG_PREFIX = "[TRAIN-DIAG]"


STAGES_ORDERED = [
    "preflight",
    "data_load",
    "feature_engineering",
    "feature_selection",
    "prepare_data",
    "artifact_check",
]


class DiagnosticStop(Exception):
    """Raised when a diagnostic stage matches --stop-after and should exit cleanly."""


class DiagnosticStageFailed(Exception):
    """Raised when a diagnostic stage raises an unhandled exception."""


@dataclass
class DiagnosticConfig:
    enabled: bool = False
    stop_after: str | None = None

    def should_stop_after(self, stage: str) -> bool:
        if not self.enabled or not self.stop_after:
            return False
        return self.stop_after == stage


@contextmanager
def diagnostic_stage(name: str, config: DiagnosticConfig):
    if not config.enabled:
        yield
        return

    print(f"{DIAG_PREFIX} START {name}", flush=True)
    start = time.time()
    try:
        yield
    except DiagnosticStop:
        raise
    except DiagnosticStageFailed:
        raise
    except Exception as exc:
        elapsed = time.time() - start
        print(f"{DIAG_PREFIX} FAILED {name} ({elapsed:.2f}s)", flush=True)
        print(f"{DIAG_PREFIX} Exception type: {type(exc).__name__}", flush=True)
        print(f"{DIAG_PREFIX} Exception message: {exc}", flush=True)
        traceback.print_exc()
        raise DiagnosticStageFailed(f"Stage '{name}' failed") from exc
    else:
        elapsed = time.time() - start
        print(f"{DIAG_PREFIX} OK {name} ({elapsed:.2f}s)", flush=True)

    if config.should_stop_after(name):
        print(f"{DIAG_PREFIX} Stopping after '{name}' as requested (--stop-after={config.stop_after}).", flush=True)
        raise DiagnosticStop(name)


def diagnostic_noop(name: str, config: DiagnosticConfig, reason: str = "disabled") -> None:
    """Print a SKIP marker for a stage that did not run, then check for early stop."""
    if not config.enabled:
        return
    print(f"{DIAG_PREFIX} START {name}", flush=True)
    print(f"{DIAG_PREFIX} SKIP {name} {reason}", flush=True)
    print(f"{DIAG_PREFIX} OK {name} (0.00s)", flush=True)
    if config.should_stop_after(name):
        print(f"{DIAG_PREFIX} Stopping after '{name}' as requested (--stop-after={config.stop_after}).", flush=True)
        raise DiagnosticStop(name)


def print_data_summary(merged_df: pd.DataFrame, full_df: pd.DataFrame | None = None) -> None:
    print(f"{DIAG_PREFIX} merged rows: {len(merged_df)}", flush=True)
    if full_df is not None:
        print(f"{DIAG_PREFIX} engineered rows: {len(full_df)}", flush=True)
        print(f"{DIAG_PREFIX} engineered cols: {len(full_df.columns)}", flush=True)

        targets = ["PTS", "REB", "AST", "STL", "BLK", "TOV"]
        present = [t for t in targets if t in full_df.columns]
        missing = [t for t in targets if t not in full_df.columns]
        print(f"{DIAG_PREFIX} target columns present: {', '.join(present)}" if present else f"{DIAG_PREFIX} target columns present: none", flush=True)
        print(f"{DIAG_PREFIX} missing target columns: {', '.join(missing) if missing else 'none'}", flush=True)
    else:
        cols = list(merged_df.columns)
        print(f"{DIAG_PREFIX} columns: {cols}", flush=True)


def print_selection_summary(manifest_payload: dict | None) -> None:
    if manifest_payload is None:
        print(f"{DIAG_PREFIX} feature selection enabled: false", flush=True)
        return
    print(f"{DIAG_PREFIX} feature selection enabled: true", flush=True)
    by_target = manifest_payload.get("selected_features_by_target", {})
    print(f"{DIAG_PREFIX} selected features by target:", flush=True)
    for target, feats in sorted(by_target.items()):
        print(f"{DIAG_PREFIX}   {target}: {len(feats)}", flush=True)
