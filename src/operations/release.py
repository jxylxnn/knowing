"""Content-addressed external release evidence for a sealed Model-v2 candidate.

Promotion gates answer "is this candidate good enough?" from the bundle itself.
Release evidence answers "may this sealed candidate be released?" from evidence
that lives outside the bundle: seven consecutive strict shadow slates over the
complete expected calendar, recomputed here rather than read from
``promotion_decision.json``.

The evidence record is immutable and content-addressed by its own canonical
JSON, so releasing the same candidate twice publishes one identical file and a
later state of the shadow store produces a different ``evidence_id`` instead of
silently rewriting history. There is deliberately no mutable "latest" pointer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
from numbers import Integral
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

from src.models.bundle import (
    _validate_and_evaluate_promotion_evidence,
    validate_v2_bundle,
)
from src.models.versioning import ModelBundleManifest
from src.utils.file_lock import exclusive_file_lock


RELEASE_EVIDENCE_SCHEMA_VERSION = "release_evidence_v1"
DEFAULT_RELEASE_EVIDENCE_ROOT = "models/release_evidence"

# Fields that describe where the record is stored and when it was written.
# Neither participates in the content address.
_IDENTITY_EXCLUDED = frozenset({"evidence_id", "created_at"})


@dataclass(frozen=True)
class ReleaseEvidenceRecord:
    """A published release-evidence payload and its content-addressed location."""

    bundle_id: str
    evidence_id: str
    path: Path
    eligible: bool
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class ReleaseEvidenceVerification:
    """Result of re-deriving one evidence record from its live dependencies."""

    bundle_id: str
    evidence_id: str
    path: Path
    eligible: bool
    model_evidence_eligible: bool
    shadow_evidence_eligible: bool
    dirty_worktree: bool
    failure_reasons: tuple[str, ...]


class ReleaseEvidenceStore:
    """Publish and verify external release evidence for sealed v2 candidates."""

    def __init__(self, root: str | Path = DEFAULT_RELEASE_EVIDENCE_ROOT) -> None:
        self.root = Path(root)

    def path_for(self, bundle_id: str, evidence_id: str) -> Path:
        return (
            self.root
            / _safe_path_segment(bundle_id, "bundle ID")
            / f"{evidence_id}.json"
        )

    def build(
        self,
        *,
        candidate_dir: str | Path,
        shadow_store,
        run_store,
        data_dir: str | Path,
        horizons: Iterable[str],
        expected_calendar: Mapping[str, Iterable[str]],
    ) -> ReleaseEvidenceRecord:
        """Recompute and publish release evidence; identical content is reused."""

        normalized_horizons = _normalize_horizons(horizons)
        normalized_calendar = _normalize_calendar(expected_calendar)
        root, manifest, validation = _validate_candidate(candidate_dir)
        model_evidence = _recompute_model_evidence(root)
        shadow_evidence = _recompute_shadow_evidence(
            shadow_store=shadow_store,
            bundle_id=manifest.bundle_id,
            horizons=normalized_horizons,
            calendar=normalized_calendar,
            run_store=run_store,
            data_dir=data_dir,
        )
        dirty_worktree = bool(manifest.dirty_worktree)
        eligible = bool(
            model_evidence["eligible"]
            and shadow_evidence["eligible"]
            and not dirty_worktree
        )
        payload: dict[str, Any] = {
            "schema_version": RELEASE_EVIDENCE_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "bundle_id": manifest.bundle_id,
            "bundle_manifest_sha256": _sha256_file(
                root / ModelBundleManifest.FILE_NAME
            ),
            "source_snapshot_id": validation.source_snapshot_id,
            "horizons": list(normalized_horizons),
            "expected_calendar": {
                day: list(games) for day, games in normalized_calendar.items()
            },
            "model_evidence": model_evidence,
            "shadow_evidence": shadow_evidence,
            "dirty_worktree": dirty_worktree,
            "eligible": eligible,
            "failure_reasons": _failure_reasons(
                model_evidence, shadow_evidence, dirty_worktree
            ),
        }
        payload = _json_native(payload)
        evidence_id = _evidence_id(payload)
        payload["evidence_id"] = evidence_id
        path, published = self._publish(
            self.path_for(manifest.bundle_id, evidence_id), payload
        )
        return ReleaseEvidenceRecord(
            bundle_id=str(published["bundle_id"]),
            evidence_id=str(published["evidence_id"]),
            path=path,
            eligible=bool(published["eligible"]),
            payload=published,
        )

    def verify(
        self,
        path: str | Path,
        *,
        candidate_dir: str | Path,
        shadow_store,
        run_store,
        data_dir: str | Path,
        horizons: Iterable[str],
        expected_calendar: Mapping[str, Iterable[str]],
    ) -> ReleaseEvidenceVerification:
        """Re-derive one record and reject stale, mistargeted, or tampered evidence."""

        target = Path(path)
        payload = _read_json_object(target, "release evidence")
        if payload.get("schema_version") != RELEASE_EVIDENCE_SCHEMA_VERSION:
            raise ValueError(
                "Release evidence schema version is not "
                f"{RELEASE_EVIDENCE_SCHEMA_VERSION}"
            )
        bundle_id = payload.get("bundle_id")
        evidence_id = payload.get("evidence_id")
        if not isinstance(bundle_id, str) or not bundle_id.strip():
            raise ValueError("Release evidence is missing its candidate bundle ID")
        if not isinstance(evidence_id, str) or not evidence_id.strip():
            raise ValueError("Release evidence is missing its evidence ID")
        created_at = payload.get("created_at")
        if created_at is not None and not isinstance(created_at, str):
            raise ValueError("Release evidence created_at must be a string")
        if _evidence_id(payload) != evidence_id:
            raise ValueError("Release evidence content does not match its evidence ID")
        if target.stem != evidence_id or target.parent.name != bundle_id:
            raise ValueError(
                "Release evidence path does not match its content address"
            )

        normalized_horizons = _normalize_horizons(horizons)
        normalized_calendar = _normalize_calendar(expected_calendar)
        root, manifest, validation = _validate_candidate(candidate_dir)
        if manifest.bundle_id != bundle_id:
            raise ValueError(
                "Release evidence does not belong to the supplied candidate"
            )
        manifest_digest = _sha256_file(root / ModelBundleManifest.FILE_NAME)
        if payload.get("bundle_manifest_sha256") != manifest_digest:
            raise ValueError(
                "Release evidence bundle manifest digest does not match the candidate"
            )
        if payload.get("source_snapshot_id") != validation.source_snapshot_id:
            raise ValueError(
                "Release evidence source snapshot does not match the candidate"
            )
        _assert_unchanged(
            payload.get("horizons"), list(normalized_horizons), "horizons"
        )
        _assert_unchanged(
            payload.get("expected_calendar"),
            {day: list(games) for day, games in normalized_calendar.items()},
            "expected calendar",
        )

        model_evidence = _recompute_model_evidence(root)
        _assert_unchanged(
            payload.get("model_evidence"), model_evidence, "model evidence"
        )
        shadow_evidence = _recompute_shadow_evidence(
            shadow_store=shadow_store,
            bundle_id=manifest.bundle_id,
            horizons=normalized_horizons,
            calendar=normalized_calendar,
            run_store=run_store,
            data_dir=data_dir,
        )
        _assert_unchanged(
            payload.get("shadow_evidence"), shadow_evidence, "shadow evidence"
        )
        dirty_worktree = bool(manifest.dirty_worktree)
        _assert_unchanged(
            payload.get("dirty_worktree"), dirty_worktree, "worktree status"
        )
        eligible = bool(
            model_evidence["eligible"]
            and shadow_evidence["eligible"]
            and not dirty_worktree
        )
        _assert_unchanged(payload.get("eligible"), eligible, "overall eligibility")
        reasons = _failure_reasons(model_evidence, shadow_evidence, dirty_worktree)
        _assert_unchanged(payload.get("failure_reasons"), reasons, "failure reasons")

        return ReleaseEvidenceVerification(
            bundle_id=bundle_id,
            evidence_id=evidence_id,
            path=target,
            eligible=eligible,
            model_evidence_eligible=bool(model_evidence["eligible"]),
            shadow_evidence_eligible=bool(shadow_evidence["eligible"]),
            dirty_worktree=dirty_worktree,
            failure_reasons=tuple(reasons),
        )

    def _publish(
        self, path: Path, payload: Mapping[str, Any]
    ) -> tuple[Path, dict[str, Any]]:
        """Atomically publish canonical JSON once; identical content is reused.

        Returns the published path together with the authoritative payload, so
        a caller never reports a document that differs from what is on disk.
        Reused evidence keeps its original ``created_at``.
        """

        identity = _identity_payload(payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_name(f".{path.name}.lock")
        with exclusive_file_lock(lock_path):
            if path.exists():
                existing = _read_json_object(path, "release evidence")
                if _canonical_bytes(_identity_payload(existing)) != _canonical_bytes(
                    identity
                ):
                    raise ValueError(
                        "Existing release evidence differs from the computed "
                        f"payload: {path}"
                    )
            else:
                encoded = _encode(payload)
                descriptor, temporary = tempfile.mkstemp(
                    prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
                )
                try:
                    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                        handle.write(encoded)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, path)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
            published = _read_json_object(path, "release evidence")
        return path, published


def _validate_candidate(candidate_dir: str | Path):
    """Seal-check a candidate and reload the manifest that evidence binds."""

    root = Path(candidate_dir)
    validation = validate_v2_bundle(root, promotion=False)
    manifest = ModelBundleManifest.load(root / ModelBundleManifest.FILE_NAME)
    if manifest.bundle_id != validation.bundle_id:
        raise ValueError(
            "Candidate manifest identity disagrees with its validation report"
        )
    return root, manifest, validation


def _recompute_model_evidence(root: Path) -> dict[str, Any]:
    """Recompute static promotion evidence from the immutable scorecards."""

    try:
        decision = _validate_and_evaluate_promotion_evidence(root)
    except ValueError as exc:
        return {
            "recomputed": False,
            "eligible": False,
            "checks": {},
            "reasons": [f"promotion evidence could not be recomputed: {exc}"],
        }
    return {
        "recomputed": True,
        "eligible": bool(decision.eligible),
        "checks": {
            str(name): bool(value) for name, value in dict(decision.checks).items()
        },
        "reasons": [str(reason) for reason in decision.reasons],
    }


def _recompute_shadow_evidence(
    *,
    shadow_store,
    bundle_id: str,
    horizons: tuple[str, ...],
    calendar: Mapping[str, list[str]],
    run_store,
    data_dir,
) -> dict[str, Any]:
    """Recompute exact shadow-slate evidence for one candidate and calendar."""

    if shadow_store is None or not hasattr(shadow_store, "promotion_evidence"):
        raise ValueError("A ShadowSlateStore is required to recompute shadow evidence")
    evidence = shadow_store.promotion_evidence(
        bundle_id=bundle_id,
        horizons=list(horizons),
        expected_calendar={day: list(games) for day, games in calendar.items()},
        run_store=run_store,
        data_dir=data_dir,
    )
    if not isinstance(evidence, Mapping):
        raise ValueError("Shadow evidence must be a mapping")
    payload = dict(evidence)
    payload["eligible"] = bool(payload.get("eligible"))
    return payload


def _failure_reasons(
    model_evidence: Mapping[str, Any],
    shadow_evidence: Mapping[str, Any],
    dirty_worktree: bool,
) -> list[str]:
    """Name every gate that blocks release; never soften a failed gate."""

    reasons: list[str] = []
    if not model_evidence.get("eligible"):
        detail = "; ".join(str(item) for item in model_evidence.get("reasons") or ())
        if model_evidence.get("recomputed"):
            head = "model promotion evidence is not eligible"
        else:
            head = "model promotion evidence could not be recomputed"
        reasons.append(f"{head}: {detail}" if detail else head)
    if not shadow_evidence.get("eligible"):
        detail = shadow_evidence.get("reason")
        if not detail:
            streak = shadow_evidence.get("consecutive_successes")
            required = shadow_evidence.get("required_consecutive_successes")
            detail = (
                f"{streak} of {required} consecutive successful slates"
                if required is not None
                else "insufficient consecutive successful slates"
            )
        reasons.append(f"shadow slate evidence is not eligible: {detail}")
    if dirty_worktree:
        reasons.append("bundle was built from a dirty worktree")
    return reasons


def _normalize_horizons(horizons: Iterable[str]) -> tuple[str, ...]:
    if horizons is None or isinstance(horizons, (str, bytes)) or not isinstance(
        horizons, Iterable
    ):
        raise ValueError("horizons must be a non-empty collection of names")
    values: list[str] = []
    for horizon in horizons:
        if not isinstance(horizon, str) or not horizon.strip():
            raise ValueError("horizon names must be non-empty strings")
        values.append(horizon)
    if not values:
        raise ValueError("horizons must not be empty")
    if len(set(values)) != len(values):
        raise ValueError("horizons must not contain duplicates")
    return tuple(sorted(values))


def _normalize_calendar(
    expected_calendar: Mapping[str, Iterable[str]],
) -> dict[str, list[str]]:
    if not isinstance(expected_calendar, Mapping):
        raise ValueError(
            "expected_calendar must be a mapping of ISO dates to game IDs"
        )
    if not expected_calendar:
        raise ValueError("expected_calendar must not be empty")
    normalized: dict[str, list[str]] = {}
    for raw_day, games in expected_calendar.items():
        day = _normalize_iso_date(raw_day)
        if day in normalized:
            raise ValueError(f"expected_calendar repeats the date {day}")
        normalized[day] = _normalize_game_ids(games, day)
    return dict(sorted(normalized.items()))


def _normalize_iso_date(raw_day) -> str:
    if isinstance(raw_day, datetime):
        return raw_day.date().isoformat()
    if isinstance(raw_day, date):
        return raw_day.isoformat()
    if not isinstance(raw_day, str):
        raise ValueError(f"expected_calendar date is not ISO formatted: {raw_day!r}")
    try:
        parsed = date.fromisoformat(raw_day)
    except ValueError as exc:
        raise ValueError(
            f"expected_calendar date is not ISO formatted: {raw_day!r}"
        ) from exc
    if parsed.isoformat() != raw_day:
        raise ValueError(f"expected_calendar date is not ISO formatted: {raw_day!r}")
    return parsed.isoformat()


def _normalize_game_ids(games: Iterable[str], day: str) -> list[str]:
    if games is None or isinstance(games, (str, bytes)) or not isinstance(
        games, Iterable
    ):
        raise ValueError(f"expected_calendar[{day}] must list game IDs")
    identifiers: list[str] = []
    for game in games:
        if isinstance(game, bool) or not isinstance(game, (str, Integral)):
            raise ValueError(f"expected_calendar[{day}] contains a non-identifier game")
        identifier = game if isinstance(game, str) else str(int(game))
        if not identifier.strip():
            raise ValueError(f"expected_calendar[{day}] contains an empty game ID")
        identifiers.append(identifier)
    if not identifiers:
        raise ValueError(f"expected_calendar[{day}] must list at least one game")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"expected_calendar[{day}] repeats a game ID")
    return sorted(identifiers)


def _evidence_id(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        _canonical_bytes(_identity_payload(payload))
    ).hexdigest()


def _identity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in _IDENTITY_EXCLUDED
    }


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _json_native(payload: Any) -> Any:
    """Round-trip through canonical JSON so tuples never leak into payloads."""

    return json.loads(_canonical_bytes(payload))


def _encode(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _assert_unchanged(stored: Any, recomputed: Any, label: str) -> None:
    if _canonical_bytes(stored) != _canonical_bytes(recomputed):
        raise ValueError(f"Release evidence {label} is stale or tampered")


def _safe_path_segment(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or os.sep in value
    ):
        raise ValueError(f"Release evidence {label} is not a safe path segment")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "DEFAULT_RELEASE_EVIDENCE_ROOT",
    "RELEASE_EVIDENCE_SCHEMA_VERSION",
    "ReleaseEvidenceRecord",
    "ReleaseEvidenceStore",
    "ReleaseEvidenceVerification",
]
