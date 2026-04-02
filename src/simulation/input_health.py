from __future__ import annotations

from typing import Any, Dict, Iterable, List, TypedDict


class InputHealthStatus(TypedDict):
    source_key: str
    status: str
    required: bool
    message: str
    details: Dict[str, Any]


def build_input_health(
    source_key: str,
    status: str,
    *,
    required: bool,
    message: str,
    details: Dict[str, Any] | None = None,
) -> InputHealthStatus:
    return {
        'source_key': source_key,
        'status': status,
        'required': required,
        'message': message,
        'details': details or {},
    }


def summarize_input_health(statuses: Iterable[InputHealthStatus]) -> Dict[str, Any]:
    items: List[InputHealthStatus] = [dict(item) for item in statuses]
    counts = {'success': 0, 'fallback': 0, 'failed': 0, 'disabled': 0}
    degraded_sources: List[str] = []
    hard_failures: List[str] = []

    for item in items:
        status = item.get('status', 'failed')
        if status not in counts:
            status = 'failed'
        counts[status] += 1
        if status != 'success':
            degraded_sources.append(item['source_key'])
        if item.get('required') and status == 'failed':
            hard_failures.append(item['source_key'])

    overall_status = 'healthy' if not degraded_sources else 'degraded'
    return {
        'overall_status': overall_status,
        'counts': counts,
        'degraded_sources': degraded_sources,
        'hard_failures': hard_failures,
        'sources': items,
    }
