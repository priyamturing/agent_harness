"""Langfuse tracing helpers shared across CLI runners."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator, Optional


def _coerce_metadata(metadata: Dict[str, Optional[str]]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        text = str(value)
        result[key] = text[:200]
    return result


@contextmanager
def langfuse_run_context(
    *,
    session_id: str,
    run_label: str,
    model_name: str,
    scenario_id: str,
    run_number: int,
    batch_alias: Optional[str] = None,
) -> Iterator[None]:
    """Propagate session metadata so each trace stays tied to a single run."""
    try:
        from turing_rl_sdk import is_langfuse_enabled
    except ImportError:
        yield
        return

    if not is_langfuse_enabled():
        yield
        return

    try:
        from langfuse import propagate_attributes
    except Exception:  # pragma: no cover - optional dependency
        propagate_attributes = None  # type: ignore

    metadata = _coerce_metadata(
        {
            "session_id": session_id,
            "run_label": run_label,
            "model": model_name,
            "scenario_id": scenario_id,
            "run_number": str(run_number),
            "batch_alias": batch_alias,
        }
    )

    trace_identifier = f"{model_name}-{scenario_id}-{run_number}"
    metadata["trace_identifier"] = trace_identifier

    if propagate_attributes is None:
        yield
        return

    with propagate_attributes(session_id=session_id, metadata=metadata):
        yield
