"""Verifier execution helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import httpx

if TYPE_CHECKING:
    from .base import Verifier, VerificationContext, VerifierResult


async def execute_verifiers(
    verifiers: list["Verifier"],
    sql_runner_url: str,
    database_id: str,
    http_client: Optional[httpx.AsyncClient] = None,
) -> list["VerifierResult"]:
    """Execute a list of verifiers and return results.

    Args:
        verifiers: List of verifier instances
        sql_runner_url: SQL runner endpoint URL
        database_id: Database ID for isolation
        http_client: Optional HTTP client (created if not provided)

    Returns:
        List of verifier results
    """
    from .base import VerificationContext

    if not verifiers:
        return []

    owns_client = http_client is None
    if http_client is None:
        http_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0))

    context = VerificationContext(
        sql_runner_url=sql_runner_url,
        database_id=database_id,
        http_client=http_client,
    )

    results: list[VerifierResult] = []

    try:
        for verifier in verifiers:
            result = await verifier.verify(context)
            results.append(result)
    finally:
        if owns_client:
            await http_client.aclose()

    return results

