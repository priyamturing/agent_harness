"""Harness verifier execution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import httpx
from rich.console import Console
from rich.table import Table

from .prompts import Scenario, Verifier


@dataclass(frozen=True)
class VerifierResult:
    """Outcome of a single verifier execution."""

    verifier: Verifier
    success: bool
    actual_value: Optional[object]
    expected_value: Optional[object]
    comparison_type: Optional[str]
    error: Optional[str] = None


def _collect_verifiers(scenario: Scenario) -> list[Verifier]:
    verifiers: list[Verifier] = []
    for prompt in scenario.prompts:
        verifiers.extend(prompt.verifiers)
    return verifiers


def _extract_scalar_value(response_json: dict) -> Optional[object]:
    """Best-effort extraction of the first scalar value from SQL runner response."""

    if not response_json:
        return None

    # Common formats: {"rows": [[1]], "columns": ["COUNT(*)"]}
    rows = response_json.get("rows")
    if isinstance(rows, list) and rows:
        first_row = rows[0]
        if isinstance(first_row, list) and first_row:
            return first_row[0]

    # Alternate format: {"data": [{"COUNT(*)": 1}]}
    data = response_json.get("data")
    if isinstance(data, list) and data:
        first_entry = data[0]
        if isinstance(first_entry, dict):
            for value in first_entry.values():
                return value

    # Some implementations wrap further
    result = response_json.get("result")
    if isinstance(result, dict):
        return _extract_scalar_value(result)

    return None


def _compare(actual: object, expected: object, comparison: str | None) -> bool:
    if comparison is None:
        return False
    comparison = comparison.lower()
    if comparison == "equals":
        return actual == expected
    return False


async def evaluate_verifiers(
    scenario: Scenario,
    *,
    sql_runner_url: str,
    database_id: str,
    client: Optional[httpx.AsyncClient] = None,
) -> list[VerifierResult]:
    """Execute verifiers and return their results without rendering output."""

    verifiers = _collect_verifiers(scenario)
    if not verifiers:
        return []

    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            # Add retries and better connection handling
            transport=httpx.AsyncHTTPTransport(retries=2),
        )

    results: list[VerifierResult] = []

    try:
        for verifier in verifiers:
            if verifier.verifier_type != "database_state":
                results.append(
                    VerifierResult(
                        verifier=verifier,
                        success=False,
                        actual_value=None,
                        expected_value=None,
                        comparison_type=None,
                        error=f"Unsupported verifier type '{verifier.verifier_type}'",
                    )
                )
                continue

            config = verifier.validation_config
            query = config.get("query")
            expected_value = config.get("expected_value")
            comparison_type = config.get("comparison_type")
            if not query:
                results.append(
                    VerifierResult(
                        verifier=verifier,
                        success=False,
                        actual_value=None,
                        expected_value=expected_value,
                        comparison_type=comparison_type,
                        error="Verifier missing SQL query.",
                    )
                )
                continue

            try:
                response = await client.post(
                    sql_runner_url,
                    headers={
                        "x-database-id": database_id,
                        "Content-Type": "application/json",
                    },
                    json={
                        "query": query,
                        "read_only": True,
                        "return_data": True,
                    },
                )
                response.raise_for_status()
                payload = response.json()
                actual_value = _extract_scalar_value(payload)
                success = _compare(actual_value, expected_value, comparison_type)
                results.append(
                    VerifierResult(
                        verifier=verifier,
                        success=success,
                        actual_value=actual_value,
                        expected_value=expected_value,
                        comparison_type=comparison_type,
                        error=None if success else "Comparison failed.",
                    )
                )
            except httpx.ReadError as exc:
                # Network read errors - connection dropped during response
                error_msg = f"Network error (connection dropped): {exc.__class__.__name__}"
                results.append(
                    VerifierResult(
                        verifier=verifier,
                        success=False,
                        actual_value=None,
                        expected_value=expected_value,
                        comparison_type=comparison_type,
                        error=error_msg,
                    )
                )
            except httpx.HTTPStatusError as exc:
                # HTTP error status codes
                error_msg = f"HTTP {exc.response.status_code}: {exc.response.text[:100]}"
                results.append(
                    VerifierResult(
                        verifier=verifier,
                        success=False,
                        actual_value=None,
                        expected_value=expected_value,
                        comparison_type=comparison_type,
                        error=error_msg,
                    )
                )
            except httpx.RequestError as exc:
                # Connection errors, timeouts, etc.
                error_msg = f"Request error: {exc.__class__.__name__}"
                results.append(
                    VerifierResult(
                        verifier=verifier,
                        success=False,
                        actual_value=None,
                        expected_value=expected_value,
                        comparison_type=comparison_type,
                        error=error_msg,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                # Catch-all for other errors
                error_msg = str(exc) or exc.__class__.__name__
                results.append(
                    VerifierResult(
                        verifier=verifier,
                        success=False,
                        actual_value=None,
                        expected_value=expected_value,
                        comparison_type=comparison_type,
                        error=error_msg,
                    )
                )
    finally:
        if owns_client:
            try:
                await client.aclose()
            except Exception:  # noqa: BLE001
                # Silently ignore errors during cleanup
                pass

    return results


async def run_verifiers(
    scenario: Scenario,
    *,
    sql_runner_url: str,
    database_id: str,
    console: Console,
    client: Optional[httpx.AsyncClient] = None,
) -> list[VerifierResult]:
    """Execute verifiers and render a summary table to the provided console."""

    results = await evaluate_verifiers(
        scenario,
        sql_runner_url=sql_runner_url,
        database_id=database_id,
        client=client,
    )

    if not results:
        return results

    table = Table(title="Verifier results", show_lines=True)
    table.add_column("Verifier")
    table.add_column("Comparison")
    table.add_column("Expected")
    table.add_column("Actual")
    table.add_column("Status")

    for result in results:
        comparison = result.comparison_type or "-"
        expected = repr(result.expected_value)
        actual = repr(result.actual_value) if result.error is None else "-"
        status = "[green]PASS[/green]" if result.success else "[red]FAIL[/red]"
        if result.error and not result.success:
            status = f"[red]FAIL[/red]\n[dim]{result.error}[/dim]"

        label = result.verifier.name or result.verifier.verifier_type
        table.add_row(label, comparison, expected, actual, status)

    console.print(table)
    return results


def render_verifier_summary(results: Sequence[VerifierResult], *, console: Console) -> None:
    """Render a summary table of verifier outcomes."""

    table = Table(title="Verifier Summary")
    table.add_column("#", justify="right")
    table.add_column("Verifier")
    table.add_column("Comparison")
    table.add_column("Expected", overflow="fold")
    table.add_column("Actual", overflow="fold")
    table.add_column("Status")

    for idx, result in enumerate(results, start=1):
        comparison = result.comparison_type or "-"
        expected = repr(result.expected_value)
        actual = repr(result.actual_value) if result.error is None else "-"
        status = "[green]PASS[/green]" if result.success else "[red]FAIL[/red]"
        label = result.verifier.name or result.verifier.verifier_type
        if result.error and not result.success:
            status = f"[red]FAIL[/red]\n[dim]{result.error}[/dim]"

        table.add_row(
            str(idx),
            label,
            comparison,
            expected,
            actual,
            status,
        )

    console.print(table)


__all__ = ["run_verifiers", "render_verifier_summary", "VerifierResult"]
