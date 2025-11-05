"""Database-based verifier using SQL runner endpoint."""

from __future__ import annotations

from typing import Any, Optional

from .base import Verifier, VerificationContext, VerifierResult


def _extract_scalar_value(response_json: dict) -> Any:
    """Best-effort extraction of first scalar value from SQL runner response."""
    if not response_json:
        return None

    # Common format: {"rows": [[1]], "columns": ["COUNT(*)"]}
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

    # Nested result format
    result = response_json.get("result")
    if isinstance(result, dict):
        return _extract_scalar_value(result)

    return None


def _compare(actual: Any, expected: Any, comparison: str | None) -> bool:
    """Compare actual vs expected using comparison type.
    
    Supported comparisons (with aliases):
    - equals / eq / == : exact equality (works with None)
    - greater_than / gt / > : actual > expected (requires non-None values)
    - less_than / lt / < : actual < expected (requires non-None values)
    - greater_than_equal / greater_than_or_equal / greater_than_or_equal_to / gte / >= : actual >= expected (requires non-None values)
    - less_than_equal / less_than_or_equal / less_than_or_equal_to / lte / <= : actual <= expected (requires non-None values)
    
    Raises:
        ValueError: If comparison type is unsupported
        TypeError: If actual is None for ordered comparisons
    """
    if comparison is None:
        raise ValueError("Comparison type must be specified")

    comparison = comparison.lower().strip()
    
    # Equals - can handle None
    if comparison in ("equals", "eq", "==", "equal"):
        return actual == expected
    
    # Ordered comparisons - require non-None values
    if actual is None:
        raise TypeError(
            f"Cannot perform ordered comparison '{comparison}' with None value. "
            f"SQL query returned None."
        )
    
    # Type check for ordered comparisons
    try:
        # Greater than
        if comparison in ("greater_than", "gt", ">"):
            return actual > expected
        
        # Less than
        elif comparison in ("less_than", "lt", "<"):
            return actual < expected
        
        # Greater than or equal
        elif comparison in ("greater_than_equal", "greater_than_or_equal", "greater_than_or_equal_to", "gte", ">=", "greater_or_equal"):
            return actual >= expected
        
        # Less than or equal
        elif comparison in ("less_than_equal", "less_than_or_equal", "less_than_or_equal_to", "lte", "<=", "less_or_equal"):
            return actual <= expected
    except TypeError as e:
        raise TypeError(
            f"Cannot compare values: actual={actual!r} ({type(actual).__name__}) "
            f"vs expected={expected!r} ({type(expected).__name__}). "
            f"Ensure SQL query returns comparable types (numbers, dates, strings)."
        ) from e
    
    else:
        raise ValueError(
            f"Unsupported comparison type: '{comparison}'. "
            f"Supported: equals/eq/==, greater_than/gt/>, less_than/lt/<, "
            f"greater_than_equal/greater_than_or_equal/greater_than_or_equal_to/gte/>=, "
            f"less_than_equal/less_than_or_equal/less_than_or_equal_to/lte/<="
        )


class DatabaseVerifier(Verifier):
    """Verifier that executes SQL queries and compares results.

    Uses the /sql-runner endpoint to execute queries against
    the MCP server's database.
    
    Supported comparison types (with aliases):
    - equals / eq / == : exact equality (actual == expected)
    - greater_than / gt / > : actual > expected
    - less_than / lt / < : actual < expected
    - greater_than_equal / greater_than_or_equal / greater_than_or_equal_to / gte / >= : actual >= expected
    - less_than_equal / less_than_or_equal / less_than_or_equal_to / lte / <= : actual <= expected
    """

    def __init__(
        self,
        query: str,
        expected_value: Any,
        comparison: str = "equals",
        name: Optional[str] = None,
    ):
        """Initialize database verifier.

        Args:
            query: SQL query to execute
            expected_value: Expected value from query
            comparison: Comparison type. Supports:
                - equals, eq, ==
                - greater_than, gt, >
                - less_than, lt, <
                - greater_than_equal, greater_than_or_equal, greater_than_or_equal_to, gte, >=
                - less_than_equal, less_than_or_equal, less_than_or_equal_to, lte, <=
            name: Optional display name
            
        Raises:
            ValueError: If comparison type is unsupported
        """
        super().__init__(name or "DatabaseVerifier")
        self.query = query
        self.expected_value = expected_value
        self.comparison = comparison

    async def verify(self, context: VerificationContext) -> VerifierResult:
        """Execute SQL query and compare result."""
        try:
            response = await context.http_client.post(
                context.sql_runner_url,
                headers={
                    "x-database-id": context.database_id,
                    "Content-Type": "application/json",
                },
                json={
                    "query": self.query,
                    "read_only": True,
                    "return_data": True,
                },
            )
            response.raise_for_status()

            payload = response.json()
            actual_value = _extract_scalar_value(payload)
            success = _compare(actual_value, self.expected_value, self.comparison)

            return VerifierResult(
                name=self.name,
                success=success,
                expected_value=self.expected_value,
                actual_value=actual_value,
                comparison_type=self.comparison,
                error=None if success else "Comparison failed",
                metadata={"query": self.query},
            )

        except Exception as exc:
            return VerifierResult(
                name=self.name,
                success=False,
                expected_value=self.expected_value,
                actual_value=None,
                comparison_type=self.comparison,
                error=str(exc),
                metadata={"query": self.query},
            )

