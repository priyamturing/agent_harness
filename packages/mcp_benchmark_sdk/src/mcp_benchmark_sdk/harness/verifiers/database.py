"""Database-based verifier using SQL runner endpoint."""

from __future__ import annotations

from typing import Any, Optional

import httpx

from ...agents import DATABASE_VERIFIER_TIMEOUT_SECONDS, derive_sql_runner_url
from .base import Verifier, VerifierResult


def _extract_scalar_value(response_json: dict) -> Any:
    """Extract first scalar value from SQL runner response.
    
    Supports multiple response formats:
    - rows/columns format: {"rows": [[val]], "columns": ["col"]}
    - data format: {"data": [{"col": val}]}
    - nested result format: {"result": {...}}
    
    Validates that the query returns exactly one column to avoid ambiguity.
    
    Args:
        response_json: Response from SQL runner endpoint
        
    Returns:
        The scalar value from the first row, first column
        
    Raises:
        ValueError: If query returns multiple columns (ambiguous which to use)
    """
    if not response_json:
        return None

    value = _try_extract_from_rows_format(response_json)
    if value is not None:
        return value
    
    value = _try_extract_from_data_format(response_json)
    if value is not None:
        return value
    
    # Try nested result format (recursive)
    result = response_json.get("result")
    if isinstance(result, dict):
        return _extract_scalar_value(result)
    
    return None


def _try_extract_from_rows_format(response_json: dict) -> Any:
    """Try to extract value from rows/columns format.
    
    Format: {"rows": [[value]], "columns": ["column_name"]}
    """
    rows = response_json.get("rows")
    if not isinstance(rows, list) or not rows:
        return None
    
    first_row = rows[0]
    if not isinstance(first_row, list):
        return None
    
    # Validate single column
    columns = response_json.get("columns")
    if columns and isinstance(columns, list) and len(columns) > 1:
        raise ValueError(
            f"Query returned {len(columns)} columns: {columns}. "
            f"Database verifier queries must return exactly 1 column. "
            f"Use aggregate functions (COUNT(*), SUM(column), etc.) or select a single column."
        )
    
    return first_row[0] if first_row else None


def _try_extract_from_data_format(response_json: dict) -> Any:
    """Try to extract value from data format.
    
    Format: {"data": [{"column_name": value}]}
    """
    data = response_json.get("data")
    if not isinstance(data, list) or not data:
        return None
    
    first_entry = data[0]
    if not isinstance(first_entry, dict):
        return None
    
    # Validate single column
    if len(first_entry) > 1:
        cols = list(first_entry.keys())
        raise ValueError(
            f"Query returned {len(cols)} columns: {cols}. "
            f"Database verifier queries must return exactly 1 column. "
            f"Use aggregate functions (COUNT(*), SUM(column), etc.) or select a single column."
        )
    
    # Return the single value
    for value in first_entry.values():
        return value
    
    return None


def _compare(actual: Any, expected: Any, comparison: Optional[str]) -> bool:
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
    
    if comparison in ("equals", "eq", "==", "equal"):
        return actual == expected
    
    if actual is None:
        raise TypeError(
            f"Cannot perform ordered comparison '{comparison}' with None value. "
            f"SQL query returned None."
        )
    
    try:
        if comparison in ("greater_than", "gt", ">"):
            return actual > expected
        
        elif comparison in ("less_than", "lt", "<"):
            return actual < expected
        
        elif comparison in ("greater_than_equal", "greater_than_or_equal", "greater_than_or_equal_to", "gte", ">=", "greater_or_equal"):
            return actual >= expected
        
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
        mcp_url: str,
        database_id: str,
        comparison: str = "equals",
        name: Optional[str] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        """Initialize database verifier.

        Args:
            query: SQL query to execute
            expected_value: Expected value from query
            mcp_url: MCP server URL (SQL runner URL will be derived from this)
            database_id: Database ID for isolation
            comparison: Comparison type. Supports:
                - equals, eq, ==
                - greater_than, gt, >
                - less_than, lt, <
                - greater_than_equal, greater_than_or_equal, greater_than_or_equal_to, gte, >=
                - less_than_equal, less_than_or_equal, less_than_or_equal_to, lte, <=
            name: Optional display name
            http_client: Optional HTTP client (will create one if not provided)
            
        Raises:
            ValueError: If comparison type is unsupported
        """
        super().__init__(name or "DatabaseVerifier")
        self.query = query
        self.expected_value = expected_value
        self.sql_runner_url = derive_sql_runner_url(mcp_url)
        self.database_id = database_id
        self.comparison = comparison
        self._http_client = http_client
        self._owns_client = http_client is None

    async def verify(self) -> VerifierResult:
        """Execute SQL query and compare result."""
        http_client = self._http_client
        
        if http_client is None:
            http_client = httpx.AsyncClient(timeout=DATABASE_VERIFIER_TIMEOUT_SECONDS)
        
        try:
            response = await http_client.post(
                self.sql_runner_url,
                headers={
                    "x-database-id": self.database_id,
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

        except httpx.HTTPError as exc:
            return VerifierResult(
                name=self.name,
                success=False,
                expected_value=self.expected_value,
                actual_value=None,
                comparison_type=self.comparison,
                error=f"{type(exc).__name__}: {exc}",
                metadata={"query": self.query},
            )
        finally:
            if self._owns_client and http_client is not None:
                await http_client.aclose()

