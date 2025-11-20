"""Database-based verifier using SQL runner endpoint."""

from __future__ import annotations

from typing import Any, Optional, Union

import httpx

from ...constants import DATABASE_VERIFIER_TIMEOUT_SECONDS
from ...agents import derive_sql_runner_url
from .base import Verifier, VerifierResult
from .utils import ComparisonType, compare


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


class DatabaseVerifier(Verifier):
    """Verifier that executes SQL queries and compares results.

    Uses the /sql-runner endpoint to execute queries against
    the MCP server's database.
    
    Supported comparison types (with aliases):
    - equals / eq / == / equal : exact equality (actual == expected)
    - not_equal / not_equals / neq / ne / != : inequality (actual != expected)
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
        comparison: Union[ComparisonType, str] = "equals",
        name: Optional[str] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        """Initialize database verifier.

        Args:
            query: SQL query to execute
            expected_value: Expected value from query
            mcp_url: MCP server URL (SQL runner URL will be derived from this)
            database_id: Database ID for isolation
            comparison: Comparison type as ComparisonType enum or string. Supports:
                - ComparisonType.EQUALS or strings: equals, eq, ==, equal
                - ComparisonType.NOT_EQUAL or strings: not_equal, not_equals, neq, ne, !=
                - ComparisonType.GREATER_THAN or strings: greater_than, gt, >
                - ComparisonType.LESS_THAN or strings: less_than, lt, <
                - ComparisonType.GREATER_THAN_EQUAL or strings: greater_than_equal, gte, >=, etc.
                - ComparisonType.LESS_THAN_EQUAL or strings: less_than_equal, lte, <=, etc.
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
        self.comparison = ComparisonType.from_string(comparison) if isinstance(comparison, str) else comparison
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
            success = compare(actual_value, self.expected_value, self.comparison)

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

