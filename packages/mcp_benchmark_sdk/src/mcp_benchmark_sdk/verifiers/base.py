"""Base verifier interface and data structures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import httpx


@dataclass
class VerificationContext:
    """Context for executing verifiers.

    Provides access to:
    - SQL runner endpoint
    - Database ID for isolation
    - HTTP client for requests
    """

    sql_runner_url: str
    database_id: str
    http_client: httpx.AsyncClient


@dataclass
class VerifierResult:
    """Result from executing a verifier."""

    name: str
    success: bool
    expected_value: Optional[Any]
    actual_value: Optional[Any]
    comparison_type: Optional[str]
    error: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class Verifier(ABC):
    """Abstract base class for verifiers.

    Verifiers validate agent execution results by checking
    system state, API responses, database contents, etc.
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    async def verify(self, context: VerificationContext) -> VerifierResult:
        """Execute verification and return result.

        Args:
            context: Verification context with database ID, HTTP client, etc.

        Returns:
            VerifierResult with success status and details
        """
        ...

