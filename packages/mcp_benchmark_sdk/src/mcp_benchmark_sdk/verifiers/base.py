"""Base verifier interface and data structures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


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
    
    Each verifier is self-contained and receives all dependencies
    through its constructor.
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    async def verify(self) -> VerifierResult:
        """Execute verification and return result.

        Returns:
            VerifierResult with success status and details
        """
        ...

