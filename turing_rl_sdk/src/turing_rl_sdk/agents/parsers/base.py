"""Base response parser interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from langchain_core.messages import AIMessage

from ..tasks import ToolCall


@dataclass
class ParsedResponse:
    """Parsed LLM response with extracted components."""

    content: str  # Primary text content
    reasoning: list[str]  # Human-readable reasoning/thinking traces
    raw_reasoning: list[str]  # Raw reasoning payloads for artifacts
    tool_calls: list[ToolCall]  # Extracted tool calls


class ResponseParser(Protocol):
    """Protocol for parsing provider-specific LLM responses."""

    def parse(self, message: AIMessage) -> ParsedResponse:
        """Parse an AI message into structured components.

        Args:
            message: LangChain AIMessage from the model

        Returns:
            ParsedResponse with extracted content, reasoning, and tool calls
        """
        ...

