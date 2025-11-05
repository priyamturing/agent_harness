"""Google Gemini response parser."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from ..tasks import ToolCall
from .base import ParsedResponse
from ._common import collect_reasoning_chunks, format_json


class GoogleResponseParser:
    """Parser for Google Gemini model responses.

    Handles:
    - Thinking budget responses
    - Standard text content
    - Thoughts/reasoning traces
    """

    def parse(self, message: AIMessage) -> ParsedResponse:
        """Parse Google Gemini AIMessage."""
        primary_chunks: list[str] = []
        reasoning_chunks: list[str] = []
        raw_reasoning: list[str] = []
        content = message.content

        # Parse content
        if isinstance(content, str):
            if content:
                primary_chunks.append(content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue

                block_type = block.get("type")

                # Text content
                if block_type in {"text", None}:
                    text = block.get("text")
                    if text:
                        primary_chunks.append(text)

                # Thoughts/thinking (Gemini thinking budget)
                elif block_type in {"thinking", "thought"}:
                    thought_text = block.get("text") or block.get("thinking")
                    if thought_text:
                        reasoning_chunks.append(thought_text)
                        raw_reasoning.append(format_json({"thought": thought_text}))

                # Fallback
                else:
                    text = block.get("text")
                    if text:
                        primary_chunks.append(text)

        # Check for thoughts in additional_kwargs
        if hasattr(message, "additional_kwargs"):
            thoughts = message.additional_kwargs.get("thoughts")
            if thoughts:
                collect_reasoning_chunks(thoughts, reasoning_chunks)
                raw_reasoning.append(format_json({"thoughts": thoughts}))

        # Extract tool calls
        tool_calls = []
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        name=tc.get("name", ""),
                        arguments=tc.get("args", {}),
                        id=tc.get("id"),
                    )
                )

        primary_text = "\n".join(chunk for chunk in primary_chunks if chunk)
        return ParsedResponse(
            content=primary_text,
            reasoning=reasoning_chunks,
            raw_reasoning=raw_reasoning,
            tool_calls=tool_calls,
        )

