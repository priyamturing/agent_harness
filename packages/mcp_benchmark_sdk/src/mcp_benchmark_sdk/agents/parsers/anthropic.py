"""Anthropic response parser."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from ..tasks import ToolCall
from .base import ParsedResponse
from ._common import collect_reasoning_chunks, format_json


class AnthropicResponseParser:
    """Parser for Anthropic Claude model responses.

    Handles:
    - thinking blocks with extended thinking
    - message blocks
    - text content blocks
    """

    def parse(self, message: AIMessage) -> ParsedResponse:
        """Parse Anthropic AIMessage."""
        primary_chunks: list[str] = []
        reasoning_chunks: list[str] = []
        raw_reasoning: list[str] = []
        content = message.content

        # Parse content blocks
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

                # Thinking blocks (Claude with extended thinking)
                elif block_type == "thinking":
                    thinking_text = block.get("thinking") or block.get("text")
                    signature = block.get("signature")

                    if thinking_text:
                        reasoning_chunks.append(thinking_text)

                    entry_payload: dict[str, object] = {}
                    if thinking_text:
                        entry_payload["thinking"] = thinking_text
                    if signature:
                        entry_payload["signature"] = signature
                    if entry_payload:
                        raw_reasoning.append(format_json(entry_payload))

                # Message blocks (Anthropic-specific)
                elif block_type == "message":
                    text = block.get("text")
                    if text:
                        primary_chunks.append(text)

                # Tool result blocks
                elif block_type == "tool_result":
                    text = block.get("content")
                    if isinstance(text, str):
                        primary_chunks.append(text)

                # Fallback
                else:
                    text = block.get("text")
                    if text:
                        primary_chunks.append(text)

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

