"""xAI Grok response parser."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from ..tasks import ToolCall
from .base import ParsedResponse
from ._common import collect_reasoning_chunks, format_json


class XAIResponseParser:
    """Parser for xAI Grok model responses.

    Handles:
    - Grok reasoning traces
    - Encrypted reasoning content
    - Standard text output
    """

    def parse(self, message: AIMessage) -> ParsedResponse:
        """Parse xAI Grok AIMessage."""
        primary_chunks: list[str] = []
        reasoning_chunks: list[str] = []
        raw_reasoning: list[str] = []
        content = message.content

        if isinstance(content, str):
            if content:
                primary_chunks.append(content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue

                block_type = block.get("type")

                if block_type in {"output_text", "text", None}:
                    text = block.get("text")
                    if text:
                        primary_chunks.append(text)

                elif block_type == "reasoning":
                    summary = block.get("summary")
                    if summary:
                        raw_reasoning.append(format_json({"summary": summary}))
                    try:
                        collect_reasoning_chunks(block, reasoning_chunks)
                    except RecursionError as e:
                        reasoning_chunks.append(f"[Error: {e!r}]")

                elif block_type == "thinking":
                    thinking_text = block.get("thinking") or block.get("text")
                    if thinking_text:
                        reasoning_chunks.append(thinking_text)
                        raw_reasoning.append(format_json({"thinking": thinking_text}))

                else:
                    text = block.get("text")
                    if text:
                        primary_chunks.append(text)

        if hasattr(message, "additional_kwargs"):
            extra_reasoning = message.additional_kwargs.get("reasoning")
            if extra_reasoning:
                if isinstance(extra_reasoning, dict):
                    summary = extra_reasoning.get("summary")
                    if summary:
                        raw_reasoning.append(format_json({"summary": summary}))
                try:
                    collect_reasoning_chunks(extra_reasoning, reasoning_chunks)
                except RecursionError as e:
                    reasoning_chunks.append(f"[Error: {e!r}]")

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

