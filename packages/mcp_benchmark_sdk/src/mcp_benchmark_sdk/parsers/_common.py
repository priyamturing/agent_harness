"""Common utilities for response parsing."""

import json
from typing import Any


def format_json(value: object) -> str:
    """Best-effort JSON formatting for debug output."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except TypeError:
        text = str(value)
        max_len = 1200
        if len(text) > max_len:
            return text[:max_len] + "…"
        return text


def collect_reasoning_chunks(
    reasoning_block: object, 
    chunks: list[str],
    max_depth: int = 10
) -> None:
    """Normalize reasoning data from various providers into text snippets.

    Recursively extracts readable text from nested reasoning structures.
    
    Args:
        reasoning_block: Reasoning data structure to parse
        chunks: List to append extracted text chunks to
        max_depth: Maximum recursion depth (default: 10)
    
    Raises:
        RecursionError: If nesting depth exceeds max_depth (security protection)
    """
    if max_depth < 0:
        raise RecursionError(
            f"Reasoning structure nesting depth exceeded maximum limit of 10 levels. "
            f"This may indicate malformed or malicious data. "
            f"The current task will be marked as failed."
        )
    
    if reasoning_block is None:
        return

    if isinstance(reasoning_block, str):
        if reasoning_block:
            chunks.append(reasoning_block)
        return

    if isinstance(reasoning_block, dict):
        text = reasoning_block.get("text")
        if isinstance(text, str) and text:
            chunks.append(text)

        summary = reasoning_block.get("summary")
        if isinstance(summary, list):
            for entry in summary:
                collect_reasoning_chunks(entry, chunks, max_depth - 1)
        elif isinstance(summary, dict):
            collect_reasoning_chunks(summary, chunks, max_depth - 1)

        steps = reasoning_block.get("steps")
        if isinstance(steps, list):
            for step in steps:
                collect_reasoning_chunks(step, chunks, max_depth - 1)
        return

    if isinstance(reasoning_block, list):
        for entry in reasoning_block:
            collect_reasoning_chunks(entry, chunks, max_depth - 1)

