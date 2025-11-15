"""Common utilities for response parsing."""

import json
from typing import Any, Optional

from ..constants import REASONING_MAX_DEPTH, REASONING_MAX_TEXT_LENGTH


def format_json(value: object) -> str:
    """Best-effort JSON formatting for debug output."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except TypeError:
        text = str(value)
        if len(text) > REASONING_MAX_TEXT_LENGTH:
            return text[:REASONING_MAX_TEXT_LENGTH] + "…"
        return text


def collect_reasoning_chunks(
    reasoning_block: object, 
    chunks: list[str],
    max_depth: int = REASONING_MAX_DEPTH,
    _initial_depth: Optional[int] = None
) -> None:
    """Normalize reasoning data from various providers into text snippets.

    Recursively extracts readable text from nested reasoning structures.
    
    Args:
        reasoning_block: Reasoning data structure to parse
        chunks: List to append extracted text chunks to
        max_depth: Maximum recursion depth
        _initial_depth: Internal parameter to track original max_depth
    
    Raises:
        RecursionError: If nesting depth exceeds max_depth (security protection)
    """
    if _initial_depth is None:
        _initial_depth = max_depth
    
    if max_depth <= 0:
        raise RecursionError(
            f"Reasoning structure nesting depth exceeded maximum limit of {_initial_depth} levels. "
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
                collect_reasoning_chunks(entry, chunks, max_depth - 1, _initial_depth)
        elif isinstance(summary, dict):
            collect_reasoning_chunks(summary, chunks, max_depth - 1, _initial_depth)

        steps = reasoning_block.get("steps")
        if isinstance(steps, list):
            for step in steps:
                collect_reasoning_chunks(step, chunks, max_depth - 1, _initial_depth)
        return

    if isinstance(reasoning_block, list):
        for entry in reasoning_block:
            collect_reasoning_chunks(entry, chunks, max_depth - 1, _initial_depth)

