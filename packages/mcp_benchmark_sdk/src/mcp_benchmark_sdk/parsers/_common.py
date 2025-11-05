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


def collect_reasoning_chunks(reasoning_block: object, chunks: list[str]) -> None:
    """Normalize reasoning data from various providers into text snippets.

    Recursively extracts readable text from nested reasoning structures.
    """
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
                collect_reasoning_chunks(entry, chunks)
        elif isinstance(summary, dict):
            collect_reasoning_chunks(summary, chunks)

        steps = reasoning_block.get("steps")
        if isinstance(steps, list):
            for step in steps:
                collect_reasoning_chunks(step, chunks)
        return

    if isinstance(reasoning_block, list):
        for entry in reasoning_block:
            collect_reasoning_chunks(entry, chunks)

