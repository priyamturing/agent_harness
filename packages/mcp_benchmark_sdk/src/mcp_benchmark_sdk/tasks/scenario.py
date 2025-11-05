"""Scenario data structures for JSON harness files."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence


@dataclass(frozen=True)
class VerifierDefinition:
    """Verifier definition from JSON harness."""

    verifier_type: str
    validation_config: dict[str, Any]
    name: Optional[str]


@dataclass(frozen=True)
class ScenarioPrompt:
    """Single prompt within a scenario."""

    prompt_text: str
    expected_tools: Sequence[str]
    verifiers: Sequence[VerifierDefinition]


@dataclass(frozen=True)
class Scenario:
    """Scenario loaded from JSON harness file."""

    scenario_id: str
    name: str
    description: str
    prompts: Sequence[ScenarioPrompt]
    metadata: dict[str, Any]
    conversation_mode: bool

