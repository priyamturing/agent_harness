"""Scenario data structures for JSON harness files.

These are CLI-specific types for loading benchmark harness files.
They are not used by the SDK runtime - only for harness parsing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence


@dataclass(frozen=True)
class VerifierDefinition:
    """Verifier definition from JSON harness."""

    verifier_type: str
    validation_config: dict[str, Any]
    name: Optional[str]

    def __post_init__(self) -> None:
        """Validate verifier definition fields after initialization."""
        if not self.verifier_type or not self.verifier_type.strip():
            raise ValueError(
                "VerifierDefinition.verifier_type cannot be empty. "
                "Provide a valid verifier type (e.g., 'database_state')."
            )


@dataclass(frozen=True)
class ScenarioPrompt:
    """Single prompt within a scenario."""

    prompt_text: str
    expected_tools: Sequence[str]
    verifiers: Sequence[VerifierDefinition]

    def __post_init__(self) -> None:
        """Validate prompt fields after initialization."""
        if not self.prompt_text or not self.prompt_text.strip():
            raise ValueError(
                "ScenarioPrompt.prompt_text cannot be empty. "
                "Provide a non-empty prompt string."
            )


@dataclass(frozen=True)
class Scenario:
    """Scenario loaded from JSON harness file."""

    scenario_id: str
    name: str
    description: str
    prompts: Sequence[ScenarioPrompt]
    metadata: dict[str, Any]
    conversation_mode: bool

    def __post_init__(self) -> None:
        """Validate scenario fields after initialization."""
        if not self.prompts:
            raise ValueError(
                f"Scenario '{self.scenario_id}' must have at least one prompt. "
                "Provide a non-empty prompts sequence."
            )
        
        if not self.scenario_id or not self.scenario_id.strip():
            raise ValueError(
                "Scenario.scenario_id cannot be empty. "
                "Provide a non-empty scenario identifier."
            )


