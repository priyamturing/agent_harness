"""Scenario data structures for JSON harness files."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence


@dataclass(frozen=True)
class VerifierDefinition:
    """Verifier definition loaded from JSON harness file.
    
    Attributes:
        verifier_type (str): Type identifier for the verifier, e.g., 
            "database_state". Used to instantiate the correct verifier class.
        validation_config (dict[str, Any]): Configuration dict passed to the 
            verifier for validation logic. Contents depend on verifier type.
        name (Optional[str]): Optional human-readable name for this verifier.
    """

    verifier_type: str
    validation_config: dict[str, Any]
    name: Optional[str]

    def __post_init__(self) -> None:
        """Validate verifier definition after initialization.
        
        Raises:
            ValueError: If verifier_type is empty or whitespace.
        """
        if not self.verifier_type or not self.verifier_type.strip():
            raise ValueError(
                "VerifierDefinition.verifier_type cannot be empty. "
                "Provide a valid verifier type (e.g., 'database_state')."
            )


@dataclass(frozen=True)
class ScenarioPrompt:
    """Single prompt within a scenario, loaded from JSON harness.
    
    Attributes:
        prompt_text (str): The actual prompt text to send to the agent. 
            Must be non-empty.
        expected_tools (Sequence[str]): List of tool names expected to be 
            called during execution. Used for validation.
        verifiers (Sequence[VerifierDefinition]): List of verifiers to run 
            after execution to validate the outcome.
    """

    prompt_text: str
    expected_tools: Sequence[str]
    verifiers: Sequence[VerifierDefinition]

    def __post_init__(self) -> None:
        """Validate prompt after initialization.
        
        Raises:
            ValueError: If prompt_text is empty or whitespace.
        """
        if not self.prompt_text or not self.prompt_text.strip():
            raise ValueError(
                "ScenarioPrompt.prompt_text cannot be empty. "
                "Provide a non-empty prompt string."
            )


@dataclass(frozen=True)
class Scenario:
    """Complete scenario loaded from JSON harness file.
    
    Represents a test scenario with prompt(s), expected behaviors, and verifiers.
    Note: This SDK currently only supports single-prompt scenarios.
    
    Attributes:
        scenario_id (str): Unique identifier for this scenario. Must be non-empty.
        name (str): Human-readable scenario name.
        description (str): Longer description of what this scenario tests.
        prompts (Sequence[ScenarioPrompt]): List of prompts in this scenario. 
            Must contain exactly 1 prompt (multi-prompt not supported).
        metadata (dict[str, Any]): Additional metadata for this scenario, 
            such as tags, categories, or custom fields.
        conversation_mode (bool): Whether this scenario uses conversation mode.
            Currently unused as multi-turn is not supported.
    """

    scenario_id: str
    name: str
    description: str
    prompts: Sequence[ScenarioPrompt]
    metadata: dict[str, Any]
    conversation_mode: bool

    def __post_init__(self) -> None:
        """Validate scenario after initialization.
        
        Raises:
            ValueError: If prompts is empty, contains more than 1 prompt, 
                or scenario_id is empty.
        """
        if not self.prompts:
            raise ValueError(
                f"Scenario '{self.scenario_id}' must have at least one prompt. "
                "Provide a non-empty prompts sequence."
            )
        
        if len(self.prompts) > 1:
            raise ValueError(
                f"Scenario '{self.scenario_id}' contains {len(self.prompts)} prompts, "
                f"but this SDK only supports single-prompt scenarios (exactly 1 prompt per scenario). "
                f"Multi-prompt/multi-turn scenarios are not supported. "
                f"If this harness file came from an external service, you need to split it into "
                f"{len(self.prompts)} separate scenarios, each with one prompt."
            )
        
        if not self.scenario_id or not self.scenario_id.strip():
            raise ValueError(
                "Scenario.scenario_id cannot be empty. "
                "Provide a non-empty scenario identifier."
            )
