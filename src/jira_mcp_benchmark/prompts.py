"""Prompt parsing utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class Verifier:
    """Represents a verification step defined in the harness."""

    verifier_type: str
    validation_config: dict
    name: Optional[str]


@dataclass(frozen=True)
class ScenarioPrompt:
    """Represents a single user prompt in a scenario."""

    prompt_text: str
    expected_tools: Sequence[str]
    verifiers: Sequence[Verifier]


@dataclass(frozen=True)
class Scenario:
    """Scenario metadata and prompts loaded from the benchmark file."""

    scenario_id: str
    name: str
    description: str
    prompts: Sequence[ScenarioPrompt]
    metadata: dict
    conversation_mode: bool


def load_scenarios(path: Path) -> list[Scenario]:
    """Load the benchmark scenarios from a JSON file."""

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    scenarios: List[Scenario] = []
    for entry in payload.get("scenarios", []):
        prompts: List[ScenarioPrompt] = []
        for prompt in entry.get("prompts", []):
            verifiers = []
            verifier_obj = prompt.get("verifier")
            if verifier_obj:
                if isinstance(verifier_obj, dict):
                    verifier_iterable = [verifier_obj]
                elif isinstance(verifier_obj, (list, tuple)):
                    verifier_iterable = list(verifier_obj)
                else:
                    raise TypeError(
                        "Verifier definitions must be a dict or sequence of dicts; "
                        f"received {type(verifier_obj).__name__!r}."
                    )
                for raw_verifier in verifier_iterable:
                    if not isinstance(raw_verifier, dict):
                        raise TypeError(
                            "Each verifier entry must be a mapping; "
                            f"received {type(raw_verifier).__name__!r}."
                        )
                    verifiers.append(
                        Verifier(
                            verifier_type=raw_verifier.get("verifier_type", ""),
                            validation_config=raw_verifier.get("validation_config", {}),
                            name=raw_verifier.get("name"),
                        )
                    )
            raw_expected_tools = prompt.get("expected_tools") or ()
            if isinstance(raw_expected_tools, str):
                expected_tools: Sequence[str] = (raw_expected_tools,)
            else:
                try:
                    expected_tools = tuple(raw_expected_tools)
                except TypeError:
                    expected_tools = (str(raw_expected_tools),)

            prompts.append(
                ScenarioPrompt(
                    prompt_text=prompt.get("prompt_text", ""),
                    expected_tools=expected_tools,
                    verifiers=tuple(verifiers),
                )
            )

        scenarios.append(
            Scenario(
                scenario_id=entry.get("scenario_id", "unknown"),
                name=entry.get("name", "Unnamed Scenario"),
                description=entry.get("description", ""),
                prompts=tuple(prompts),
                metadata=entry.get("metadata", {}),
                conversation_mode=bool(entry.get("conversation_mode", False)),
            )
        )

    return scenarios


def scenario_summary(scenario: Scenario, *, include_expected_tools: bool = False) -> str:
    """Return a short textual summary to feed models."""

    parts: list[str] = [
        f"Scenario: {scenario.name}",
        f"ID: {scenario.scenario_id}",
        f"Description: {scenario.description}",
        f"Conversation mode: {'enabled' if scenario.conversation_mode else 'disabled'}",
    ]

    if include_expected_tools:
        expected_tool_names = {
            tool for prompt in scenario.prompts for tool in prompt.expected_tools
        }
        tool_list = ", ".join(sorted(expected_tool_names)) if expected_tool_names else "N/A"
        parts.append(f"Expected tools: {tool_list}")

    return "\n".join(parts)
