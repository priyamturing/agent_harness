"""Load JSON harness files and convert to SDK Task objects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from mcp_benchmark_sdk import (
    DatabaseVerifier,
    MCPConfig,
    Task,
    Verifier,
)

from .scenario import Scenario, ScenarioPrompt, VerifierDefinition


def load_harness_file(path: Path) -> list[Scenario]:
    """Load scenarios from a JSON harness file.

    Args:
        path: Path to JSON harness file

    Returns:
        List of Scenario objects
    """
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    scenarios: list[Scenario] = []

    for entry in payload.get("scenarios", []):
        prompts: list[ScenarioPrompt] = []

        for prompt_entry in entry.get("prompts", []):
            # Parse verifiers
            verifiers: list[VerifierDefinition] = []
            verifier_obj = prompt_entry.get("verifier")

            if verifier_obj:
                if isinstance(verifier_obj, dict):
                    verifier_iterable = [verifier_obj]
                elif isinstance(verifier_obj, (list, tuple)):
                    verifier_iterable = list(verifier_obj)
                else:
                    raise TypeError(
                        "Verifier must be dict or list of dicts; "
                        f"got {type(verifier_obj).__name__}"
                    )

                for raw_verifier in verifier_iterable:
                    if not isinstance(raw_verifier, dict):
                        raise TypeError(
                            "Each verifier must be a dict; "
                            f"got {type(raw_verifier).__name__}"
                        )

                    verifiers.append(
                        VerifierDefinition(
                            verifier_type=raw_verifier.get("verifier_type", ""),
                            validation_config=raw_verifier.get("validation_config", {}),
                            name=raw_verifier.get("name"),
                        )
                    )

            # Parse expected tools
            raw_expected_tools = prompt_entry.get("expected_tools") or ()
            if isinstance(raw_expected_tools, str):
                expected_tools = (raw_expected_tools,)
            else:
                try:
                    expected_tools = tuple(raw_expected_tools)
                except TypeError:
                    expected_tools = (str(raw_expected_tools),)

            prompts.append(
                ScenarioPrompt(
                    prompt_text=prompt_entry.get("prompt_text", ""),
                    expected_tools=expected_tools,
                    verifiers=tuple(verifiers),
                )
            )

        scenarios.append(
            Scenario(
                scenario_id=entry.get("scenario_id") or entry.get("name", "unknown"),
                name=entry.get("name", "Unnamed Scenario"),
                description=entry.get("description", ""),
                prompts=tuple(prompts),
                metadata=entry.get("metadata", {}),
                conversation_mode=bool(entry.get("conversation_mode", False)),
            )
        )

    return scenarios


def scenario_to_task(
    scenario: Scenario,
    mcps: list[MCPConfig],
    database_id: Optional[str] = None,
) -> tuple[Task, list[VerifierDefinition]]:
    """Convert Scenario to SDK Task and verifier definitions.
    
    This function decouples verification from task execution,
    allowing CLI orchestrators to control when and how verification happens.
    
    Returns verifier definitions instead of instantiated verifiers since
    verifiers need runtime context (sql_runner_url, database_id, http_client)
    which is only available during execution.

    Args:
        scenario: Scenario from harness file
        mcps: List of MCP configurations
        database_id: Optional database ID for isolation

    Returns:
        Tuple of (Task, list of VerifierDefinitions) for independent orchestration
    """
    # Collect all verifier definitions from all prompts
    verifier_defs: list[VerifierDefinition] = []

    for prompt in scenario.prompts:
        for verifier_def in prompt.verifiers:
            verifier_defs.append(verifier_def)

    # Build prompt text
    # If conversation_mode, merge all prompts; otherwise use first
    if scenario.conversation_mode:
        prompt_text = "\n".join(p.prompt_text for p in scenario.prompts)
    else:
        prompt_text = scenario.prompts[0].prompt_text if scenario.prompts else ""

    task = Task(
        prompt=prompt_text,
        mcps=mcps,
        metadata={
            "scenario_id": scenario.scenario_id,
            "scenario_name": scenario.name,
            "description": scenario.description,
            "conversation_mode": scenario.conversation_mode,
            **scenario.metadata,
        },
        database_id=database_id,
        conversation_mode=scenario.conversation_mode,
    )
    
    return task, verifier_defs


def _create_verifier_from_definition(
    verifier_def: VerifierDefinition,
    sql_runner_url: str,
    database_id: str,
    http_client: Any,
) -> Verifier:
    """Create SDK Verifier from harness definition.

    Args:
        verifier_def: Verifier definition from harness
        sql_runner_url: SQL runner endpoint URL
        database_id: Database ID for isolation
        http_client: HTTP client for making requests

    Returns:
        Verifier instance

    Raises:
        ValueError: If verifier type is unsupported or config is invalid
    """
    if verifier_def.verifier_type == "database_state":
        config = verifier_def.validation_config
        query = config.get("query")
        expected_value = config.get("expected_value")
        comparison = config.get("comparison_type", "equals")

        if not query:
            raise ValueError(
                f"Verifier '{verifier_def.name or 'unnamed'}': "
                f"database_state verifier requires 'query' in validation_config"
            )

        return DatabaseVerifier(
            query=query,
            expected_value=expected_value,
            sql_runner_url=sql_runner_url,
            database_id=database_id,
            comparison=comparison,
            name=verifier_def.name,
            http_client=http_client,
        )

    raise ValueError(
        f"Unsupported verifier type: '{verifier_def.verifier_type}'. "
        f"Supported types: database_state"
    )


__all__ = ["load_harness_file", "scenario_to_task"]

