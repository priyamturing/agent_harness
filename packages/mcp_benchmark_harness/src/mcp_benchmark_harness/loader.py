"""Load JSON harness files and convert to SDK Task objects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import httpx
from mcp_benchmark_agents import DatabaseVerifier, MCPConfig, Task, Verifier

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


def load_harness_directory(directory: Path) -> dict[str, list[Scenario]]:
    """Load all JSON harness files from a directory.

    Args:
        directory: Path to directory containing JSON files

    Returns:
        Dict mapping file stem to list of scenarios

    Raises:
        ValueError: If directory doesn't exist or contains no JSON files
    """
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    json_files = sorted(directory.glob("*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found in directory: {directory}")

    result: dict[str, list[Scenario]] = {}
    
    for json_file in json_files:
        try:
            scenarios = load_harness_file(json_file)
            if scenarios:
                result[json_file.stem] = scenarios
        except Exception as exc:
            raise ValueError(f"Failed to load {json_file.name}: {exc}") from exc

    if not result:
        raise ValueError(f"No valid scenarios found in directory: {directory}")

    return result


def scenario_to_task(
    scenario: Scenario,
    mcp: MCPConfig,
    database_id: Optional[str] = None,
) -> tuple[Task, list[VerifierDefinition]]:
    """Convert Scenario to SDK Task and verifier definitions.
    
    This function decouples verification from task execution,
    allowing orchestrators to control when and how verification happens.
    
    Returns verifier definitions instead of instantiated verifiers since
    verifiers need runtime context (mcp_url, database_id, http_client)
    which is only available during execution.

    Args:
        scenario: Scenario from harness file
        mcp: MCP configuration
        database_id: Optional database ID for isolation

    Returns:
        Tuple of (Task, list of VerifierDefinitions) for independent orchestration
    """
    verifier_defs: list[VerifierDefinition] = []

    for prompt in scenario.prompts:
        for verifier_def in prompt.verifiers:
            verifier_defs.append(verifier_def)

    if scenario.conversation_mode:
        prompt_text = "\n".join(p.prompt_text for p in scenario.prompts)
    else:
        prompt_text = scenario.prompts[0].prompt_text if scenario.prompts else ""

    task = Task(
        prompt=prompt_text,
        mcp=mcp,
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


def create_verifier_from_definition(
    verifier_def: VerifierDefinition,
    mcp_url: str,
    database_id: str,
    http_client: httpx.AsyncClient,
) -> Verifier:
    """Create SDK Verifier from harness definition.

    Args:
        verifier_def: Verifier definition from harness
        mcp_url: MCP server URL (SQL runner URL will be derived from this)
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
            mcp_url=mcp_url,
            database_id=database_id,
            comparison=comparison,
            name=verifier_def.name,
            http_client=http_client,
        )

    raise ValueError(
        f"Unsupported verifier type: '{verifier_def.verifier_type}'. "
        f"Supported types: database_state"
    )


class HarnessLoader:
    """Utility class for loading harness files with a fluent API."""

    def __init__(self):
        self.scenarios: list[Scenario] = []
        self.file_map: dict[str, list[Scenario]] = {}

    def load_file(self, path: Path) -> "HarnessLoader":
        """Load scenarios from a single file.

        Args:
            path: Path to JSON harness file

        Returns:
            Self for chaining
        """
        scenarios = load_harness_file(path)
        self.scenarios.extend(scenarios)
        self.file_map[path.stem] = scenarios
        return self

    def load_directory(self, directory: Path) -> "HarnessLoader":
        """Load scenarios from all JSON files in a directory.

        Args:
            directory: Path to directory containing JSON files

        Returns:
            Self for chaining
        """
        file_scenarios = load_harness_directory(directory)
        for file_stem, scenarios in file_scenarios.items():
            self.scenarios.extend(scenarios)
            self.file_map[file_stem] = scenarios
        return self

    def get_scenarios(self) -> list[Scenario]:
        """Get all loaded scenarios.

        Returns:
            List of all loaded scenarios
        """
        return self.scenarios

    def get_file_map(self) -> dict[str, list[Scenario]]:
        """Get mapping of file stems to scenarios.

        Returns:
            Dict mapping file stem to list of scenarios
        """
        return self.file_map
