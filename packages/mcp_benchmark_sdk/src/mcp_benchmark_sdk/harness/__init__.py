"""Test harness for running benchmarks against agents."""

from .agent_factory import create_agent, DEFAULT_SYSTEM_PROMPT
from .loader import HarnessLoader, load_harness_file, load_harness_directory, scenario_to_task, create_verifier_from_definition
from .orchestrator import TestHarness, TestHarnessConfig, RunResult
from .scenario import Scenario, ScenarioPrompt, VerifierDefinition

__all__ = [
    "create_agent",
    "DEFAULT_SYSTEM_PROMPT",
    "HarnessLoader",
    "load_harness_file",
    "load_harness_directory",
    "scenario_to_task",
    "create_verifier_from_definition",
    "TestHarness",
    "TestHarnessConfig",
    "RunResult",
    "Scenario",
    "ScenarioPrompt",
    "VerifierDefinition",
]
