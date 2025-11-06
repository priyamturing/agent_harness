"""Test harness for running benchmarks against agents."""

from .agent_factory import create_agent, DEFAULT_SYSTEM_PROMPT
from .loader import HarnessLoader, load_harness_file, load_harness_directory, scenario_to_task
from .orchestrator import TestHarness, TestHarnessConfig, RunResult
from .scenario import Scenario, ScenarioPrompt, VerifierDefinition

__all__ = [
    "create_agent",
    "DEFAULT_SYSTEM_PROMPT",
    "HarnessLoader",
    "load_harness_file",
    "load_harness_directory",
    "scenario_to_task",
    "TestHarness",
    "TestHarnessConfig",
    "RunResult",
    "Scenario",
    "ScenarioPrompt",
    "VerifierDefinition",
]
