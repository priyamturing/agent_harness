"""Test harness orchestrator for running benchmarks."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, overload
from uuid import uuid4

import httpx

from ..agents.mcp.config import MCPConfig
from ..agents.runtime.context import RunContext
from ..agents.runtime.events import RunObserver
from ..agents.tasks.result import Result
from ..constants import HTTP_CLIENT_TIMEOUT_SECONDS
from .verifiers import Verifier, VerifierResult
from .loader import create_verifier_from_definition, load_harness_directory, load_harness_file, round_robin_configs, scenario_to_task
from .scenario import Scenario, VerifierDefinition

if TYPE_CHECKING:
    from ..agents import Agent


@dataclass
class RunResult:
    """Result of a single test run.
    
    Attributes:
        model: Model identifier used for this run
        scenario_id: Unique scenario identifier
        scenario_name: Human-readable scenario name
        run_number: Run number (for multiple runs per scenario)
        success: Overall success status (agent + verifiers)
        result: Agent execution result (None if agent failed to initialize/run)
        verifier_results: Results from all verifiers
        error: Error message if run failed
        metadata: Additional metadata about the run
        prompt_text: Prompt text associated with the scenario
        execution_time_ms: Duration of the run in milliseconds
    """

    model: str
    scenario_id: str
    scenario_name: str
    run_number: int
    success: bool
    result: Optional[Result]
    verifier_results: list[VerifierResult]
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    prompt_text: Optional[str] = None
    execution_time_ms: Optional[int] = None
    
    def get_conversation_history(self) -> list[dict[str, Any]]:
        """Extract conversation history from result messages.
        
        Delegates to the Result.get_conversation_history() method which formats
        messages in a clean, user-friendly format.
        
        Returns:
            List of conversation entries in clean format:
            - Messages: {"type": "message", "role": "user|assistant|system", "content": "...", "reasoning": [...]}
            - Tool calls: {"type": "tool_call", "tool": "tool_name", "args": {...}}
            - Tool results: {"type": "tool_result", "tool": "tool_name", "output": {...}}
        """
        if not self.result:
            return []
        
        return self.result.get_conversation_history()

    def to_dict(self) -> dict[str, Any]:
        """Convert RunResult to dictionary for serialization.
        
        Returns:
            Dictionary representation suitable for JSON export
        """
        return {
            "model": self.model,
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "run_number": self.run_number,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
            "prompt_text": self.prompt_text,
            "execution_time_ms": self.execution_time_ms,
            "conversation": self.get_conversation_history(),
        "verifier_results": [
            {
                "name": vr.name,
                "success": vr.success,
                "expected_value": vr.expected_value if hasattr(vr, "expected_value") else None,
                "actual_value": vr.actual_value if hasattr(vr, "actual_value") else None,
                "comparison": vr.comparison_type if hasattr(vr, "comparison_type") else None,
                "error": vr.error if hasattr(vr, "error") else None,
            }
            for vr in self.verifier_results
        ],
            "reasoning_traces": self.result.reasoning_traces if self.result and self.result.reasoning_traces else [],
            "steps": self.result.metadata.get("steps") if self.result else None,
            "database_id": self.result.database_id if self.result else None,
        }


@dataclass
class ModelResultFile:
    """Serializable payload for a single model/harness combination."""

    harness_name: str
    model_name: str
    payload: dict[str, Any]


@dataclass
class ResultBundle(Sequence[RunResult]):
    """Wrapper around run results with helpers for building exports."""

    harness_name: str
    run_results: list[RunResult]
    model_names: Optional[list[str]] = None

    def __post_init__(self) -> None:
        self._run_results = list(self.run_results)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._run_results)

    def __iter__(self):  # pragma: no cover - trivial
        return iter(self._run_results)

    @overload
    def __getitem__(self, index: int) -> RunResult: ...
    
    @overload
    def __getitem__(self, index: slice) -> Sequence[RunResult]: ...

    def __getitem__(self, index: int | slice) -> RunResult | Sequence[RunResult]:
        return self._run_results[index]

    def build_model_reports(
        self,
        *,
        reset_database_between_runs: bool = False,
        status: str = "completed",
    ) -> list[ModelResultFile]:
        """Build serialized report payloads grouped by harness/model.
        
        Auto-detects runs_per_prompt from the actual run results.
        """
        grouped = self._group_runs_by_file_and_model()
        reports: list[ModelResultFile] = []

        for (harness_name, model_name), runs in grouped.items():
            platform_name = infer_platform_from_model(model_name)
            ordered_runs = sorted(
                runs,
                key=lambda r: (r.scenario_id, r.run_number),
            )
            
            runs_per_prompt = max((r.run_number for r in ordered_runs), default=1)
            
            payload = {
                "execution_metadata": {
                    "harness_name": harness_name,
                    "status": status,
                    "models_tested": [
                        {
                            "model_name": model_name,
                            "platform_name": platform_name,
                            "platform_config_id": model_name,
                        }
                    ],
                    "multipass_config": {
                        "runs_per_prompt": runs_per_prompt,
                        "reset_database_between_runs": reset_database_between_runs,
                    },
                },
                "results": [
                    self._format_run_entry(run_result, platform_name)
                    for run_result in ordered_runs
                ],
            }
            reports.append(
                ModelResultFile(
                    harness_name=harness_name,
                    model_name=model_name,
                    payload=payload,
                )
            )

        return reports

    def _group_runs_by_file_and_model(self) -> dict[tuple[str, str], list[RunResult]]:
        grouped: dict[tuple[str, str], list[RunResult]] = {}
        for result in self._run_results:
            file_stem = (result.metadata or {}).get("file_stem") or self.harness_name
            key = (file_stem, result.model)
            grouped.setdefault(key, []).append(result)
        return grouped

    def _format_run_entry(self, run_result: RunResult, platform_name: str) -> dict[str, Any]:
        prompt_text = run_result.prompt_text or ""
        entry: dict[str, Any] = {
            "model_name": run_result.model,
            "platform_name": platform_name,
            "scenario_id": run_result.scenario_id,
            "prompt_index": 0,
            "run_number": run_result.run_number,
            "prompt_text": prompt_text,
            "response": self._build_response(run_result),
            "execution_time_ms": run_result.execution_time_ms or 0,
            "overall_success": self._verifier_success(run_result),
            "error_occurred": self._error_occurred(run_result),
            "error_message": self._error_message(run_result),
            "comments": [],
        }

        entry["database_id"] = run_result.result.database_id if run_result.result else None

        if run_result.result and run_result.result.langsmith_url:
            entry["langsmith_url"] = run_result.result.langsmith_url
        if run_result.result and run_result.result.langfuse_url:
            entry["langfuse_url"] = run_result.result.langfuse_url

        return entry

    def _build_response(self, run_result: RunResult) -> list[dict[str, Any]]:
        if not run_result.result:
            return []

        entries = run_result.result.build_benchmark_response()

        if run_result.verifier_results:
            entries.append(
                {
                    "type": "verifier_results",
                    "results": [
                        self._format_verifier_result(verifier)
                        for verifier in run_result.verifier_results
                    ],
                }
            )

        return entries

    def _format_verifier_result(self, verifier: VerifierResult) -> dict[str, Any]:
        metadata = verifier.metadata or {}
        verifier_type = metadata.get("verifier_type") or verifier.name
        query = metadata.get("query")

        if verifier.success:
            if (
                verifier.expected_value is not None
                and verifier.actual_value is not None
                and verifier.expected_value == verifier.actual_value
            ):
                message = "Values match"
            else:
                message = "Verifier passed"
        else:
            if verifier.expected_value is not None and verifier.actual_value is not None:
                message = (
                    f"Expected {verifier.expected_value}, got {verifier.actual_value}"
                )
            else:
                message = verifier.error or "Verifier failed"

        return {
            "success": verifier.success,
            "message": message,
            "expected_value": verifier.expected_value,
            "actual_value": verifier.actual_value,
            "comparison_type": verifier.comparison_type,
            "query": query,
            "verifier_type": verifier_type,
            "multi_gym_failure": False,
            "gyms_tried": 1,
            "all_gym_errors": [],
        }

    def _verifier_success(self, run_result: RunResult) -> bool:
        if not run_result.verifier_results:
            return False
        return all(verifier.success for verifier in run_result.verifier_results)

    def _error_occurred(self, run_result: RunResult) -> bool:
        if run_result.result is None:
            return True
        return bool(run_result.result.error)

    def _error_message(self, run_result: RunResult) -> Optional[str]:
        if not self._error_occurred(run_result):
            return None
        if run_result.result is None:
            return run_result.error
        return run_result.result.error or run_result.error


@dataclass
class TestHarnessConfig:
    """Configuration for test harness execution."""

    mcp: MCPConfig
    max_steps: int = 1000
    tool_call_limit: int = 1000
    temperature: float = 0.1
    max_output_tokens: Optional[int] = None
    max_concurrent_runs: int = 20
    runs_per_scenario: int = 1


class TestHarness:
    """Orchestrates benchmark execution across scenarios and models.
    
    Provides a simple, HUD-like experience for running benchmarks:
    
    ```python
    harness = TestHarness(
        harness_path=Path("benchmarks/task1.json"),
        config=TestHarnessConfig(
            mcp=mcp_config,
        )
    )
    
    harness.add_observer_factory(lambda: MyUIObserver())
    
    results = await harness.run(
        models=["gpt-4o", "claude-sonnet-4-5"],
        agent_factory=create_agent_from_string,
    )
    ```
    """

    def __init__(
        self,
        harness_path: Path,
        config: TestHarnessConfig,
    ):
        """Initialize test harness.

        Args:
            harness_path: Path to harness file or directory
            config: Test harness configuration
        """
        self.harness_path = harness_path
        self.config = config
        self.scenarios: list[Scenario] = []
        self.file_map: dict[str, list[Scenario]] = {}
        self.observer_factories: list[Callable[[], RunObserver]] = []
        
        self._load_scenarios()

    def _load_scenarios(self) -> None:
        """Load scenarios from harness path."""
        if not self.harness_path.exists():
            raise ValueError(f"Harness path does not exist: {self.harness_path}")

        if self.harness_path.is_dir():
            self.file_map = load_harness_directory(self.harness_path)
            for scenarios in self.file_map.values():
                self.scenarios.extend(scenarios)
        else:
            scenarios = load_harness_file(self.harness_path)
            self.scenarios = scenarios
            self.file_map[self.harness_path.stem] = scenarios

    def _default_harness_name(self) -> str:
        if self.harness_path.is_dir():
            return self.harness_path.name
        return self.harness_path.stem

    def add_observer_factory(self, factory: Callable[[], RunObserver]) -> None:
        """Add observer factory for creating observers per run.

        Args:
            factory: Callable that creates a new observer instance
        """
        self.observer_factories.append(factory)

    async def run(
        self,
        models: list[str],
        agent_factory: Callable[..., Agent],
        observer_config: Optional[dict[str, Any]] = None,
    ) -> ResultBundle:
        """Run benchmarks across all models and scenarios.

        Args:
            models: List of model identifiers
            agent_factory: Factory function to create agents from model strings
            observer_config: Optional configuration to pass to observer factories

        Returns:
            ResultBundle containing all run results and aggregation helpers
        """
        run_configs = self._build_run_configs(models)
        
        async with httpx.AsyncClient(timeout=HTTP_CLIENT_TIMEOUT_SECONDS) as http_client:
            semaphore = asyncio.Semaphore(self.config.max_concurrent_runs)
            tasks = []

            for run_config in run_configs:
                task = self._run_single(
                    run_config=run_config,
                    agent_factory=agent_factory,
                    http_client=http_client,
                    semaphore=semaphore,
                    observer_config=observer_config,
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

        run_results: list[RunResult] = []
        for config, result in zip(run_configs, results):
            if isinstance(result, Exception):
                run_results.append(
                    RunResult(
                        model=config["model"],
                        scenario_id=config["scenario"].scenario_id,
                        scenario_name=config["scenario"].name,
                        run_number=config["run_num"],
                        success=False,
                        result=None,
                        verifier_results=[],
                        error=str(result),
                    )
                )
            elif isinstance(result, RunResult):
                run_results.append(result)

        return ResultBundle(
            harness_name=self._default_harness_name(),
            run_results=run_results,
            model_names=models,
        )

    def _build_run_configs(self, models: list[str]) -> list[dict[str, Any]]:
        """Build run configurations for all model/scenario combinations.

        Args:
            models: List of model identifiers

        Returns:
            List of run configuration dicts
        """
        configs_by_model: dict[str, list[dict[str, Any]]] = {}
        
        for model in models:
            for file_stem, scenarios in self.file_map.items():
                for scenario in scenarios:
                    for run_num in range(1, self.config.runs_per_scenario + 1):
                        label_parts = [model]
                        if len(self.file_map) > 1:
                            label_parts.append(file_stem)
                        label_parts.append(scenario.scenario_id)
                        if self.config.runs_per_scenario > 1:
                            label_parts.append(f"r{run_num}")
                        
                        configs_by_model.setdefault(model, []).append(
                            {
                                "label": "_".join(label_parts),
                                "model": model,
                                "scenario": scenario,
                                "file_stem": file_stem,
                                "run_num": run_num,
                            }
                        )
        
        return round_robin_configs(configs_by_model)



    async def _run_single(
        self,
        run_config: dict[str, Any],
        agent_factory: Callable[..., Agent],
        http_client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        observer_config: Optional[dict[str, Any]],
    ) -> RunResult:
        """Run a single benchmark task.

        Args:
            run_config: Run configuration dict
            agent_factory: Factory to create agent
            http_client: Shared HTTP client
            semaphore: Concurrency semaphore
            observer_config: Optional observer configuration

        Returns:
            RunResult
        """
        async with semaphore:
            scenario = run_config["scenario"]
            model = run_config["model"]
            run_num = run_config["run_num"]
            prompt_text = scenario.prompts[0].prompt_text if scenario.prompts else ""
            start_time = time.perf_counter()

            try:
                agent = agent_factory(
                    model,
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_output_tokens,
                    tool_call_limit=self.config.tool_call_limit,
                )

                task, verifier_defs = scenario_to_task(scenario, self.config.mcp)

                run_context = RunContext()

                for factory in self.observer_factories:
                    observer = factory()
                    if observer_config and hasattr(observer, "configure"):
                        observer.configure(run_config["label"], observer_config)  # type: ignore
                    run_context.add_observer(observer)

                if not self.config.mcp:
                    raise ValueError("MCP configuration is required to run benchmarks")
                mcp_url = self.config.mcp.url

                verifier_runner = VerifierRunner(
                    verifier_defs,
                    run_context,
                    http_client=http_client,
                    mcp_url=mcp_url,
                )

                result = await agent.run(
                    task,
                    max_steps=self.config.max_steps,
                    run_context=run_context,
                )

                verifier_results = await verifier_runner.run_verifiers()

                all_verifiers_passed = (
                    all(v.success for v in verifier_results) if verifier_results else True
                )
                final_success = result.success and all_verifiers_passed

                if not all_verifiers_passed:
                    failed = [v.name for v in verifier_results if not v.success]
                    error = f"Verifiers failed: {', '.join(failed)}"
                else:
                    error = result.error

                elapsed_ms = int((time.perf_counter() - start_time) * 1000)

                return RunResult(
                    model=model,
                    scenario_id=scenario.scenario_id,
                    scenario_name=scenario.name,
                    run_number=run_num,
                    success=final_success,
                    result=result,
                    verifier_results=verifier_results,
                    error=error,
                    metadata={
                        "file_stem": run_config["file_stem"],
                        "temperature": self.config.temperature,
                    },
                    prompt_text=prompt_text,
                    execution_time_ms=elapsed_ms,
                )

            except Exception as exc:
                elapsed_ms = int((time.perf_counter() - start_time) * 1000)
                return RunResult(
                    model=model,
                    scenario_id=scenario.scenario_id,
                    scenario_name=scenario.name,
                    run_number=run_num,
                    success=False,
                    result=None,
                    verifier_results=[],
                    error=str(exc),
                    metadata={
                        "file_stem": run_config.get("file_stem"),
                        "temperature": self.config.temperature,
                    },
                    prompt_text=prompt_text,
                    execution_time_ms=elapsed_ms,
                )


class VerifierRunner:
    """Orchestrates verifier execution independently of agent.
    
    Creates verifiers once and reuses them across multiple verification runs.
    Uses shared HTTP client for efficiency across parallel tasks.
    """

    def __init__(
        self,
        verifier_defs: list[VerifierDefinition],
        run_context: RunContext,
        http_client: httpx.AsyncClient,
        mcp_url: Optional[str] = None,
    ):
        """Initialize verifier runner.
        
        Args:
            verifier_defs: List of verifier definitions from harness
            run_context: Runtime context with database ID
            http_client: Shared HTTP client (managed by caller)
            mcp_url: MCP server URL (used to derive SQL runner URL)
        """
        self._verifier_entries: list[tuple[Verifier, str]] = []
        
        if not verifier_defs or not mcp_url:
            return
        
        self._verifier_entries = [
            (
                create_verifier_from_definition(
                    verifier_def,
                    mcp_url,
                    run_context.database_id,
                    http_client,
                ),
                verifier_def.verifier_type,
            )
            for verifier_def in verifier_defs
        ]

    async def run_verifiers(self) -> list[VerifierResult]:
        """Execute all verifiers (reuses pre-built verifiers).
        
        Returns:
            List of verifier results
        """
        results: list[VerifierResult] = []
        for verifier, verifier_type in self._verifier_entries:
            result = await verifier.verify()
            metadata = dict(result.metadata or {})
            metadata.setdefault("verifier_type", verifier_type)
            result.metadata = metadata
            results.append(result)
        return results


def infer_platform_from_model(model_name: str) -> str:
    """Best-effort mapping from model name to provider platform."""
    model_lower = model_name.lower()
    if "claude" in model_lower or "anthropic" in model_lower:
        return "anthropic"
    if "gpt" in model_lower or "openai" in model_lower or model_lower.startswith("o1"):
        return "openai"
    if "gemini" in model_lower or "google" in model_lower:
        return "google"
    if "grok" in model_lower or "xai" in model_lower:
        return "xai"
    if "qwen" in model_lower:
        return "qwen"
    return "unknown"
