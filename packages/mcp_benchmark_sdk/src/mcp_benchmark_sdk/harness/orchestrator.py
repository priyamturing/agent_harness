"""Test harness orchestrator for running benchmarks."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import uuid4

import httpx

from ..agents import Agent
from ..mcp import MCPConfig
from ..runtime import RunContext, RunObserver
from ..tasks import Result
from ..verifiers import Verifier, VerifierResult
from .loader import create_verifier_from_definition, load_harness_directory, load_harness_file, scenario_to_task
from .scenario import Scenario, VerifierDefinition


@dataclass
class RunResult:
    """Result of a single test run."""

    model: str
    scenario_id: str
    scenario_name: str
    run_number: int
    success: bool
    result: Result
    verifier_results: list[VerifierResult]
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TestHarnessConfig:
    """Configuration for test harness execution."""

    mcps: list[MCPConfig]
    sql_runner_url: Optional[str] = None
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
            mcps=[mcp_config],
            sql_runner_url="http://localhost:8015/api/sql-runner",
        )
    )
    
    # Attach observers for UI
    harness.add_observer_factory(lambda: MyUIObserver())
    
    # Run benchmarks
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
    ) -> list[RunResult]:
        """Run benchmarks across all models and scenarios.

        Args:
            models: List of model identifiers
            agent_factory: Factory function to create agents from model strings
            observer_config: Optional configuration to pass to observer factories

        Returns:
            List of run results
        """
        run_configs = self._build_run_configs(models)
        
        async with httpx.AsyncClient(timeout=30.0) as http_client:
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
                        result=None,  # type: ignore
                        verifier_results=[],
                        error=str(result),
                    )
                )
            elif isinstance(result, RunResult):
                run_results.append(result)

        return run_results

    def _build_run_configs(self, models: list[str]) -> list[dict[str, Any]]:
        """Build run configurations for all model/scenario combinations.

        Args:
            models: List of model identifiers

        Returns:
            List of run configuration dicts
        """
        configs = []
        
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
                        
                        configs.append({
                            "label": "_".join(label_parts),
                            "model": model,
                            "scenario": scenario,
                            "file_stem": file_stem,
                            "run_num": run_num,
                        })
        
        return configs

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

            try:
                agent = agent_factory(
                    model,
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_output_tokens,
                    tool_call_limit=self.config.tool_call_limit,
                )

                task, verifier_defs = scenario_to_task(scenario, self.config.mcps)

                run_context = RunContext(
                    sql_runner_url=self.config.sql_runner_url,
                )

                for factory in self.observer_factories:
                    observer = factory()
                    if observer_config and hasattr(observer, "configure"):
                        observer.configure(run_config["label"], observer_config)  # type: ignore
                    run_context.add_observer(observer)

                verifier_runner = VerifierRunner(
                    verifier_defs,
                    run_context,
                    http_client=http_client,
                )

                result = await agent.run(
                    task,
                    max_steps=self.config.max_steps,
                    run_context=run_context,
                )

                verifier_results = await verifier_runner.run_verifiers()

                result.verifier_results = verifier_results

                all_verifiers_passed = (
                    all(v.success for v in verifier_results) if verifier_results else True
                )
                final_success = result.success and all_verifiers_passed

                if not all_verifiers_passed:
                    failed = [v.name for v in verifier_results if not v.success]
                    error = f"Verifiers failed: {', '.join(failed)}"
                else:
                    error = result.error

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
                )

            except Exception as exc:
                return RunResult(
                    model=model,
                    scenario_id=scenario.scenario_id,
                    scenario_name=scenario.name,
                    run_number=run_num,
                    success=False,
                    result=None,  # type: ignore
                    verifier_results=[],
                    error=str(exc),
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
    ):
        """Initialize verifier runner.
        
        Args:
            verifier_defs: List of verifier definitions from harness
            run_context: Runtime context with SQL runner URL and database ID
            http_client: Shared HTTP client (managed by caller)
        """
        self.verifiers: list[Verifier] = []
        
        if not verifier_defs or not run_context.sql_runner_url:
            return
        
        self.verifiers = [
            create_verifier_from_definition(
                verifier_def,
                run_context.sql_runner_url,
                run_context.database_id,
                http_client,
            )
            for verifier_def in verifier_defs
        ]

    async def run_verifiers(self) -> list[VerifierResult]:
        """Execute all verifiers (reuses pre-built verifiers).
        
        Returns:
            List of verifier results
        """
        results: list[VerifierResult] = []
        for verifier in self.verifiers:
            result = await verifier.verify()
            results.append(result)
        return results
