"""Verifier orchestration for CLI."""

from typing import Any, Optional

from mcp_benchmark_sdk import RunContext
from mcp_benchmark_sdk.tasks.scenario import VerifierDefinition
from mcp_benchmark_sdk.verifiers import Verifier, VerifierResult

from .harness_loader import _create_verifier_from_definition


class VerifierRunner:
    """Orchestrates verifier execution independently of agent.
    
    This class provides a clean interface for running verifiers
    after agent execution completes, maintaining separation between
    execution and verification concerns.
    
    Verifiers are built once from definitions and reused for efficiency.
    """

    def __init__(
        self,
        verifier_defs: list[VerifierDefinition],
        run_context: RunContext,
    ):
        """Initialize with verifier definitions and runtime context.
        
        Args:
            verifier_defs: List of verifier definitions from harness
            run_context: Runtime context with SQL runner URL and database ID
        """
        self.verifiers: list[Verifier] = []
        
        if not verifier_defs or not run_context.sql_runner_url:
            return
        
        # Build verifiers once with runtime context
        for verifier_def in verifier_defs:
            verifier = _create_verifier_from_definition(
                verifier_def,
                run_context.sql_runner_url,
                run_context.database_id,
                run_context.get_http_client(),
            )
            self.verifiers.append(verifier)

    async def run_verifiers(self) -> list[VerifierResult]:
        """Execute all verifiers.
        
        Returns:
            List of verifier results
        """
        if not self.verifiers:
            return []
        
        results: list[VerifierResult] = []
        for verifier in self.verifiers:
            result = await verifier.verify()
            results.append(result)
        
        return results

