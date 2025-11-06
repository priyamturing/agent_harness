"""Verifier orchestration for CLI."""

import httpx

from mcp_benchmark_sdk import RunContext
from mcp_benchmark_sdk.verifiers import Verifier, VerifierResult

from .harness_loader import _create_verifier_from_definition
from .scenario import VerifierDefinition


class VerifierRunner:
    """Orchestrates verifier execution independently of agent.
    
    Creates verifiers once and reuses them across multiple verification runs.
    Uses shared HTTP client provided by CLI for efficiency across parallel tasks.
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
            http_client: Shared HTTP client (managed by CLI)
        """
        self.verifiers: list[Verifier] = []
        
        if not verifier_defs or not run_context.sql_runner_url:
            return
        
        # Build verifiers ONCE with shared HTTP client
        self.verifiers = [
            _create_verifier_from_definition(
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

