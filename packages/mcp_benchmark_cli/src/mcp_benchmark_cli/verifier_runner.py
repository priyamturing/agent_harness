"""Verifier orchestration for CLI."""

from typing import Optional

from mcp_benchmark_sdk import RunContext
from mcp_benchmark_sdk.verifiers import Verifier, VerifierResult, execute_verifiers


class VerifierRunner:
    """Orchestrates verifier execution independently of agent.
    
    This class provides a clean interface for running verifiers
    after agent execution completes, maintaining separation between
    execution and verification concerns.
    """

    async def run_verifiers(
        self,
        verifiers: list[Verifier],
        run_context: RunContext,
    ) -> list[VerifierResult]:
        """Execute verifiers using SDK's execute_verifiers utility.
        
        Args:
            verifiers: List of verifiers to execute
            run_context: Runtime context with SQL runner URL and database ID
            
        Returns:
            List of verifier results
        """
        if not verifiers:
            return []
        
        if not run_context.sql_runner_url:
            return []
        
        results = await execute_verifiers(
            verifiers,
            run_context.sql_runner_url,
            run_context.database_id,
            run_context.get_http_client(),
        )
        
        return results

