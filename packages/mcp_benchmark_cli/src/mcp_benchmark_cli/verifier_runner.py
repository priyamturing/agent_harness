"""Verifier orchestration for CLI.

Note: This module now re-exports from mcp_benchmark_sdk.harness.orchestrator.
The VerifierRunner has been moved to the SDK for broader use.
"""

from mcp_benchmark_sdk.harness.orchestrator import VerifierRunner

__all__ = ["VerifierRunner"]

