"""Load JSON harness files and convert to SDK Task objects.

Note: This module now re-exports from mcp_benchmark_sdk.harness.
The harness functionality has been moved to the SDK for broader use.
"""

from __future__ import annotations

from mcp_benchmark_sdk.harness import (
    load_harness_file,
    scenario_to_task,
    create_verifier_from_definition as _create_verifier_from_definition,
)

__all__ = ["load_harness_file", "scenario_to_task", "_create_verifier_from_definition"]
