"""Simple example showing how to use the SDK's TestHarness.

This demonstrates the HUD-like experience for running benchmarks.
"""

import asyncio
from pathlib import Path

from mcp_benchmark_sdk import (
    TestHarness,
    TestHarnessConfig,
    MCPConfig,
    create_agent,
    RunObserver,
)


class SimpleConsoleObserver(RunObserver):
    """Simple console observer for progress tracking."""

    def __init__(self, label: str):
        self.label = label

    async def on_message(self, role: str, content: str, metadata=None):
        if role == "assistant":
            print(f"[{self.label}] Agent: {content[:100]}...")

    async def on_tool_call(self, tool_name, arguments, result, is_error=False):
        status = "✗" if is_error else "✓"
        print(f"[{self.label}] Tool {status}: {tool_name}")

    async def on_status(self, message: str, level: str = "info"):
        print(f"[{self.label}] {message}")


async def main():
    mcp_config = MCPConfig(
        name="jira",
        url="http://localhost:8015/mcp",
        transport="streamable_http",
    )

    harness = TestHarness(
        harness_path=Path("task4.json"),
        config=TestHarnessConfig(
            mcps=[mcp_config],
            sql_runner_url="http://localhost:8015/api/sql-runner",
            max_steps=1000,
            tool_call_limit=1000,
            temperature=0.1,
            runs_per_scenario=1,
            max_concurrent_runs=5,
        )
    )

    def observer_factory():
        return SimpleConsoleObserver("benchmark")

    harness.add_observer_factory(observer_factory)

    print(f"Loaded {len(harness.scenarios)} scenario(s)")
    print(f"Running benchmarks...")

    results = await harness.run(
        models=["gpt-4o"],
        agent_factory=create_agent,
    )

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    for result in results:
        status = "✓ PASS" if result.success else "✗ FAIL"
        print(f"{result.model} - {result.scenario_id}: {status}")
        if result.error:
            print(f"  Error: {result.error}")
        
        # Show conversation summary
        conversation = result.get_conversation_history()
        print(f"  Conversation: {len(conversation)} messages")
        
        if result.verifier_results:
            print(f"  Verifiers:")
            for v in result.verifier_results:
                v_status = "✓" if v.success else "✗"
                print(f"    {v_status} {v.name}: {v.message}")
        
        # Export to dict for saving (includes full conversation)
        result_dict = result.to_dict()
        print(f"  Result dict keys: {list(result_dict.keys())}")

    successful = sum(1 for r in results if r.success)
    print(f"\nTotal: {len(results)} | Passed: {successful} | Failed: {len(results) - successful}")
    
    # Example: Save first result to JSON
    if results:
        import json
        first_result = results[0].to_dict()
        print(f"\nFirst result conversation has {len(first_result['conversation'])} messages")
        print("Sample conversation entry:")
        if first_result['conversation']:
            print(json.dumps(first_result['conversation'][0], indent=2)[:200] + "...")


if __name__ == "__main__":
    asyncio.run(main())
