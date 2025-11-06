"""Example showing how to save artifacts with full conversation history."""

import asyncio
import json
from pathlib import Path

from mcp_benchmark_sdk import (
    TestHarness,
    TestHarnessConfig,
    MCPConfig,
    create_agent,
)


async def main():
    """Run benchmarks and save artifacts with conversation history."""
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
        )
    )

    print(f"Running benchmarks for {len(harness.scenarios)} scenario(s)...")

    results = await harness.run(
        models=["gpt-4o"],
        agent_factory=create_agent,
    )

    # Create artifacts directory
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    print(f"\nSaving {len(results)} result(s) to {artifacts_dir}/")

    for result in results:
        # Export complete result to dict
        result_dict = result.to_dict()
        
        # Create filename
        filename = f"{result.model}_{result.scenario_id}_run{result.run_number}.json"
        filepath = artifacts_dir / filename
        
        # Save to file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Saved {filename}")
        print(f"    - Success: {result.success}")
        print(f"    - Conversation: {len(result_dict['conversation'])} messages")
        print(f"    - Verifiers: {len(result_dict['verifier_results'])} results")
        print(f"    - Reasoning traces: {len(result_dict['reasoning_traces'])} traces")

    print(f"\n✓ All artifacts saved to {artifacts_dir}/")
    
    # Show example of accessing conversation
    if results:
        print("\nExample conversation entry:")
        conversation = results[0].get_conversation_history()
        if conversation:
            first_msg = conversation[0]
            print(f"  Role: {first_msg['role']}")
            print(f"  Content: {first_msg['content'][:100]}...")
            if 'tool_calls' in first_msg:
                print(f"  Tool calls: {len(first_msg['tool_calls'])}")


if __name__ == "__main__":
    asyncio.run(main())
