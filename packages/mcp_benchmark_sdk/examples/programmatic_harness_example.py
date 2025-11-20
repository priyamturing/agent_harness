"""Example of creating a test harness programmatically without a JSON file.

This demonstrates how to use TestHarness.from_scenarios() to create scenarios
directly in Python code instead of loading them from a JSON file.
"""

import asyncio

from mcp_benchmark_sdk import (
    TestHarness,
    TestHarnessConfig,
    MCPConfig,
    create_agent,
    Scenario,
    ScenarioPrompt,
    VerifierDefinition,
    RunObserver,
)


class SimpleConsoleObserver(RunObserver):
    """Simple console observer for progress tracking."""

    def __init__(self, label: str = "harness"):
        self.label = label

    async def on_message(self, role: str, content: str, metadata=None):
        if role == "assistant":
            print(f"[{self.label}] Agent: {content[:80]}...")

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

    scenario1 = Scenario(
        scenario_id="create_bug_simple",
        name="Create Bug Issue - Simple",
        description="Test if agent can create a bug issue",
        prompts=[
            ScenarioPrompt(
                prompt_text="Create a bug issue in project DEMO with summary 'Login button broken'",
                expected_tools=["create_issue"],
                verifiers=[
                    VerifierDefinition(
                        verifier_type="database_state",
                        validation_config={
                            "query": "SELECT COUNT(*) FROM issue WHERE summary = 'Login button broken'",
                            "expected_value": 1,
                            "comparison_type": "equals"
                        },
                        name="check_issue_created"
                    )
                ]
            )
        ],
        metadata={"difficulty": "easy", "category": "basic"},
        conversation_mode=False
    )

    scenario2 = Scenario(
        scenario_id="create_task_with_assignee",
        name="Create Task with Assignee",
        description="Test if agent can create a task and assign it",
        prompts=[
            ScenarioPrompt(
                prompt_text="Create a task in DEMO project with summary 'Update documentation' and assign it to admin",
                expected_tools=["create_issue"],
                verifiers=[
                    VerifierDefinition(
                        verifier_type="database_state",
                        validation_config={
                            "query": "SELECT COUNT(*) FROM issue WHERE summary = 'Update documentation' AND assignee_id IS NOT NULL",
                            "expected_value": 1,
                            "comparison_type": "equals"
                        },
                        name="check_task_created_with_assignee"
                    )
                ]
            )
        ],
        metadata={"difficulty": "medium", "category": "advanced"},
        conversation_mode=False
    )

    harness = TestHarness.from_scenarios(
        scenarios=[scenario1, scenario2],
        config=TestHarnessConfig(
            mcp=mcp_config,
            max_steps=50,
            tool_call_limit=100,
            temperature=0.1,
            runs_per_scenario=1,
            max_concurrent_runs=2,
        ),
        harness_name="programmatic_test"
    )

    def observer_factory():
        return SimpleConsoleObserver("benchmark")

    harness.add_observer_factory(observer_factory)

    print(f"Created harness with {len(harness.scenarios)} scenario(s) programmatically")
    print("Running benchmarks...\n")

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
        
        if result.verifier_results:
            print(f"  Verifiers:")
            for v in result.verifier_results:
                v_status = "✓" if v.success else "✗"
                msg = v.error if v.error else f"{v.actual_value} vs {v.expected_value}"
                print(f"    {v_status} {v.name}: {msg}")

    successful = sum(1 for r in results if r.success)
    print(f"\nTotal: {len(results)} | Passed: {successful} | Failed: {len(results) - successful}")


if __name__ == "__main__":
    asyncio.run(main())










