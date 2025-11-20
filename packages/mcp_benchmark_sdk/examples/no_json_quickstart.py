"""Quick reference: Running a test harness WITHOUT a JSON file.

Shows the minimal code needed to create and run scenarios programmatically.
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
)


async def main():
    mcp_config = MCPConfig(
        name="jira",
        url="http://localhost:8015/mcp",
        transport="streamable_http",
    )

    scenario = Scenario(
        scenario_id="test_1",
        name="Create Bug Issue",
        description="Simple bug creation test",
        prompts=[
            ScenarioPrompt(
                prompt_text="Create a bug in DEMO project with summary 'Test Bug'",
                expected_tools=["create_issue"],
                verifiers=[
                    VerifierDefinition(
                        verifier_type="database_state",
                        validation_config={
                            "query": "SELECT COUNT(*) FROM issue WHERE summary = 'Test Bug'",
                            "expected_value": 1,
                            "comparison_type": "equals"
                        },
                        name="check_created"
                    )
                ]
            )
        ],
        metadata={},
        conversation_mode=False
    )

    harness = TestHarness.from_scenarios(
        scenarios=[scenario],
        config=TestHarnessConfig(mcp=mcp_config)
    )

    results = await harness.run(
        models=["gpt-4o"],
        agent_factory=create_agent
    )

    for r in results:
        print(f"{r.scenario_id}: {'✓ PASS' if r.success else '✗ FAIL'}")


if __name__ == "__main__":
    asyncio.run(main())










