import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp_benchmark_sdk.agents.mcp.config import MCPConfig  # noqa: E402
from mcp_benchmark_sdk.harness.orchestrator import (  # noqa: E402
    TestHarness,
    TestHarnessConfig,
    _round_robin_configs,
)


def _write_harness(tmp_path: Path, scenario_count: int = 1) -> Path:
    scenarios = []
    for idx in range(1, scenario_count + 1):
        scenarios.append(
            {
                "scenario_id": f"scenario-{idx}",
                "name": f"Scenario {idx}",
                "prompts": [
                    {
                        "prompt_text": f"Prompt {idx}",
                    }
                ],
            }
        )

    payload = {"scenarios": scenarios}
    path = tmp_path / "harness.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _create_harness(tmp_path: Path, runs_per_scenario: int = 1) -> TestHarness:
    harness_path = _write_harness(tmp_path)
    config = TestHarnessConfig(
        mcp=MCPConfig(name="test", url="http://localhost:9000/mcp"),
        runs_per_scenario=runs_per_scenario,
    )
    return TestHarness(harness_path, config)


def test_build_run_configs_uses_round_robin_order(tmp_path: Path) -> None:
    harness = _create_harness(tmp_path, runs_per_scenario=2)
    configs = harness._build_run_configs(["model-a", "model-b"])

    labels = [config["label"] for config in configs]
    assert labels == [
        "model-a_scenario-1_r1",
        "model-b_scenario-1_r1",
        "model-a_scenario-1_r2",
        "model-b_scenario-1_r2",
    ]


def test_round_robin_handles_uneven_model_counts() -> None:
    configs = _round_robin_configs(
        {
            "model-a": [{"label": "a1"}, {"label": "a2"}, {"label": "a3"}],
            "model-b": [{"label": "b1"}],
            "model-c": [],
        }
    )

    assert [config["label"] for config in configs] == ["a1", "b1", "a2", "a3"]
