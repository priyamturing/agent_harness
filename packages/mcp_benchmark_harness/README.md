# MCP Benchmark Harness

`mcp-benchmark-harness` provides the orchestration layer for running scripted
agent benchmarks. It loads JSON harness files, converts them into agent tasks,
spins up observers, and manages verifier execution.

```bash
pip install mcp-benchmark-harness
```

The harness depends on `mcp-benchmark-agents` for the underlying agent runtime,
so installing it gives you everything needed to load MCP configs, run agents,
and verify outcomes. The top-level `mcp-benchmark-sdk` meta package bundles the
agents and harness libraries together for convenience.
