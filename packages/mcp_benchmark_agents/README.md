# MCP Benchmark Agents

This library packages everything needed to build and run MCP-compatible agents:

- Provider-specific agent implementations (Anthropic, OpenAI, Google, xAI)
- Shared runtime utilities (tasks, MCP client loader, tool runner, verifiers)
- Response parsers, telemetry helpers, and retry utilities

Install it directly with:

```bash
pip install mcp-benchmark-agents
```

The harness package depends on this library, and the high-level `mcp-benchmark-sdk`
meta package simply bundles both together.
