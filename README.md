# llama-index-tools-a2a

[![PyPI version](https://badge.fury.io/py/llama-index-tools-a2a.svg)](https://badge.fury.io/py/llama-index-tools-a2a)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Support](https://img.shields.io/pypi/pyversions/llama-index-tools-a2a.svg)](https://pypi.org/project/llama-index-tools-a2a/)
[![Build Status](https://github.com/shreyanshgautam3/llama-index-tools-a2a/actions/workflows/publish.yml/badge.svg)](https://github.com/shreyanshgautam3/llama-index-tools-a2a/actions)

**A2A (Agent-to-Agent) Protocol integration for LlamaIndex**

Enable your LlamaIndex agents to discover and invoke other A2A-compatible agents as tools. Implements Google's A2A protocol specification for universal agent interoperability.

## Why?

Agents are becoming composable building blocks, and the A2A (Agent-to-Agent) protocol is the emerging standard for letting independently built agents discover and call each other. While Google's A2A team has an inbound prototype, this package provides a complete, importable integration for LlamaIndex — covering both outbound client (connecting to remote A2A agents) and inbound server (exposing LlamaIndex Workflows as A2A agents) with FunctionTool conversion, partial parameter support, and multi-turn conversation handling.

## Features

- **Outbound Client**: Connect to any A2A agent via its AgentCard URL
- **Skill Discovery**: Automatically discover available tools/skills from remote agents
- **Skill-to-tool Conversion**: Converts AgentCard skill metadata (name, description, examples) to LlamaIndex FunctionTools
- **Inbound Server**: Expose any LlamaIndex Workflow as an A2A-capable agent
- **Async Support**: Full async/await support for non-blocking operation
- **Partial Params**: Support for pre-filling common parameters across tools
- **Multi-turn Conversations**: `contextId` propagation for stateful agent interactions

## Installation

```bash
pip install llama-index-tools-a2a
```

Or with uv:
```bash
uv add llama-index-tools-a2a
```

With dev dependencies:
```bash
pip install "llama-index-tools-a2a[dev]"
```

## Quick Start

### Discovering Tools from an A2A Agent

```python
from llama_index.tools.a2a import aget_tools_from_a2a_card

# Connect to a remote A2A agent and get its tools
tools = await aget_tools_from_a2a_card(
    "https://weather-agent.example.com/.well-known/agent-card.json"
)

# Use tools in your LlamaIndex workflow (FunctionAgent is recommended over deprecated ReActAgent)
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-4o-mini")
agent = FunctionAgent(tools=tools, llm=llm)
result = await agent.run("What's the weather in Tokyo?")
```

### Filtering Specific Tools

```python
# Only use specific skills
tools = await aget_tools_from_a2a_card(
    "https://agent.example.com/.well-known/agent-card.json",
    allowed_tools=["search_web", "calculate"],
)
```

### Pre-filling Common Parameters

```python
# Apply default parameters across all tools
tools = await aget_tools_from_a2a_card(
    "https://agent.example.com/.well-known/agent-card.json",
    global_partial_params={"api_key": "my-secret-key"},
)
```

### Exposing a Workflow as an A2A Agent (Inbound)

```python
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, step
from llama_index.tools.a2a import workflow_as_a2a


class MyStartEvent(StartEvent):
    query: str


class MyWorkflow(Workflow):
    @step
    async def run(self, start_event: MyStartEvent) -> StopEvent:
        return StopEvent(result=f"Processed: {start_event.query}")


# Create the A2A-compatible FastAPI app
app = workflow_as_a2a(MyWorkflow())

# Run with uvicorn: uvicorn my_module:app --host 0.0.0.0 --port 8000
# The agent will be discoverable at http://localhost:8000/.well-known/agent-card.json
```

## Architecture

This integration mirrors the existing `llama-index-tools-mcp` pattern:

| Concept | MCP Equivalent | A2A v0.3 | A2A v1.0 |
|---------|---------------|----------|----------|
| Tool discovery | `list_tools()` | AgentCard.skills[] | AgentCard.skills[] |
| Tool invocation | `call_tool(name, args)` | `message/send` + `tasks/get` | `SendMessage` + `GetTask` |
| Inbound server | `MCPServer` | ASGI app (`workflow_as_a2a()`) | ASGI app (`workflow_as_a2a()`) |
| Workflow wrapping | `workflow_as_mcp()` | `workflow_as_a2a()` | `workflow_as_a2a()` |

## Protocol Version

This package supports **both A2A v0.3 and v1.0** via automatic protocol detection:

- **Outbound client** (`BasicA2AClient`): Auto-detects version from AgentCard `supportedInterfaces` field (wire format). Sends `A2A-Version: 1.0` header for v1.0 servers, uses v1.0 method names (`SendMessage`, `GetTask`), falls back to v0.3 methods (`message/send`, `tasks/get`) for compatibility. Force a version with `protocol_version="0.3"` or `"1.0"` constructor argument.
- **Inbound server** (`workflow_as_a2a`): Generates v1.0 AgentCard with `supportedInterfaces[]` by default (`protocol_version="1.0"`) — note: underlying a2a-sdk 0.3.26 request handler only processes v0.3 method names regardless of AgentCard format.
- **Models**: Includes v1.0 types (`AgentInterface`, `Task`, `TaskStatus`, `Artifact`, `SendMessageResult`) with discriminated union Part types (no `kind` field).

Based on `a2a-sdk==0.3.26` with v1.0 wire protocol additions. See [CHANGELOG](CHANGELOG.md) for version history.

## Limitations

- **Authentication**: No built-in auth handling — relies on caller to supply headers/tokens via `BasicA2AClient` constructor.
- **Streaming**: Inbound server declares `streaming: false` (correct) — actual SSE/event streaming is not implemented in `WorkflowAgentExecutor`.
- **Task polling**: `call_skill` uses `SendMessage` (v1.0) or `message/send` (v0.3) — fire-and-forget; no automatic polling for long-running tasks. Use `get_task()` to retrieve results.
- **Error handling**: Error responses from agents are wrapped in `TaskResult.error` but not systematically classified.
- **Testing**: 34 tests (13 unit + 12 v1.0 + 9 integration) — integration tests use `ASGITransport` against local server; no tests against external A2A v1.0 reference servers.
- **Multi-turn**: `contextId` is passed through but conversation state management is caller responsibility (use `A2AToolSpec.context_id` property).
- **Skill schemas**: A2A v1.0 skills do not include JSON input schemas; tools accept natural language `message` parameter (pass JSON in text for structured input).
- **Underlying SDK**: Uses `a2a-sdk==0.3.26` — inbound server handles only v0.3 method names regardless of AgentCard format version.

## Usage Examples

See [`examples/a2a.ipynb`](examples/a2a.ipynb) for a full end-to-end Jupyter notebook demonstrating:
- Creating an inbound A2A agent with `workflow_as_a2a()`
- Connecting as an outbound client via `BasicA2AClient`
- Converting skills to LlamaIndex FunctionTools with `A2AToolSpec`
- Using A2A tools inside a LlamaIndex ReActAgent
- Filtering skills and pre-filling partial parameters

Run the example server standalone:
```bash
python examples/a2a_server.py --port 8000
# Then visit http://localhost:8000/.well-known/agent-card.json
```

## Project Status

- **v0.1.4** (2026-11-16): Fixed A2A v1.0 protocol compliance — v1.0 method names (`SendMessage`, `GetTask`), discriminated union Part types (no `kind`), contextId on Message object, skill name embedded in message text, proper task state handling (`input-required`, `failed`, `canceled`), v0.3 method name fixes (`message/send`, not `tasks/send`/`messages/send`), dead code removal, 9 new integration tests with client-server round-trip verification ✅
- **v0.1.3** (2026-11-15): A2A v1.0 dual-version support, multi-turn contextId, honest feature naming, FunctionAgent examples ✅
- **v0.1.2** (2026-11-14): CI stability — strict asyncio mode, debug output, protocol version docs ✅
- **v0.1.1** (2026-11-12): PyPI publishing fix — version alignment, CI workflow reliability ✅
- **v0.1.0** (2026-11-12): Initial release — Outbound client, inbound server, tool spec conversion, 13 passing tests ✅

## API Reference

### Outbound Client

- `BasicA2AClient(agent_card_url, protocol_version=None)` — Connect to an A2A agent
  - `get_agent_card()` — Discover agent metadata
  - `list_skills()` — List available skills
  - `call_skill(name, arguments)` — Invoke a skill (v1.0: `SendMessage`, v0.3: `tasks/send`)
  - `send_message(message, context_id)` — Send natural language message (v1.0: `SendMessage`, v0.3: `messages/send`)
  - `get_task(task_id)` — Get task status/result (v1.0: `GetTask`, v0.3: `tasks/get`)
  - `protocol_version` property — Get detected protocol version ("0.3" or "1.0")

### Tool Conversion

- `A2AToolSpec(client)` — Convert A2A skills to LlamaIndex FunctionTools
  - `context_id` property — Get/set context ID for multi-turn conversations
  - `reset_context()` — Clear context ID for new conversation
- `aget_tools_from_a2a_card(url)` — Async convenience function
- `get_tools_from_a2a_card(url)` — Sync convenience function

### Inbound Server

- `workflow_as_a2a(workflow, protocol_version="1.0")` — Expose a Workflow as an A2A agent
- `create_agent_card(workflow, protocol_version="1.0")` — Create AgentCard from a Workflow

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

This project is an independent implementation of the A2A integration approach proposed in LlamaIndex issue [#24104](https://github.com/run-llama/llama_index/issues/24104). It is maintained as a standalone package per LlamaIndex's integration policy.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Google A2A Team for designing and publishing the open A2A protocol specification
- LlamaIndex community for the MCP integration pattern this builds upon

## Built with

Developed with [Claude Code](https://claude.com/claude-code).
