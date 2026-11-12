# llama-index-tools-a2a

[![PyPI version](https://badge.fury.io/py/llama-index-tools-a2a.svg)](https://badge.fury.io/py/llama-index-tools-a2a)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Support](https://img.shields.io/pypi/pyversions/llama-index-tools-a2a.svg)](https://pypi.org/project/llama-index-tools-a2a/)

**A2A (Agent-to-Agent) Protocol integration for LlamaIndex**

Enable your LlamaIndex agents to discover and invoke other A2A-compatible agents as tools. Implements Google's A2A protocol specification for universal agent interoperability.

## Features

- **Outbound Client**: Connect to any A2A agent via its AgentCard URL
- **Skill Discovery**: Automatically discover available tools/skills from remote agents
- **Schema Conversion**: Maps A2A skill schemas to LlamaIndex FunctionTools
- **Inbound Server**: Expose any LlamaIndex Workflow as an A2A-capable agent
- **Async Support**: Full async/await support for non-blocking operation
- **Partial Params**: Support for pre-filling common parameters across tools

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

# Use tools in your LlamaIndex workflow
from llama_index.core.agent import ReActAgent

agent = ReActAgent.from_tools(tools)
result = await agent.chat("What's the weather in Tokyo?")
```

### Filtering Specific Tools

```python
# Only use specific skills
tools = await aget_tools_from_a2a_card(
    "https://agent.example.com/.well-known/agent-card.json",
    allowed_tools=["search_web", "calculate"]
)
```

### Pre-filling Common Parameters

```python
# Apply default parameters across all tools
tools = await aget_tools_from_a2a_card(
    "https://agent.example.com/.well-known/agent-card.json",
    global_partial_params={"api_key": "my-secret-key"}
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

| Concept | MCP Equivalent | A2A Implementation |
|---------|---------------|-------------------|
| Tool discovery | `list_tools()` | AgentCard.skills[] |
| Tool invocation | `call_tool(name, args)` | `message/send` + `tasks/get` |
| Inbound server | `MCPServer` | ASGI app (`workflow_as_a2a()`) |
| Workflow wrapping | `workflow_as_mcp()` | `workflow_as_a2a()` |

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

- **Milestone 1**: Package scaffolded ✅
- **Milestone 2**: Outbound client with AgentCard discovery ✅
- **Milestone 3**: Inbound ASGI server (`workflow_as_a2a()`) ✅
- **Milestone 4**: Integration tests (13 tests passing) ✅
- **Milestone 5**: Documentation & cookbook ✅

## API Reference

### Outbound Client

- `BasicA2AClient(agent_card_url)` — Connect to an A2A agent
  - `get_agent_card()` — Discover agent metadata
  - `list_skills()` — List available skills
  - `call_skill(name, arguments)` — Invoke a skill
  - `send_message(message)` — Send natural language message

### Tool Conversion

- `A2AToolSpec(client)` — Convert A2A skills to LlamaIndex FunctionTools
- `aget_tools_from_a2a_card(url)` — Async convenience function
- `get_tools_from_a2a_card(url)` — Sync convenience function

### Inbound Server

- `workflow_as_a2a(workflow)` — Expose a Workflow as an A2A agent
- `create_agent_card(workflow)` — Create AgentCard from a Workflow

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

This project was initially developed as a contribution to the LlamaIndex ecosystem under issue [#24104](https://github.com/run-llama/llama_index/issues/24104) in collaboration with the Google A2A team, and is now maintained as a standalone package per LlamaIndex's integration policy.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Google A2A Team for protocol design and mentorship
- LlamaIndex community for the MCP integration pattern this builds upon
