# Contributing to llama-index-tools-a2a

Thanks for your interest in contributing!

## Reporting Issues

- Use [GitHub Issues](https://github.com/shreyanshgautam3/llama-index-tools-a2a/issues) for bugs and feature requests
- Include Python version, package version, and a minimal reproduction script

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
pip install -e ".[dev]"
pytest tests/ -v
ruff check .
ruff format .
mypy llama_index/tools/a2a/
```

## Pull Requests

1. Fork the repo and create a feature branch
2. Make your changes with tests
3. Ensure all tests pass: `pytest tests/ -v`
4. Run the linter: `ruff check . && ruff format .`
5. Submit a PR with a clear description

## Architecture

This package mirrors the `llama-index-tools-mcp` pattern:

| Component | Purpose |
|-----------|---------|
| `BasicA2AClient` | Outbound: connect to remote A2A agents |
| `A2AToolSpec` | Convert A2A skills → LlamaIndex FunctionTools |
| `workflow_as_a2a()` | Inbound: expose Workflows as A2A agents |
| `create_agent_card()` | Generate AgentCard from Workflow metadata |

## License

MIT License — see LICENSE for details.
