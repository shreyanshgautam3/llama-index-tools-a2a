## [0.1.4] - 2026-11-16

### Fixed - Critical A2A Protocol Compliance
- **v1.0 Client Method Names**: Client now uses correct `SendMessage` and `GetTask` (PascalCase) instead of v0.3 `message/send`/`tasks/get`
- **v1.0 Part Structure**: Removed `kind` field from Parts — now uses discriminated union (`TextPart`, `FilePart`, `DataPart`) per v1.0 spec
- **contextId Placement**: v1.0 `contextId` now correctly placed on `Message` object, not as top-level parameter
- **Skill Name in Message**: v1.0 `SendMessage` embeds skill name in message text as JSON: `{"skill": "name", "message": "..."}`
- **Task State Handling**: `get_task()` and `call_skill()` now properly handle v0.3/v1.0 states: `input-required`, `failed`, `canceled`
- **v0.3 Method Names Fixed**: Corrected to `message/send` (not `tasks/send` or `messages/send` which don't exist in a2a-sdk 0.3.26)
- **Version Detection**: Auto-detection now ONLY checks `supportedInterfaces` (wire format), not `additionalInterfaces` (a2a-sdk internal)
- **Dead Code Removed**: `DefaultInMemoryTokenStorage`, `auth` parameter, `tool_call_logs_callback` parameter, unused imports
- **Streaming Claim Fixed**: Server now correctly advertises `streaming: false` in AgentCapabilities
- **Return Type Consistency**: `call_skill` and `send_message` both return `TaskOutput` for v1.0 (not raw `Artifact`)
- **Message Response Handling**: Both methods now handle `message/send` and `SendMessage` returning `Message` instead of `Task`
- **Removed v1.0 Server Support**: Inbound server only produces v0.3 AgentCards (a2a-sdk 0.3.26 limitation). `protocol_version="1.0"` now raises `NotImplementedError`.

### Added
- **9 Integration Tests** (`tests/test_integration.py`): Full client-server round-trip verification
  - v0.3: `message/send` → `tasks/get` flow
  - Version auto-detection from AgentCard (v0.3 only, since server only produces v0.3)
  - `BasicA2AClient.call_skill()` against real server
- **12 v1.0 Client Protocol Tests** (`tests/test_client_v10.py`): Unit tests for v1.0 outbound client compliance

### Changed
- **Skill Model**: `input_schema` and `output_schema` default to `None` (not empty dict) to match A2A spec
- **v0.3 call_skill**: Arguments serialized as JSON (not Python str repr); accepts `message` key from A2AToolSpec
- **Inbound Server**: Simplified to v0.3 only; removed `protocol_version` parameter from `workflow_as_a2a` and `create_agent_card`
- **README**: Updated to honestly reflect v0.3 server limitation; v1.0 client paths marked as experimental

## [0.1.3] - 2026-11-15

### Added
- **A2A Protocol v1.0 Support**: Dual version support (v0.3 + v1.0) via automatic detection
  - `BasicA2AClient` now auto-detects protocol version from AgentCard
  - Sends `A2A-Version: 1.0` header for v1.0 servers
  - Uses v1.0 JSON-RPC method names (`SendMessage`, `GetTask`) when appropriate
  - Falls back to v0.3 methods (`message/send`, `tasks/get`, `tasks/send`) for compatibility
  - New `protocol_version` parameter to force a specific version
  - New `get_task()` method for retrieving task status (v1.0: `GetTask`, v0.3: `tasks/get`)
- **Multi-turn conversation support**: `A2AToolSpec.context_id` property for propagating context across tool calls
- **v1.0 AgentCard models**: `AgentInterface`, `Task`, `TaskStatus`, `Artifact`, `SendMessageResult`
- **Inbound server v1.0 support**: `workflow_as_a2a()` now accepts `protocol_version` parameter
  - Generates `supportedInterfaces[]` for v1.0 cards
  - Maintains v0.3 format (`url` + `preferred_transport`) for backward compatibility

### Changed
- **Skill-to-tool conversion** (`A2AToolSpec`): Updated to reflect A2A v1.0 reality
  - Skills no longer require JSON input schemas (per v1.0 spec)
  - Tools now accept natural language `message` parameter
  - Structured input should be passed as JSON in message text
  - Added `context_id` parameter for multi-turn conversations
- **README**: Added **Limitations** section with 7 known constraints
- **README**: Updated version history (v0.1.2 first, then v0.1.1)
- **README**: Corrected "independent implementation" claim (removed unverified Google collaboration)
- **README**: Added **Protocol Version** section clarifying current v0.3 base with v1.0 support
- **Examples**: `examples/a2a.ipynb` updated to use `FunctionAgent` (ReActAgent deprecated in LlamaIndex 0.13+)
- **Test config**: `asyncio_mode = "strict"` in pyproject.toml for CI consistency

### Fixed
- Client now properly handles v1.0 `SendMessage` response structure (task or message)
- Server generates valid v1.0 AgentCard with `supportedInterfaces`
- Tool spec no longer assumes skills have `input_schema` (v1.0 doesn't provide it)

## [0.1.2] - 2026-11-14

### Added
- Test CI workflow (`.github/workflows/tests.yml`) running pytest on Python 3.10/3.11/3.12
- "Why?" section in README explaining the A2A interoperability problem
- "Built with Claude Code" footer in README

### Fixed
- Clean commit history: squashed 6 publish/debug commits into single release commit

## [0.1.1] - 2026-11-12

### Fixed
- PyPI publishing workflow: version bump to match release tag
- Test dependencies: pytest-asyncio, fastapi, uvicorn added to dev extras

### Changed
- Removed `asyncio_mode = "auto"` from pytest config (not needed with pytest-asyncio 1.x)
- CI workflow uses API token auth for reliable publishing

## [Unreleased]

### Added
- **Outbound client**: `BasicA2AClient` for connecting to remote A2A agents
  - `get_agent_card()` — discover agent metadata and capabilities
  - `list_skills()` — list available skills on a remote agent
  - `call_skill(name, arguments)` — invoke a skill via tasks/send
  - `send_message(message)` — send a natural language message via messages/send
- **Tool spec**: `A2AToolSpec` converts A2A skills to LlamaIndex `FunctionTool`s
  - Supports `allowed_tools` filtering
  - Supports `global_partial_params` and `partial_params_by_tool`
- **Inbound server**: `workflow_as_a2a(workflow)` exposes any LlamaIndex Workflow as an A2A agent
  - Implements JSON-RPC methods: `message/send`, `tasks/get`
  - Serves `/.well-known/agent-card.json` for discovery
  - Built on `A2AFastAPIApplication` from `a2a-sdk`
- **Workflow executor**: `WorkflowAgentExecutor(AgentExecutor)` wraps a LlamaIndex `Workflow`
- **Pydantic models**: `AgentCard`, `Skill`, `Message`, `Task`, `TaskResult`, etc.
- **Utility functions**: `aget_tools_from_a2a_card()`, `get_tools_from_a2a_card()`, `create_agent_card()`
- **Cookbook**: `examples/a2a.ipynb` demonstrating end-to-end A2A flow
- **Server example**: `examples/a2a_server.py` standalone A2A server script
- **Tests**: 13 integration tests covering server endpoints, client methods, and tool spec conversion

### Dependencies
- `a2a-sdk>=0.1.0,<1` (tested against 0.3.26)
- `llama-index-core>=0.13.0,<0.15`
- `pydantic>=2,<3`
- `httpx>=0.27,<1`