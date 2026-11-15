"""
Pydantic models for A2A protocol types.

Defines the core data structures used in the A2A (Agent-to-Agent) protocol.
Supports both v0.3 and v1.0 protocol versions.

- AgentCard: Discovery document describing agent capabilities
- Skill: Individual capability definition
- Task: Unit of work exchanged between agents
- Message: Communication unit with parts (text, file, etc.)
- Part: Content fragment within a message
- TaskResult: Response from skill/task execution
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class Part(BaseModel):
    """A single content part within a message or task input/output."""

    kind: Literal["text", "file", "image"]
    text: str | None = None
    file_uri: str | None = None
    mime_type: str | None = None


class Message(BaseModel):
    """A message exchanged between agents."""

    role: Literal["user", "assistant", "system"]
    parts: list[Part]
    message_id: str | None = None
    context_id: str | None = None


class TaskInput(BaseModel):
    """Input to a task."""

    parts: list[Part]


class TaskOutput(BaseModel):
    """Output from a task."""

    parts: list[Part]
    status: str | None = None


class Task(BaseModel):
    """A task represents a unit of work to be performed by an agent (v1.0)."""

    id: str
    context_id: str
    status: "TaskStatus"
    history: list[Message] = Field(default_factory=list)
    artifacts: list["Artifact"] | None = None
    metadata: dict[str, Any] | None = Field(default_factory=dict)
    extensions: dict[str, Any] | None = None


class TaskStatus(BaseModel):
    """Task status in v1.0."""

    state: Literal["submitted", "working", "input-required", "completed", "failed", "canceled"]
    message: Message | None = None
    timestamp: str | None = None


class Artifact(BaseModel):
    """Task artifact in v1.0."""

    artifact_id: str
    parts: list[Part]
    index: int | None = None
    append: bool | None = None
    last_chunk: bool | None = None
    metadata: dict[str, Any] | None = None


class Capability(BaseModel):
    """Describes a specific capability an agent offers."""

    name: str
    description: str | None = None
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    tags: list[str] | None = Field(default_factory=list)


class Skill(Capability):
    """A named capability that can be invoked by other agents."""

    name: str
    endpoint: str | None = None
    input_schema: dict[str, Any] | None = Field(default_factory=dict)
    output_schema: dict[str, Any] | None = Field(default_factory=dict)


class AgentCapabilities(BaseModel):
    """Agent capabilities as defined by A2A protocol."""

    streaming: bool = False
    push_notifications: bool = False


class AgentProvider(BaseModel):
    """Agent provider information."""

    organization: str
    url: str


class AgentInterface(BaseModel):
    """Agent interface declaration for v1.0 additionalInterfaces (a2a-sdk)."""

    url: str
    transport: Literal["JSONRPC", "GRPC", "HTTP+JSON"]


class AgentCard(BaseModel):
    """
    The AgentCard is the discovery document for an A2A agent.
    It describes the agent's identity, capabilities, and endpoints.

    Typically served at /.well-known/agent-card.json

    v1.0: Uses supportedInterfaces[] instead of url + preferred_transport
    """

    model_config = {"populate_by_name": True}

    name: str
    description: str | None = None
    version: str = "1.0.0"
    skills: list[Skill] = Field(default_factory=list)
    capabilities: AgentCapabilities | None = None
    provider: AgentProvider | None = None
    authentication: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = Field(default_factory=dict)
    default_input_modes: list[str] | None = None
    default_output_modes: list[str] | None = None
    # v0.3 fields (kept for backward compatibility)
    url: str | None = None
    preferred_transport: str | None = None
    protocol_version: str | None = None
    # v1.0 fields - a2a-sdk uses additionalInterfaces, wire format also uses additionalInterfaces
    additional_interfaces: list[AgentInterface] = Field(
        default_factory=list, alias="additionalInterfaces"
    )
    # Alias for backward compatibility - some docs refer to this as supportedInterfaces
    supported_interfaces: list[AgentInterface] = Field(
        default_factory=list, alias="supportedInterfaces"
    )

    @property
    def primary_url(self) -> str:
        """Get the primary URL for the agent (v1.0: first additionalInterface/
        supportedInterface, v0.3: url field)."""
        # Check both field names for v1.0 compatibility
        interfaces = self.additional_interfaces or self.supported_interfaces
        if interfaces:
            return interfaces[0].url
        return self.url or ""


class TaskResult(BaseModel):
    """Result of a task/skill execution."""

    task_id: str
    output: TaskOutput | None = None
    error: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = Field(default_factory=dict)


class SendMessageResult(BaseModel):
    """Result of SendMessage call (v1.0) - wraps task or message."""

    task: Task | None = None
    message: Message | None = None


class JSONRPCResponse(BaseModel):
    """Standard JSON-RPC 2.0 response."""

    jsonrpc: str = "2.0"
    result: Any | None = None
    error: dict[str, Any] | None = None
    id: int | None = None


class JSONRPCError(BaseModel):
    """JSON-RPC 2.0 error object."""

    code: int
    message: str
    data: Any | None = None
