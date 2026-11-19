"""
Pydantic models for A2A protocol types.

Defines the core data structures used in the A2A (Agent-to-Agent) protocol.
Supports both v0.3 and v1.0 protocol versions.

- AgentCard: Discovery document describing agent capabilities
- Skill: Individual capability definition
- Task: Unit of work exchanged between agents
- Message: Communication unit with parts (text, file, etc.)
- Part: Content fragment within a message (v1.0: discriminated union, no 'kind' field)
- TaskResult: Response from skill/task execution
"""

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class Part(BaseModel):
    """Unified v1.0 Part model (discriminated union by field presence).

    Per A2A v1.0 spec, a Part has one of:
    - text: str (TextPart)
    - url: str + optional mediaType, filename (FilePart by URL)
    - raw: bytes + optional mediaType, filename (FilePart by base64)
    - data: dict (DataPart)
    - mediaType: str (shared metadata)
    - filename: str (shared metadata)
    """

    # Text content
    text: str | None = None
    # File by URL
    url: str | None = None
    # File by base64 (raw bytes)
    raw: bytes | None = None
    # Structured data
    data: dict[str, Any] | None = None
    # Optional metadata for file/data parts
    media_type: str | None = Field(default=None, alias="mediaType")
    filename: str | None = None

    model_config = {"populate_by_name": True}

    @field_validator("raw", mode="before")
    @classmethod
    def _decode_base64(cls, v):
        """Decode base64 string to bytes if needed."""
        if isinstance(v, str):
            import base64

            return base64.b64decode(v)
        return v

    def is_text(self) -> bool:
        return self.text is not None

    def is_file(self) -> bool:
        return self.url is not None or self.raw is not None

    def is_data(self) -> bool:
        return self.data is not None


# v0.3 compatibility: PartWithKind for legacy wire format
class PartWithKind(BaseModel):
    """Legacy v0.3 Part with 'kind' field for backward compatibility."""

    kind: Literal["text", "file", "image"]
    text: str | None = None
    file_uri: str | None = None
    mime_type: str | None = None


class Message(BaseModel):
    """A message exchanged between agents.

    v1.0:
    - role uses ROLE_USER / ROLE_AGENT (not "user"/"assistant")
    - messageId is auto-generated UUID if not provided
    - metadata field for skill name and other extensions
    Uses camelCase aliases for wire format.
    """

    role: Literal["ROLE_USER", "ROLE_AGENT", "system"] = "ROLE_USER"
    parts: list[Part]
    message_id: str | None = Field(default=None, alias="messageId")
    context_id: str | None = Field(default=None, alias="contextId")
    metadata: dict[str, Any] | None = None

    model_config = {"populate_by_name": True}

    def __init__(self, **data):
        # Auto-generate messageId for v1.0 if not provided
        if "message_id" not in data and "messageId" not in data:
            data["message_id"] = str(uuid.uuid4())
        super().__init__(**data)


class TaskInput(BaseModel):
    """Input to a task."""

    parts: list[Part]


class TaskOutput(BaseModel):
    """Output from a task."""

    parts: list[Part]
    status: str | None = None


class Task(BaseModel):
    """A task represents a unit of work to be performed by an agent (v1.0).

    Uses camelCase aliases for wire format.
    """

    id: str
    context_id: str = Field(alias="contextId")
    status: "TaskStatus"
    history: list[Message] = Field(default_factory=list)
    artifacts: list["Artifact"] | None = None
    metadata: dict[str, Any] | None = Field(default_factory=dict)
    extensions: dict[str, Any] | None = None

    model_config = {"populate_by_name": True}


class TaskStatus(BaseModel):
    """Task status in v1.0.

    Uses camelCase aliases for wire format.
    v1.0 states: TASK_STATE_SUBMITTED, TASK_STATE_WORKING, TASK_STATE_COMPLETED,
    TASK_STATE_FAILED, TASK_STATE_CANCELED, TASK_STATE_REJECTED,
    TASK_STATE_INPUT_REQUIRED, TASK_STATE_AUTH_REQUIRED, TASK_STATE_UNSPECIFIED
    v0.3 states (kept for wire compatibility when parsing raw dicts): submitted,
    working, input-required, completed, failed, canceled, rejected, auth-required
    """

    state: Literal[
        # v1.0 forms
        "TASK_STATE_SUBMITTED",
        "TASK_STATE_WORKING",
        "TASK_STATE_COMPLETED",
        "TASK_STATE_FAILED",
        "TASK_STATE_CANCELED",
        "TASK_STATE_REJECTED",
        "TASK_STATE_INPUT_REQUIRED",
        "TASK_STATE_AUTH_REQUIRED",
        "TASK_STATE_UNSPECIFIED",
        # v0.3 forms (for backward compatibility when parsing raw dicts)
        "submitted",
        "working",
        "input-required",
        "completed",
        "failed",
        "canceled",
        "rejected",
        "auth-required",
    ]
    message: Message | None = None
    timestamp: str | None = None

    model_config = {"populate_by_name": True}


class Artifact(BaseModel):
    """Task artifact in v1.0.

    Uses camelCase aliases for wire format.
    """

    artifact_id: str = Field(alias="artifactId")
    parts: list[Part]
    index: int | None = None
    append: bool | None = None
    last_chunk: bool | None = Field(default=None, alias="lastChunk")
    metadata: dict[str, Any] | None = None

    model_config = {"populate_by_name": True}


class Capability(BaseModel):
    """Describes a specific capability an agent offers."""

    name: str
    description: str | None = None
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    tags: list[str] | None = Field(default_factory=list)


class Skill(Capability):
    """A named capability that can be invoked by other agents."""

    id: str = Field(default_factory=lambda: "")
    name: str
    input_modes: list[str] | None = None
    output_modes: list[str] | None = None
    examples: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    # input_schema and output_schema are not standard A2A fields; removed per A2A spec


class AgentCapabilities(BaseModel):
    """Agent capabilities as defined by A2A protocol."""

    streaming: bool = False
    push_notifications: bool = False


class AgentProvider(BaseModel):
    """Agent provider information."""

    organization: str
    url: str


class AgentInterface(BaseModel):
    """Agent interface declaration for v1.0 supportedInterfaces.

    v1.0 uses protocolBinding + protocolVersion (not transport).
    """

    url: str
    protocol_binding: str = Field(alias="protocolBinding")
    protocol_version: str = Field(alias="protocolVersion")

    model_config = {"populate_by_name": True}


class AgentInterfaceV03(BaseModel):
    """Agent interface declaration for v0.3 additionalInterfaces.

    v0.3 uses url + transport (not protocolBinding/protocolVersion).
    """

    url: str
    transport: str | None = None

    model_config = {"populate_by_name": True}


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
    preferred_transport: str | None = Field(default=None, alias="preferredTransport")
    protocol_version: str | None = None
    # v1.0 fields - a2a-sdk uses additionalInterfaces, wire format also uses additionalInterfaces
    additional_interfaces: list[AgentInterfaceV03] | None = Field(
        default=None, alias="additionalInterfaces"
    )
    # v1.0 supportedInterfaces - uses protocolBinding + protocolVersion
    supported_interfaces: list[AgentInterface] | None = Field(
        default=None, alias="supportedInterfaces"
    )

    @property
    def primary_url(self) -> str:
        """Get the primary URL for the agent.

        Mode-dependent logic:
        - v0.3: use card.url; fall back to first additionalInterfaces entry
          only if url is empty/None.
        - v1.0: use first supportedInterfaces entry whose protocolBinding is
          JSONRPC; return "" if none found.

        Note: AgentCard model itself does not know the protocol version.
        The caller (client) should use this in combination with protocol_version
        to get the correct URL. For standalone use, this property returns the
        v0.3-style url or the first additionalInterfaces URL.
        """
        # v0.3 style: url field takes priority
        if self.url:
            return self.url
        # Fall back to first additionalInterfaces entry
        if self.additional_interfaces:
            return self.additional_interfaces[0].url
        # v1.0: first supportedInterfaces with JSONRPC binding
        if self.supported_interfaces:
            for iface in self.supported_interfaces:
                if iface.protocol_binding.upper() == "JSONRPC":
                    return iface.url
        return ""

    def primary_url_for_version(self, version: str) -> str:
        """Get the primary URL appropriate for the given protocol version.

        Mode-dependent URL selection:
        - version "0.3": use card.url; fall back to first additionalInterfaces
          entry only if url is empty/None.
        - version "1.0": use first supportedInterfaces entry whose
          protocolBinding is JSONRPC; return "" if none found.
        """
        if version == "0.3":
            if self.url:
                return self.url
            if self.additional_interfaces:
                return self.additional_interfaces[0].url
            return ""
        # v1.0
        if self.supported_interfaces:
            for iface in self.supported_interfaces:
                if iface.protocol_binding.upper() == "JSONRPC":
                    return iface.url
        return ""


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
