"""
A2A (Agent-to-Agent) Protocol integration for LlamaIndex.

This module provides outbound and inbound A2A protocol support,
mirroring the MCP integration pattern.

Outbound: Connect to an A2A agent card URL and discover tools
Inbound: Expose a Workflow as an A2A-capable ASGI app
"""

# Use relative imports to avoid circular import issues
# Import models first (no dependencies on other modules in this package)
from .models import (
    AgentCard,
    AgentCapabilities,
    AgentInterface,
    AgentProvider,
    Skill,
)
# Import base (depends only on models)
from .base import A2AToolSpec
# Import client (depends only on models)
from .client import BasicA2AClient
# Import server (depends on models, a2a-sdk)
from .server import (
    WorkflowAgentExecutor,
    create_agent_card,
    workflow_as_a2a,
)
# Import utils (depends on base and client)
from .utils import (
    aget_tools_from_a2a_card,
    get_tools_from_a2a_card,
)

__all__ = [
    "A2AToolSpec",
    "BasicA2AClient",
    "workflow_as_a2a",
    "create_agent_card",
    "WorkflowAgentExecutor",
    "get_tools_from_a2a_card",
    "aget_tools_from_a2a_card",
]
