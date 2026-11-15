"""
Basic A2A Client for connecting to A2A agents.

Mirrors the BasicMCPClient pattern but adapted for the A2A protocol.
Handles AgentCard discovery and task-based skill invocation.

Supports both A2A protocol v0.3 and v1.0 via version detection and A2A-Version header.
"""

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx

from llama_index.tools.a2a.models import (
    AgentCard,
    Message,
    Part,
    SendMessageResult,
    Task,
    TaskOutput,
    TaskResult,
)

logger = logging.getLogger(__name__)


class DefaultInMemoryTokenStorage:
    """Simple in-memory token storage for A2A authentication."""

    def __init__(self):
        self._tokens: dict | None = None
        self._client_info: dict | None = None

    async def get_tokens(self) -> dict | None:
        return self._tokens

    async def set_tokens(self, tokens: dict) -> None:
        self._tokens = tokens

    async def get_client_info(self) -> dict | None:
        return self._client_info

    async def set_client_info(self, client_info: dict) -> None:
        self._client_info = client_info


class BasicA2AClient:
    """
    Basic A2A client for connecting to an A2A agent.

    This client discovers the agent's capabilities via its AgentCard
    and provides methods to invoke skills through the A2A protocol.

    Supports both A2A v0.3 (message/send, tasks/get) and v1.0 (SendMessage, GetTask)
    via automatic version detection and the A2A-Version header.

    Args:
        agent_card_url: The URL of the agent card (e.g., https://agent.example.com/.well-known/agent-card.json).
        timeout: The timeout for HTTP operations in seconds. Default is 30.
        headers: Optional headers to pass with requests.
        auth: Optional authentication handler.
        tool_call_logs_callback: Async function to capture logs from skill calls.
        protocol_version: Force a specific protocol version ("0.3" or "1.0"). Default: auto-detect.

    """

    def __init__(
        self,
        agent_card_url: str,
        timeout: int = 30,
        headers: dict[str, str] | None = None,
        auth: Any | None = None,
        tool_call_logs_callback: Callable[[list[str]], Awaitable[Any]] | None = None,
        protocol_version: str | None = None,
    ):
        self.agent_card_url = agent_card_url
        self.timeout = timeout
        self.base_headers = headers or {}
        self.auth = auth
        self.tool_call_logs_callback = tool_call_logs_callback
        self._forced_protocol_version = protocol_version

        # Cached agent card
        self._agent_card: AgentCard | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._detected_version: str | None = None

    def _get_request_headers(self) -> dict[str, str]:
        """Get headers for A2A requests, including A2A-Version header for v1.0."""
        headers = dict(self.base_headers)
        headers["Content-Type"] = "application/json"
        if self._detected_version == "1.0":
            headers["A2A-Version"] = "1.0"
        return headers

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create an HTTP client with configured timeouts."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
            )
        return self._http_client

    async def _discover_agent_card(self) -> AgentCard:
        """Discover and parse the agent card from the given URL."""
        if self._agent_card is not None:
            return self._agent_card

        client = await self._get_http_client()
        response = await client.get(self.agent_card_url, headers=self.base_headers)
        response.raise_for_status()

        self._agent_card = AgentCard.model_validate(response.json())

        # Detect protocol version
        if self._forced_protocol_version:
            self._detected_version = self._forced_protocol_version
        else:
            # v1.0 has additional_interfaces (a2a-sdk field for supportedInterfaces)
            # v0.3 has url + preferred_transport
            if getattr(self._agent_card, 'additional_interfaces', None):
                self._detected_version = "1.0"
            else:
                self._detected_version = "0.3"

        logger.info(f"Detected A2A protocol version: {self._detected_version}")
        return self._agent_card

    @property
    def protocol_version(self) -> str:
        """Get the detected/forced protocol version."""
        if self._detected_version is None:
            # Trigger discovery
            import asyncio
            asyncio.create_task(self._discover_agent_card())
        return self._detected_version or "0.3"

    @property
    def base_url(self) -> str:
        """Get the base URL for API calls."""
        if self._agent_card:
            # v1.0: use first additional_interface
            if getattr(self._agent_card, 'additional_interfaces', None):
                return self._agent_card.additional_interfaces[0].url
            # v0.3: use url field
            return self._agent_card.url or ""
        return ""

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    @asynccontextmanager
    async def _session(self) -> AsyncIterator["BasicA2AClient"]:
        """Context manager for session lifecycle."""
        try:
            yield self
        finally:
            await self.close()

    # ---- Discovery Methods ----

    async def get_agent_card(self) -> AgentCard:
        """
        Get the agent's capabilities from its AgentCard.

        Returns:
            The parsed AgentCard containing agent metadata and skills.
        """
        return await self._discover_agent_card()

    async def list_skills(self) -> list[Any]:
        """
        List all available skills on the A2A agent.

        Returns:
            A list of skill descriptors with name, description, and input_schema.
        """
        card = await self._discover_agent_card()
        return card.skills

    # ---- Skill Invocation Methods ----

    async def _send_v1_request(self, method: str, params: dict) -> dict:
        """Send a v1.0 JSON-RPC request with A2A-Version header."""
        client = await self._get_http_client()
        request_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        }
        response = await client.post(
            self.base_url, json=request_payload, headers=self._get_request_headers()
        )
        response.raise_for_status()
        return response.json()

    async def _send_v03_request(self, endpoint: str, method: str, params: dict) -> dict:
        """Send a v0.3 JSON-RPC request."""
        client = await self._get_http_client()
        request_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        }
        response = await client.post(
            endpoint, json=request_payload, headers=self._get_request_headers()
        )
        response.raise_for_status()
        return response.json()

    async def call_skill(
        self,
        skill_name: str,
        arguments: dict | None = None,
    ) -> TaskResult:
        """
        Call a specific skill on the A2A agent.

        This method creates a task, sends it to the agent, and polls
        for the result using the A2A task protocol.

        Args:
            skill_name: The name of the skill to invoke.
            arguments: Optional arguments to pass to the skill.

        Returns:
            The task result containing the output.
        """
        card = await self._discover_agent_card()

        # Build the input parts
        input_parts = [Part(kind="text", text=str(arguments) if arguments else "")]

        if self.protocol_version == "1.0":
            # v1.0: Use SendMessage with message parameter
            params = {
                "message": Message(
                    role="user",
                    parts=input_parts,
                ).model_dump(exclude_none=True)
            }
            result = await self._send_v1_request("SendMessage", params)
            raw_result = result.get("result", {})
            # v1.0 SendMessage returns either task or message
            send_result = SendMessageResult.model_validate(raw_result)
            if send_result.task:
                return TaskResult(
                    task_id=send_result.task.id,
                    output=send_result.task.artifacts[0] if send_result.task.artifacts else None,
                    metadata=send_result.task.metadata,
                )
            elif send_result.message:
                return TaskResult(
                    task_id="",
                    output=TaskOutput(parts=send_result.message.parts),
                )
            return TaskResult(task_id="")
        else:
            # v0.3: Use tasks/send
            skill_endpoint = card.url
            for skill in card.skills:
                if skill.name == skill_name:
                    skill_endpoint = skill.endpoint or card.url
                    break

            params = {
                "task": {
                    "name": skill_name,
                    "input": {"parts": [p.model_dump() for p in input_parts]},
                }
            }
            result = await self._send_v03_request(skill_endpoint, "tasks/send", params)
            raw_result = result.get("result", {})
            if "id" in raw_result and "task_id" not in raw_result:
                raw_result["task_id"] = raw_result.pop("id")
            return TaskResult.model_validate(raw_result)

    async def send_message(
        self,
        message: str,
        context_id: str | None = None,
    ) -> TaskResult:
        """
        Send a natural language message to the agent.

        Args:
            message: The message to send.
            context_id: Optional context ID for stateful conversations.

        Returns:
            The agent's response as a TaskResult.
        """
        card = await self._discover_agent_card()

        input_parts = [Part(kind="text", text=message)]

        if self.protocol_version == "1.0":
            # v1.0: Use SendMessage
            params = {
                "message": Message(
                    role="user",
                    parts=input_parts,
                ).model_dump(exclude_none=True)
            }
            if context_id:
                params["contextId"] = context_id
            result = await self._send_v1_request("SendMessage", params)
            raw_result = result.get("result", {})
            send_result = SendMessageResult.model_validate(raw_result)
            if send_result.task:
                return TaskResult(
                    task_id=send_result.task.id,
                    output=send_result.task.artifacts[0] if send_result.task.artifacts else None,
                    metadata=send_result.task.metadata,
                )
            elif send_result.message:
                return TaskResult(
                    task_id=send_result.task.id if send_result.task else "",
                    output=TaskOutput(parts=send_result.message.parts),
                )
            return TaskResult(task_id="")
        else:
            # v0.3: Use messages/send
            params = {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": message}],
                }
            }
            if context_id:
                params["contextId"] = context_id

            result = await self._send_v03_request(card.url, "messages/send", params)
            raw_result = result.get("result", {})
            if "id" in raw_result and "task_id" not in raw_result:
                raw_result["task_id"] = raw_result.pop("id")
            return TaskResult.model_validate(raw_result)

    async def get_task(self, task_id: str) -> TaskResult:
        """
        Get task status/result (v1.0: GetTask, v0.3: tasks/get).

        Args:
            task_id: The task ID to retrieve.

        Returns:
            The task result.
        """
        card = await self._discover_agent_card()

        if self.protocol_version == "1.0":
            result = await self._send_v1_request("GetTask", {"id": task_id})
            raw_result = result.get("result", {})
            task = Task.model_validate(raw_result)
            return TaskResult(
                task_id=task.id,
                output=task.artifacts[0] if task.artifacts else None,
                metadata=task.metadata,
            )
        else:
            skill_endpoint = card.url
            result = await self._send_v03_request(skill_endpoint, "tasks/get", {"id": task_id})
            raw_result = result.get("result", {})
            if "id" in raw_result and "task_id" not in raw_result:
                raw_result["task_id"] = raw_result.pop("id")
            return TaskResult.model_validate(raw_result)
