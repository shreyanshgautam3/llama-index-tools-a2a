"""
Basic A2A Client for connecting to A2A agents.

Mirrors the BasicMCPClient pattern but adapted for the A2A protocol.
Handles AgentCard discovery and task-based skill invocation.

Supports both A2A protocol v0.3 and v1.0 via version detection and A2A-Version header.

v1.0 Protocol Notes:
- Method names: SendMessage, GetTask (PascalCase)
- Part structure: discriminated union (TextPart, FilePart, DataPart) - NO 'kind' field
- contextId: placed ON the Message object, not as top-level param
- skill_name: embedded in message text as JSON {"skill": "name", "message": "..."}
- A2A-Version: 1.0 header required for v1.0 servers
"""

import json
import logging
from collections.abc import AsyncIterator
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
    TextPart,
)

logger = logging.getLogger(__name__)


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
        protocol_version: Force a specific protocol version ("0.3" or "1.0"). Default: auto-detect.

    """

    def __init__(
        self,
        agent_card_url: str,
        timeout: int = 30,
        headers: dict[str, str] | None = None,
        protocol_version: str | None = None,
    ):
        self.agent_card_url = agent_card_url
        self.timeout = timeout
        self.base_headers = headers or {}
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
            # v1.0 wire format uses supportedInterfaces (not additionalInterfaces)
            # additionalInterfaces is a2a-sdk 0.3.26 internal field name
            # Only detect v1.0 if supported_interfaces is present
            has_supported = getattr(self._agent_card, 'supported_interfaces', None)
            if has_supported:
                self._detected_version = "1.0"
            else:
                self._detected_version = "0.3"

        logger.info(f"Detected A2A protocol version: {self._detected_version}")
        return self._agent_card

    @property
    def protocol_version(self) -> str:
        """Get the detected/forced protocol version (synchronous access)."""
        # Note: this is a sync property. Call get_agent_card() first for reliable detection.
        return self._detected_version or "0.3"

    async def detect_protocol_version(self) -> str:
        """Async method to ensure discovery and return protocol version."""
        if self._detected_version is None:
            await self._discover_agent_card()
        return self._detected_version or "0.3"

    @property
    def base_url(self) -> str:
        """Get the base URL for API calls."""
        if self._agent_card:
            return self._agent_card.primary_url
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

    def _build_v1_message(
        self, skill_name: str, message_text: str, context_id: str | None = None
    ) -> Message:
        """Build a v1.0 Message with skill_name embedded in text and contextId on the Message."""
        # v1.0: embed skill name in message text as JSON
        message_payload = json.dumps({"skill": skill_name, "message": message_text})
        parts = [TextPart(text=message_payload)]
        msg = Message(role="user", parts=parts)
        if context_id:
            msg.context_id = context_id
        return msg

    def _build_v1_parts(self, message_text: str) -> list[Part]:
        """Build v1.0 parts (discriminated union, no 'kind' field)."""
        return [TextPart(text=message_text)]

    def _build_v03_parts(self, message_text: str) -> list[dict]:
        """Build v0.3 parts (with 'kind' field for wire compatibility)."""
        return [{"kind": "text", "text": message_text}]

    async def call_skill(
        self,
        skill_name: str,
        arguments: dict | None = None,
    ) -> TaskResult:
        """
        Call a specific skill on the A2A agent.

        This method sends a message to the agent via SendMessage (v1.0) or
        message/send (v0.3) and returns the task result. It does NOT poll;
        use get_task() to retrieve results for long-running tasks.

        Args:
            skill_name: The name of the skill to invoke.
            arguments: Optional arguments to pass to the skill.
                       For v1.0, expected keys: "message" (str), "context_id" (str, optional).
                       For v0.3, the dict is stringified as task input.

        Returns:
            The task result containing the output.
        """
        card = await self._discover_agent_card()

        if self.protocol_version == "1.0":
            # v1.0: Extract message and context_id from arguments
            message_text = ""
            context_id = None
            if arguments:
                message_text = arguments.get("message", "")
                context_id = arguments.get("context_id")

            # Build v1.0 message with skill_name embedded
            msg = self._build_v1_message(skill_name, message_text, context_id)

            params = {
                "message": msg.model_dump(exclude_none=True)
            }

            result = await self._send_v1_request("SendMessage", params)
            raw_result = result.get("result", {})
            # v1.0 SendMessage returns either task or message
            send_result = SendMessageResult.model_validate(raw_result)
            if send_result.task:
                artifact = send_result.task.artifacts[0] if send_result.task.artifacts else None
                return TaskResult(
                    task_id=send_result.task.id,
                    output=TaskOutput(parts=artifact.parts) if artifact else None,
                    metadata=send_result.task.metadata,
                )
            elif send_result.message:
                return TaskResult(
                    task_id="",
                    output=TaskOutput(parts=send_result.message.parts),
                )
            return TaskResult(task_id="")
        else:
            # v0.3: Use message/send with task-like input structure
            # Serialize arguments as proper JSON (not Python str repr)
            import json
            message_text = json.dumps(arguments) if arguments else "{}"
            input_parts = self._build_v03_parts(message_text)

            skill_endpoint = card.url
            for skill in card.skills:
                if skill.name == skill_name:
                    skill_endpoint = skill.endpoint or card.url
                    break

            # v0.3 requires messageId in the message object
            import uuid
            params = {
                "message": {
                    "messageId": f"msg-{uuid.uuid4().hex[:8]}",
                    "role": "user",
                    "parts": input_parts,
                }
            }
            result = await self._send_v03_request(skill_endpoint, "message/send", params)
            raw_result = result.get("result", {})
            if "id" in raw_result and "task_id" not in raw_result:
                raw_result["task_id"] = raw_result.pop("id")
            # v0.3 returns a full Task object - convert to TaskResult
            task_id = raw_result.get("id") or raw_result.get("task_id") or ""
            artifacts = raw_result.get("artifacts", [])
            output = None
            if artifacts:
                # Get parts from first artifact
                parts = artifacts[0].get("parts", [])
                # Convert v0.3 parts (with 'kind') to our Part model
                converted_parts = []
                for p in parts:
                    if "text" in p:
                        converted_parts.append(TextPart(text=p["text"]))
                output = TaskOutput(parts=converted_parts)
            return TaskResult(
                task_id=task_id,
                output=output,
                error=None,
                metadata=raw_result.get("metadata", {}),
            )

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

        if self.protocol_version == "1.0":
            # v1.0: Use SendMessage with contextId on Message
            msg = Message(
                role="user",
                parts=self._build_v1_parts(message),
                context_id=context_id,
            )
            params = {
                "message": msg.model_dump(exclude_none=True)
            }
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
            # v0.3: Use message/send
            params = {
                "message": {
                    "role": "user",
                    "parts": self._build_v03_parts(message),
                }
            }
            if context_id:
                params["contextId"] = context_id

            result = await self._send_v03_request(card.url, "message/send", params)
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

        if not task_id:
            return TaskResult(
                task_id="",
                error={"code": -32602, "message": "Invalid task_id: empty string"},
            )

        if self.protocol_version == "1.0":
            result = await self._send_v1_request("GetTask", {"id": task_id})
            # Check for JSON-RPC error
            if "error" in result:
                return TaskResult(
                    task_id=task_id,
                    error=result["error"],
                )
            raw_result = result.get("result", {})
            try:
                task = Task.model_validate(raw_result)
                # v1.0: Check task state for input-required, failed, canceled
                task_state = task.status.state if task.status else "unknown"
                if task_state == "input-required":
                    return TaskResult(
                        task_id=task.id,
                        error={
                            "code": -32001,
                            "message": "Task requires additional input",
                            "data": {"state": task_state, "message": task.status.message},
                        },
                    )
                if task_state == "failed":
                    return TaskResult(
                        task_id=task.id,
                        error={
                            "code": -32002,
                            "message": "Task failed",
                            "data": {"state": task_state, "message": task.status.message},
                        },
                    )
                if task_state == "canceled":
                    return TaskResult(
                        task_id=task.id,
                        error={
                            "code": -32003,
                            "message": "Task was canceled",
                            "data": {"state": task_state},
                        },
                    )
                return TaskResult(
                    task_id=task.id,
                    output=task.artifacts[0] if task.artifacts else None,
                    metadata=task.metadata,
                )
            except Exception as e:
                return TaskResult(
                    task_id=task_id,
                    error={"code": -32603, "message": f"Failed to parse task result: {e}"},
                )
        else:
            skill_endpoint = card.url
            result = await self._send_v03_request(skill_endpoint, "tasks/get", {"id": task_id})
            if "error" in result:
                return TaskResult(
                    task_id=task_id,
                    error=result["error"],
                )
            raw_result = result.get("result", {})
            if "id" in raw_result and "task_id" not in raw_result:
                raw_result["task_id"] = raw_result.pop("id")
            try:
                return TaskResult.model_validate(raw_result)
            except Exception as e:
                return TaskResult(
                    task_id=task_id,
                    error={"code": -32603, "message": f"Failed to parse task result: {e}"},
                )
