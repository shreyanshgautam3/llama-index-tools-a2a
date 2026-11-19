"""
Basic A2A Client for connecting to A2A agents.

Mirrors the BasicMCPClient pattern but adapted for the A2A protocol.
Handles AgentCard discovery and task-based skill invocation.

Supports both A2A protocol v0.3 and v1.0 via explicit protocol_version parameter.

v1.0 Protocol Notes:
- Method names: SendMessage, GetTask (PascalCase)
- Part structure: discriminated union (TextPart, FilePart, DataPart) - NO 'kind' field
- contextId: placed ON the Message object, not as top-level param
- skill_name: sent in message.metadata, not embedded in message text
- A2A-Version: 1.0 header required for v1.0 servers

v0.3 Protocol Notes:
- Method names: message/send, tasks/get
- Part structure: with 'kind' field (text, file, image)
- contextId: placed ON the Message object
"""

import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx

from .models import (
    AgentCard,
    Artifact,
    Message,
    Part,
    SendMessageResult,
    Task,
    TaskOutput,
    TaskResult,
)

logger = logging.getLogger(__name__)


class A2AJSONRPCError(RuntimeError):
    """JSON-RPC error with structured code, message, and data.

    Raised from _rpc when a JSON-RPC response contains an "error" field.
    Callers should copy code/message into TaskResult.error to preserve
    the real JSON-RPC error code instead of hard-coding -32601.
    """

    def __init__(self, code: int, message: str, data: Any = None, method: str = ""):
        self.code = code
        self.message = message
        self.data = data
        self.method = method
        desc = f"JSON-RPC error from '{method}': code={code} message={message!r} data={data}"
        super().__init__(desc)


class BasicA2AClient:
    """
    Basic A2A client for connecting to an A2A agent.

    This client discovers the agent's capabilities via its AgentCard
    and provides methods to invoke skills through the A2A protocol.

    Supports both A2A v0.3 (message/send, tasks/get) and v1.0 (SendMessage, GetTask)
    via explicit protocol_version parameter. Default is "0.3" for backward compatibility.

    Args:
        agent_card_url: The URL of the agent card (e.g., https://agent.example.com/.well-known/agent-card.json).
        timeout: The timeout for HTTP operations in seconds. Default is 30.
        headers: Optional headers to pass with requests.
        protocol_version: Explicit protocol version ("0.3" or "1.0"). Default: "0.3".
    """

    def __init__(
        self,
        agent_card_url: str,
        timeout: int = 30,
        headers: dict[str, str] | None = None,
        protocol_version: str = "0.3",
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
        if self.protocol_version == "1.0":
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
        """Discover and parse the agent card from the given URL.

        Version detection: v1.0 is opt-in only via protocol_version="1.0" constructor parameter.
        No auto-detection from supportedInterfaces - must be explicitly requested.
        """
        if self._agent_card is not None:
            return self._agent_card

        client = await self._get_http_client()
        response = await client.get(self.agent_card_url, headers=self.base_headers)
        response.raise_for_status()

        self._agent_card = AgentCard.model_validate(response.json())

        # Use explicit protocol_version if provided, otherwise default to v0.3
        # v1.0 is opt-in - no auto-detection from supportedInterfaces
        if self._forced_protocol_version:
            self._detected_version = self._forced_protocol_version
        else:
            self._detected_version = "0.3"
        logger.info(f"Using A2A protocol version: {self._detected_version} (explicit)")
        return self._agent_card

    @property
    def protocol_version(self) -> str:
        """Get the explicit/forced protocol version."""
        return self._detected_version or self._forced_protocol_version or "0.3"

    async def detect_protocol_version(self) -> str:
        """Async method to ensure discovery and return protocol version."""
        if self._detected_version is None:
            await self._discover_agent_card()
        return self.protocol_version

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

    async def _rpc(self, method: str, params: dict) -> dict:
        """Send a JSON-RPC request and return the "result" field, or raise on error.

        Handles both v0.3 and v1.0. Checks the JSON-RPC response for an "error"
        field before returning, so callers never see a swallowed error as an
        empty dict.

        Args:
            method: The JSON-RPC method name.
            params: The params object to send.

        Returns:
            The "result" field from the JSON-RPC response.

        Raises:
            httpx.HTTPStatusError: If the HTTP request fails.
            RuntimeError: If the JSON-RPC response contains an error.
        """
        client = await self._get_http_client()
        request_payload = {
            "jsonrpc": "2.0",
            "id": uuid.uuid4().hex,  # Fresh id per request (Task 5)
            "method": method,
            "params": params,
        }
        response = await client.post(
            self.base_url, json=request_payload, headers=self._get_request_headers()
        )
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            err = data["error"]
            raise A2AJSONRPCError(
                code=err.get("code", -32603),
                message=err.get("message", "Unknown JSON-RPC error"),
                data=err.get("data"),
                method=method,
            )
        return data.get("result", {})

    def _build_v1_message(
        self, skill_name: str, message_text: str, context_id: str | None = None
    ) -> Message:
        """Build a v1.0 Message with skill_name in metadata and contextId on the Message."""
        # v1.0: put skill name in message.metadata, not in text
        msg = Message(
            role="ROLE_USER",
            parts=[Part(text=message_text)],
            context_id=context_id,
        )
        if skill_name:
            msg.metadata = {"skill": skill_name}
        return msg

    def _build_v1_parts(self, message_text: str) -> list[Part]:
        """Build v1.0 parts (discriminated union, no 'kind' field)."""
        return [Part(text=message_text)]

    def _build_v03_parts(self, message_text: str) -> list[dict]:
        """Build v0.3 parts (with 'kind' field for wire compatibility)."""
        return [{"kind": "text", "text": message_text}]

    def _build_v03_message(self, message_text: str, context_id: str | None = None) -> dict:
        """Build a v0.3 message with messageId and optional contextId."""
        return {
            "messageId": f"msg-{uuid.uuid4().hex[:8]}",
            "role": "user",
            "parts": self._build_v03_parts(message_text),
            **({"contextId": context_id} if context_id else {}),
        }

    def _map_task_state(self, state: str) -> dict | None:
        """Map task state to error dict if terminal error state.

        Accepts both v0.3 (lowercase) and v1.0 (TASK_STATE_*) forms.
        Returns distinct code+message for each error state.
        completed/working/submitted/unspecified return None.
        """
        # Normalize v1.0 form to lowercase for unified handling
        normalized = state.lower()

        if normalized in ("task_state_input_required", "input-required"):
            return {"code": -32001, "message": "Task requires additional input"}
        if normalized in ("task_state_failed", "failed"):
            return {"code": -32002, "message": "Task failed"}
        if normalized in ("task_state_canceled", "canceled"):
            return {"code": -32003, "message": "Task was canceled"}
        if normalized in ("task_state_rejected", "rejected"):
            return {"code": -32004, "message": "Task was rejected"}
        if normalized in ("task_state_auth_required", "auth-required"):
            return {"code": -32005, "message": "Authentication required"}
        # TASK_STATE_COMPLETED, TASK_STATE_WORKING, TASK_STATE_SUBMITTED,
        # TASK_STATE_UNSPECIFIED, and their v0.3 equivalents return None
        return None

    def _validate_send_result(self, raw_result: dict) -> SendMessageResult:
        """Validate a raw SendMessage result dict as SendMessageResult.

        Task 2: shared handling for both call_skill and send_message (v1.0 branch).
        On validation failure, returns a SendMessageResult with a task carrying
        an error TaskResult - callers check task.error before proceeding.

        Returns:
            A SendMessageResult. If the raw result was not parseable,
            task is set to None and error_info is accessible via _send_result_error.
        """
        # Use a sentinel to avoid a new return type; store the error for the caller
        self._last_send_result_error: TaskResult | None = None
        try:
            return SendMessageResult.model_validate(raw_result)
        except Exception as e:
            self._last_send_result_error = TaskResult(
                task_id="",
                error={"code": -32603, "message": f"Failed to parse SendMessage result: {e}"},
            )
            return SendMessageResult()

    def _check_send_result_error(self) -> TaskResult | None:
        """Return the TaskResult error sentinel from _validate_send_result, or None."""
        result = getattr(self, "_last_send_result_error", None)
        self._last_send_result_error = None
        return result

    def _flatten_artifacts(self, artifacts: list[Artifact] | list[dict] | None) -> list[Part]:
        """Flatten artifacts into a list of Parts (shared by v1.0 and v0.3)."""
        all_parts = []
        if artifacts:
            for artifact in artifacts:
                parts = artifact.parts if hasattr(artifact, "parts") else artifact.get("parts", [])
                for p in parts:
                    if hasattr(p, "model_dump"):
                        # It's a Part model, keep as-is
                        all_parts.append(p)
                    elif isinstance(p, dict):
                        # It's a dict, convert to Part
                        all_parts.append(Part.model_validate(p))
        return all_parts

    # Task 5c: _to_task_result and _to_task_result_v03 are kept as separate methods.
    # Merging them into a single flow would require a shared intermediate structure
    # that both versions feed into. The only version-specific step is how the raw
    # payload becomes a common structure: v1.0 uses Pydantic model_validate (Task),
    # v0.3 uses raw dict manipulation (id→task_id rename). The downstream
    # artifact-flattening + error-mapping logic is already shared via _flatten_artifacts
    # and _map_task_state, which both methods call. Merging would mean duplicating
    # that shared logic in one place or introducing an artificial intermediate type,
    # so the two methods are kept separate with their respective version-specific
    # parsing steps documented here.
    def _to_task_result(self, task: Task, task_state: str) -> TaskResult:
        """Convert a v1.0 Task to TaskResult with artifact flattening and error mapping."""
        error_info = self._map_task_state(task_state)
        if error_info:
            # Use server's error message if available
            server_message = None
            if task.status and task.status.message and task.status.message.parts:
                for part in task.status.message.parts:
                    if part.text:
                        server_message = part.text
                        break
            return TaskResult(
                task_id=task.id,
                error={
                    **error_info,
                    "data": {"state": task_state, "message": server_message},
                },
            )
        # Flatten artifacts; if none, check status message for working/input-required states
        all_parts = self._flatten_artifacts(task.artifacts)
        if not all_parts and task.status and task.status.message and task.status.message.parts:
            all_parts.extend(task.status.message.parts)
        return TaskResult(
            task_id=task.id,
            output=TaskOutput(parts=all_parts) if all_parts else None,
            metadata=task.metadata,
        )

    def _to_task_result_v03(self, raw_result: dict) -> TaskResult:
        """Convert a v0.3 Task dict to TaskResult with artifact flattening and error mapping."""
        if "id" in raw_result and "task_id" not in raw_result:
            raw_result["task_id"] = raw_result.pop("id")
        task_id = raw_result.get("task_id") or raw_result.get("id") or ""

        # Check task state for v0.3 using shared helper
        task_state = raw_result.get("status", {}).get("state", "unknown")
        error_info = self._map_task_state(task_state)
        if error_info:
            return TaskResult(
                task_id=task_id,
                error={
                    **error_info,
                    "data": {
                        "state": task_state,
                        "message": raw_result.get("status", {}).get("message"),
                    },
                },
            )

        # Handle case where message/send returns a Message instead of a Task
        if "artifacts" not in raw_result and "parts" in raw_result:
            # It's a Message response
            parts = raw_result.get("parts", [])
            converted_parts = []
            for p in parts:
                if "text" in p:
                    converted_parts.append(Part(text=p["text"]))
            return TaskResult(
                task_id="",
                output=TaskOutput(parts=converted_parts),
            )

        # Convert Task to TaskResult
        # Task 5b: when a v0.3 Task has no artifacts, fall back to status.message
        # parts (matching v1.0 _to_task_result behavior)
        artifacts = raw_result.get("artifacts", [])
        all_parts = self._flatten_artifacts(artifacts)
        if not all_parts:
            status_msg = raw_result.get("status", {}).get("message")
            if status_msg and isinstance(status_msg, dict) and status_msg.get("parts"):
                for p in status_msg["parts"]:
                    if isinstance(p, dict):
                        if "text" in p:
                            all_parts.append(Part(text=p["text"]))
                        elif "url" in p or "raw" in p:
                            all_parts.append(Part.model_validate(p))
                    elif hasattr(p, "model_dump"):
                        all_parts.append(p)
        output = TaskOutput(parts=all_parts) if all_parts else None
        return TaskResult(
            task_id=task_id,
            output=output,
            error=None,
            metadata=raw_result.get("metadata", {}),
        )

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
        await self._discover_agent_card()

        if self.protocol_version == "1.0":
            # v1.0: Extract message and context_id from arguments
            message_text = ""
            context_id = None
            if arguments:
                message_text = arguments.get("message", "")
                context_id = arguments.get("context_id")

            # Build v1.0 message with skill_name in metadata
            msg = self._build_v1_message(skill_name, message_text, context_id)

            params = {"message": msg.model_dump(by_alias=True, exclude_none=True)}

            try:
                raw_result = await self._rpc("SendMessage", params)
            except A2AJSONRPCError as e:
                return TaskResult(
                    task_id="",
                    error={"code": e.code, "message": e.message, "data": e.data},
                )

            # v1.0 SendMessage returns either task or message
            send_result = self._validate_send_result(raw_result)
            err = self._check_send_result_error()
            if err:
                return err
            if send_result.task:
                task_state = send_result.task.status.state if send_result.task.status else "unknown"
                return self._to_task_result(send_result.task, task_state)
            elif send_result.message:
                # SendMessage returned a Message directly (not a Task)
                return TaskResult(
                    task_id="",
                    output=TaskOutput(parts=send_result.message.parts),
                )
            return TaskResult(task_id="")
        else:
            # v0.3: Use message/send with task-like input structure
            # For v0.3, there's no standard "skill" concept - the skill is implicit from the card.
            # The message text should be the workflow input directly.
            # If arguments has a "message" key, use its value as the message text.
            # Otherwise, JSON-serialize the entire arguments dict.
            if arguments and "message" in arguments and isinstance(arguments["message"], str):
                # Called from A2AToolSpec: message is the actual text to send
                message_text = arguments["message"]
            else:
                # Direct call_skill call: serialize the arguments as workflow input
                message_text = json.dumps(arguments) if arguments else "{}"

            # v0.3: Extract context_id from arguments (FIX: was missing)
            context_id = arguments.get("context_id") if arguments else None

            input_parts = self._build_v03_parts(message_text)

            # v0.3 uses base URL, not skill-specific endpoint (endpoint is not an A2A field)
            # v0.3 has no standard way to specify skill - it's implicit from the card

            # v0.3 requires messageId in the message object
            params = {
                "message": {
                    "messageId": f"msg-{uuid.uuid4().hex[:8]}",
                    "role": "user",
                    "parts": input_parts,
                    **({"contextId": context_id} if context_id else {}),
                }
            }
            try:
                raw_result = await self._rpc("message/send", params)
            except A2AJSONRPCError as e:
                return TaskResult(
                    task_id="",
                    error={"code": e.code, "message": e.message, "data": e.data},
                )

            if "id" in raw_result and "task_id" not in raw_result:
                raw_result["task_id"] = raw_result.pop("id")
            # v0.3 returns a full Task object - convert to TaskResult
            return self._to_task_result_v03(raw_result)

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
        await self._discover_agent_card()

        if self.protocol_version == "1.0":
            # v1.0: Use SendMessage with contextId on Message
            msg = Message(
                role="ROLE_USER",
                parts=self._build_v1_parts(message),
                context_id=context_id,
            )
            params = {"message": msg.model_dump(by_alias=True, exclude_none=True)}
            try:
                raw_result = await self._rpc("SendMessage", params)
            except A2AJSONRPCError as e:
                return TaskResult(
                    task_id="",
                    error={"code": e.code, "message": e.message, "data": e.data},
                )
            send_result = self._validate_send_result(raw_result)
            err = self._check_send_result_error()
            if err:
                return err
            if send_result.task:
                return self._to_task_result(
                    send_result.task,
                    send_result.task.status.state if send_result.task.status else "unknown",
                )
            elif send_result.message:
                # SendMessage returned a Message directly (not a Task)
                return TaskResult(
                    task_id="",
                    output=TaskOutput(parts=send_result.message.parts),
                )
            return TaskResult(task_id="")
        else:
            # v0.3: Use message/send
            params = {"message": self._build_v03_message(message, context_id)}
            # v0.3: contextId is NOT a top-level param; it goes inside message if needed

            try:
                raw_result = await self._rpc("message/send", params)
            except A2AJSONRPCError as e:
                return TaskResult(
                    task_id="",
                    error={"code": e.code, "message": e.message, "data": e.data},
                )
            if "id" in raw_result and "task_id" not in raw_result:
                raw_result["task_id"] = raw_result.pop("id")
            return self._to_task_result_v03(raw_result)

    async def get_task(self, task_id: str) -> TaskResult:
        """
        Get task status/result (v1.0: GetTask, v0.3: tasks/get).

        Args:
            task_id: The task ID to retrieve.

        Returns:
            The task result.
        """
        await self._discover_agent_card()

        if not task_id:
            return TaskResult(
                task_id="",
                error={"code": -32602, "message": "Invalid task_id: empty string"},
            )

        if self.protocol_version == "1.0":
            try:
                raw_result = await self._rpc("GetTask", {"id": task_id})
            except A2AJSONRPCError as e:
                return TaskResult(
                    task_id=task_id,
                    error={"code": e.code, "message": e.message, "data": e.data},
                )
            # v1.0 GetTask returns {"task": {...}} - extract task before validating
            task_data = raw_result
            if isinstance(raw_result, dict):
                task_data = raw_result.get("task", raw_result)
            try:
                task = Task.model_validate(task_data)
                # v1.0: Check task state for input-required, failed, canceled using shared helper
                task_state = task.status.state if task.status else "unknown"
                error_info = self._map_task_state(task_state)
                if error_info:
                    return TaskResult(
                        task_id=task.id,
                        error={
                            **error_info,
                            "data": {"state": task_state, "message": task.status.message},
                        },
                    )
                # Return ALL artifacts combined; if none, check status message for working state
                all_parts = []
                if task.artifacts:
                    for artifact in task.artifacts:
                        all_parts.extend(artifact.parts)
                if not all_parts and task.status and task.status.message:
                    msg = task.status.message
                    if msg.parts:
                        all_parts.extend(msg.parts)
                return TaskResult(
                    task_id=task.id,
                    output=TaskOutput(parts=all_parts) if all_parts else None,
                    metadata=task.metadata,
                )
            except Exception as e:
                return TaskResult(
                    task_id=task_id,
                    error={"code": -32603, "message": f"Failed to parse task result: {e}"},
                )
        else:
            try:
                raw_result = await self._rpc("tasks/get", {"id": task_id})
            except A2AJSONRPCError as e:
                return TaskResult(
                    task_id=task_id,
                    error={"code": e.code, "message": e.message, "data": e.data},
                )
            if "id" in raw_result and "task_id" not in raw_result:
                raw_result["task_id"] = raw_result.pop("id")
            try:
                return self._to_task_result_v03(raw_result)
            except Exception as e:
                return TaskResult(
                    task_id=task_id,
                    error={"code": -32603, "message": f"Failed to parse task result: {e}"},
                )
