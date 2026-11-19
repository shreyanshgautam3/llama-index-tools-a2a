"""
Tests using real v1.0 wire format fixtures.

These tests validate that the client correctly handles the actual JSON-RPC
responses from a v1.0-compliant A2A server (like the a2a-samples helloworld).
"""

import json
from typing import Any

import httpx
import pytest

from llama_index.tools.a2a_tools.client import BasicA2AClient
from llama_index.tools.a2a_tools.models import Message, Part, Task, TaskResult
from tests.fixtures import (
    V10_AGENT_CARD,
    V10_ERROR_METHOD_NOT_FOUND,
    V10_GET_TASK_COMPLETED,
    V10_GET_TASK_WORKING,
    V10_SEND_MESSAGE_DIRECT,
    V10_SEND_MESSAGE_FAILED,
    V10_SEND_MESSAGE_MULTI_PART,
    V10_SEND_MESSAGE_RESULT,
    V10_SEND_MESSAGE_WORKING,
)


class MockTransport:
    """Mock transport that returns predefined fixture responses."""

    def __init__(self, responses: dict[str, dict[str, Any]]):
        self.responses = responses
        self.call_count = 0
        self.last_request: httpx.Request | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.call_count += 1
        self.last_request = request

        # Get the JSON-RPC method from the request body
        body = request.read()
        if isinstance(body, bytes):
            body = body.decode("utf-8")
        req_data = json.loads(body)
        method = req_data.get("method", "")

        # Find matching fixture
        for fixture_name, fixture_data in self.responses.items():
            if method in fixture_name.lower():
                response = httpx.Response(200, json=fixture_data, request=request)
                return response

        # Default: return first fixture
        first_fixture = next(iter(self.responses.values()))
        response = httpx.Response(200, json=first_fixture, request=request)
        return response


class MockAsyncClient:
    """Mock httpx.AsyncClient that uses our MockTransport."""

    def __init__(self, transport: MockTransport):
        self._transport = transport

    async def post(self, url: str, json: dict, headers: dict | None = None) -> httpx.Response:
        # Create a mock request
        request = httpx.Request("POST", url, json=json, headers=headers)
        return await self._transport.handle_async_request(request)

    async def get(self, url: str, headers: dict | None = None) -> httpx.Response:
        request = httpx.Request("GET", url, headers=headers)
        return await self._transport.handle_async_request(request)

    async def aclose(self):
        pass


def make_client(transport: MockTransport) -> BasicA2AClient:
    """Create a BasicA2AClient with mocked HTTP client and pre-loaded agent card."""
    mock_client = MockAsyncClient(transport)

    # Create client and inject the mock HTTP client directly
    client = BasicA2AClient(
        agent_card_url="http://localhost:9999/.well-known/agent-card.json",
        protocol_version="1.0",
    )
    from llama_index.tools.a2a_tools.models import AgentCard

    client._agent_card = AgentCard.model_validate(V10_AGENT_CARD["result"])
    client._detected_version = "1.0"
    client._http_client = mock_client  # Inject mock client directly
    return client


class TestV10FixturesSendMessage:
    """Test send_message with real v1.0 wire format fixtures."""

    @pytest.mark.asyncio
    async def test_send_message_v10_completed_task(self):
        """Test send_message returning a completed task (v10_send_message_result)."""
        transport = MockTransport({"send_message": V10_SEND_MESSAGE_RESULT})
        client = make_client(transport)

        result = await client.send_message("hi")

        assert isinstance(result, TaskResult)
        assert result.task_id == "task-001"
        assert result.output is not None
        assert len(result.output.parts) == 1
        assert result.output.parts[0].text == "Hello, World! I have received your request (hi)"

    @pytest.mark.asyncio
    async def test_send_message_v10_working_status(self):
        """Test send_message returning working status (v10_send_message_working)."""
        transport = MockTransport({"send_message": V10_SEND_MESSAGE_WORKING})
        client = make_client(transport)

        result = await client.send_message("hello")

        assert isinstance(result, TaskResult)
        assert result.task_id == "task-002"
        assert result.output is not None
        assert len(result.output.parts) == 1
        assert result.output.parts[0].text == "Processing request..."

    @pytest.mark.asyncio
    async def test_send_message_v10_direct_message(self):
        """Test send_message returning a direct message (v10_send_message_direct)."""
        transport = MockTransport({"send_message": V10_SEND_MESSAGE_DIRECT})
        client = make_client(transport)

        result = await client.send_message("how can I help")

        assert isinstance(result, TaskResult)
        assert result.task_id == ""  # No task returned, just message
        assert result.output is not None
        assert len(result.output.parts) == 1
        assert result.output.parts[0].text == "Hello, how can I help you?"

    @pytest.mark.asyncio
    async def test_send_message_v10_failed(self):
        """Test send_message returning failed task (v10_send_message_failed)."""
        transport = MockTransport({"send_message": V10_SEND_MESSAGE_FAILED})
        client = make_client(transport)

        result = await client.send_message("trigger error")

        assert isinstance(result, TaskResult)
        assert result.task_id == "task-004"
        assert result.error is not None
        # The main message is "Task failed"; server's message is in error.data.message
        assert "Task failed" == result.error["message"]
        assert result.error["data"]["state"] == "failed"
        assert result.error["data"]["message"] == "An error occurred while processing the request."

    @pytest.mark.asyncio
    async def test_send_message_v10_multi_part_artifacts(self):
        """Test send_message with multi-part artifacts (v10_send_message_multi_part)."""
        transport = MockTransport({"send_message": V10_SEND_MESSAGE_MULTI_PART})
        client = make_client(transport)

        result = await client.send_message("send me files")

        assert isinstance(result, TaskResult)
        assert result.task_id == "task-007"
        assert result.output is not None
        assert len(result.output.parts) == 4

        # Text part
        assert result.output.parts[0].text == "Here is the text content"

        # File by URL
        assert result.output.parts[1].url == "https://example.com/document.pdf"
        assert result.output.parts[1].media_type == "application/pdf"
        assert result.output.parts[1].filename == "document.pdf"

        # File by base64 (raw bytes)
        assert result.output.parts[2].raw == b"Hello, World!"
        assert result.output.parts[2].media_type == "text/plain"
        assert result.output.parts[2].filename == "hello.txt"

        # Data part
        assert result.output.parts[3].data == {
            "key": "value",
            "numbers": [1, 2, 3],
            "nested": {"field": "data"},
        }


class TestV10FixturesGetTask:
    """Test get_task with real v1.0 wire format fixtures."""

    @pytest.mark.asyncio
    async def test_get_task_v10_completed(self):
        """Test get_task returning completed task (v10_get_task_completed)."""
        transport = MockTransport({"get_task": V10_GET_TASK_COMPLETED})
        client = make_client(transport)

        task = await client.get_task("task-005")

        assert isinstance(task, TaskResult)
        assert task.task_id == "task-005"
        assert task.output is not None
        assert len(task.output.parts) == 1
        assert task.output.parts[0].text == "Processed: hello"

    @pytest.mark.asyncio
    async def test_get_task_v10_working(self):
        """Test get_task returning working task with history (v10_get_task_working)."""
        transport = MockTransport({"get_task": V10_GET_TASK_WORKING})
        client = make_client(transport)

        task = await client.get_task("task-006")

        assert isinstance(task, TaskResult)
        assert task.task_id == "task-006"
        assert task.output is not None
        assert len(task.output.parts) == 1
        assert task.output.parts[0].text == "Still working..."


class TestV10FixturesAgentCard:
    """Test AgentCard parsing with real v1.0 wire format."""

    def test_agent_card_v10_additional_interfaces(self):
        """Test AgentCard with additionalInterfaces (v10_agent_card)."""
        from llama_index.tools.a2a_tools.models import AgentCard

        card = AgentCard.model_validate(V10_AGENT_CARD["result"])

        assert card.name == "Hello World Agent"
        assert card.version == "0.0.1"
        assert card.capabilities is not None
        assert card.capabilities.streaming is True
        assert card.capabilities.push_notifications is False
        assert len(card.skills) == 1
        assert card.skills[0].id == "echo_bot"
        assert card.skills[0].name == "Echo Bot"
        # v1.0: additionalInterfaces
        assert card.additional_interfaces is not None
        assert len(card.additional_interfaces) == 1
        assert card.additional_interfaces[0].url == "http://127.0.0.1:9999"
        assert card.additional_interfaces[0].protocol_binding == "JSONRPC"
        assert card.additional_interfaces[0].protocol_version == "1.0"
        # primary_url should return first additionalInterface URL
        assert card.primary_url == "http://127.0.0.1:9999"


class TestV10FixturesErrorHandling:
    """Test JSON-RPC error handling with real v1.0 error fixtures."""

    @pytest.mark.asyncio
    async def test_v10_method_not_found_error(self):
        """Test handling of method not found error (v10_error_method_not_found)."""
        transport = MockTransport({"error": V10_ERROR_METHOD_NOT_FOUND})
        client = make_client(transport)

        result = await client.send_message("hi")

        assert isinstance(result, TaskResult)
        assert result.error is not None
        assert "Method not found" in result.error["message"]
        assert result.error["code"] == -32601


class TestV10FixturesRoundtrip:
    """Test model serialization roundtrip with real fixtures."""

    def test_message_v10_roundtrip(self):
        """Test Message model roundtrip with v1.0 role values."""
        # From v10_send_message_direct fixture
        msg_data = V10_SEND_MESSAGE_DIRECT["result"]["message"]

        # Parse with our model
        msg = Message.model_validate(msg_data)

        assert msg.role == "ROLE_AGENT"
        assert msg.parts[0].text == "Hello, how can I help you?"
        assert msg.context_id == "ctx-003"
        assert msg.message_id is not None

        # Serialize back to wire format (camelCase)
        wire = msg.model_dump(by_alias=True, exclude_none=True)

        assert wire["role"] == "ROLE_AGENT"
        assert wire["parts"][0]["text"] == "Hello, how can I help you?"
        assert wire["contextId"] == "ctx-003"
        assert "messageId" in wire

    def test_task_v10_roundtrip(self):
        """Test Task model roundtrip with v1.0 fields."""
        task_data = V10_GET_TASK_COMPLETED["result"]["task"]

        task = Task.model_validate(task_data)

        assert task.id == "task-005"
        assert task.context_id == "ctx-005"
        assert task.status.state == "completed"
        assert len(task.artifacts) == 1

        # Serialize back to wire format
        wire = task.model_dump(by_alias=True, exclude_none=True)

        assert wire["id"] == "task-005"
        assert wire["contextId"] == "ctx-005"
        assert wire["status"]["state"] == "completed"
        assert wire["artifacts"][0]["artifactId"] == "task-005-artifact"
        assert wire["artifacts"][0]["parts"][0]["text"] == "Processed: hello"
        assert wire["artifacts"][0]["lastChunk"] is True

    def test_part_v10_all_types_roundtrip(self):
        """Test Part model with all v1.0 part types."""
        # From v10_send_message_multi_part fixture
        artifacts = V10_SEND_MESSAGE_MULTI_PART["result"]["task"]["artifacts"]

        # Text part
        text_part = Part.model_validate(artifacts[0]["parts"][0])
        assert text_part.text == "Here is the text content"
        assert text_part.is_text()

        # File by URL
        file_url_part = Part.model_validate(artifacts[1]["parts"][0])
        assert file_url_part.url == "https://example.com/document.pdf"
        assert file_url_part.media_type == "application/pdf"
        assert file_url_part.filename == "document.pdf"
        assert file_url_part.is_file()

        # File by base64 (raw bytes)
        file_raw_part = Part.model_validate(artifacts[2]["parts"][0])
        assert file_raw_part.raw == b"Hello, World!"
        assert file_raw_part.media_type == "text/plain"
        assert file_raw_part.filename == "hello.txt"
        assert file_raw_part.is_file()

        # Data part
        data_part = Part.model_validate(artifacts[3]["parts"][0])
        assert data_part.data == {"key": "value", "numbers": [1, 2, 3], "nested": {"field": "data"}}
        assert data_part.is_data()

        # All serialize correctly to wire format
        for part in [text_part, file_url_part, file_raw_part, data_part]:
            wire = part.model_dump(by_alias=True, exclude_none=True)
            assert wire  # Should not be empty


class TestV10FixturesContextIdHandling:
    """Test contextId handling in v1.0 with fixtures."""

    @pytest.mark.asyncio
    async def test_send_message_preserves_context_id(self):
        """Test that contextId from user is preserved in request."""
        transport = MockTransport({"send_message": V10_SEND_MESSAGE_RESULT})
        client = make_client(transport)

        await client.send_message("hi", context_id="my-custom-context")

        # Verify the request sent had contextId
        req = transport.last_request
        body = req.read()
        if isinstance(body, bytes):
            body = body.decode("utf-8")
        req_data = json.loads(body)

        # In v1.0, contextId is on the message
        params = req_data.get("params", {})
        message = params.get("message", {})
        assert message.get("contextId") == "my-custom-context"

    @pytest.mark.asyncio
    async def test_get_task_sends_task_id(self):
        """Test get_task includes taskId in request (contextId is not a param for GetTask)."""
        transport = MockTransport({"get_task": V10_GET_TASK_COMPLETED})
        client = make_client(transport)

        await client.get_task("task-005")

        req = transport.last_request
        body = req.read()
        if isinstance(body, bytes):
            body = body.decode("utf-8")
        req_data = json.loads(body)

        params = req_data.get("params", {})
        # GetTask takes id (taskId)
        assert params.get("id") == "task-005"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
