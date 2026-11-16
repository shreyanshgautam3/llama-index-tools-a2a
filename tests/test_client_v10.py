"""Tests for A2A v1.0 protocol compliance in the client."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llama_index.tools.a2a.client import BasicA2AClient
from llama_index.tools.a2a.models import (
    AgentCard,
    AgentInterface,
    Skill,
)


@pytest.mark.asyncio
async def test_version_detection_v10_supported_interfaces():
    """Test that v1.0 is detected ONLY from supported_interfaces (wire format)."""
    mock_card = AgentCard(
        name="TestAgent",
        url="http://example.com",
        skills=[Skill(name="skill1", description="A skill")],
        supported_interfaces=[AgentInterface(url="http://example.com", transport="JSONRPC")],
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_card.model_dump(by_alias=True)

    with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=mock_response)):
        client = BasicA2AClient("http://example.com/.well-known/agent-card.json")
        await client.get_agent_card()
        assert client.protocol_version == "1.0"


@pytest.mark.asyncio
async def test_version_detection_v03_no_supported_interfaces():
    """Test that v0.3 is detected when supported_interfaces is absent."""
    mock_card = AgentCard(
        name="TestAgent",
        url="http://example.com",
        skills=[Skill(name="skill1", description="A skill")],
        preferred_transport="JSONRPC",
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_card.model_dump(by_alias=True)

    with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=mock_response)):
        client = BasicA2AClient("http://example.com/.well-known/agent-card.json")
        await client.get_agent_card()
        assert client.protocol_version == "0.3"


@pytest.mark.asyncio
async def test_version_detection_not_fooled_by_additional_interfaces():
    """Test that additional_interfaces (a2a-sdk internal) does NOT trigger v1.0 detection."""
    mock_card = AgentCard(
        name="TestAgent",
        url="http://example.com",
        skills=[Skill(name="skill1", description="A skill")],
        # additional_interfaces is a2a-sdk 0.3.26 internal field
        # It should NOT cause v1.0 detection
        additional_interfaces=[AgentInterface(url="http://example.com", transport="JSONRPC")],
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_card.model_dump(by_alias=True)

    with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=mock_response)):
        client = BasicA2AClient("http://example.com/.well-known/agent-card.json")
        await client.get_agent_card()
        # Should be v0.3 because supported_interfaces is absent
        assert client.protocol_version == "0.3"


@pytest.mark.asyncio
async def test_call_skill_v10_method_name():
    """Test that v1.0 uses SendMessage method name."""
    mock_card = AgentCard(
        name="TestAgent",
        url="http://example.com",
        skills=[Skill(name="my-skill", description="A skill")],
        supported_interfaces=[AgentInterface(url="http://example.com", transport="JSONRPC")],
    )

    # Mock response matches SendMessageResult structure (snake_case for Pydantic models)
    mock_post_response = MagicMock()
    mock_post_response.status_code = 200
    mock_post_response.json.return_value = {
        "jsonrpc": "2.0",
        "result": {
            "task": {
                "id": "task-123",
                "context_id": "ctx-456",
                "status": {"state": "completed", "timestamp": "2024-01-01T00:00:00Z"},
                "artifacts": [{"artifact_id": "art-1", "parts": [{"text": "result"}], "index": 0}],
            }
        },
        "id": 1,
    }

    mock_get_response = MagicMock()
    mock_get_response.status_code = 200
    mock_get_response.json.return_value = mock_card.model_dump(by_alias=True)

    with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=mock_get_response)):
        with patch(
            "httpx.AsyncClient.post", new=AsyncMock(return_value=mock_post_response)
        ) as mock_post:
            client = BasicA2AClient(
                "http://example.com/.well-known/agent-card.json",
                protocol_version="1.0",  # Force v1.0
            )
            result = await client.call_skill(
                "my-skill", arguments={"message": "hello", "context_id": "ctx-1"}
            )

            # Verify SendMessage was called (not tasks/send)
            call_args = mock_post.call_args
            request_payload = call_args[1]["json"]
            assert request_payload["method"] == "SendMessage"
            assert "A2A-Version" in call_args[1]["headers"]
            assert call_args[1]["headers"]["A2A-Version"] == "1.0"

            # Verify contextId is on Message, not top-level params
            params = request_payload["params"]
            assert "contextId" not in params  # Not at top level
            assert "message" in params
            assert params["message"]["context_id"] == "ctx-1"

            # Verify skill_name is embedded in message text
            msg_text = params["message"]["parts"][0]["text"]
            msg_data = json.loads(msg_text)
            assert msg_data["skill"] == "my-skill"
            assert msg_data["message"] == "hello"

            assert result.task_id == "task-123"


@pytest.mark.asyncio
async def test_send_message_v10_context_id_on_message():
    """Test that v1.0 send_message puts contextId on Message object."""
    mock_card = AgentCard(
        name="TestAgent",
        url="http://example.com",
        skills=[],
        supported_interfaces=[AgentInterface(url="http://example.com", transport="JSONRPC")],
    )

    # Mock response matches SendMessageResult with message
    mock_post_response = MagicMock()
    mock_post_response.status_code = 200
    mock_post_response.json.return_value = {
        "jsonrpc": "2.0",
        "result": {
            "message": {
                "role": "assistant",
                "parts": [{"text": "response"}],
                "contextId": "ctx-789",
            }
        },
        "id": 1,
    }

    mock_get_response = MagicMock()
    mock_get_response.status_code = 200
    mock_get_response.json.return_value = mock_card.model_dump(by_alias=True)

    with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=mock_get_response)):
        with patch(
            "httpx.AsyncClient.post", new=AsyncMock(return_value=mock_post_response)
        ) as mock_post:
            client = BasicA2AClient(
                "http://example.com/.well-known/agent-card.json",
                protocol_version="1.0",
            )
            result = await client.send_message("Hello", context_id="ctx-789")

            call_args = mock_post.call_args
            request_payload = call_args[1]["json"]
            assert request_payload["method"] == "SendMessage"

            params = request_payload["params"]
            assert "contextId" not in params  # Not at top level
            assert params["message"]["context_id"] == "ctx-789"

            # Result should have task_id from message
            assert result.task_id == ""


@pytest.mark.asyncio
async def test_get_task_v10_method_name():
    """Test that v1.0 uses GetTask method name."""
    mock_card = AgentCard(
        name="TestAgent",
        url="http://example.com",
        skills=[],
        supported_interfaces=[AgentInterface(url="http://example.com", transport="JSONRPC")],
    )

    # Mock response matches Task structure (GetTask returns Task directly) - snake_case for Pydantic
    mock_post_response = MagicMock()
    mock_post_response.status_code = 200
    mock_post_response.json.return_value = {
        "jsonrpc": "2.0",
        "result": {
            "id": "task-123",
            "context_id": "ctx-456",
            "status": {"state": "completed", "timestamp": "2024-01-01T00:00:00Z"},
            "artifacts": [{"artifact_id": "art-1", "parts": [{"text": "result"}], "index": 0}],
        },
        "id": 1,
    }

    mock_get_response = MagicMock()
    mock_get_response.status_code = 200
    mock_get_response.json.return_value = mock_card.model_dump(by_alias=True)

    with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=mock_get_response)):
        with patch(
            "httpx.AsyncClient.post", new=AsyncMock(return_value=mock_post_response)
        ) as mock_post:
            client = BasicA2AClient(
                "http://example.com/.well-known/agent-card.json",
                protocol_version="1.0",
            )
            result = await client.get_task("task-123")

            call_args = mock_post.call_args
            request_payload = call_args[1]["json"]
            assert request_payload["method"] == "GetTask"
            assert request_payload["params"]["id"] == "task-123"

            assert result.task_id == "task-123"


@pytest.mark.asyncio
async def test_get_task_v10_handles_input_required():
    """Test that v1.0 get_task handles input-required state."""
    mock_card = AgentCard(
        name="TestAgent",
        url="http://example.com",
        skills=[],
        supported_interfaces=[AgentInterface(url="http://example.com", transport="JSONRPC")],
    )

    # Mock response with input-required state - snake_case for Pydantic
    mock_post_response = MagicMock()
    mock_post_response.status_code = 200
    mock_post_response.json.return_value = {
        "jsonrpc": "2.0",
        "result": {
            "id": "task-123",
            "context_id": "ctx-456",
            "status": {
                "state": "input-required",
                "timestamp": "2024-01-01T00:00:00Z",
                "message": {
                    "role": "assistant",
                    "parts": [{"text": "Please provide more info"}],
                    "message_id": "msg-1",
                },
            },
        },
        "id": 1,
    }

    mock_get_response = MagicMock()
    mock_get_response.status_code = 200
    mock_get_response.json.return_value = mock_card.model_dump(by_alias=True)

    with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=mock_get_response)):
        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_post_response)):
            client = BasicA2AClient(
                "http://example.com/.well-known/agent-card.json",
                protocol_version="1.0",
            )
            result = await client.get_task("task-123")

            assert result.error is not None
            assert result.error["code"] == -32001
            assert "input" in result.error["message"].lower()


@pytest.mark.asyncio
async def test_get_task_v10_handles_failed():
    """Test that v1.0 get_task handles failed state."""
    mock_card = AgentCard(
        name="TestAgent",
        url="http://example.com",
        skills=[],
        supported_interfaces=[AgentInterface(url="http://example.com", transport="JSONRPC")],
    )

    # Mock response with failed state - snake_case for Pydantic
    mock_post_response = MagicMock()
    mock_post_response.status_code = 200
    mock_post_response.json.return_value = {
        "jsonrpc": "2.0",
        "result": {
            "id": "task-123",
            "context_id": "ctx-456",
            "status": {
                "state": "failed",
                "timestamp": "2024-01-01T00:00:00Z",
                "message": {
                    "role": "assistant",
                    "parts": [{"text": "Internal error"}],
                    "message_id": "msg-1",
                },
            },
        },
        "id": 1,
    }

    mock_get_response = MagicMock()
    mock_get_response.status_code = 200
    mock_get_response.json.return_value = mock_card.model_dump(by_alias=True)

    with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=mock_get_response)):
        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_post_response)):
            client = BasicA2AClient(
                "http://example.com/.well-known/agent-card.json",
                protocol_version="1.0",
            )
            result = await client.get_task("task-123")

            assert result.error is not None
            assert result.error["code"] == -32002
            assert "failed" in result.error["message"].lower()


@pytest.mark.asyncio
async def test_get_task_v10_handles_canceled():
    """Test that v1.0 get_task handles canceled state."""
    mock_card = AgentCard(
        name="TestAgent",
        url="http://example.com",
        skills=[],
        supported_interfaces=[AgentInterface(url="http://example.com", transport="JSONRPC")],
    )

    # Mock response with canceled state - snake_case for Pydantic
    mock_post_response = MagicMock()
    mock_post_response.status_code = 200
    mock_post_response.json.return_value = {
        "jsonrpc": "2.0",
        "result": {
            "id": "task-123",
            "context_id": "ctx-456",
            "status": {
                "state": "canceled",
                "timestamp": "2024-01-01T00:00:00Z",
            },
        },
        "id": 1,
    }

    mock_get_response = MagicMock()
    mock_get_response.status_code = 200
    mock_get_response.json.return_value = mock_card.model_dump(by_alias=True)

    with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=mock_get_response)):
        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_post_response)):
            client = BasicA2AClient(
                "http://example.com/.well-known/agent-card.json",
                protocol_version="1.0",
            )
            result = await client.get_task("task-123")

            assert result.error is not None
            assert result.error["code"] == -32003
            assert "canceled" in result.error["message"].lower()


@pytest.mark.asyncio
async def test_v03_uses_message_send_not_tasks_send():
    """Test that v0.3 uses message/send (not tasks/send which doesn't exist)."""
    mock_card = AgentCard(
        name="TestAgent",
        url="http://example.com",
        skills=[Skill(name="my-skill", description="A skill")],
        preferred_transport="JSONRPC",
    )

    mock_post_response = MagicMock()
    mock_post_response.status_code = 200
    mock_post_response.json.return_value = {
        "jsonrpc": "2.0",
        "result": {
            "id": "task-123",
            "status": {"state": "completed"},
            "artifacts": [{"parts": [{"kind": "text", "text": "result"}]}],
        },
        "id": 1,
    }

    mock_get_response = MagicMock()
    mock_get_response.status_code = 200
    mock_get_response.json.return_value = mock_card.model_dump(by_alias=True)

    with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=mock_get_response)):
        with patch(
            "httpx.AsyncClient.post", new=AsyncMock(return_value=mock_post_response)
        ) as mock_post:
            client = BasicA2AClient(
                "http://example.com/.well-known/agent-card.json",
                protocol_version="0.3",
            )
            await client.call_skill("my-skill", arguments={"key": "value"})

            call_args = mock_post.call_args
            request_payload = call_args[1]["json"]
            # v0.3 should use message/send, NOT tasks/send
            assert request_payload["method"] == "message/send"


@pytest.mark.asyncio
async def test_v03_send_message_uses_message_send():
    """Test that v0.3 send_message uses message/send (not messages/send)."""
    mock_card = AgentCard(
        name="TestAgent",
        url="http://example.com",
        skills=[],
        preferred_transport="JSONRPC",
    )

    mock_post_response = MagicMock()
    mock_post_response.status_code = 200
    mock_post_response.json.return_value = {
        "jsonrpc": "2.0",
        "result": {
            "id": "task-789",
            "status": {"state": "completed"},
        },
        "id": 1,
    }

    mock_get_response = MagicMock()
    mock_get_response.status_code = 200
    mock_get_response.json.return_value = mock_card.model_dump(by_alias=True)

    with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=mock_get_response)):
        with patch(
            "httpx.AsyncClient.post", new=AsyncMock(return_value=mock_post_response)
        ) as mock_post:
            client = BasicA2AClient(
                "http://example.com/.well-known/agent-card.json",
                protocol_version="0.3",
            )
            await client.send_message("Hello")

            call_args = mock_post.call_args
            request_payload = call_args[1]["json"]
            assert request_payload["method"] == "message/send"


@pytest.mark.asyncio
async def test_v10_parts_no_kind_field():
    """Test that v1.0 parts don't have 'kind' field (discriminated union)."""
    mock_card = AgentCard(
        name="TestAgent",
        url="http://example.com",
        skills=[],
        supported_interfaces=[AgentInterface(url="http://example.com", transport="JSONRPC")],
    )

    # Mock response matches SendMessageResult with message
    mock_post_response = MagicMock()
    mock_post_response.status_code = 200
    mock_post_response.json.return_value = {
        "jsonrpc": "2.0",
        "result": {
            "message": {
                "role": "assistant",
                "parts": [{"text": "response"}],
            }
        },
        "id": 1,
    }

    mock_get_response = MagicMock()
    mock_get_response.status_code = 200
    mock_get_response.json.return_value = mock_card.model_dump(by_alias=True)

    with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=mock_get_response)):
        with patch(
            "httpx.AsyncClient.post", new=AsyncMock(return_value=mock_post_response)
        ) as mock_post:
            client = BasicA2AClient(
                "http://example.com/.well-known/agent-card.json",
                protocol_version="1.0",
            )
            await client.send_message("Hello")

            call_args = mock_post.call_args
            request_payload = call_args[1]["json"]
            parts = request_payload["params"]["message"]["parts"]
            # v1.0 parts should NOT have 'kind' field
            for part in parts:
                assert "kind" not in part
                assert "text" in part
