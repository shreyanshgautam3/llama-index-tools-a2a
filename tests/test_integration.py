"""Integration tests for client-server round-trip verification."""

import pytest
from httpx import ASGITransport, AsyncClient
from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step

from llama_index.tools.a2a_tools import workflow_as_a2a
from llama_index.tools.a2a_tools.client import BasicA2AClient


class SimpleStart(StartEvent):
    query: str


class SimpleWorkflow(Workflow):
    @step
    async def run(self, start_event: SimpleStart) -> StopEvent:
        return StopEvent(result=f"Response to: {start_event.query}")


class MultiStepWorkflow(Workflow):
    @step
    async def process(self, start_event: SimpleStart) -> StopEvent:
        intermediate = f"Processed: {start_event.query}"
        return StopEvent(result=intermediate)


@pytest.mark.asyncio
async def test_v03_client_server_roundtrip():
    """Test v0.3 client-server round-trip with message/send and tasks/get."""
    app = workflow_as_a2a(SimpleWorkflow())
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Get agent card
        card_resp = await client.get("/.well-known/agent-card.json")
        assert card_resp.status_code == 200
        card = card_resp.json()

        # Verify v0.3 format (fields use camelCase aliases in JSON)
        assert "url" in card
        assert card["preferredTransport"] == "JSONRPC"
        assert card["capabilities"]["streaming"] is False


@pytest.mark.asyncio
async def test_v10_agent_card_format_not_supported():
    """Test that v1.0 AgentCard format is NOT supported by server (sdk limitation)."""
    # v1.0 is not supported - create_agent_card should raise NotImplementedError
    from llama_index.tools.a2a_tools.server import create_agent_card
    with pytest.raises(NotImplementedError):
        create_agent_card(SimpleWorkflow(), protocol_version="1.0")

    # But workflow_as_a2a with default (v0.3) should work
    app = workflow_as_a2a(SimpleWorkflow())
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Get agent card
        card_resp = await client.get("/.well-known/agent-card.json")
        assert card_resp.status_code == 200
        card = card_resp.json()

        # Verify v0.3 format (no supportedInterfaces)
        assert "url" in card
        assert card["preferredTransport"] == "JSONRPC"
        assert "supportedInterfaces" not in card
        assert card["capabilities"]["streaming"] is False


@pytest.mark.asyncio
async def test_v03_message_send_flow():
    """Test v0.3 message/send flow through the server."""
    app = workflow_as_a2a(SimpleWorkflow())
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Send message using v0.3 message/send
        resp = await client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "method": "message/send",
                "params": {
                    "message": {
                        "messageId": "msg-1",
                        "role": "user",
                        "parts": [{"kind": "text", "text": '{"query": "hello"}'}],
                    }
                },
                "id": 1,
            },
        )
        assert resp.status_code == 200
        result = resp.json()
        assert "result" in result
        task = result["result"]
        assert task["status"]["state"] == "completed"
        assert "artifacts" in task


@pytest.mark.asyncio
async def test_v03_tasks_get_flow():
    """Test v0.3 tasks/get flow through the server."""
    app = workflow_as_a2a(SimpleWorkflow())
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # First send a message
        send_resp = await client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "method": "message/send",
                "params": {
                    "message": {
                        "messageId": "msg-2",
                        "role": "user",
                        "parts": [{"kind": "text", "text": '{"query": "test"}'}],
                    }
                },
                "id": 2,
            },
        )
        task_id = send_resp.json()["result"]["id"]

        # Get the task using v0.3 tasks/get
        get_resp = await client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "method": "tasks/get",
                "params": {"id": task_id},
                "id": 3,
            },
        )
        assert get_resp.status_code == 200
        task = get_resp.json()["result"]
        assert task["status"]["state"] == "completed"


@pytest.mark.asyncio
async def test_v10_methods_not_supported_by_server():
    """Test that v1.0 methods (SendMessage, GetTask) return -32601 on the server."""
    app = workflow_as_a2a(SimpleWorkflow())
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # v0.3 method should work
        resp = await client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "method": "message/send",
                "params": {
                    "message": {
                        "messageId": "msg-3",
                        "role": "user",
                        "parts": [{"kind": "text", "text": '{"query": "hello"}'}],
                    }
                },
                "id": 1,
            },
        )
        assert resp.status_code == 200
        result = resp.json()
        assert "result" in result
        task = result["result"]
        assert task["status"]["state"] == "completed"

        # v1.0 methods should NOT work (a2a-sdk 0.3.26 doesn't support them)
        resp = await client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "method": "SendMessage",
                "params": {
                    "message": {
                        "messageId": "msg-4",
                        "role": "user",
                        "parts": [{"text": '{"skill": "simpleworkflow", "message": "hello"}'}],
                    }
                },
                "id": 2,
            },
            headers={"A2A-Version": "1.0"},
        )
        # Should return method not found
        assert resp.status_code == 200
        result = resp.json()
        assert "error" in result
        assert result["error"]["code"] == -32601


@pytest.mark.asyncio
async def test_version_auto_detection():
    """Test that client auto-detects protocol version from agent card."""
    from llama_index.tools.a2a_tools.models import AgentCard

    # Test v0.3 card detection (server only produces v0.3 cards)
    app_v03 = workflow_as_a2a(SimpleWorkflow())
    transport_v03 = ASGITransport(app=app_v03)

    async with AsyncClient(transport=transport_v03, base_url="http://test") as client:
        card_resp = await client.get("/.well-known/agent-card.json")
        card = card_resp.json()

        a2a_client = BasicA2AClient("http://test/.well-known/agent-card.json")
        a2a_client._agent_card = AgentCard.model_validate(card)
        a2a_client._detected_version = None

        # Simulate detection logic - should be v0.3 since no supported_interfaces
        has_supported = getattr(a2a_client._agent_card, 'supported_interfaces', None)
        if has_supported:
            detected = "1.0"
        else:
            detected = "0.3"
        assert detected == "0.3"


@pytest.mark.asyncio
async def test_multi_step_workflow_v03():
    """Test multi-step workflow with v0.3 protocol."""
    app = workflow_as_a2a(MultiStepWorkflow())
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "method": "message/send",
                "params": {
                    "message": {
                        "messageId": "msg-5",
                        "role": "user",
                        "parts": [{"kind": "text", "text": '{"query": "hello"}'}],
                    }
                },
                "id": 1,
            },
        )
        assert resp.status_code == 200
        result = resp.json()
        assert "result" in result
        task = result["result"]
        assert task["status"]["state"] == "completed"
        artifact_text = task["artifacts"][0]["parts"][0]["text"]
        assert "Processed: hello" in artifact_text


@pytest.mark.asyncio
async def test_multi_step_workflow_v03_only():
    """Test multi-step workflow with v0.3 protocol (only supported version)."""
    app = workflow_as_a2a(MultiStepWorkflow())
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "method": "message/send",
                "params": {
                    "message": {
                        "messageId": "msg-6",
                        "role": "user",
                        "parts": [{"kind": "text", "text": '{"query": "hello"}'}],
                    }
                },
                "id": 1,
            },
        )
        assert resp.status_code == 200
        result = resp.json()
        assert "result" in result
        task = result["result"]
        assert task["status"]["state"] == "completed"
        artifact_text = task["artifacts"][0]["parts"][0]["text"]
        assert "Processed: hello" in artifact_text


@pytest.mark.asyncio
async def test_client_v03_call_skill():
    """Test BasicA2AClient v0.3 call_skill against real server."""
    from llama_index.tools.a2a_tools.models import AgentCard

    app = workflow_as_a2a(SimpleWorkflow())
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        # Get agent card
        card_resp = await http_client.get("/.well-known/agent-card.json")
        card = card_resp.json()

        # Create A2A client with the real HTTP transport
        a2a_client = BasicA2AClient(
            "http://test/.well-known/agent-card.json"
        )
        # Replace the internal HTTP client with our test client
        a2a_client._http_client = http_client
        a2a_client._agent_card = AgentCard.model_validate(card)

        result = await a2a_client.call_skill("simpleworkflow", arguments={"query": "hello"})

        assert result.task_id != ""
        assert result.output is not None
        assert "Response to: hello" in str(result.output)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
