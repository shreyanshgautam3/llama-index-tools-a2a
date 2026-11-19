"""
Test fixtures loader for v1.0 A2A protocol.

Loads real v1.0 wire format responses from JSON fixtures.
These fixtures represent the actual JSON-RPC responses from a v1.0-compliant A2A server.
"""

import json
from pathlib import Path
from typing import Any

FIXTURES_DIR = Path(__file__).parent


def load_fixture(name: str) -> dict[str, Any]:
    """Load a fixture by name (without .json extension)."""
    path = FIXTURES_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Fixture not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_all_fixtures() -> dict[str, dict[str, Any]]:
    """Load all fixtures in the fixtures directory."""
    fixtures = {}
    for path in FIXTURES_DIR.glob("*.json"):
        with open(path, encoding="utf-8") as f:
            fixtures[path.stem] = json.load(f)
    return fixtures


# Pre-load common fixtures for convenience
V10_SEND_MESSAGE_RESULT = load_fixture("v10_send_message_result")
V10_SEND_MESSAGE_WORKING = load_fixture("v10_send_message_working")
V10_SEND_MESSAGE_DIRECT = load_fixture("v10_send_message_direct")
V10_SEND_MESSAGE_FAILED = load_fixture("v10_send_message_failed")
V10_GET_TASK_COMPLETED = load_fixture("v10_get_task_completed")
V10_GET_TASK_WORKING = load_fixture("v10_get_task_working")
V10_AGENT_CARD = load_fixture("v10_agent_card")
V10_ERROR_METHOD_NOT_FOUND = load_fixture("v10_error_method_not_found")
V10_SEND_MESSAGE_MULTI_PART = load_fixture("v10_send_message_multi_part")
