"""Call one MCP tool on the server under test, the way a chat client does.

The published client speaks the HTTP API only, and some behaviour exists just for
MCP callers. This is the streamable-HTTP handshake and nothing more, so the suite
needs no MCP SDK.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
_INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-03-26",
        "capabilities": {},
        "clientInfo": {"name": "hindsight-system-tests", "version": "0"},
    },
}


def _reply(response: httpx.Response, request_id: int) -> dict[str, Any]:
    """The JSON-RPC reply to ``request_id``, whether it came as JSON or as an event stream."""
    response.raise_for_status()
    if "text/event-stream" not in response.headers.get("content-type", ""):
        return response.json()
    for line in response.text.splitlines():
        if line.startswith("data:"):
            message = json.loads(line[len("data:") :])
            if message.get("id") == request_id:
                return message
    raise AssertionError(f"no reply to request {request_id} in the event stream: {response.text[:500]}")


async def call_mcp_tool(base_url: str, bank_id: str, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Run ``tool`` against ``bank_id`` and return its decoded result."""
    headers = {**_HEADERS, "X-Bank-Id": bank_id}
    async with httpx.AsyncClient(base_url=base_url, timeout=60) as http:
        opened = await http.post("/mcp/", headers=headers, json=_INITIALIZE)
        _reply(opened, 1)
        session_id = opened.headers.get("mcp-session-id")
        if session_id:
            headers["mcp-session-id"] = session_id
        await http.post("/mcp/", headers=headers, json={"jsonrpc": "2.0", "method": "notifications/initialized"})
        called = await http.post(
            "/mcp/",
            headers=headers,
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": tool, "arguments": arguments}},
        )
    reply = _reply(called, 2)
    assert "error" not in reply, reply
    result = reply["result"]
    assert not result.get("isError"), result
    return json.loads(result["content"][0]["text"])
