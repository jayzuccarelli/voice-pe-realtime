"""The harness's view of a fake house that lives in another process.

When the assistant under test is the broker, its tools run inside the fake
house server (evals/fake_house_server.py), not in the harness. This executor
resets that house to the scenario's state when the harness creates it, and
reads the final state and the tool log back when the harness grades, so the
broker is scored on exactly the checks raw GPT-Live is.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any


class RemoteHouse:
    def __init__(self, base_url: str, initial_state: dict[str, Any]) -> None:
        self.base_url = base_url.rstrip("/")
        self._request("/reset", {"initial_state": initial_state})

    def _request(self, path: str, body: dict[str, Any] | None = None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base_url + path, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())

    @property
    def executions(self) -> list[dict[str, Any]]:
        return self._request("/executions")

    def execute(self, name: str, arguments: dict[str, Any], *, call_id: str) -> dict[str, Any]:
        raise RuntimeError(
            f"The harness tried to run {name} itself; with the broker under test every tool call "
            "must come from the broker, through the fake house's MCP endpoint"
        )

    def snapshot(self) -> dict[str, Any]:
        return self._request("/snapshot")
