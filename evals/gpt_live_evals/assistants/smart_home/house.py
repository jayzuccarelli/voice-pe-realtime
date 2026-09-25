"""A fake house that answers Home Assistant's MCP tools and remembers what they did.

The eval needs a house it can check afterwards and must never touch the real
one. This models the devices from a depersonalized snapshot of Jay's Home
Assistant, matches targets the way HA's intents do (a name alias, an area, a
domain), and records every call so the grader can compare what was asked with
what happened. The same object backs raw GPT-Live in the harness and, through
server.py, the broker under test, so both are graded on one definition of
"the lights went off".
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

SEED_FILE = Path(__file__).with_name("house_seed.json")
TOOLS_FILE = Path(__file__).with_name("ha_tools.json")
FIXED_NOW = {"date": "2026-09-25", "time": "19:30:00", "weekday": "Thursday"}
SWITCHABLE = {"light", "switch", "fan", "media_player", "climate"}


def load_seed() -> dict[str, Any]:
    return json.loads(SEED_FILE.read_text(encoding="utf-8"))


def device_key(entity: dict[str, Any]) -> str:
    """One stable, readable key per device: all its names, as HA lists them."""
    return ", ".join(entity["names"])


class House:
    def __init__(self, initial_state: dict[str, Any] | None = None) -> None:
        seed = load_seed()
        self.entities: list[dict[str, Any]] = copy.deepcopy(seed["entities"])
        self.lists: dict[str, list[str]] = copy.deepcopy(seed["lists"])
        self.tv_apps: dict[str, str] = {}
        self.executions: list[dict[str, Any]] = []
        self._seed_states = {device_key(e): e["state"] for e in self.entities}
        self._seed_lists = copy.deepcopy(self.lists)
        self._watched = set()
        for key, state in ((initial_state or {}).get("devices") or {}).items():
            self._by_key(key)["state"] = state
            self._watched.add(key)

    # ---- state -----------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """What the scenario set up plus whatever changed since, and nothing else.

        The harness checks "nothing happened" as final == initial state, and a
        scenario's initial state names only the devices it cares about. A
        full-house dump would never equal that, so empty sections are left out
        and untouched devices are not listed.
        """
        devices = {
            device_key(e): e["state"]
            for e in self.entities
            if device_key(e) in self._watched or e["state"] != self._seed_states[device_key(e)]
        }
        out: dict[str, Any] = {}
        if devices:
            out["devices"] = devices
        attributes = {device_key(e): e["attributes"] for e in self.entities if e.get("attributes")}
        if attributes:
            out["attributes"] = attributes
        if self.tv_apps:
            out["tv_apps"] = dict(self.tv_apps)
        if self.lists != self._seed_lists:
            out["lists"] = copy.deepcopy(self.lists)
        return out

    def _by_key(self, key: str) -> dict[str, Any]:
        for e in self.entities:
            if device_key(e) == key:
                return e
        raise KeyError(f"no device {key!r} in the seed")

    def _match(self, args: dict[str, Any]) -> list[dict[str, Any]]:
        name = (args.get("name") or "").strip().lower()
        area = (args.get("area") or "").strip().lower()
        domains = args.get("domain") or []
        if isinstance(domains, str):
            domains = [domains]
        out = []
        for e in self.entities:
            if name and name not in (n.lower() for n in e["names"]):
                continue
            if area and area != (e.get("area") or "").lower():
                continue
            if domains and e["domain"] not in domains:
                continue
            if not name and not area and not domains:
                continue
            out.append(e)
        return out

    # ---- tools -----------------------------------------------------------

    def execute(self, name: str, arguments: dict[str, Any], *, call_id: str = "") -> dict[str, Any]:
        args = {k: v for k, v in (arguments or {}).items() if v is not None}
        handler = getattr(self, f"_tool_{name}", None)
        result = handler(args) if handler else _done([])
        self.executions.append({
            "call_id": call_id,
            "name": name,
            "arguments": copy.deepcopy(arguments or {}),
            "status": "completed" if result.get("success") else "failed",
            "output": copy.deepcopy(result),
        })
        return result

    def _switch(self, args: dict[str, Any], new_state: str) -> dict[str, Any]:
        matched = [e for e in self._match(args) if e["domain"] in SWITCHABLE]
        if not matched:
            return _no_match(args)
        done, failed = [], []
        for e in matched:
            if e["state"] == "unavailable":
                failed.append(e)
                continue
            e["state"] = new_state
            done.append(e)
        return _done(done, failed)

    def _tool_HassTurnOn(self, args):
        return self._switch(args, "on")

    def _tool_HassTurnOff(self, args):
        return self._switch(args, "off")

    def _set(self, args, domains, state=None, **attrs):
        matched = [e for e in self._match(args) if e["domain"] in domains]
        if not matched:
            return _no_match(args)
        for e in matched:
            if state:
                e["state"] = state
            e.setdefault("attributes", {}).update({k: v for k, v in attrs.items() if v is not None})
        return _done(matched)

    def _tool_HassLightSet(self, args):
        return self._set(args, {"light"}, "on", brightness=args.get("brightness"), color=args.get("color"),
                         temperature=args.get("temperature"))

    def _tool_HassMediaPause(self, args):
        return self._set(args, {"media_player"}, "paused")

    def _tool_HassMediaUnpause(self, args):
        return self._set(args, {"media_player"}, "playing")

    def _tool_HassMediaNext(self, args):
        return self._set(args, {"media_player"})

    def _tool_HassMediaPrevious(self, args):
        return self._set(args, {"media_player"})

    def _tool_HassSetVolume(self, args):
        return self._set(args, {"media_player"}, volume_level=args.get("volume_level"))

    def _tool_HassSetVolumeRelative(self, args):
        return self._set(args, {"media_player"}, volume_step=args.get("volume_step"))

    def _tool_HassMediaPlayerMute(self, args):
        return self._set(args, {"media_player"}, is_volume_muted=True)

    def _tool_HassMediaPlayerUnmute(self, args):
        return self._set(args, {"media_player"}, is_volume_muted=False)

    def _tool_HassMediaSearchAndPlay(self, args):
        return self._set(args, {"media_player"}, "playing", media=args.get("search_query"))

    def _tool_HassClimateSetTemperature(self, args):
        return self._set(args, {"climate"}, target_temperature=args.get("temperature"))

    def _tool_HassFanSetSpeed(self, args):
        return self._set(args, {"fan"}, "on", percentage=args.get("percentage"))

    def _tool_HassListAddItem(self, args):
        self.lists.setdefault(args.get("name") or "Shopping List", []).append(args.get("item", ""))
        return _done([])

    def _tool_HassListCompleteItem(self, args):
        return self._tool_HassListRemoveItem(args)

    def _tool_HassListRemoveItem(self, args):
        items = self.lists.get(args.get("name") or "Shopping List", [])
        if args.get("item") in items:
            items.remove(args["item"])
        return _done([])

    def _tool_todo_get_items(self, args):
        items = self.lists.get(args.get("todo_list") or "Shopping List", [])
        return {"success": True, "result": {"items": [{"summary": i, "status": "needs_action"} for i in items]}}

    def _tool_GetDateTime(self, args):
        return {"success": True, "result": FIXED_NOW}

    def _tool_tv_launch_app(self, args):
        room = (args.get("room") or "").strip().lower()
        tvs = [e for e in self.entities if e["domain"] == "media_player" and (e.get("area") or "").lower() == room
               and any("tv" in n.lower() or "frame" in n.lower() or "samsung" in n.lower() for n in e["names"])]
        if not tvs:
            return {"success": False, "error": f"No TV found in room {args.get('room')!r}"}
        self.tv_apps[room] = args.get("app", "")
        for e in tvs:
            e["state"] = "on"
        return {"success": True, "result": f"Launched {args.get('app')} on the {room} TV"}

    def _tool_tv_remote(self, args):
        return {"success": True, "result": f"Sent {args.get('action')} to the {args.get('room')} TV"}

    def _tool_GetLiveContext(self, args):
        lines = ["Live Context: An overview of the areas and the devices in this smart home:"]
        for e in self.entities:
            lines.append(f"- names: {', '.join(e['names'])}\n  domain: {e['domain']}\n  state: '{e['state']}'")
            if e.get("area"):
                lines.append(f"  areas: {e['area']}")
        return {"success": True, "result": "\n".join(lines)}


def _done(done: list[dict[str, Any]], failed: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "success": True,
        "result": {
            "response_type": "action_done",
            "data": {
                "success": [{"name": e["names"][0], "type": "entity"} for e in done],
                "failed": [{"name": e["names"][0], "type": "entity"} for e in failed or []],
            },
        },
    }


def _no_match(args: dict[str, Any]) -> dict[str, Any]:
    target = ", ".join(f"{k}={v}" for k, v in args.items() if k in ("name", "area", "domain"))
    return {"success": False, "error": f"MatchFailedError: no device matched {target or 'nothing'}"}
