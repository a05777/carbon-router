from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class BridgeToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]

    def arguments_json(self) -> str:
        return json.dumps(self.arguments, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class ParsedBridgeAnswer:
    kind: Literal["final", "tool_calls"]
    content: str | None
    tool_calls: tuple[BridgeToolCall, ...]
    source_text: str


def advertised_tool_names(tools: list[dict[str, Any]] | None) -> set[str]:
    names: set[str] = set()
    for tool in tools or []:
        name = tool.get("name")
        function = tool.get("function")
        if not name and isinstance(function, dict):
            name = function.get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return names


def _decode_protocol_object(answer: str) -> dict[str, Any] | None:
    candidate = answer.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        first_line, separator, remainder = candidate.partition("\n")
        if separator and first_line.lower() in {"```", "```json"}:
            candidate = remainder.rsplit("```", 1)[0].strip()
    if not candidate.startswith("{") or not candidate.endswith("}"):
        return None
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or value.get("bridge_version") != "1":
        return None
    return value


def parse_bridge_answer(
    answer: str, tools: list[dict[str, Any]] | None = None
) -> ParsedBridgeAnswer:
    protocol = _decode_protocol_object(answer)
    if protocol is None:
        return ParsedBridgeAnswer("final", answer, (), answer)

    answer_type = protocol.get("type")
    if answer_type == "final" and isinstance(protocol.get("content"), str):
        return ParsedBridgeAnswer("final", protocol["content"], (), answer)

    raw_calls = protocol.get("tool_calls")
    if answer_type != "tool_calls" or not isinstance(raw_calls, list) or not raw_calls:
        return ParsedBridgeAnswer("final", answer, (), answer)

    allowed_names = advertised_tool_names(tools)
    if not allowed_names:
        return ParsedBridgeAnswer("final", answer, (), answer)
    parsed_calls: list[BridgeToolCall] = []
    for raw_call in raw_calls:
        if not isinstance(raw_call, dict):
            return ParsedBridgeAnswer("final", answer, (), answer)
        name = raw_call.get("name")
        arguments = raw_call.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return ParsedBridgeAnswer("final", answer, (), answer)
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(arguments, dict)
            or name not in allowed_names
        ):
            return ParsedBridgeAnswer("final", answer, (), answer)
        call_id = raw_call.get("id")
        if not isinstance(call_id, str) or not call_id:
            call_id = f"call_{uuid.uuid4().hex[:24]}"
        parsed_calls.append(BridgeToolCall(call_id, name, arguments))

    return ParsedBridgeAnswer("tool_calls", None, tuple(parsed_calls), answer)
