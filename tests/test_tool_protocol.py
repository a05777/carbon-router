import json

from web_ai_clipboard_bridge.tool_protocol import parse_bridge_answer


TOOLS = [
    {
        "type": "function",
        "name": "read_file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    }
]


def test_plain_text_and_final_envelope_become_final_answers() -> None:
    plain = parse_bridge_answer("ordinary answer", TOOLS)
    assert plain.kind == "final"
    assert plain.content == "ordinary answer"

    wrapped = parse_bridge_answer(
        json.dumps(
            {"bridge_version": "1", "type": "final", "content": "wrapped answer"}
        ),
        TOOLS,
    )
    assert wrapped.kind == "final"
    assert wrapped.content == "wrapped answer"


def test_tool_envelope_accepts_json_fence_and_object_arguments() -> None:
    answer = """```json
{"bridge_version":"1","type":"tool_calls","tool_calls":[{"id":"call_read","name":"read_file","arguments":{"path":"README.md"}}]}
```"""
    parsed = parse_bridge_answer(answer, TOOLS)
    assert parsed.kind == "tool_calls"
    assert parsed.tool_calls[0].call_id == "call_read"
    assert parsed.tool_calls[0].name == "read_file"
    assert parsed.tool_calls[0].arguments == {"path": "README.md"}


def test_unknown_tool_is_not_forwarded_as_a_tool_call() -> None:
    answer = json.dumps(
        {
            "bridge_version": "1",
            "type": "tool_calls",
            "tool_calls": [{"name": "delete_everything", "arguments": {}}],
        }
    )
    parsed = parse_bridge_answer(answer, TOOLS)
    assert parsed.kind == "final"
    assert parsed.content == answer


def test_tool_envelope_is_not_activated_without_advertised_tools() -> None:
    answer = json.dumps(
        {
            "bridge_version": "1",
            "type": "tool_calls",
            "tool_calls": [{"name": "read_file", "arguments": {"path": "a.txt"}}],
        }
    )
    parsed = parse_bridge_answer(answer, tools=None)
    assert parsed.kind == "final"
    assert parsed.content == answer
