from web_ai_clipboard_bridge.formatter import (
    DEFAULT_SYSTEM_PROMPT,
    FINAL_INSTRUCTION,
    TOOL_PROTOCOL_INSTRUCTION,
    format_messages,
)
from web_ai_clipboard_bridge.models import ChatMessage


def test_formatter_preserves_roles_and_multimodal_content() -> None:
    prompt = format_messages(
        [
            ChatMessage(role="system", content="Be concise."),
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "Fix this."},
                    {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}},
                ],
            ),
        ]
    )
    assert "## Message 1: System" in prompt
    assert "Fix this." in prompt
    assert "https://example.test/a.png" in prompt
    assert DEFAULT_SYSTEM_PROMPT in prompt
    assert FINAL_INSTRUCTION in prompt


def test_formatter_includes_advertised_tools() -> None:
    prompt = format_messages(
        [ChatMessage(role="user", content="Inspect the file.")],
        tools=[
            {
                "type": "function",
                "function": {"name": "read_file", "parameters": {"type": "object"}},
            }
        ],
    )
    assert "Tools Advertised" in prompt
    assert '"name": "read_file"' in prompt
    assert TOOL_PROTOCOL_INSTRUCTION in prompt
    assert '"type": "tool_calls"' in prompt
    assert '"type": "final"' in prompt


def test_formatter_accepts_custom_bridge_system_prompt() -> None:
    prompt = format_messages(
        [ChatMessage(role="user", content="Return a patch.")],
        system_prompt="Custom bridge rules.",
    )
    assert "Custom bridge rules." in prompt
    assert DEFAULT_SYSTEM_PROMPT not in prompt
