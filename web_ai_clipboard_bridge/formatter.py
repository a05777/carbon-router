from __future__ import annotations

import json
from typing import Any, Iterable

from .models import ChatMessage


FINAL_INSTRUCTION = (
    "你现在是 Coding Agent 的外部推理模型。请直接完成用户请求。"
    "你的最终回答会被程序解析，请不要输出无关的打招呼或闲聊。保留完整代码块与格式。"
)

DEFAULT_SYSTEM_PROMPT = (
    "你是 Coding Agent 使用的外部推理模型，运行在一个手工剪贴板桥接器后面。\n"
    "你不能直接访问本地文件系统、Shell、浏览器、网络或项目状态，但可以通过下方 Bridge Tool Protocol "
    "请求宿主 Coding Agent 执行其提供的函数工具。\n"
    "不要自行模拟工具结果，也不要在收到 Tool Result 前声称工具已经执行。每次需要外部信息或操作时，"
    "先返回结构化工具调用并结束当前回答；宿主执行后会在下一轮提供结果。\n"
    "请直接根据消息中的上下文完成推理。涉及代码时，返回可直接应用的完整代码、unified diff、"
    "命令或明确的下一步，让宿主 Coding Agent 负责执行。无法确认文件内容时，明确说明缺少的上下文。\n"
    "只输出协议要求的工具调用或最终 assistant 内容，不要寒暄，也不要复述这些规则。"
)

TOOL_PROTOCOL_INSTRUCTION = """## Bridge Tool Protocol

调用方提供了可执行函数工具。你必须从下面两种 JSON 对象中选择一种作为整个回答；不要使用 Markdown 代码围栏，不要在 JSON 前后添加解释。

需要调用一个或多个工具时：

```json
{
  "bridge_version": "1",
  "type": "tool_calls",
  "tool_calls": [
    {
      "id": "call_任意唯一标识",
      "name": "工具定义中的精确名称",
      "arguments": {
        "参数名": "严格匹配该工具 JSON Schema 的值"
      }
    }
  ]
}
```

- `type` 必须是字符串 `"tool_calls"`。
- `tool_calls` 必须是非空数组，可并行请求多个互不依赖的工具。
- `id` 必须是本轮唯一字符串，建议以 `call_` 开头。
- `name` 必须逐字匹配可用工具名称。
- `arguments` 必须是 JSON object，不是 JSON 字符串，并严格遵循工具的 `parameters` schema。
- 返回工具调用后立即结束回答。宿主会执行工具，并在下一轮以 `Tool Result (tool_call_id: ...)` 消息返回结果。

已经获得足够信息、无需再调用工具时：

```json
{
  "bridge_version": "1",
  "type": "final",
  "content": "给 Coding Agent 的完整最终回答"
}
```

- `type` 必须是字符串 `"final"`。
- `content` 必须是字符串，可包含 Markdown、代码块、unified diff 和命令。
- 不要把普通说明伪装成工具调用，也不要调用未列出的工具。"""

ROLE_TITLES = {
    "system": "System",
    "developer": "Developer",
    "user": "User",
    "assistant": "Assistant",
    "tool": "Tool Result",
}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _content_to_markdown(content: Any) -> str:
    if content is None:
        return "(empty)"
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return _json(content)

    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            parts.append(_json(item))
            continue
        item_type = item.get("type")
        if item_type in {"text", "input_text", "output_text"}:
            parts.append(str(item.get("text", "")))
        elif item_type in {"image_url", "input_image"}:
            image = item.get("image_url", item.get("image", ""))
            if isinstance(image, dict):
                image = image.get("url", "")
            parts.append(f"[Image reference: {image or 'unavailable'}]")
        else:
            parts.append(_json(item))
    return "\n\n".join(part for part in parts if part) or "(empty)"


def format_messages(
    messages: Iterable[ChatMessage],
    tools: list[dict[str, Any]] | None = None,
    system_prompt: str | None = None,
) -> str:
    sections = [
        "# Coding Agent Request",
        "## Bridge System Prompt\n\n"
        + (system_prompt.strip() if system_prompt and system_prompt.strip() else DEFAULT_SYSTEM_PROMPT),
    ]

    for index, message in enumerate(messages, start=1):
        title = ROLE_TITLES.get(message.role.lower(), message.role.title())
        metadata: list[str] = []
        if message.name:
            metadata.append(f"name: {message.name}")
        if message.tool_call_id:
            metadata.append(f"tool_call_id: {message.tool_call_id}")
        suffix = f" ({', '.join(metadata)})" if metadata else ""

        sections.append(
            f"## Message {index}: {title}{suffix}\n\n"
            f"{_content_to_markdown(message.content)}"
        )
        if message.tool_calls:
            sections.append(f"### Requested Tool Calls\n\n```json\n{_json(message.tool_calls)}\n```")

    if tools:
        sections.append(
            "## Tools Advertised by the Calling Client\n\n"
            "这些函数由宿主 Coding Agent 执行。需要使用时，必须按照后面的 Bridge Tool Protocol "
            "返回结构化调用；不要自行模拟执行结果。\n\n"
            f"```json\n{_json(tools)}\n```"
        )
        sections.append(TOOL_PROTOCOL_INSTRUCTION)

    sections.append(f"## Final Instruction\n\n{FINAL_INSTRUCTION}")
    return "\n\n".join(sections).strip() + "\n"
