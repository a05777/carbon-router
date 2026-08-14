from __future__ import annotations

import asyncio
import json
import math
import time
import uuid
from typing import Any, AsyncIterator

from .models import ChatMessage, ResponsesRequest
from .state import BridgeState, PendingRequest
from .tool_protocol import BridgeToolCall, ParsedBridgeAnswer


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _input_item_to_messages(item: Any, default_role: str = "user") -> list[ChatMessage]:
    if isinstance(item, str):
        return [ChatMessage(role=default_role, content=item)]
    if not isinstance(item, dict):
        return [ChatMessage(role=default_role, content=_json(item))]

    item_type = item.get("type")
    if item_type in {None, "message"} or "role" in item:
        return [
            ChatMessage(
                role=str(item.get("role", default_role)),
                content=item.get("content"),
                name=item.get("name"),
            )
        ]

    if item_type in {"function_call", "custom_tool_call"}:
        name = item.get("name", "unknown_tool")
        arguments = item.get("arguments", item.get("input", "{}"))
        return [
            ChatMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    {
                        "id": item.get("call_id", item.get("id", "call_unknown")),
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                ],
            )
        ]

    if item_type in {"function_call_output", "custom_tool_call_output"}:
        return [
            ChatMessage(
                role="tool",
                content=item.get("output", ""),
                tool_call_id=item.get("call_id"),
            )
        ]

    return [
        ChatMessage(
            role=default_role,
            content=f"Responses API input item (`{item_type}`):\n\n```json\n{_json(item)}\n```",
        )
    ]


def responses_to_messages(payload: ResponsesRequest) -> list[ChatMessage]:
    messages: list[ChatMessage] = []

    if payload.instructions is not None:
        instruction_items = (
            payload.instructions if isinstance(payload.instructions, list) else [payload.instructions]
        )
        for item in instruction_items:
            messages.extend(_input_item_to_messages(item, default_role="developer"))

    if payload.input is not None:
        input_items = payload.input if isinstance(payload.input, list) else [payload.input]
        for item in input_items:
            messages.extend(_input_item_to_messages(item))

    if payload.previous_response_id:
        messages.append(
            ChatMessage(
                role="system",
                content=(
                    "The caller supplied previous_response_id="
                    f"{payload.previous_response_id}. The bridge is stateless, so only the input "
                    "items included in this request are available as conversation context."
                ),
            )
        )

    return messages


def _token_usage(prompt: str, answer: str) -> dict[str, Any]:
    input_tokens = max(1, math.ceil(len(prompt) / 4))
    output_tokens = max(1, math.ceil(len(answer) / 4))
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": input_tokens + output_tokens,
    }


def _message_output_item(message_id: str, answer: str, status: str) -> dict[str, Any]:
    return {
        "id": message_id,
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": (
            []
            if status == "in_progress"
            else [{"type": "output_text", "text": answer, "annotations": []}]
        ),
    }


def _function_output_item(
    item_id: str, call: BridgeToolCall, status: str, arguments: str | None = None
) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "function_call",
        "status": status,
        "call_id": call.call_id,
        "name": call.name,
        "arguments": call.arguments_json() if arguments is None else arguments,
    }


def response_object(
    pending: PendingRequest,
    payload: ResponsesRequest,
    answer: ParsedBridgeAnswer | None,
    message_id: str | None,
    function_item_ids: list[str] | None = None,
) -> dict[str, Any]:
    completed = answer is not None
    output: list[dict[str, Any]] = []
    output_text = None
    if completed and answer.kind == "final":
        output_text = answer.content or ""
        output.append(_message_output_item(message_id or "", output_text, "completed"))
    elif completed:
        output = [
            _function_output_item(item_id, call, "completed")
            for item_id, call in zip(function_item_ids or [], answer.tool_calls)
        ]
    response: dict[str, Any] = {
        "id": pending.request_id,
        "object": "response",
        "created_at": pending.created_unix,
        "status": "completed" if completed else "in_progress",
        "background": bool(payload.model_extra and payload.model_extra.get("background", False)),
        "error": None,
        "incomplete_details": None,
        "instructions": payload.instructions,
        "max_output_tokens": payload.max_output_tokens,
        "model": payload.model,
        "output": output,
        "parallel_tool_calls": payload.parallel_tool_calls,
        "previous_response_id": payload.previous_response_id,
        "reasoning": payload.reasoning or {"effort": None, "summary": None},
        "store": True if payload.store is None else payload.store,
        "temperature": payload.temperature,
        "text": payload.text or {"format": {"type": "text"}},
        "tool_choice": payload.tool_choice,
        "tools": payload.tools or [],
        "top_p": payload.top_p,
        "truncation": payload.truncation,
        "usage": _token_usage(pending.prompt, answer.source_text) if completed else None,
        "user": payload.user,
        "metadata": payload.metadata or {},
    }
    if completed:
        response["completed_at"] = int(time.time())
        response["output_text"] = output_text or ""
    return response


def _event(event_type: str, sequence_number: int, payload: dict[str, Any]) -> str:
    data = {"type": event_type, "sequence_number": sequence_number, **payload}
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _answer_chunks(answer: str, size: int = 192) -> list[str]:
    return [answer[index : index + size] for index in range(0, len(answer), size)] or [""]


async def stream_response_events(
    bridge: BridgeState,
    pending: PendingRequest,
    payload: ResponsesRequest,
    answer: ParsedBridgeAnswer,
) -> AsyncIterator[str]:
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    function_item_ids = [f"fc_{uuid.uuid4().hex[:24]}" for _ in answer.tool_calls]
    item_in_progress = _message_output_item(message_id, "", "in_progress")
    part_in_progress = {"type": "output_text", "text": "", "annotations": []}
    outcome = "cancelled"
    sequence_number = 0
    try:
        in_progress = response_object(
            pending,
            payload,
            None,
            message_id,
            function_item_ids,
        )
        yield _event("response.created", sequence_number, {"response": in_progress})
        sequence_number += 1
        yield _event("response.in_progress", sequence_number, {"response": in_progress})
        sequence_number += 1
        if answer.kind == "tool_calls":
            for output_index, (item_id, call) in enumerate(
                zip(function_item_ids, answer.tool_calls)
            ):
                yield _event(
                    "response.output_item.added",
                    sequence_number,
                    {
                        "output_index": output_index,
                        "item": _function_output_item(item_id, call, "in_progress", ""),
                    },
                )
                sequence_number += 1
                for chunk in _answer_chunks(call.arguments_json()):
                    yield _event(
                        "response.function_call_arguments.delta",
                        sequence_number,
                        {
                            "item_id": item_id,
                            "output_index": output_index,
                            "delta": chunk,
                        },
                    )
                    sequence_number += 1
                    await asyncio.sleep(0)
                yield _event(
                    "response.function_call_arguments.done",
                    sequence_number,
                    {
                        "item_id": item_id,
                        "output_index": output_index,
                        "name": call.name,
                        "arguments": call.arguments_json(),
                    },
                )
                sequence_number += 1
                yield _event(
                    "response.output_item.done",
                    sequence_number,
                    {
                        "output_index": output_index,
                        "item": _function_output_item(item_id, call, "completed"),
                    },
                )
                sequence_number += 1
        else:
            yield _event(
                "response.output_item.added",
                sequence_number,
                {"output_index": 0, "item": item_in_progress},
            )
            sequence_number += 1
            yield _event(
                "response.content_part.added",
                sequence_number,
                {
                    "item_id": message_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": part_in_progress,
                },
            )
            sequence_number += 1
            for chunk in _answer_chunks(answer.content or ""):
                yield _event(
                    "response.output_text.delta",
                    sequence_number,
                    {
                        "item_id": message_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": chunk,
                        "logprobs": [],
                    },
                )
                sequence_number += 1
                await asyncio.sleep(0)

            completed_part = {
                "type": "output_text",
                "text": answer.content or "",
                "annotations": [],
            }
            completed_item = _message_output_item(message_id, answer.content or "", "completed")
            yield _event(
                "response.output_text.done",
                sequence_number,
                {
                    "item_id": message_id,
                    "output_index": 0,
                    "content_index": 0,
                    "text": answer.content or "",
                    "logprobs": [],
                },
            )
            sequence_number += 1
            yield _event(
                "response.content_part.done",
                sequence_number,
                {
                    "item_id": message_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": completed_part,
                },
            )
            sequence_number += 1
            yield _event(
                "response.output_item.done",
                sequence_number,
                {"output_index": 0, "item": completed_item},
            )
            sequence_number += 1
        yield _event(
            "response.completed",
            sequence_number,
            {
                "response": response_object(
                    pending,
                    payload,
                    answer,
                    message_id,
                    function_item_ids,
                )
            },
        )
        outcome = "completed"
    finally:
        await bridge.release(pending, outcome)


def completed_response(
    pending: PendingRequest, payload: ResponsesRequest, answer: ParsedBridgeAnswer
) -> dict[str, Any]:
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    function_item_ids = [f"fc_{uuid.uuid4().hex[:24]}" for _ in answer.tool_calls]
    return response_object(
        pending,
        payload,
        answer,
        message_id=message_id,
        function_item_ids=function_item_ids,
    )
