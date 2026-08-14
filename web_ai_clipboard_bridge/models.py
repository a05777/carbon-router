from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str = Field(min_length=1)
    content: Any = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    response_format: dict[str, Any] | None = None


class ResponsesRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = Field(min_length=1)
    input: Any = None
    instructions: Any = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = "auto"
    parallel_tool_calls: bool = True
    previous_response_id: str | None = None
    store: bool | None = True
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    max_output_tokens: int | None = None
    reasoning: dict[str, Any] | None = None
    text: dict[str, Any] | None = None
    truncation: Any = "disabled"
    metadata: dict[str, Any] | None = None
    user: str | None = None


class SubmitAnswerRequest(BaseModel):
    request_id: str = Field(min_length=1)
    answer: str = Field(min_length=1)
