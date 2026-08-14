from __future__ import annotations

import asyncio
import json
import logging
import math
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse

from .config import Settings
from .formatter import format_messages
from .models import ChatCompletionRequest, ResponsesRequest, SubmitAnswerRequest
from .responses import completed_response, responses_to_messages, stream_response_events
from .state import (
    BridgeBusyError,
    BridgeState,
    BridgeTimeoutError,
    InvalidSubmissionError,
    PendingRequest,
)
from .tool_protocol import ParsedBridgeAnswer, parse_bridge_answer


logger = logging.getLogger("clipboard_bridge")
STATIC_DIR = Path(__file__).parent / "static"
MODEL_ID = "web-ai-clipboard"


def _error(message: str, error_type: str, code: str, status: int) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": code,
            }
        },
    )


def _usage(prompt: str, answer: str) -> dict[str, int]:
    prompt_tokens = max(1, math.ceil(len(prompt) / 4))
    completion_tokens = max(1, math.ceil(len(answer) / 4))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _completion(request: PendingRequest, answer: ParsedBridgeAnswer) -> dict[str, Any]:
    if answer.kind == "tool_calls":
        message: dict[str, Any] = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments_json(),
                    },
                }
                for call in answer.tool_calls
            ],
        }
        finish_reason = "tool_calls"
    else:
        message = {"role": "assistant", "content": answer.content or ""}
        finish_reason = "stop"
    return {
        "id": request.request_id,
        "object": "chat.completion",
        "created": request.created_unix,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
        "usage": _usage(request.prompt, answer.source_text),
        "system_fingerprint": "web-ai-clipboard-bridge",
    }


def _sse_data(payload: dict[str, Any] | str) -> str:
    body = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return f"data: {body}\n\n"


def _stream_chunk(
    request: PendingRequest,
    delta: dict[str, Any],
    finish_reason: str | None = None,
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": request.request_id,
        "object": "chat.completion.chunk",
        "created": request.created_unix,
        "model": request.model,
        "choices": [
            {"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish_reason}
        ],
        "system_fingerprint": "web-ai-clipboard-bridge",
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


def _answer_chunks(answer: str, size: int = 192) -> list[str]:
    return [answer[index : index + size] for index in range(0, len(answer), size)] or [""]


async def _stream_response(
    bridge: BridgeState,
    pending: PendingRequest,
    answer: ParsedBridgeAnswer,
    include_usage: bool,
) -> AsyncIterator[str]:
    outcome = "cancelled"
    try:
        yield _sse_data(_stream_chunk(pending, {"role": "assistant", "content": ""}))
        if answer.kind == "tool_calls":
            for index, call in enumerate(answer.tool_calls):
                yield _sse_data(
                    _stream_chunk(
                        pending,
                        {
                            "tool_calls": [
                                {
                                    "index": index,
                                    "id": call.call_id,
                                    "type": "function",
                                    "function": {
                                        "name": call.name,
                                        "arguments": call.arguments_json(),
                                    },
                                }
                            ]
                        },
                    )
                )
                await asyncio.sleep(0)
            yield _sse_data(_stream_chunk(pending, {}, finish_reason="tool_calls"))
        else:
            for chunk in _answer_chunks(answer.content or ""):
                yield _sse_data(_stream_chunk(pending, {"content": chunk}))
                await asyncio.sleep(0)
            yield _sse_data(_stream_chunk(pending, {}, finish_reason="stop"))
        if include_usage:
            usage_chunk = _stream_chunk(pending, {})
            usage_chunk["choices"] = []
            usage_chunk["usage"] = _usage(pending.prompt, answer.source_text)
            yield _sse_data(usage_chunk)
        yield _sse_data("[DONE]")
        outcome = "completed"
    finally:
        await bridge.release(pending, outcome)


def create_app(settings: Settings) -> FastAPI:
    bridge = BridgeState(settings.timeout_seconds)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        banner = (
            "\n"
            "+--------------------------------------------------+\n"
            "|          Web AI Clipboard Bridge  v0.3.0         |\n"
            f"|  UI: http://{settings.host}:{settings.port}/ui"
            + " " * max(0, 26 - len(settings.host) - len(str(settings.port)))
            + "|\n"
            "+--------------------------------------------------+"
        )
        logger.info(banner)
        yield

    app = FastAPI(
        title="Web AI Clipboard Bridge",
        version="0.3.0",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.bridge = bridge
    app.state.settings = settings

    @app.middleware("http")
    async def secure_local_routes(request: Request, call_next: Any) -> Any:
        started = time.monotonic()
        api_route = request.url.path == "/v1" or request.url.path.startswith("/v1/")
        request_id = f"req_{secrets.token_hex(12)}"

        if api_route:
            authorization = request.headers.get("authorization", "")
            scheme, _, token = authorization.partition(" ")
            authenticated = (
                scheme.lower() == "bearer"
                and bool(token)
                and secrets.compare_digest(token, settings.api_key)
            )
            if not authenticated:
                response = _error(
                    "Invalid or missing API key.",
                    "authentication_error",
                    "invalid_api_key",
                    401,
                )
                response.headers["WWW-Authenticate"] = "Bearer"
                response.headers["x-request-id"] = request_id
                logger.warning("%s %s -> 401", request.method, request.url.path)
                return response

        if request.url.path.startswith("/api/"):
            client_host = request.client.host if request.client else ""
            if client_host not in {"127.0.0.1", "::1", "localhost", "testclient"}:
                return _error(
                    "The Web UI API is available only from the local machine.",
                    "permission_error",
                    "local_access_only",
                    403,
                )
            if request.method not in {"GET", "HEAD"} and request.headers.get(
                "x-bridge-ui"
            ) != "clipboard-bridge":
                return _error(
                    "Missing Web UI request header.",
                    "permission_error",
                    "invalid_ui_request",
                    403,
                )
        response = await call_next(request)
        if api_route:
            if response.status_code == 404:
                logger.warning(
                    "Unknown OpenAI API route: %s %s",
                    request.method,
                    request.url.path,
                )
                response = _error(
                    f"No route for {request.method} {request.url.path}.",
                    "invalid_request_error",
                    "not_found",
                    404,
                )
            response.headers["x-request-id"] = request_id
            response.headers["openai-processing-ms"] = str(
                max(0, round((time.monotonic() - started) * 1000))
            )
            logger.info(
                "%s %s -> %s in %.3fs",
                request.method,
                request.url.path,
                response.status_code,
                time.monotonic() - started,
            )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> Any:
        if request.url.path == "/v1" or request.url.path.startswith("/v1/"):
            first = exc.errors()[0] if exc.errors() else {}
            location = ".".join(str(part) for part in first.get("loc", [])[1:]) or None
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": first.get("msg", "Invalid request body."),
                        "type": "invalid_request_error",
                        "param": location,
                        "code": "validation_error",
                    }
                },
            )
        return await request_validation_exception_handler(request, exc)

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse("/ui", status_code=307)

    @app.get("/ui", include_in_schema=False)
    async def ui() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/healthz", include_in_schema=False)
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/status", include_in_schema=False)
    async def status() -> dict[str, Any]:
        return await bridge.snapshot()

    @app.post("/api/submit", include_in_schema=False)
    async def submit(payload: SubmitAnswerRequest) -> Any:
        answer = payload.answer.strip()
        if not answer:
            return _error(
                "Answer cannot be blank.",
                "invalid_request_error",
                "blank_answer",
                422,
            )
        try:
            await bridge.submit(payload.request_id, answer)
        except InvalidSubmissionError as exc:
            return _error(str(exc), "invalid_request_error", "invalid_submission", 409)
        return {"ok": True, "request_id": payload.request_id}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": MODEL_ID,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local",
                }
            ],
        }

    @app.get("/v1/models/{model_id}")
    async def retrieve_model(model_id: str) -> Any:
        if model_id != MODEL_ID:
            return _error(
                f"The model '{model_id}' does not exist.",
                "invalid_request_error",
                "model_not_found",
                404,
            )
        return {
            "id": MODEL_ID,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: ChatCompletionRequest) -> Any:
        prompt = format_messages(payload.messages, payload.tools, settings.system_prompt)
        messages = [message.model_dump(mode="json") for message in payload.messages]
        try:
            pending = await bridge.claim(payload.model, prompt, messages)
        except BridgeBusyError:
            response = _error(
                "Bridge is currently busy processing another request.",
                "rate_limit_error",
                "bridge_busy",
                429,
            )
            response.headers["Retry-After"] = "1"
            return response

        if payload.stream:
            try:
                raw_answer = await bridge.wait_for_answer(pending)
            except BridgeTimeoutError:
                await bridge.release(pending, "timed_out")
                return _error(
                    "Timed out waiting for an answer from the Web UI.",
                    "timeout_error",
                    "bridge_timeout",
                    504,
                )
            except BaseException:
                await bridge.release(pending, "cancelled")
                raise
            answer = parse_bridge_answer(raw_answer, payload.tools)
            return StreamingResponse(
                _stream_response(
                    bridge,
                    pending,
                    answer,
                    bool(
                        payload.stream_options
                        and payload.stream_options.get("include_usage")
                    ),
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        outcome = "cancelled"
        try:
            raw_answer = await bridge.wait_for_answer(pending)
            answer = parse_bridge_answer(raw_answer, payload.tools)
            outcome = "completed"
            return _completion(pending, answer)
        except BridgeTimeoutError:
            outcome = "timed_out"
            return _error(
                "Timed out waiting for an answer from the Web UI.",
                "timeout_error",
                "bridge_timeout",
                504,
            )
        finally:
            await bridge.release(pending, outcome)

    @app.post("/v1/responses")
    async def create_response(payload: ResponsesRequest) -> Any:
        response_messages = responses_to_messages(payload)
        prompt = format_messages(response_messages, payload.tools, settings.system_prompt)
        messages = [message.model_dump(mode="json") for message in response_messages]
        try:
            pending = await bridge.claim(
                payload.model,
                prompt,
                messages,
                id_prefix="resp_",
            )
        except BridgeBusyError:
            response = _error(
                "Bridge is currently busy processing another request.",
                "rate_limit_error",
                "bridge_busy",
                429,
            )
            response.headers["Retry-After"] = "1"
            return response

        if payload.stream:
            try:
                raw_answer = await bridge.wait_for_answer(pending)
            except BridgeTimeoutError:
                await bridge.release(pending, "timed_out")
                return _error(
                    "Timed out waiting for an answer from the Web UI.",
                    "timeout_error",
                    "bridge_timeout",
                    504,
                )
            except BaseException:
                await bridge.release(pending, "cancelled")
                raise
            answer = parse_bridge_answer(raw_answer, payload.tools)
            return StreamingResponse(
                stream_response_events(bridge, pending, payload, answer),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        outcome = "cancelled"
        try:
            raw_answer = await bridge.wait_for_answer(pending)
            answer = parse_bridge_answer(raw_answer, payload.tools)
            outcome = "completed"
            return completed_response(pending, payload, answer)
        except BridgeTimeoutError:
            outcome = "timed_out"
            return _error(
                "Timed out waiting for an answer from the Web UI.",
                "timeout_error",
                "bridge_timeout",
                504,
            )
        finally:
            await bridge.release(pending, outcome)

    return app
