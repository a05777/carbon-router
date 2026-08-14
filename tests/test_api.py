import asyncio
import json

import httpx
import pytest

from web_ai_clipboard_bridge.config import Settings
from web_ai_clipboard_bridge.server import create_app


@pytest.fixture
def settings() -> Settings:
    return Settings(host="127.0.0.1", port=8000, api_key="test-secret", timeout_seconds=1)


@pytest.fixture
def app(settings: Settings):
    return create_app(settings)


async def api_client(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


@pytest.mark.asyncio
async def test_models_requires_key_and_returns_model(app):
    async with await api_client(app) as client:
        unauthenticated = await client.get("/v1/models")
        assert unauthenticated.status_code == 401
        response = await client.get(
            "/v1/models", headers={"Authorization": "Bearer test-secret"}
        )
        assert response.status_code == 200
        assert response.json()["data"][0]["id"] == "web-ai-clipboard"


@pytest.mark.asyncio
async def test_chat_waits_for_ui_submission_and_returns_openai_shape(app):
    async with await api_client(app) as client:
        task = asyncio.create_task(
            client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer test-secret"},
                json={
                    "model": "web-ai-clipboard",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
        )
        request_id = None
        for _ in range(20):
            status = await client.get("/api/status")
            if status.json()["current"]:
                request_id = status.json()["current"]["request_id"]
                break
            await asyncio.sleep(0.01)
        assert request_id
        submitted = await client.post(
            "/api/submit",
            headers={"X-Bridge-UI": "clipboard-bridge"},
            json={"request_id": request_id, "answer": "world"},
        )
        assert submitted.status_code == 200
        response = await task
        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["content"] == "world"


@pytest.mark.asyncio
async def test_second_request_is_rejected_while_first_is_waiting(app):
    async with await api_client(app) as client:
        first = asyncio.create_task(
            client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer test-secret"},
                json={"model": "one", "messages": [{"role": "user", "content": "a"}]},
            )
        )
        for _ in range(20):
            if (await client.get("/api/status")).json()["current"]:
                break
            await asyncio.sleep(0.01)
        second = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-secret"},
            json={"model": "two", "messages": [{"role": "user", "content": "b"}]},
        )
        assert second.status_code == 429
        request_id = (await client.get("/api/status")).json()["current"]["request_id"]
        await client.post(
            "/api/submit",
            headers={"X-Bridge-UI": "clipboard-bridge"},
            json={"request_id": request_id, "answer": "done"},
        )
        assert (await first).status_code == 200


@pytest.mark.asyncio
async def test_streaming_response_is_sse(app):
    async with await api_client(app) as client:
        task = asyncio.create_task(
            client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer test-secret"},
                json={
                    "model": "web-ai-clipboard",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
        )
        for _ in range(20):
            current = (await client.get("/api/status")).json()["current"]
            if current:
                break
            await asyncio.sleep(0.01)
        await client.post(
            "/api/submit",
            headers={"X-Bridge-UI": "clipboard-bridge"},
            json={"request_id": current["request_id"], "answer": "streamed"},
        )
        response = await task
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert "streamed" in response.text
        assert "data: [DONE]" in response.text


@pytest.mark.asyncio
async def test_timeout_returns_openai_error(app):
    async with await api_client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-secret"},
            json={"model": "web-ai-clipboard", "messages": [{"role": "user", "content": "wait"}]},
        )
        assert response.status_code == 504
        assert response.json()["error"]["code"] == "bridge_timeout"


@pytest.mark.asyncio
async def test_stream_timeout_returns_http_504_before_sse_headers():
    app = create_app(
        Settings(
            host="127.0.0.1",
            port=8000,
            api_key="test-secret",
            timeout_seconds=0.01,
        )
    )
    async with await api_client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-secret"},
            json={
                "model": "web-ai-clipboard",
                "stream": True,
                "messages": [{"role": "user", "content": "wait"}],
            },
        )
        assert response.status_code == 504
        assert response.headers["content-type"].startswith("application/json")
        assert response.json()["error"]["code"] == "bridge_timeout"


@pytest.mark.asyncio
async def test_responses_api_waits_for_submission_and_returns_official_shape(app):
    async with await api_client(app) as client:
        task = asyncio.create_task(
            client.post(
                "/v1/responses",
                headers={"Authorization": "Bearer test-secret"},
                json={
                    "model": "web-ai-clipboard",
                    "instructions": "Act as a coding agent.",
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "Fix the bug."}],
                        }
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "name": "read_file",
                            "description": "Read a file",
                            "parameters": {"type": "object"},
                        }
                    ],
                },
            )
        )
        current = None
        for _ in range(20):
            current = (await client.get("/api/status")).json()["current"]
            if current:
                break
            await asyncio.sleep(0.01)
        assert current
        assert current["request_id"].startswith("resp_")
        assert "Act as a coding agent." in current["prompt"]
        assert "Fix the bug." in current["prompt"]
        assert '"name": "read_file"' in current["prompt"]

        await client.post(
            "/api/submit",
            headers={"X-Bridge-UI": "clipboard-bridge"},
            json={"request_id": current["request_id"], "answer": "Fixed."},
        )
        response = await task
        assert response.status_code == 200
        body = response.json()
        assert body["id"].startswith("resp_")
        assert body["object"] == "response"
        assert body["status"] == "completed"
        assert body["output"][0]["id"].startswith("msg_")
        assert body["output"][0]["content"][0] == {
            "type": "output_text",
            "text": "Fixed.",
            "annotations": [],
        }
        assert body["output_text"] == "Fixed."
        assert body["usage"]["total_tokens"] >= 2

        response_type = pytest.importorskip("openai.types.responses.response")
        parsed = response_type.Response.model_validate(body)
        assert parsed.output_text == "Fixed."


@pytest.mark.asyncio
async def test_responses_stream_uses_official_named_events(app):
    async with await api_client(app) as client:
        task = asyncio.create_task(
            client.post(
                "/v1/responses",
                headers={"Authorization": "Bearer test-secret"},
                json={
                    "model": "web-ai-clipboard",
                    "input": "Hello",
                    "stream": True,
                },
            )
        )
        current = None
        for _ in range(20):
            current = (await client.get("/api/status")).json()["current"]
            if current:
                break
            await asyncio.sleep(0.01)
        assert current
        await client.post(
            "/api/submit",
            headers={"X-Bridge-UI": "clipboard-bridge"},
            json={"request_id": current["request_id"], "answer": "Hello back"},
        )
        response = await task
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        for event_type in (
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ):
            assert f"event: {event_type}\n" in response.text
            assert f'"type": "{event_type}"' in response.text
        assert "data: [DONE]" not in response.text

        responses_types = pytest.importorskip("openai.types.responses")
        pydantic = pytest.importorskip("pydantic")
        adapter = pydantic.TypeAdapter(responses_types.ResponseStreamEvent)
        events = [
            adapter.validate_python(json.loads(line.removeprefix("data: ")))
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        assert [event.sequence_number for event in events] == list(range(len(events)))


@pytest.mark.asyncio
async def test_unknown_v1_route_and_validation_use_openai_errors(app):
    async with await api_client(app) as client:
        unknown = await client.post(
            "/v1/unknown",
            headers={"Authorization": "Bearer test-secret"},
            json={},
        )
        assert unknown.status_code == 404
        assert unknown.json()["error"]["code"] == "not_found"
        assert unknown.headers["x-request-id"].startswith("req_")

        invalid = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-secret"},
            json={"model": "web-ai-clipboard"},
        )
        assert invalid.status_code == 400
        assert invalid.json()["error"]["type"] == "invalid_request_error"


@pytest.mark.asyncio
async def test_retrieve_model_matches_openai_model_endpoint(app):
    async with await api_client(app) as client:
        response = await client.get(
            "/v1/models/web-ai-clipboard",
            headers={"Authorization": "Bearer test-secret"},
        )
        assert response.status_code == 200
        assert response.json()["object"] == "model"


@pytest.mark.asyncio
async def test_chat_completion_converts_bridge_protocol_to_tool_calls(app):
    async with await api_client(app) as client:
        task = asyncio.create_task(
            client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer test-secret"},
                json={
                    "model": "web-ai-clipboard",
                    "messages": [{"role": "user", "content": "Read the file."}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "parameters": {"type": "object"},
                            },
                        }
                    ],
                },
            )
        )
        current = None
        for _ in range(20):
            current = (await client.get("/api/status")).json()["current"]
            if current:
                break
            await asyncio.sleep(0.01)
        assert current
        assert "Bridge Tool Protocol" in current["prompt"]
        answer = json.dumps(
            {
                "bridge_version": "1",
                "type": "tool_calls",
                "tool_calls": [
                    {
                        "id": "call_read",
                        "name": "read_file",
                        "arguments": {"path": "README.md"},
                    }
                ],
            }
        )
        await client.post(
            "/api/submit",
            headers={"X-Bridge-UI": "clipboard-bridge"},
            json={"request_id": current["request_id"], "answer": answer},
        )
        response = await task
        choice = response.json()["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"]["content"] is None
        tool_call = choice["message"]["tool_calls"][0]
        assert tool_call["id"] == "call_read"
        assert tool_call["function"]["name"] == "read_file"
        assert json.loads(tool_call["function"]["arguments"]) == {"path": "README.md"}


@pytest.mark.asyncio
async def test_responses_api_converts_bridge_protocol_to_function_call(app):
    async with await api_client(app) as client:
        task = asyncio.create_task(
            client.post(
                "/v1/responses",
                headers={"Authorization": "Bearer test-secret"},
                json={
                    "model": "web-ai-clipboard",
                    "input": "Read the file.",
                    "tools": [
                        {
                            "type": "function",
                            "name": "read_file",
                            "parameters": {"type": "object"},
                        }
                    ],
                },
            )
        )
        current = None
        for _ in range(20):
            current = (await client.get("/api/status")).json()["current"]
            if current:
                break
            await asyncio.sleep(0.01)
        assert current
        answer = json.dumps(
            {
                "bridge_version": "1",
                "type": "tool_calls",
                "tool_calls": [
                    {
                        "id": "call_read",
                        "name": "read_file",
                        "arguments": {"path": "README.md"},
                    }
                ],
            }
        )
        await client.post(
            "/api/submit",
            headers={"X-Bridge-UI": "clipboard-bridge"},
            json={"request_id": current["request_id"], "answer": answer},
        )
        response = await task
        body = response.json()
        item = body["output"][0]
        assert item["type"] == "function_call"
        assert item["call_id"] == "call_read"
        assert item["name"] == "read_file"
        assert json.loads(item["arguments"]) == {"path": "README.md"}
        assert body["output_text"] == ""

        response_type = pytest.importorskip("openai.types.responses.response")
        parsed = response_type.Response.model_validate(body)
        assert parsed.output[0].type == "function_call"


@pytest.mark.asyncio
async def test_responses_stream_emits_official_function_call_events(app):
    async with await api_client(app) as client:
        task = asyncio.create_task(
            client.post(
                "/v1/responses",
                headers={"Authorization": "Bearer test-secret"},
                json={
                    "model": "web-ai-clipboard",
                    "input": "Run the command.",
                    "stream": True,
                    "tools": [
                        {
                            "type": "function",
                            "name": "shell",
                            "parameters": {"type": "object"},
                        }
                    ],
                },
            )
        )
        current = None
        for _ in range(20):
            current = (await client.get("/api/status")).json()["current"]
            if current:
                break
            await asyncio.sleep(0.01)
        assert current
        answer = json.dumps(
            {
                "bridge_version": "1",
                "type": "tool_calls",
                "tool_calls": [
                    {
                        "id": "call_shell",
                        "name": "shell",
                        "arguments": {"command": "pwd"},
                    }
                ],
            }
        )
        await client.post(
            "/api/submit",
            headers={"X-Bridge-UI": "clipboard-bridge"},
            json={"request_id": current["request_id"], "answer": answer},
        )
        response = await task
        assert "event: response.function_call_arguments.delta" in response.text
        assert "event: response.function_call_arguments.done" in response.text
        assert "event: response.output_text.delta" not in response.text

        responses_types = pytest.importorskip("openai.types.responses")
        pydantic = pytest.importorskip("pydantic")
        adapter = pydantic.TypeAdapter(responses_types.ResponseStreamEvent)
        events = [
            adapter.validate_python(json.loads(line.removeprefix("data: ")))
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        assert [event.sequence_number for event in events] == list(range(len(events)))
