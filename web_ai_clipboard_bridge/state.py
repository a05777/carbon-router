from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any


logger = logging.getLogger("clipboard_bridge")


class BridgeStatus(str, Enum):
    IDLE = "IDLE"
    WAITING_FOR_INPUT = "WAITING_FOR_INPUT"
    PROCESSING = "PROCESSING"


class BridgeBusyError(Exception):
    pass


class BridgeTimeoutError(Exception):
    pass


class InvalidSubmissionError(Exception):
    pass


@dataclass(slots=True)
class PendingRequest:
    request_id: str
    model: str
    prompt: str
    messages: list[dict[str, Any]]
    created_unix: int
    started_monotonic: float
    future: asyncio.Future[str]


class BridgeState:
    def __init__(self, timeout_seconds: float) -> None:
        self.timeout_seconds = timeout_seconds
        self._lock = asyncio.Lock()
        self._status = BridgeStatus.IDLE
        self._current: PendingRequest | None = None
        self._history: list[dict[str, Any]] = []

    async def claim(
        self,
        model: str,
        prompt: str,
        messages: list[dict[str, Any]],
        id_prefix: str = "chatcmpl-bridge-",
    ) -> PendingRequest:
        async with self._lock:
            if self._current is not None:
                raise BridgeBusyError

            request_id = f"{id_prefix}{uuid.uuid4().hex[:24]}"
            request = PendingRequest(
                request_id=request_id,
                model=model,
                prompt=prompt,
                messages=messages,
                created_unix=int(time.time()),
                started_monotonic=time.monotonic(),
                future=asyncio.get_running_loop().create_future(),
            )
            self._current = request
            self._status = BridgeStatus.WAITING_FOR_INPUT
            self._history.insert(
                0,
                {
                    "request_id": request_id,
                    "model": model,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "status": "waiting",
                    "duration_seconds": None,
                },
            )
            self._history = self._history[:10]
            logger.info("Request %s is waiting for a clipboard answer", request_id)
            return request

    async def wait_for_answer(self, request: PendingRequest) -> str:
        elapsed = time.monotonic() - request.started_monotonic
        remaining = max(0.0, self.timeout_seconds - elapsed)
        try:
            return await asyncio.wait_for(asyncio.shield(request.future), timeout=remaining)
        except TimeoutError as exc:
            raise BridgeTimeoutError from exc

    async def submit(self, request_id: str, answer: str) -> None:
        async with self._lock:
            if self._current is None:
                raise InvalidSubmissionError("There is no request waiting for an answer.")
            if self._current.request_id != request_id:
                raise InvalidSubmissionError(
                    "This answer belongs to a stale request. Refresh the UI and try again."
                )
            if self._status is not BridgeStatus.WAITING_FOR_INPUT:
                raise InvalidSubmissionError("The current request is already being processed.")

            self._status = BridgeStatus.PROCESSING
            self._current.future.set_result(answer)
            logger.info("Answer submitted for %s", request_id)

    async def release(self, request: PendingRequest, outcome: str) -> None:
        async with self._lock:
            if self._current is None or self._current.request_id != request.request_id:
                return

            duration = round(time.monotonic() - request.started_monotonic, 3)
            if not request.future.done():
                request.future.cancel()
            for item in self._history:
                if item["request_id"] == request.request_id:
                    item["status"] = outcome
                    item["duration_seconds"] = duration
                    break
            self._current = None
            self._status = BridgeStatus.IDLE
            logger.info(
                "Request %s finished with status=%s in %.3fs",
                request.request_id,
                outcome,
                duration,
            )

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            current = self._current
            return {
                "status": self._status.value,
                "ui_status": {
                    BridgeStatus.IDLE: "READY",
                    BridgeStatus.WAITING_FOR_INPUT: "WAITING_FOR_USER",
                    BridgeStatus.PROCESSING: "PROCESSING",
                }[self._status],
                "current": (
                    {
                        "request_id": current.request_id,
                        "model": current.model,
                        "prompt": current.prompt,
                        "messages": current.messages,
                        "created": current.created_unix,
                        "timeout_seconds": self.timeout_seconds,
                    }
                    if current
                    else None
                ),
                "history": [dict(item) for item in self._history],
            }
