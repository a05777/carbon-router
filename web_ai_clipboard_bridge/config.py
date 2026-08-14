from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


@dataclass(frozen=True, slots=True)
class Settings:
    host: str
    port: int
    api_key: str
    timeout_seconds: float
    system_prompt: str = ""

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        host = os.getenv("HOST", "127.0.0.1").strip()
        api_key = os.getenv("GATEWAY_API_KEY", "").strip()

        if host not in LOOPBACK_HOSTS:
            raise ValueError(
                "HOST must be a loopback address (127.0.0.1, localhost, or ::1). "
                "This bridge refuses public bindings."
            )
        if not api_key or api_key == "your-secret-key":
            raise ValueError(
                "Set GATEWAY_API_KEY in .env to a non-placeholder secret before starting."
            )

        try:
            port = int(os.getenv("PORT", "8000"))
        except ValueError as exc:
            raise ValueError("PORT must be an integer.") from exc
        if not 1 <= port <= 65535:
            raise ValueError("PORT must be between 1 and 65535.")

        try:
            timeout_seconds = float(os.getenv("TIMEOUT_SECONDS", "1800"))
        except ValueError as exc:
            raise ValueError("TIMEOUT_SECONDS must be a number.") from exc
        if timeout_seconds <= 0:
            raise ValueError("TIMEOUT_SECONDS must be greater than zero.")

        return cls(
            host=host,
            port=port,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            system_prompt=os.getenv("BRIDGE_SYSTEM_PROMPT", "").strip(),
        )
