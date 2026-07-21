from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Callable
from typing import Any

from fastapi import Request, Response

from app.config import get_settings


class JsonFormatter(logging.Formatter):
    """Small JSON-lines formatter for container logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in getattr(record, "structured", {}).items():
            payload[key] = value
        return json.dumps(payload, default=str, sort_keys=True)


def configure_logging() -> None:
    """Configure process-wide logging once for local and container runtime."""

    root = logging.getLogger()
    if getattr(root, "_agentic_logging_configured", False):
        return

    settings = get_settings()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(settings.log_level.upper())
    setattr(root, "_agentic_logging_configured", True)


async def request_logging_middleware(
    request: Request,
    call_next: Callable[[Request], Any],
) -> Response:
    """Log one compact record per HTTP request."""

    logger = logging.getLogger("app.api")
    started_at = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        latency_ms = (time.perf_counter() - started_at) * 1000
        logger.exception(
            "request_failed",
            extra={
                "structured": {
                    "method": request.method,
                    "path": request.url.path,
                    "latency_ms": round(latency_ms, 2),
                }
            },
        )
        raise

    latency_ms = (time.perf_counter() - started_at) * 1000
    logger.info(
        "request_completed",
        extra={
            "structured": {
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": round(latency_ms, 2),
            }
        },
    )
    return response
