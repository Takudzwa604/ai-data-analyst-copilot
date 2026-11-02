import logging
import re
import time
import uuid

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.exceptions import error_body
from app.core.logging import request_id_ctx

logger = logging.getLogger("app.access")

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}


class RequestContextMiddleware:
    """Assigns request IDs, rejects oversized bodies early, adds security headers
    and writes one structured access-log line per request.

    Implemented as pure ASGI (not BaseHTTPMiddleware) so streaming responses and
    background tasks are unaffected.
    """

    def __init__(self, app: ASGIApp, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        incoming_id = headers.get("x-request-id", "")
        request_id = incoming_id if _REQUEST_ID_RE.match(incoming_id) else uuid.uuid4().hex
        request_id_ctx.set(request_id)

        start = time.perf_counter()
        status_code = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                response_headers = MutableHeaders(scope=message)
                response_headers["X-Request-ID"] = request_id
                for name, value in _SECURITY_HEADERS.items():
                    response_headers.setdefault(name, value)
            await send(message)

        try:
            content_length = headers.get("content-length", "")
            if content_length.isdigit() and int(content_length) > self.max_body_bytes:
                response = JSONResponse(
                    error_body("payload_too_large", "Request body exceeds the allowed size"),
                    status_code=413,
                )
                await response(scope, receive, send_wrapper)
                return
            await self.app(scope, receive, send_wrapper)
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "%s %s %d %.1fms",
                scope["method"],
                scope["path"],
                status_code,
                duration_ms,
                extra={
                    "http_method": scope["method"],
                    "http_path": scope["path"],
                    "http_status": status_code,
                    "duration_ms": round(duration_ms, 1),
                    "client_ip": scope["client"][0] if scope.get("client") else None,
                },
            )
