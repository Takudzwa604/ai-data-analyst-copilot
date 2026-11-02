import logging
from typing import Any, ClassVar

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import request_id_ctx

logger = logging.getLogger(__name__)


class AppError(Exception):
    """Base class for errors that map to a well-defined HTTP response."""

    status_code: int = 500
    code: str = "internal_error"
    message: str = "An unexpected error occurred"
    headers: ClassVar[dict[str, str] | None] = None

    def __init__(self, message: str | None = None, *, details: Any = None) -> None:
        self.message = message or self.message
        self.details = details
        super().__init__(self.message)


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"
    message = "Resource not found"


class ConflictError(AppError):
    status_code = 409
    code = "conflict"
    message = "Resource already exists"


class AuthenticationError(AppError):
    status_code = 401
    code = "authentication_failed"
    message = "Could not validate credentials"
    headers: ClassVar[dict[str, str] | None] = {"WWW-Authenticate": "Bearer"}


class PermissionDeniedError(AppError):
    status_code = 403
    code = "permission_denied"
    message = "You do not have permission to perform this action"


class InvalidRequestError(AppError):
    """Request is well-formed but semantically invalid (e.g. unknown column)."""

    status_code = 422
    code = "invalid_request"
    message = "The request could not be processed"


class PayloadTooLargeError(AppError):
    status_code = 413
    code = "payload_too_large"
    message = "Payload too large"


class UnsupportedMediaTypeError(AppError):
    status_code = 415
    code = "unsupported_media_type"
    message = "Unsupported file type"


class AIServiceError(AppError):
    status_code = 502
    code = "ai_service_error"
    message = "The AI analyst could not complete the request"


class AIUnavailableError(AppError):
    status_code = 503
    code = "ai_unavailable"
    message = "The AI analyst is currently unavailable"


def error_body(code: str, message: str, details: Any = None) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message, "request_id": request_id_ctx.get()}
    if details is not None:
        error["details"] = details
    return {"error": error}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            error_body(exc.code, exc.message, exc.details),
            status_code=exc.status_code,
            headers=exc.headers,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            error_body("http_error", str(exc.detail)),
            status_code=exc.status_code,
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = jsonable_encoder(exc.errors(), exclude={"input", "ctx", "url"})
        return JSONResponse(
            error_body("validation_error", "Request validation failed", details),
            status_code=422,
        )

    @app.exception_handler(Exception)
    async def _unhandled_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(error_body(AppError.code, AppError.message), status_code=500)
