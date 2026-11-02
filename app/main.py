import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.api.router import api_router
from app.api.routes import health
from app.core.config import Settings, get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging
from app.core.middleware import RequestContextMiddleware
from app.db.base import Base
from app.db.session import engine

logger = logging.getLogger(__name__)

# Multipart overhead allowance on top of the file size limit.
_BODY_OVERHEAD_BYTES = 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    if settings.auto_create_tables:
        logger.warning("AUTO_CREATE_TABLES is enabled; use Alembic migrations in production")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    logger.info(
        "Starting %s v%s (%s)", settings.app_name, settings.app_version, settings.environment
    )
    yield
    await engine.dispose()
    logger.info("Shutdown complete")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        lifespan=lifespan,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url="/redoc" if settings.docs_enabled else None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
    )

    # Middleware added last runs first (outermost).
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
            expose_headers=["X-Request-ID"],
        )
    app.add_middleware(
        RequestContextMiddleware,
        max_body_bytes=settings.max_upload_size_bytes + _BODY_OVERHEAD_BYTES,
    )

    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(api_router, prefix=settings.api_v1_prefix)
    return app


app = create_app()
