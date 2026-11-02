import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.api.deps import DbSession, SettingsDep, StorageDep

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)


@router.get("/health")
async def liveness(settings: SettingsDep) -> dict[str, str]:
    """Process is up. Does not touch dependencies."""
    return {"status": "ok", "version": settings.app_version}


@router.get("/health/ready")
async def readiness(db: DbSession, storage: StorageDep) -> JSONResponse:
    """Dependencies (database, storage) are reachable."""
    checks: dict[str, str] = {}
    try:
        await db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception:
        logger.exception("Readiness check: database unavailable")
        checks["database"] = "unavailable"
    checks["storage"] = "ok" if storage.is_writable() else "unavailable"

    healthy = all(v == "ok" for v in checks.values())
    return JSONResponse(
        {"status": "ok" if healthy else "degraded", "checks": checks},
        status_code=200 if healthy else 503,
    )
