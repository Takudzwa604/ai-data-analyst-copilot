from typing import Any

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from app.api.deps import DatasetSource, SettingsDep
from app.schemas.analytics import (
    AggregateRequest,
    CorrelationRequest,
    CorrelationResult,
    TabularResult,
    TimeSeriesRequest,
)
from app.services import analytics

router = APIRouter(prefix="/datasets/{dataset_id}/analytics", tags=["analytics"])


@router.post("/aggregate", response_model=TabularResult)
async def aggregate(
    request: AggregateRequest, source: DatasetSource, settings: SettingsDep
) -> dict[str, Any]:
    """Filter, group and aggregate rows (a safe, structured `SELECT ... GROUP BY`)."""
    return await run_in_threadpool(analytics.aggregate, source, request, settings.max_result_rows)


@router.post("/correlation", response_model=CorrelationResult)
async def correlation(request: CorrelationRequest, source: DatasetSource) -> dict[str, Any]:
    """Pairwise correlation matrix between numeric columns."""
    return await run_in_threadpool(analytics.correlation, source, request)


@router.post("/timeseries", response_model=TabularResult)
async def timeseries(
    request: TimeSeriesRequest, source: DatasetSource, settings: SettingsDep
) -> dict[str, Any]:
    """Resample value columns into fixed time buckets."""
    return await run_in_threadpool(analytics.timeseries, source, request, settings.max_result_rows)
