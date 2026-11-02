from typing import Annotated, Any

from fastapi import APIRouter, File, Form, Query, Response, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from app.api.deps import (
    CurrentUser,
    DatasetSource,
    DbSession,
    OwnedDataset,
    SettingsDep,
    StorageDep,
)
from app.models import Dataset
from app.schemas.common import Page
from app.schemas.dataset import DatasetPreview, DatasetProfile, DatasetRead, DatasetUpdate
from app.services import analytics
from app.services import datasets as dataset_service

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("", response_model=DatasetRead, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    user: CurrentUser,
    db: DbSession,
    storage: StorageDep,
    settings: SettingsDep,
    file: Annotated[UploadFile, File(description="CSV, TSV, XLSX, JSON or Parquet file")],
    name: Annotated[str | None, Form(max_length=255)] = None,
    description: Annotated[str | None, Form(max_length=2000)] = None,
) -> Dataset:
    return await dataset_service.create_dataset(
        db, storage, settings, user, file, name, description
    )


@router.get("", response_model=Page[DatasetRead])
async def list_datasets(
    user: CurrentUser,
    db: DbSession,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    search: Annotated[str | None, Query(max_length=255)] = None,
) -> dict[str, Any]:
    items, total = await dataset_service.list_datasets(
        db, user, limit=limit, offset=offset, search=search
    )
    return {"items": items, "total": total, "limit": limit, "offset": offset}


@router.get("/{dataset_id}", response_model=DatasetRead)
async def get_dataset(dataset: OwnedDataset) -> Dataset:
    return dataset


@router.patch("/{dataset_id}", response_model=DatasetRead)
async def update_dataset(dataset: OwnedDataset, payload: DatasetUpdate, db: DbSession) -> Dataset:
    return await dataset_service.update_dataset(db, dataset, payload)


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(dataset: OwnedDataset, db: DbSession, storage: StorageDep) -> Response:
    await dataset_service.delete_dataset(db, storage, dataset)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{dataset_id}/preview", response_model=DatasetPreview)
async def preview_dataset(
    source: DatasetSource,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 50,
) -> dict[str, Any]:
    return await run_in_threadpool(analytics.preview, source, offset, limit)


@router.get("/{dataset_id}/profile", response_model=DatasetProfile)
async def profile_dataset(source: DatasetSource) -> dict[str, Any]:
    return await run_in_threadpool(analytics.profile, source)
