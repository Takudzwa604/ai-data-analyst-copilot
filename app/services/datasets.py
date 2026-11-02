import logging
import uuid
from pathlib import Path

import pandas as pd
from fastapi import UploadFile
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.exceptions import NotFoundError, UnsupportedMediaTypeError
from app.models import Dataset, User
from app.schemas.dataset import DatasetUpdate
from app.services.ingestion import SUPPORTED_EXTENSIONS, ingest_file
from app.services.storage import DataFrameCache, LocalStorage, copy_with_limit

logger = logging.getLogger(__name__)


async def create_dataset(
    db: AsyncSession,
    storage: LocalStorage,
    settings: Settings,
    owner: User,
    upload: UploadFile,
    name: str | None,
    description: str | None,
) -> Dataset:
    filename = Path(upload.filename or "").name
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedMediaTypeError(
            f"Unsupported file type '{extension or 'none'}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    dataset_id = uuid.uuid4()
    storage_key = f"{owner.id}/{dataset_id}.parquet"
    destination = storage.path_for(storage_key)
    temp_path = storage.new_temp_path(extension)

    try:
        size_bytes = await run_in_threadpool(
            copy_with_limit, upload.file, temp_path, settings.max_upload_size_bytes
        )
        result = await run_in_threadpool(ingest_file, temp_path, extension, destination)
    finally:
        temp_path.unlink(missing_ok=True)

    dataset = Dataset(
        id=dataset_id,
        owner_id=owner.id,
        name=(name or Path(filename).stem or "Untitled dataset")[:255],
        description=description,
        original_filename=filename[:512],
        file_format=extension.lstrip("."),
        storage_key=storage_key,
        size_bytes=size_bytes,
        row_count=result.row_count,
        column_count=result.column_count,
        columns=result.columns,
    )
    db.add(dataset)
    try:
        await db.commit()
    except Exception:
        await db.rollback()
        storage.delete(storage_key)
        raise

    logger.info(
        "Dataset created",
        extra={"dataset_id": str(dataset.id), "rows": result.row_count, "bytes": size_bytes},
    )
    return dataset


async def get_dataset_for_user(db: AsyncSession, user: User, dataset_id: uuid.UUID) -> Dataset:
    dataset = await db.get(Dataset, dataset_id)
    # 404 rather than 403 so other users' dataset IDs are not disclosed.
    if dataset is None or (dataset.owner_id != user.id and not user.is_superuser):
        raise NotFoundError("Dataset not found")
    return dataset


async def list_datasets(
    db: AsyncSession, owner: User, *, limit: int, offset: int, search: str | None
) -> tuple[list[Dataset], int]:
    query = select(Dataset).where(Dataset.owner_id == owner.id)
    if search:
        escaped = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        query = query.where(Dataset.name.ilike(f"%{escaped}%", escape="\\"))

    total = await db.scalar(select(func.count()).select_from(query.subquery()))
    rows = await db.scalars(
        query.order_by(Dataset.created_at.desc(), Dataset.id).limit(limit).offset(offset)
    )
    return list(rows), total or 0


async def update_dataset(db: AsyncSession, dataset: Dataset, changes: DatasetUpdate) -> Dataset:
    for field, value in changes.model_dump(exclude_unset=True).items():
        if field == "name" and value is None:
            continue
        setattr(dataset, field, value)
    await db.commit()
    return dataset


async def delete_dataset(
    db: AsyncSession, storage: LocalStorage, cache: DataFrameCache, dataset: Dataset
) -> None:
    path = storage.path_for(dataset.storage_key)
    await db.delete(dataset)
    await db.commit()
    cache.invalidate(path)
    try:
        storage.delete(dataset.storage_key)
    except OSError:
        # The record is gone; an orphaned file is harmless and can be swept later.
        logger.warning("Failed to delete dataset file", extra={"storage_key": dataset.storage_key})


async def load_dataframe(
    storage: LocalStorage, cache: DataFrameCache, dataset: Dataset
) -> pd.DataFrame:
    return await run_in_threadpool(cache.load, storage.path_for(dataset.storage_key))
