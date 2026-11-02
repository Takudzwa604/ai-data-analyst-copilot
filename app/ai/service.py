"""Dataset context: the metadata Claude sees about a dataset before calling any tool."""

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.schemas import ColumnContext, DatasetContext
from app.models import Dataset, User
from app.services.datasets import get_dataset_for_user


def build_dataset_context(dataset: Dataset) -> DatasetContext:
    """Built from stored metadata only; the dataset file is not read."""
    return DatasetContext(
        dataset_name=dataset.name,
        row_count=dataset.row_count,
        columns=[
            ColumnContext(name=column["name"], type=column["kind"], description=None)
            for column in dataset.columns
        ],
    )


async def get_dataset_context(
    db: AsyncSession, user: User, dataset_id: uuid.UUID
) -> DatasetContext:
    """Context for a dataset the user can access; raises NotFoundError otherwise."""
    return build_dataset_context(await get_dataset_for_user(db, user, dataset_id))
