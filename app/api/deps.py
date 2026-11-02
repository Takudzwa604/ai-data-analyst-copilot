import uuid
from functools import lru_cache
from typing import Annotated

import pandas as pd
from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.exceptions import AuthenticationError, NotFoundError
from app.core.security import decode_token
from app.db.session import get_db
from app.models import Dataset, User
from app.services.datasets import load_dataframe
from app.services.storage import DataFrameCache, LocalStorage

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{get_settings().api_v1_prefix}/auth/login")


@lru_cache
def get_storage() -> LocalStorage:
    return LocalStorage(get_settings().storage_dir)


@lru_cache
def get_dataframe_cache() -> DataFrameCache:
    return DataFrameCache(get_settings().dataframe_cache_size)


SettingsDep = Annotated[Settings, Depends(get_settings)]
DbSession = Annotated[AsyncSession, Depends(get_db)]
StorageDep = Annotated[LocalStorage, Depends(get_storage)]
CacheDep = Annotated[DataFrameCache, Depends(get_dataframe_cache)]


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)], db: DbSession, settings: SettingsDep
) -> User:
    subject = decode_token(token, "access", settings)
    try:
        user_id = uuid.UUID(subject)
    except ValueError as exc:
        raise AuthenticationError("Invalid token") from exc

    user = await db.get(User, user_id)
    if user is None or not user.is_active:
        raise AuthenticationError("User not found or inactive")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


async def get_owned_dataset(dataset_id: uuid.UUID, user: CurrentUser, db: DbSession) -> Dataset:
    dataset = await db.get(Dataset, dataset_id)
    # 404 rather than 403 so other users' dataset IDs are not disclosed.
    if dataset is None or (dataset.owner_id != user.id and not user.is_superuser):
        raise NotFoundError("Dataset not found")
    return dataset


OwnedDataset = Annotated[Dataset, Depends(get_owned_dataset)]


async def get_dataset_frame(
    dataset: OwnedDataset, storage: StorageDep, cache: CacheDep
) -> pd.DataFrame:
    try:
        return await load_dataframe(storage, cache, dataset)
    except FileNotFoundError as exc:
        raise NotFoundError("Dataset file is missing from storage") from exc


DatasetFrame = Annotated[pd.DataFrame, Depends(get_dataset_frame)]
