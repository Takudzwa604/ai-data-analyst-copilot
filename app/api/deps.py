import logging
import uuid
from functools import lru_cache
from typing import Annotated

import anthropic
import pandas as pd
from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.analyst import AnalystService
from app.core.config import Settings, get_settings
from app.core.exceptions import AIUnavailableError, AuthenticationError, NotFoundError
from app.core.security import decode_token
from app.db.session import get_db
from app.models import Dataset, User
from app.services.datasets import get_dataset_for_user, load_dataframe
from app.services.storage import DataFrameCache, LocalStorage

logger = logging.getLogger(__name__)

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
    return await get_dataset_for_user(db, user, dataset_id)


OwnedDataset = Annotated[Dataset, Depends(get_owned_dataset)]


async def get_dataset_frame(
    dataset: OwnedDataset, storage: StorageDep, cache: CacheDep
) -> pd.DataFrame:
    try:
        return await load_dataframe(storage, cache, dataset)
    except FileNotFoundError as exc:
        raise NotFoundError("Dataset file is missing from storage") from exc


DatasetFrame = Annotated[pd.DataFrame, Depends(get_dataset_frame)]


@lru_cache
def _build_anthropic_client() -> anthropic.AsyncAnthropic:
    settings = get_settings()
    # An empty value means "unset" so the SDK can fall back to other credential sources.
    secret = settings.anthropic_api_key
    api_key = (secret.get_secret_value() if secret else "") or None
    return anthropic.AsyncAnthropic(
        api_key=api_key, timeout=settings.ai_timeout_seconds, max_retries=2
    )


def get_anthropic_client() -> anthropic.AsyncAnthropic:
    # Failures are not cached by lru_cache, so a fixed configuration is picked up on retry.
    try:
        return _build_anthropic_client()
    except anthropic.AnthropicError as exc:
        logger.error("Anthropic client unavailable: %s", exc)
        raise AIUnavailableError("The AI analyst is not configured") from exc


def get_analyst_service(
    settings: SettingsDep,
    client: Annotated[anthropic.AsyncAnthropic, Depends(get_anthropic_client)],
) -> AnalystService:
    return AnalystService(client, settings)


AnalystDep = Annotated[AnalystService, Depends(get_analyst_service)]
