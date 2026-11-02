from fastapi import APIRouter

from app.ai.schemas import ChatRequest, ChatResponse
from app.ai.service import build_dataset_context
from app.api.deps import AnalystDep, CacheDep, CurrentUser, DbSession, StorageDep
from app.core.exceptions import NotFoundError
from app.services import datasets as dataset_service

router = APIRouter(prefix="/chat", tags=["ai"])


@router.post("", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    user: CurrentUser,
    db: DbSession,
    storage: StorageDep,
    cache: CacheDep,
    analyst: AnalystDep,
) -> ChatResponse:
    """Ask a question about a dataset in plain language.

    Claude chooses from a fixed set of read-only analytics tools; it never writes SQL.
    """
    dataset = await dataset_service.get_dataset_for_user(db, user, payload.dataset_id)
    try:
        df = await dataset_service.load_dataframe(storage, cache, dataset)
    except FileNotFoundError as exc:
        raise NotFoundError("Dataset file is missing from storage") from exc
    return await analyst.answer(
        context=build_dataset_context(dataset), question=payload.question, df=df
    )
