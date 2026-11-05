from fastapi import APIRouter

from app.ai.schemas import ChatRequest, ChatResponse
from app.ai.service import build_dataset_context
from app.api.deps import AnalystDep, CurrentUser, DbSession, StorageDep
from app.services import datasets as dataset_service

router = APIRouter(prefix="/chat", tags=["ai"])


@router.post("", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    user: CurrentUser,
    db: DbSession,
    storage: StorageDep,
    analyst: AnalystDep,
) -> ChatResponse:
    """Ask a question about a dataset in plain language.

    Claude chooses from a fixed set of read-only analytics tools; it never writes SQL.
    With AI_DEMO_MODE=true a keyword matcher chooses instead, with no Claude API calls.
    """
    dataset = await dataset_service.get_dataset_for_user(db, user, payload.dataset_id)
    source = dataset_service.dataset_path(storage, dataset)
    return await analyst.answer(
        context=build_dataset_context(dataset), question=payload.question, source=source
    )
