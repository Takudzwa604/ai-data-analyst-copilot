from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from app.api.deps import CurrentUser, DbSession
from app.core.exceptions import PermissionDeniedError
from app.core.security import hash_password, verify_password
from app.models import User
from app.schemas.user import UserRead, UserUpdate

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me", response_model=UserRead)
async def read_me(user: CurrentUser) -> User:
    return user


@router.patch("/me", response_model=UserRead)
async def update_me(payload: UserUpdate, user: CurrentUser, db: DbSession) -> User:
    if "full_name" in payload.model_fields_set:
        user.full_name = payload.full_name
    if payload.password is not None:
        current = payload.current_password or ""
        if not await run_in_threadpool(verify_password, current, user.hashed_password):
            raise PermissionDeniedError("Current password is incorrect")
        user.hashed_password = await run_in_threadpool(hash_password, payload.password)
    await db.commit()
    return user
