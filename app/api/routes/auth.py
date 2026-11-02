import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, status
from fastapi.concurrency import run_in_threadpool
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.api.deps import DbSession, SettingsDep
from app.core.config import Settings
from app.core.exceptions import AuthenticationError, ConflictError
from app.core.security import (
    burn_password_check,
    create_token,
    decode_token,
    hash_password,
    password_needs_rehash,
    verify_password,
)
from app.models import User
from app.schemas.auth import RefreshRequest, TokenPair
from app.schemas.user import UserCreate, UserRead

router = APIRouter(prefix="/auth", tags=["auth"])


def _issue_tokens(user: User, settings: Settings) -> TokenPair:
    subject = str(user.id)
    return TokenPair(
        access_token=create_token(subject, "access", settings),
        refresh_token=create_token(subject, "refresh", settings),
        expires_in=settings.access_token_expire_minutes * 60,
    )


@router.post("/register", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def register(payload: UserCreate, db: DbSession) -> User:
    if await db.scalar(select(User.id).where(User.email == payload.email)):
        raise ConflictError("An account with this email already exists")

    user = User(
        email=payload.email,
        hashed_password=await run_in_threadpool(hash_password, payload.password),
        full_name=payload.full_name,
    )
    db.add(user)
    try:
        await db.commit()
    except IntegrityError as exc:  # concurrent registration with the same email
        await db.rollback()
        raise ConflictError("An account with this email already exists") from exc
    return user


@router.post("/login", response_model=TokenPair)
async def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()], db: DbSession, settings: SettingsDep
) -> TokenPair:
    """OAuth2 password flow. Send the email address as `username`."""
    user = await db.scalar(select(User).where(User.email == form.username.strip().lower()))
    if user is None:
        await run_in_threadpool(burn_password_check, form.password)
        raise AuthenticationError("Incorrect email or password")
    if not await run_in_threadpool(verify_password, form.password, user.hashed_password):
        raise AuthenticationError("Incorrect email or password")
    if not user.is_active:
        raise AuthenticationError("Account is disabled")

    if password_needs_rehash(user.hashed_password):
        user.hashed_password = await run_in_threadpool(hash_password, form.password)
        await db.commit()
    return _issue_tokens(user, settings)


@router.post("/refresh", response_model=TokenPair)
async def refresh(payload: RefreshRequest, db: DbSession, settings: SettingsDep) -> TokenPair:
    subject = decode_token(payload.refresh_token, "refresh", settings)
    try:
        user = await db.get(User, uuid.UUID(subject))
    except ValueError as exc:
        raise AuthenticationError("Invalid token") from exc
    if user is None or not user.is_active:
        raise AuthenticationError("User not found or inactive")
    return _issue_tokens(user, settings)
