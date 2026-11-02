import uuid
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Literal

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

from app.core.config import Settings
from app.core.exceptions import AuthenticationError

TokenType = Literal["access", "refresh"]

_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    try:
        return _hasher.verify(hashed, password)
    except (VerificationError, InvalidHashError):
        return False


def password_needs_rehash(hashed: str) -> bool:
    return _hasher.check_needs_rehash(hashed)


@lru_cache
def _dummy_hash() -> str:
    return _hasher.hash("timing-attack-mitigation")


def burn_password_check(password: str) -> None:
    """Spend the same time as a real verification so unknown emails can't be detected."""
    verify_password(password, _dummy_hash())


def create_token(subject: str, token_type: TokenType, settings: Settings) -> str:
    now = datetime.now(UTC)
    minutes = (
        settings.access_token_expire_minutes
        if token_type == "access"  # noqa: S105
        else settings.refresh_token_expire_minutes
    )
    payload = {
        "sub": subject,
        "type": token_type,
        "iat": now,
        "exp": now + timedelta(minutes=minutes),
        "jti": uuid.uuid4().hex,
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)


def decode_token(token: str, expected_type: TokenType, settings: Settings) -> str:
    """Validate a token and return its subject."""
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.jwt_algorithm],
            options={"require": ["exp", "iat", "sub", "type"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthenticationError("Token has expired") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthenticationError("Invalid token") from exc

    if payload["type"] != expected_type:
        raise AuthenticationError("Invalid token type")
    return payload["sub"]
