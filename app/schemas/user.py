import uuid
from datetime import datetime
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, EmailStr, Field, model_validator

NormalizedEmail = Annotated[EmailStr, AfterValidator(lambda v: v.lower())]
Password = Annotated[str, Field(min_length=8, max_length=128)]


class UserCreate(BaseModel):
    email: NormalizedEmail
    password: Password
    full_name: str | None = Field(default=None, max_length=255)


class UserUpdate(BaseModel):
    full_name: str | None = Field(default=None, max_length=255)
    password: Password | None = None
    current_password: str | None = None

    @model_validator(mode="after")
    def _require_current_password(self) -> "UserUpdate":
        if self.password is not None and not self.current_password:
            raise ValueError("current_password is required to change the password")
        return self


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: EmailStr
    full_name: str | None
    is_active: bool
    is_superuser: bool
    created_at: datetime
