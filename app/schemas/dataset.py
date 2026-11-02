import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ColumnKind = Literal["numeric", "boolean", "datetime", "categorical"]


class ColumnInfo(BaseModel):
    name: str
    dtype: str
    kind: ColumnKind


class DatasetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    owner_id: uuid.UUID
    name: str
    description: str | None
    original_filename: str
    file_format: str
    size_bytes: int
    row_count: int
    column_count: int
    columns: list[ColumnInfo]
    created_at: datetime
    updated_at: datetime


class DatasetUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=2000)


class DatasetPreview(BaseModel):
    columns: list[str]
    rows: list[dict[str, Any]]
    total_rows: int
    offset: int
    limit: int


class NumericStats(BaseModel):
    mean: float | None
    std: float | None
    min: float | None
    p25: float | None
    median: float | None
    p75: float | None
    max: float | None
    sum: float | None


class ValueCount(BaseModel):
    value: Any
    count: int


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    kind: ColumnKind
    null_count: int
    null_pct: float
    unique_count: int
    numeric_stats: NumericStats | None = None
    min_datetime: datetime | None = None
    max_datetime: datetime | None = None
    top_values: list[ValueCount] | None = None


class DatasetProfile(BaseModel):
    row_count: int
    column_count: int
    duplicate_rows: int
    memory_bytes: int
    columns: list[ColumnProfile]
