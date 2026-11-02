"""Typed models for the AI analyst: dataset context, tool inputs and the chat API.

Tool input models are what Claude sees as JSON Schema. They are deliberately narrower than
the public analytics request models (typed filter values, small row limits) and convert
into those models, so every analytics rule is still enforced by the existing services.
"""

import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.analytics import (
    AggFunc,
    AggregateRequest,
    CorrelationRequest,
    Filter,
    FilterOp,
    Metric,
    Sort,
    TimeFrequency,
    TimeSeriesRequest,
)
from app.schemas.dataset import ColumnKind

# --------------------------------------------------------------------------- dataset context


class ColumnContext(BaseModel):
    name: str
    type: ColumnKind
    description: str | None = None


class DatasetContext(BaseModel):
    dataset_name: str
    row_count: int
    columns: list[ColumnContext]


# --------------------------------------------------------------------------- tool inputs

FilterScalar = str | int | float | bool


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolFilter(ToolInput):
    column: str = Field(description="Exact column name from the dataset context.")
    op: FilterOp
    value: FilterScalar | list[FilterScalar] | None = Field(
        default=None,
        description=(
            "Comparison value. A list for 'in'/'not_in', omitted for 'is_null'/'not_null'. "
            "Dates are ISO-8601 strings, e.g. '2024-01-31'."
        ),
    )

    def to_filter(self) -> Filter:
        return Filter(column=self.column, op=self.op, value=self.value)


FiltersField = Field(
    default_factory=list,
    max_length=20,
    description="Row filters, combined with AND. Defaults to no filters.",
)


class ProfileDatasetInput(ToolInput):
    """No parameters: profiles every column."""


class PreviewDatasetInput(ToolInput):
    offset: int = Field(default=0, ge=0, description="Rows to skip. Defaults to 0.")
    limit: int = Field(
        default=20, ge=1, le=50, description="Rows to return (1-50). Defaults to 20."
    )


class AggregateDatasetInput(ToolInput):
    filters: list[ToolFilter] = FiltersField
    group_by: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Columns to group by. Empty for a single overall row.",
    )
    metrics: list[Metric] = Field(
        min_length=1,
        max_length=10,
        description="Aggregations to compute. Use func='count' with no column to count rows.",
    )
    sort: list[Sort] = Field(
        default_factory=list,
        max_length=5,
        description="Sort by output columns: group_by columns or metric output names "
        "(alias if given, else '<func>_<column>', or 'count').",
    )
    limit: int = Field(default=25, ge=1, le=100, description="Max rows (1-100). Defaults to 25.")

    def to_request(self) -> AggregateRequest:
        return AggregateRequest(
            filters=[f.to_filter() for f in self.filters],
            group_by=self.group_by,
            metrics=self.metrics,
            sort=self.sort,
            limit=self.limit,
        )


class CorrelationAnalysisInput(ToolInput):
    columns: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Numeric columns to correlate (2-20). Empty means all numeric columns.",
    )
    method: Literal["pearson", "spearman"] = Field(
        default="pearson", description="Defaults to 'pearson'. Use 'spearman' for ranks/outliers."
    )
    filters: list[ToolFilter] = FiltersField

    def to_request(self) -> CorrelationRequest:
        return CorrelationRequest(
            columns=self.columns or None,
            method=self.method,
            filters=[f.to_filter() for f in self.filters],
        )


class TimeSeriesAnalysisInput(ToolInput):
    date_column: str = Field(description="Date/time column (or text column holding dates).")
    value_columns: list[str] = Field(
        min_length=1, max_length=10, description="Columns to aggregate per period."
    )
    frequency: TimeFrequency = Field(
        default=TimeFrequency.MONTH, description="Bucket size. Defaults to 'month'."
    )
    agg: AggFunc = Field(
        default=AggFunc.SUM, description="Aggregation per bucket. Defaults to 'sum'."
    )
    filters: list[ToolFilter] = FiltersField

    def to_request(self) -> TimeSeriesRequest:
        return TimeSeriesRequest(
            date_column=self.date_column,
            value_columns=self.value_columns,
            frequency=self.frequency,
            agg=self.agg,
            filters=[f.to_filter() for f in self.filters],
        )


# --------------------------------------------------------------------------- chat API


class ChatRequest(BaseModel):
    dataset_id: uuid.UUID
    question: str = Field(min_length=1, max_length=2000)


class ToolCallRecord(BaseModel):
    tool: str
    input: dict[str, Any]
    is_error: bool
    error: str | None = None


class ChatResponse(BaseModel):
    answer: str
    tool_used: str | None = Field(description="Last tool that ran successfully, if any.")
    results: list[dict[str, Any]] = Field(description="Rows produced by `tool_used`.")
    tool_calls: list[ToolCallRecord] = Field(description="Every tool call attempted, in order.")
    model: str
