from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class FilterOp(StrEnum):
    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    IS_NULL = "is_null"
    NOT_NULL = "not_null"


VALUELESS_OPS = {FilterOp.IS_NULL, FilterOp.NOT_NULL}
LIST_OPS = {FilterOp.IN, FilterOp.NOT_IN}


class AggFunc(StrEnum):
    COUNT = "count"
    NUNIQUE = "nunique"
    SUM = "sum"
    MEAN = "mean"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    STD = "std"
    VAR = "var"


NUMERIC_ONLY_FUNCS = {AggFunc.SUM, AggFunc.MEAN, AggFunc.MEDIAN, AggFunc.STD, AggFunc.VAR}


class Filter(BaseModel):
    column: str
    op: FilterOp
    value: Any = None

    @model_validator(mode="after")
    def _check_value(self) -> "Filter":
        if self.op in LIST_OPS:
            if not isinstance(self.value, list) or not self.value:
                raise ValueError(f"'{self.op}' requires a non-empty list value")
            if len(self.value) > 1000:
                raise ValueError(f"'{self.op}' accepts at most 1000 values")
        elif self.op not in VALUELESS_OPS and (self.value is None or isinstance(self.value, list)):
            raise ValueError(f"'{self.op}' requires a scalar value")
        return self


class Metric(BaseModel):
    func: AggFunc
    column: str | None = Field(
        default=None, description="Column to aggregate. Omit with func=count to count rows."
    )
    alias: str | None = Field(default=None, min_length=1, max_length=64)

    @model_validator(mode="after")
    def _check_column(self) -> "Metric":
        if self.column is None and self.func != AggFunc.COUNT:
            raise ValueError(f"'{self.func}' requires a column")
        return self

    @property
    def output_name(self) -> str:
        if self.alias:
            return self.alias
        return f"{self.func}_{self.column}" if self.column else "count"


class Sort(BaseModel):
    column: str
    direction: Literal["asc", "desc"] = "asc"


class AggregateRequest(BaseModel):
    filters: list[Filter] = Field(default_factory=list, max_length=50)
    group_by: list[str] = Field(default_factory=list, max_length=10)
    metrics: list[Metric] = Field(
        default_factory=lambda: [Metric(func=AggFunc.COUNT)], min_length=1, max_length=25
    )
    sort: list[Sort] = Field(default_factory=list, max_length=10)
    limit: int | None = Field(default=None, gt=0)


class TabularResult(BaseModel):
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int = Field(description="Number of rows before truncation")
    truncated: bool


class CorrelationRequest(BaseModel):
    columns: list[str] | None = Field(
        default=None,
        min_length=2,
        max_length=50,
        description="Numeric columns to correlate. Defaults to all numeric columns (max 50).",
    )
    method: Literal["pearson", "spearman"] = "pearson"
    filters: list[Filter] = Field(default_factory=list, max_length=50)


class CorrelationResult(BaseModel):
    method: str
    columns: list[str]
    matrix: list[list[float | None]]


class TimeFrequency(StrEnum):
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


class TimeSeriesRequest(BaseModel):
    date_column: str
    value_columns: list[str] = Field(min_length=1, max_length=20)
    frequency: TimeFrequency = TimeFrequency.DAY
    agg: AggFunc = AggFunc.SUM
    filters: list[Filter] = Field(default_factory=list, max_length=50)
