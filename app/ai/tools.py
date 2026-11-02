"""Tool registry: the only operations the AI analyst can perform.

Each tool pairs a typed input model with a handler that delegates to the existing analytics
service. Tools never receive a dataset ID; they operate on the DataFrame the caller already
authorized and loaded, so model output cannot reach other users' data.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import BaseModel, ValidationError

from app.ai.schemas import (
    AggregateDatasetInput,
    CorrelationAnalysisInput,
    PreviewDatasetInput,
    ProfileDatasetInput,
    TimeSeriesAnalysisInput,
)
from app.core.exceptions import InvalidRequestError
from app.services import analytics

# JSON Schema keywords strict tool use does not accept; Pydantic still enforces them.
_UNSUPPORTED_SCHEMA_KEYS = {
    "title",
    "default",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
}


class ToolError(Exception):
    """A tool call that cannot run. The message is returned to Claude so it can retry."""


@dataclass(frozen=True)
class ToolContext:
    df: pd.DataFrame
    max_rows: int


@dataclass(frozen=True)
class ToolOutput:
    payload: dict[str, Any]  # sent back to Claude
    rows: list[dict[str, Any]]  # returned to the API client


Handler = Callable[[Any, ToolContext], ToolOutput]


@dataclass(frozen=True)
class AnalystTool:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: Handler

    def definition(self, *, strict: bool) -> dict[str, Any]:
        definition: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "input_schema": strict_json_schema(self.input_model),
        }
        if strict:
            definition["strict"] = True
        return definition


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic JSON Schema adapted for strict tool use: closed objects, no unsupported keys."""
    return _strictify(model.model_json_schema())


def _strictify(node: Any) -> Any:
    if isinstance(node, list):
        return [_strictify(item) for item in node]
    if not isinstance(node, dict):
        return node
    result: dict[str, Any] = {}
    for key, value in node.items():
        if key in ("properties", "$defs"):
            # Keys here are field/definition names, never schema keywords: keep them all.
            result[key] = {name: _strictify(sub) for name, sub in value.items()}
        elif key not in _UNSUPPORTED_SCHEMA_KEYS:
            result[key] = _strictify(value)
    if result.get("type") == "object" or "properties" in result:
        result.setdefault("properties", {})
        result["additionalProperties"] = False
    return result


class ToolRegistry:
    def __init__(self, tools: Iterable[AnalystTool] = ()) -> None:
        self._tools: dict[str, AnalystTool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: AnalystTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def definitions(self, *, strict: bool = True) -> list[dict[str, Any]]:
        # Sorted for a byte-stable request prefix (prompt caching).
        return [self._tools[name].definition(strict=strict) for name in self.names]

    def parse(self, name: str, raw_input: Any) -> tuple[AnalystTool, BaseModel]:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(f"Unknown tool '{name}'. Available tools: {', '.join(self.names)}")
        try:
            return tool, tool.input_model.model_validate(raw_input)
        except ValidationError as exc:
            raise ToolError(f"Invalid input for '{name}': {_format_errors(exc)}") from exc

    def execute(self, name: str, raw_input: Any, context: ToolContext) -> ToolOutput:
        tool, params = self.parse(name, raw_input)
        try:
            return tool.handler(params, context)
        except InvalidRequestError as exc:
            raise ToolError(exc.message) from exc
        except ValidationError as exc:  # raised converting into analytics request models
            raise ToolError(f"Invalid input for '{name}': {_format_errors(exc)}") from exc


def _format_errors(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors(include_url=False, include_input=False):
        location = ".".join(str(p) for p in error["loc"]) or "input"
        parts.append(f"{location}: {error['msg']}")
    return "; ".join(parts)


# --------------------------------------------------------------------------- handlers


def _profile(_: ProfileDatasetInput, ctx: ToolContext) -> ToolOutput:
    result = analytics.profile(ctx.df)
    return ToolOutput(payload=result, rows=result["columns"])


def _preview(params: PreviewDatasetInput, ctx: ToolContext) -> ToolOutput:
    result = analytics.preview(ctx.df, params.offset, min(params.limit, ctx.max_rows))
    return ToolOutput(payload=result, rows=result["rows"])


def _aggregate(params: AggregateDatasetInput, ctx: ToolContext) -> ToolOutput:
    result = analytics.aggregate(ctx.df, params.to_request(), ctx.max_rows)
    return ToolOutput(payload=result, rows=result["rows"])


def _correlation(params: CorrelationAnalysisInput, ctx: ToolContext) -> ToolOutput:
    result = analytics.correlation(ctx.df, params.to_request())
    columns = result["columns"]
    rows = [
        {"column": column, **dict(zip(columns, values, strict=True))}
        for column, values in zip(columns, result["matrix"], strict=True)
    ]
    return ToolOutput(payload=result, rows=rows)


def _timeseries(params: TimeSeriesAnalysisInput, ctx: ToolContext) -> ToolOutput:
    result = analytics.timeseries(ctx.df, params.to_request(), ctx.max_rows)
    return ToolOutput(payload=result, rows=result["rows"])


def build_default_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            AnalystTool(
                name="profile_dataset",
                description=(
                    "Summarize every column: type, null count and percentage, distinct count, "
                    "numeric stats (mean, std, min, quartiles, max, sum), date ranges and the "
                    "most frequent values. Use for data-quality questions, overviews, or when you "
                    "need actual category values before filtering."
                ),
                input_model=ProfileDatasetInput,
                handler=_profile,
            ),
            AnalystTool(
                name="preview_dataset",
                description=(
                    "Return raw rows. Use only to show example records or inspect value formats; "
                    "never to compute totals or rankings (use aggregate_dataset)."
                ),
                input_model=PreviewDatasetInput,
                handler=_preview,
            ),
            AnalystTool(
                name="aggregate_dataset",
                description=(
                    "Filter rows, group by columns and compute metrics (count, nunique, sum, mean, "
                    "median, min, max, std, var), with sorting and a row limit. Use for totals, "
                    "averages, rankings and top-N questions such as 'which products generate the "
                    "most revenue'. sum/mean/median/std/var need numeric columns."
                ),
                input_model=AggregateDatasetInput,
                handler=_aggregate,
            ),
            AnalystTool(
                name="correlation_analysis",
                description=(
                    "Pairwise correlation matrix (Pearson or Spearman, -1 to 1) between numeric "
                    "columns. Use for questions about relationships between measures."
                ),
                input_model=CorrelationAnalysisInput,
                handler=_correlation,
            ),
            AnalystTool(
                name="time_series_analysis",
                description=(
                    "Aggregate value columns into time buckets (hour, day, week, month, quarter, "
                    "year). Use for trends, growth and seasonality over time."
                ),
                input_model=TimeSeriesAnalysisInput,
                handler=_timeseries,
            ),
        ]
    )


TOOL_REGISTRY = build_default_registry()
