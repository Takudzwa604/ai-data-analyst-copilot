"""Pure, synchronous analytics over pandas DataFrames.

All functions treat their input frame as read-only (it may be shared via the cache)
and are CPU-bound: call them from a worker thread, not the event loop.
User input never reaches `eval`/`query`; columns, operators and functions are whitelisted.
"""

import json
import math
from collections.abc import Iterable
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from app.core.exceptions import InvalidRequestError
from app.schemas.analytics import (
    NUMERIC_ONLY_FUNCS,
    AggFunc,
    AggregateRequest,
    CorrelationRequest,
    Filter,
    FilterOp,
    TimeFrequency,
    TimeSeriesRequest,
)

_FREQUENCY_ALIASES = {
    TimeFrequency.HOUR: "h",
    TimeFrequency.DAY: "D",
    TimeFrequency.WEEK: "W-MON",
    TimeFrequency.MONTH: "MS",
    TimeFrequency.QUARTER: "QS",
    TimeFrequency.YEAR: "YS",
}
_MAX_AUTO_CORRELATION_COLUMNS = 50


# --------------------------------------------------------------------------- helpers


def column_kind(series: pd.Series) -> str:
    if ptypes.is_bool_dtype(series):
        return "boolean"
    if ptypes.is_numeric_dtype(series):
        return "numeric"
    if ptypes.is_datetime64_any_dtype(series):
        return "datetime"
    return "categorical"


def to_json_scalar(value: Any) -> Any:
    """Convert numpy/pandas scalars into JSON-safe Python values."""
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, pd.Timestamp | datetime | date):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def frame_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    # to_json handles NaN/NaT/inf -> null and timestamps -> ISO-8601 in one vectorized pass.
    return json.loads(df.to_json(orient="records", date_format="iso", default_handler=str))


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted({c for c in columns if c not in df.columns})
    if missing:
        raise InvalidRequestError(
            f"Unknown column(s): {', '.join(missing)}", details={"unknown_columns": missing}
        )


def _filter_columns(filters: list[Filter]) -> list[str]:
    return [f.column for f in filters]


# --------------------------------------------------------------------------- filtering


def _coerce_value(series: pd.Series, value: Any) -> Any:
    if isinstance(value, list):
        return [_coerce_value(series, v) for v in value]
    if value is None:
        return None
    kind = column_kind(series)
    try:
        if kind == "datetime":
            ts = pd.Timestamp(value)
            tz = series.dt.tz
            if tz is not None and ts.tzinfo is None:
                ts = ts.tz_localize(tz)
            elif tz is None and ts.tzinfo is not None:
                ts = ts.tz_convert(None)
            return ts
        if kind == "numeric" and isinstance(value, str):
            return float(value)
    except (ValueError, TypeError) as exc:
        raise InvalidRequestError(
            f"Value {value!r} is not valid for {kind} column '{series.name}'"
        ) from exc
    return value


def _condition(series: pd.Series, flt: Filter) -> pd.Series:
    op = flt.op
    if op == FilterOp.IS_NULL:
        return series.isna()
    if op == FilterOp.NOT_NULL:
        return series.notna()
    if op == FilterOp.CONTAINS:
        return (
            series.astype("string")
            .str.contains(str(flt.value), case=False, regex=False)
            .fillna(False)
            .astype(bool)
        )

    value = _coerce_value(series, flt.value)
    try:
        match op:
            case FilterOp.EQ:
                result = series == value
            case FilterOp.NE:
                result = series != value
            case FilterOp.GT:
                result = series > value
            case FilterOp.GTE:
                result = series >= value
            case FilterOp.LT:
                result = series < value
            case FilterOp.LTE:
                result = series <= value
            case FilterOp.IN:
                result = series.isin(value)
            case FilterOp.NOT_IN:
                result = ~series.isin(value)
            case _:  # pragma: no cover - exhaustive over FilterOp
                raise InvalidRequestError(f"Unsupported operator: {op}")
    except TypeError as exc:
        raise InvalidRequestError(
            f"Operator '{op}' cannot compare column '{flt.column}' with {flt.value!r}"
        ) from exc
    return result.fillna(False).astype(bool)


def apply_filters(df: pd.DataFrame, filters: list[Filter]) -> pd.DataFrame:
    if not filters:
        return df
    require_columns(df, _filter_columns(filters))
    mask = pd.Series(True, index=df.index)
    for flt in filters:
        mask &= _condition(df[flt.column], flt)
    return df.loc[mask]


# --------------------------------------------------------------------------- operations


def preview(df: pd.DataFrame, offset: int, limit: int) -> dict[str, Any]:
    return {
        "columns": list(df.columns),
        "rows": frame_to_records(df.iloc[offset : offset + limit]),
        "total_rows": len(df),
        "offset": offset,
        "limit": limit,
    }


def profile(df: pd.DataFrame, top_n: int = 5) -> dict[str, Any]:
    row_count = len(df)
    columns = []
    for name in df.columns:
        series = df[name]
        kind = column_kind(series)
        null_count = int(series.isna().sum())
        non_null = series.dropna()
        entry: dict[str, Any] = {
            "name": name,
            "dtype": str(series.dtype),
            "kind": kind,
            "null_count": null_count,
            "null_pct": round(null_count / row_count * 100, 2) if row_count else 0.0,
            "unique_count": int(non_null.nunique()),
        }
        if kind == "numeric" and not non_null.empty:
            numeric = non_null.astype("float64")
            quartiles = numeric.quantile([0.25, 0.5, 0.75])
            entry["numeric_stats"] = {
                "mean": to_json_scalar(numeric.mean()),
                "std": to_json_scalar(numeric.std()),
                "min": to_json_scalar(numeric.min()),
                "p25": to_json_scalar(quartiles.loc[0.25]),
                "median": to_json_scalar(quartiles.loc[0.5]),
                "p75": to_json_scalar(quartiles.loc[0.75]),
                "max": to_json_scalar(numeric.max()),
                "sum": to_json_scalar(numeric.sum()),
            }
        elif kind == "datetime" and not non_null.empty:
            entry["min_datetime"] = to_json_scalar(non_null.min())
            entry["max_datetime"] = to_json_scalar(non_null.max())
        if kind in {"categorical", "boolean"}:
            counts = non_null.value_counts().head(top_n)
            entry["top_values"] = [
                {"value": to_json_scalar(value), "count": int(count)}
                for value, count in counts.items()
            ]
        columns.append(entry)

    return {
        "row_count": row_count,
        "column_count": df.shape[1],
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_bytes": int(df.memory_usage(deep=True).sum()),
        "columns": columns,
    }


def _check_metric_types(df: pd.DataFrame, column: str, func: AggFunc) -> None:
    if func in NUMERIC_ONLY_FUNCS and column_kind(df[column]) not in {"numeric", "boolean"}:
        raise InvalidRequestError(f"'{func}' requires a numeric column; '{column}' is not numeric")


def _tabular_result(result: pd.DataFrame, limit: int) -> dict[str, Any]:
    return {
        "columns": [str(c) for c in result.columns],
        "rows": frame_to_records(result.head(limit)),
        "row_count": len(result),
        "truncated": len(result) > limit,
    }


def aggregate(df: pd.DataFrame, request: AggregateRequest, max_rows: int) -> dict[str, Any]:
    metric_columns = [m.column for m in request.metrics if m.column]
    require_columns(df, [*request.group_by, *metric_columns])
    if len(set(request.group_by)) != len(request.group_by):
        raise InvalidRequestError("group_by contains duplicate columns")

    output_names = [m.output_name for m in request.metrics]
    duplicates = {n for n in output_names if output_names.count(n) > 1 or n in request.group_by}
    if duplicates:
        raise InvalidRequestError(
            f"Duplicate output column name(s): {', '.join(sorted(duplicates))}. Use 'alias'."
        )
    for metric in request.metrics:
        if metric.column:
            _check_metric_types(df, metric.column, metric.func)

    data = apply_filters(df, request.filters)

    if request.group_by:
        grouped = data.groupby(request.group_by, dropna=False, sort=False, observed=True)
        parts = [
            (grouped.size() if m.column is None else grouped[m.column].agg(str(m.func))).rename(
                m.output_name
            )
            for m in request.metrics
        ]
        result = pd.concat(parts, axis=1).reset_index()
    else:
        row = {
            m.output_name: len(data) if m.column is None else data[m.column].agg(str(m.func))
            for m in request.metrics
        }
        result = pd.DataFrame([{k: to_json_scalar(v) for k, v in row.items()}])

    if request.sort:
        require_columns(result, [s.column for s in request.sort])
        result = result.sort_values(
            by=[s.column for s in request.sort],
            ascending=[s.direction == "asc" for s in request.sort],
            na_position="last",
            kind="stable",
        )

    limit = min(request.limit or max_rows, max_rows)
    return _tabular_result(result, limit)


def correlation(df: pd.DataFrame, request: CorrelationRequest) -> dict[str, Any]:
    if request.columns is not None:
        require_columns(df, request.columns)
        non_numeric = [c for c in request.columns if column_kind(df[c]) != "numeric"]
        if non_numeric:
            raise InvalidRequestError(
                f"Correlation requires numeric columns; not numeric: {', '.join(non_numeric)}"
            )
        columns = list(dict.fromkeys(request.columns))
    else:
        columns = [c for c in df.columns if column_kind(df[c]) == "numeric"]
        columns = columns[:_MAX_AUTO_CORRELATION_COLUMNS]

    if len(columns) < 2:
        raise InvalidRequestError("Correlation requires at least two numeric columns")

    data = apply_filters(df, request.filters)
    matrix = data[columns].astype("float64").corr(method=request.method)
    return {
        "method": request.method,
        "columns": columns,
        "matrix": [[to_json_scalar(v) for v in row] for row in matrix.to_numpy()],
    }


def timeseries(df: pd.DataFrame, request: TimeSeriesRequest, max_rows: int) -> dict[str, Any]:
    require_columns(df, [request.date_column, *request.value_columns])
    if request.date_column in request.value_columns:
        raise InvalidRequestError("date_column cannot also be a value column")
    for column in request.value_columns:
        _check_metric_types(df, column, request.agg)

    data = apply_filters(df, request.filters)
    dates = data[request.date_column]
    kind = column_kind(dates)
    if kind == "categorical":
        dates = pd.to_datetime(dates, errors="coerce", format="mixed", utc=True)
    elif kind != "datetime":
        raise InvalidRequestError(
            f"Column '{request.date_column}' is {kind}, not a date/time column"
        )

    frame = data[request.value_columns].set_axis(pd.DatetimeIndex(dates), axis=0)
    frame = frame.loc[frame.index.notna()].sort_index()
    if frame.empty:
        return _tabular_result(pd.DataFrame(columns=["period", *request.value_columns]), max_rows)

    result = frame.resample(_FREQUENCY_ALIASES[request.frequency]).agg(str(request.agg))
    result.index.name = "period"
    return _tabular_result(result.reset_index(), max_rows)
