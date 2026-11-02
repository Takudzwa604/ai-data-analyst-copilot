"""Analytics operations, executed by DuckDB.

Each operation takes a data source (a Parquet path, or a DataFrame) and is CPU-bound: call it
from a worker thread. Requests are validated against the source's schema, then compiled to SQL
with quoted identifiers and bound parameters, so user input is never interpolated into SQL.
"""

from collections.abc import Sequence
from itertools import batched
from pathlib import Path
from typing import Any

import pandas as pd

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
from app.services.duckdb_engine import (
    SOURCE,
    Column,
    DataSource,
    Query,
    open_source,
    quote_ident,
    to_json_scalar,
)

__all__ = [
    "aggregate",
    "apply_filters",
    "correlation",
    "preview",
    "profile",
    "timeseries",
    "to_json_scalar",
]

# Buckets are labelled by their start (weeks start on Monday).
_TIME_BUCKETS = {
    TimeFrequency.HOUR: ("hour", "INTERVAL 1 HOUR"),
    TimeFrequency.DAY: ("day", "INTERVAL 1 DAY"),
    TimeFrequency.WEEK: ("week", "INTERVAL 7 DAY"),
    TimeFrequency.MONTH: ("month", "INTERVAL 1 MONTH"),
    TimeFrequency.QUARTER: ("quarter", "INTERVAL 3 MONTH"),
    TimeFrequency.YEAR: ("year", "INTERVAL 1 YEAR"),
}
# Empty buckets read as 0 for additive aggregations (as pandas resample did), null otherwise.
_ZERO_WHEN_EMPTY = {AggFunc.COUNT, AggFunc.NUNIQUE, AggFunc.SUM}
_PERIOD = "__dap_period"
_MAX_AUTO_CORRELATION_COLUMNS = 50
_PROFILE_BATCH_COLUMNS = 50
_TOTAL = "__dap_total"
_ORDERING_OPS = {FilterOp.GT, FilterOp.GTE, FilterOp.LT, FilterOp.LTE}
_COMPARISON_SQL = {
    FilterOp.EQ: "=",
    FilterOp.NE: "IS DISTINCT FROM",  # nulls count as "not equal", as before
    FilterOp.GT: ">",
    FilterOp.GTE: ">=",
    FilterOp.LT: "<",
    FilterOp.LTE: "<=",
}
_AGG_SQL = {
    AggFunc.COUNT: "count({})",
    AggFunc.NUNIQUE: "count(DISTINCT {})",
    AggFunc.SUM: "coalesce(sum({}), 0)",
    AggFunc.MEAN: "avg({})",
    AggFunc.MEDIAN: "median({})",
    AggFunc.MIN: "min({})",
    AggFunc.MAX: "max({})",
    AggFunc.STD: "stddev_samp({})",
    AggFunc.VAR: "var_samp({})",
}


# --------------------------------------------------------------------------- SQL building


def _coerce_scalar(column: Column, value: Any, op: FilterOp) -> Any:
    """Convert a filter value to the column's type before binding it."""
    if value is None:
        return None
    try:
        if column.kind == "datetime":
            ts = pd.Timestamp(value)
            if "TIME ZONE" in column.type.upper():
                ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
            elif ts.tzinfo is not None:
                ts = ts.tz_convert(None)
            return ts.date() if column.type.upper() == "DATE" else ts.to_pydatetime()
        if column.kind == "numeric":
            if isinstance(value, bool):
                return int(value)
            return float(value) if isinstance(value, str) else value
        if column.kind == "categorical" and not isinstance(value, str):
            if op in _ORDERING_OPS:
                raise TypeError("ordering comparison on a text column")
            return str(value)
    except (ValueError, TypeError) as exc:
        raise InvalidRequestError(
            f"Value {value!r} is not valid for {column.kind} column '{column.name}'"
        ) from exc
    return value


def _condition(column: Column, flt: Filter) -> tuple[str, list[Any]]:
    ident = column.ident
    if flt.op == FilterOp.IS_NULL:
        return f"{ident} IS NULL", []
    if flt.op == FilterOp.NOT_NULL:
        return f"{ident} IS NOT NULL", []
    if flt.op == FilterOp.CONTAINS:
        # strpos, not LIKE: the value is matched literally, no wildcard escaping needed.
        sql = f"coalesce(strpos(lower(CAST({ident} AS VARCHAR)), lower(?)) > 0, false)"
        return sql, [str(flt.value)]
    if flt.op in (FilterOp.IN, FilterOp.NOT_IN):
        values = [_coerce_scalar(column, v, flt.op) for v in flt.value]
        membership = f"coalesce({ident} IN ({', '.join(['?'] * len(values))}), false)"
        return (membership if flt.op == FilterOp.IN else f"NOT {membership}"), values
    value = _coerce_scalar(column, flt.value, flt.op)
    return f"{ident} {_COMPARISON_SQL[flt.op]} ?", [value]


def _where(query: Query, filters: list[Filter], extra: Sequence[str] = ()) -> tuple[str, list[Any]]:
    query.require([f.column for f in filters])
    clauses, params = list(extra), []
    for flt in filters:
        clause, values = _condition(query.column(flt.column), flt)
        clauses.append(clause)
        params.extend(values)
    return (f"WHERE {' AND '.join(clauses)}" if clauses else ""), params


def _check_metric_kind(query: Query, column: str, func: AggFunc) -> None:
    if func in NUMERIC_ONLY_FUNCS and query.column(column).kind not in {"numeric", "boolean"}:
        raise InvalidRequestError(f"'{func}' requires a numeric column; '{column}' is not numeric")


def _agg_expr(query: Query, column: str | None, func: AggFunc) -> str:
    if column is None:
        return "count(*)"
    col = query.column(column)
    ident = col.ident
    if func in NUMERIC_ONLY_FUNCS and col.kind == "boolean":
        ident = f"CAST({ident} AS INTEGER)"
    return _AGG_SQL[func].format(ident)


def _limited_result(
    query: Query,
    inner_sql: str,
    params: list[Any],
    order_by: str,
    columns: list[str],
    limit: int,
) -> dict[str, Any]:
    """Run an unlimited result query; return up to `limit` rows plus the total row count."""
    rows = query.records(
        f"SELECT *, count(*) OVER () AS {_TOTAL} FROM ({inner_sql}) AS result {order_by} LIMIT ?",
        [*params, limit],
    )
    total = rows[0][_TOTAL] if rows else 0
    for row in rows:
        del row[_TOTAL]
    return {"columns": columns, "rows": rows, "row_count": total, "truncated": total > limit}


# --------------------------------------------------------------------------- filtering


def apply_filters(source: DataSource, filters: list[Filter]) -> pd.DataFrame:
    """Rows matching every filter, materialized as a DataFrame. Meant for small results."""
    if not filters and isinstance(source, pd.DataFrame):
        return source
    with open_source(source) as query:
        where, params = _where(query, filters)
        names, rows = query.fetch(f"SELECT * FROM {SOURCE} {where}", params)
    return pd.DataFrame.from_records(rows, columns=names)


# --------------------------------------------------------------------------- operations


def preview(source: DataSource, offset: int, limit: int) -> dict[str, Any]:
    with open_source(source) as query:
        total = query.one(f"SELECT count(*) FROM {SOURCE}")[0]
        rows = query.records(f"SELECT * FROM {SOURCE} LIMIT ? OFFSET ?", [limit, offset])
        columns = list(query.columns)
    return {"columns": columns, "rows": rows, "total_rows": total, "offset": offset, "limit": limit}


def _column_stat_exprs(column: Column) -> list[str]:
    # Only constant-memory aggregates here. Distinct counts and exact quartiles must hold a
    # column's values, so _column_profile computes them one column at a time.
    exprs = [f"count({column.ident})"]
    if column.kind == "numeric":
        x = f"CAST({column.ident} AS DOUBLE)"
        exprs += [
            f"avg({x})",
            f"stddev_samp({x})",
            f"min({x})",
            f"max({x})",
            f"sum({x})",
        ]
    elif column.kind == "datetime":
        exprs += [f"min({column.ident})", f"max({column.ident})"]
    return exprs


def _column_profile(query: Query, column: Column, values: Any, row_count: int, top_n: int) -> dict:
    non_null = next(values)
    null_count = row_count - non_null
    entry: dict[str, Any] = {
        "name": column.name,
        "dtype": column.type,
        "kind": column.kind,
        "null_count": null_count,
        "null_pct": round(null_count / row_count * 100, 2) if row_count else 0.0,
    }
    if column.kind == "numeric":
        mean, std, low, high, total = (next(values) for _ in range(5))
        distinct, quartiles = query.one(
            f"SELECT count(DISTINCT {column.ident}), "
            f"quantile_cont(CAST({column.ident} AS DOUBLE), [0.25, 0.5, 0.75]) FROM {SOURCE}"
        )
        entry["unique_count"] = distinct
        if non_null:
            p25, median, p75 = quartiles
            entry["numeric_stats"] = {
                key: to_json_scalar(value)
                for key, value in {
                    "mean": mean,
                    "std": std,
                    "min": low,
                    "p25": p25,
                    "median": median,
                    "p75": p75,
                    "max": high,
                    "sum": total,
                }.items()
            }
    elif column.kind == "datetime":
        low, high = next(values), next(values)
        if non_null:
            entry["min_datetime"] = to_json_scalar(low)
            entry["max_datetime"] = to_json_scalar(high)

    if column.kind in {"categorical", "boolean"}:
        # One grouped scan of this column yields both the top values and the distinct count.
        _, rows = query.fetch(
            f"SELECT {column.ident}, count(*), count(*) OVER () FROM {SOURCE} "
            f"WHERE {column.ident} IS NOT NULL GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT ?",
            [top_n],
        )
        entry["unique_count"] = rows[0][2] if rows else 0
        entry["top_values"] = [
            {"value": to_json_scalar(value), "count": count} for value, count, _ in rows
        ]
    elif column.kind == "datetime":
        sql = f"SELECT count(DISTINCT {column.ident}) FROM {SOURCE}"
        entry["unique_count"] = query.one(sql)[0]
    return entry


def profile(source: DataSource, top_n: int = 5) -> dict[str, Any]:
    with open_source(source) as query:
        columns = list(query.columns.values())
        row_count = query.one(f"SELECT count(*) FROM {SOURCE}")[0]
        profiles = []
        # One scan per batch of columns computes the counts and statistics for the batch.
        for batch in batched(columns, _PROFILE_BATCH_COLUMNS):
            exprs = [e for column in batch for e in _column_stat_exprs(column)]
            values = iter(query.one(f"SELECT {', '.join(exprs)} FROM {SOURCE}"))
            profiles += [_column_profile(query, c, values, row_count, top_n) for c in batch]
        # Distinct 64-bit row hashes rather than DISTINCT *, which would hold every row in
        # memory. A collision (two different rows counted as duplicates) is ~1e-6 at 10M rows.
        row_hash = f"hash({', '.join(c.ident for c in columns)})"
        distinct_rows = query.one(f"SELECT count(DISTINCT {row_hash}) FROM {SOURCE}")[0]

    return {
        "row_count": row_count,
        "column_count": len(profiles),
        "duplicate_rows": row_count - distinct_rows,
        "memory_bytes": None,
        "storage_bytes": source.stat().st_size if isinstance(source, Path) else None,
        "columns": profiles,
    }


def aggregate(source: DataSource, request: AggregateRequest, max_rows: int) -> dict[str, Any]:
    if len(set(request.group_by)) != len(request.group_by):
        raise InvalidRequestError("group_by contains duplicate columns")
    output_names = [m.output_name for m in request.metrics]
    duplicates = {n for n in output_names if output_names.count(n) > 1 or n in request.group_by}
    if duplicates:
        raise InvalidRequestError(
            f"Duplicate output column name(s): {', '.join(sorted(duplicates))}. Use 'alias'."
        )
    output_columns = [*request.group_by, *output_names]
    unknown_sort = sorted({s.column for s in request.sort} - set(output_columns))
    if unknown_sort:
        raise InvalidRequestError(f"Unknown sort column(s): {', '.join(unknown_sort)}")

    with open_source(source) as query:
        query.require([*request.group_by, *(m.column for m in request.metrics if m.column)])
        for metric in request.metrics:
            if metric.column:
                _check_metric_kind(query, metric.column, metric.func)

        where, params = _where(query, request.filters)
        groups = [query.column(c).ident for c in request.group_by]
        selects = [
            *groups,
            *(
                f"{_agg_expr(query, m.column, m.func)} AS {quote_ident(m.output_name)}"
                for m in request.metrics
            ),
        ]
        group_by = f"GROUP BY {', '.join(groups)}" if groups else ""
        inner = f"SELECT {', '.join(selects)} FROM {SOURCE} {where} {group_by}"

        # Explicit sort first, then group keys, so results (and LIMIT) are deterministic.
        order_terms = [
            f"{quote_ident(s.column)} {s.direction.upper()} NULLS LAST" for s in request.sort
        ]
        order_terms += [f"{quote_ident(c)} ASC NULLS LAST" for c in request.group_by]
        order_by = f"ORDER BY {', '.join(order_terms)}" if order_terms else ""

        limit = min(request.limit or max_rows, max_rows)
        return _limited_result(query, inner, params, order_by, output_columns, limit)


def _average_rank(ident: str) -> str:
    """Rank with ties sharing their average rank (pandas' default for Spearman)."""
    return f"(rank() OVER (ORDER BY {ident}) + (count(*) OVER (PARTITION BY {ident}) - 1) / 2.0)"


def _correlation_columns(query: Query, request: CorrelationRequest) -> list[Column]:
    if request.columns is None:
        numeric = [c for c in query.columns.values() if c.kind == "numeric"]
        columns = numeric[:_MAX_AUTO_CORRELATION_COLUMNS]
    else:
        query.require(request.columns)
        columns = [query.column(name) for name in dict.fromkeys(request.columns)]
        non_numeric = [c.name for c in columns if c.kind != "numeric"]
        if non_numeric:
            raise InvalidRequestError(
                f"Correlation requires numeric columns; not numeric: {', '.join(non_numeric)}"
            )
    if len(columns) < 2:
        raise InvalidRequestError("Correlation requires at least two numeric columns")
    return columns


def _pearson(query: Query, columns: list[Column], where: str, params: list[Any]) -> dict:
    """Every pair (including the diagonal) in a single scan."""
    exprs = [
        f"corr(CAST({a.ident} AS DOUBLE), CAST({b.ident} AS DOUBLE))"
        for i, a in enumerate(columns)
        for b in columns[i:]
    ]
    values = iter(query.one(f"SELECT {', '.join(exprs)} FROM {SOURCE} {where}", params))
    return {(i, j): next(values) for i in range(len(columns)) for j in range(i, len(columns))}


def _spearman(query: Query, columns: list[Column], where: str, params: list[Any]) -> dict:
    counts = query.one(
        f"SELECT count(*), {', '.join(f'count({c.ident})' for c in columns)} FROM {SOURCE} {where}",
        params,
    )
    if all(count == counts[0] for count in counts[1:]):
        # No missing values: rank each column once, then correlate the ranks.
        ranked = ", ".join(f"{_average_rank(c.ident)} AS {c.ident}" for c in columns)
        inner = f"(SELECT {ranked} FROM {SOURCE} {where}) AS ranked"
        exprs = [f"corr({a.ident}, {b.ident})" for i, a in enumerate(columns) for b in columns[i:]]
        values = iter(query.one(f"SELECT {', '.join(exprs)} FROM {inner}", params))
        return {(i, j): next(values) for i in range(len(columns)) for j in range(i, len(columns))}

    # Missing values: rank each pair over the rows where both are present (pandas semantics).
    result = {}
    for i, a in enumerate(columns):
        distinct = query.one(f"SELECT count(DISTINCT {a.ident}) FROM {SOURCE} {where}", params)[0]
        result[(i, i)] = 1.0 if distinct > 1 else None
        for j in range(i + 1, len(columns)):
            b = columns[j]
            pair_where = f"{where} AND" if where else "WHERE"
            sql = f"""
                SELECT corr(ra, rb) FROM (
                    SELECT {_average_rank(a.ident)} AS ra, {_average_rank(b.ident)} AS rb
                    FROM {SOURCE}
                    {pair_where} {a.ident} IS NOT NULL AND {b.ident} IS NOT NULL
                )
            """
            result[(i, j)] = query.one(sql, params)[0]
    return result


def correlation(source: DataSource, request: CorrelationRequest) -> dict[str, Any]:
    with open_source(source) as query:
        columns = _correlation_columns(query, request)
        where, params = _where(query, request.filters)
        compute = _pearson if request.method == "pearson" else _spearman
        pairs = compute(query, columns, where, params)

    n = len(columns)
    matrix = [[to_json_scalar(pairs[(min(i, j), max(i, j))]) for j in range(n)] for i in range(n)]
    return {"method": request.method, "columns": [c.name for c in columns], "matrix": matrix}


def timeseries(source: DataSource, request: TimeSeriesRequest, max_rows: int) -> dict[str, Any]:
    if request.date_column in request.value_columns:
        raise InvalidRequestError("date_column cannot also be a value column")

    with open_source(source) as query:
        query.require([request.date_column, *request.value_columns])
        for column in request.value_columns:
            _check_metric_kind(query, column, request.agg)

        date_col = query.column(request.date_column)
        if date_col.kind == "categorical":
            # Text dates: ISO-8601 strings parse, anything else becomes NULL and is skipped.
            timestamp = f"TRY_CAST({date_col.ident} AS TIMESTAMPTZ)"
        elif date_col.kind == "datetime":
            timestamp = (
                f"CAST({date_col.ident} AS TIMESTAMP)"
                if date_col.type == "DATE"
                else date_col.ident
            )
        else:
            raise InvalidRequestError(
                f"Column '{request.date_column}' is {date_col.kind}, not a date/time column"
            )

        unit, step = _TIME_BUCKETS[request.frequency]
        where, params = _where(query, request.filters, extra=[f"{timestamp} IS NOT NULL"])
        values = [query.column(c) for c in request.value_columns]
        aggregates = ", ".join(
            f"{_agg_expr(query, c.name, request.agg)} AS {c.ident}" for c in values
        )
        filled = ", ".join(
            f"coalesce(a.{c.ident}, 0) AS {c.ident}"
            if request.agg in _ZERO_WHEN_EMPTY
            else f"a.{c.ident}"
            for c in values
        )
        # Aggregate per bucket, then join onto every bucket between the first and last so
        # gaps appear as rows, matching the previous pandas resample output.
        inner = f"""
            WITH agg AS (
                SELECT date_trunc('{unit}', {timestamp}) AS {_PERIOD}, {aggregates}
                FROM {SOURCE} {where}
                GROUP BY 1
            ),
            periods AS (
                SELECT unnest(generate_series(min({_PERIOD}), max({_PERIOD}), {step})) AS {_PERIOD}
                FROM agg
            )
            SELECT p.{_PERIOD} AS period, {filled}
            FROM periods AS p LEFT JOIN agg AS a ON a.{_PERIOD} = p.{_PERIOD}
        """
        columns = ["period", *request.value_columns]
        return _limited_result(query, inner, params, "ORDER BY period", columns, max_rows)
