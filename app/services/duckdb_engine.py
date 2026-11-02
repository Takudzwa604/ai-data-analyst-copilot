"""DuckDB execution layer: one in-process database, a cursor per operation.

A data source (a Parquet path, or a DataFrame in tests) is exposed to SQL as the
cursor-local view `src`. Parquet files are scanned in place, so queries read only the
columns and row groups they need instead of loading whole datasets into memory.

SQL is assembled only by application code from validated requests: identifiers go
through `quote_ident`, user values are always bound as `?` parameters.
"""

import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from app.core.config import get_settings
from app.core.exceptions import InvalidRequestError

DataSource = Path | pd.DataFrame
SOURCE = "src"

_NUMERIC_TYPES = {
    "TINYINT",
    "SMALLINT",
    "INTEGER",
    "BIGINT",
    "HUGEINT",
    "UTINYINT",
    "USMALLINT",
    "UINTEGER",
    "UBIGINT",
    "UHUGEINT",
    "FLOAT",
    "DOUBLE",
}


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    kind: str

    @property
    def ident(self) -> str:
        return quote_ident(self.name)


def kind_for_type(duckdb_type: str) -> str:
    """Map a DuckDB type to the platform's column kinds."""
    upper = duckdb_type.upper()
    if upper == "BOOLEAN":
        return "boolean"
    if upper in _NUMERIC_TYPES or upper.startswith("DECIMAL"):
        return "numeric"
    if upper.startswith(("DATE", "TIMESTAMP")):
        return "datetime"
    return "categorical"


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def to_json_scalar(value: Any) -> Any:
    """Convert DuckDB/numpy/pandas scalars into JSON-safe Python values."""
    if value is None or value is pd.NaT:
        return None
    if hasattr(value, "item") and not isinstance(value, list | tuple | dict | str):
        value = value.item()  # numpy scalar
    if isinstance(value, Decimal):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, pd.Timestamp | datetime | date | time):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    if isinstance(value, bytes):
        return value.hex()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


@lru_cache
def _database() -> duckdb.DuckDBPyConnection:
    settings = get_settings()
    temp_dir = settings.duckdb_temp_dir or settings.storage_dir / ".duckdb_tmp"
    config: dict[str, Any] = {"temp_directory": str(temp_dir), "TimeZone": "UTC"}
    if settings.duckdb_threads:
        config["threads"] = settings.duckdb_threads
    if settings.duckdb_memory_limit:
        config["memory_limit"] = settings.duckdb_memory_limit
    return duckdb.connect(":memory:", config=config)


class Query:
    """A cursor with one data source bound as `src`."""

    def __init__(self, cursor: duckdb.DuckDBPyConnection, columns: list[Column]) -> None:
        self._cursor = cursor
        self.columns = {column.name: column for column in columns}

    def column(self, name: str) -> Column:
        return self.columns[name]

    def require(self, names: Sequence[str]) -> None:
        missing = sorted({n for n in names if n not in self.columns})
        if missing:
            raise InvalidRequestError(
                f"Unknown column(s): {', '.join(missing)}", details={"unknown_columns": missing}
            )

    def fetch(self, sql: str, params: Sequence[Any] = ()) -> tuple[list[str], list[tuple]]:
        try:
            result = self._cursor.execute(sql, list(params))
            names = [d[0] for d in result.description]
            return names, result.fetchall()
        except (duckdb.ConversionException, duckdb.OutOfRangeException) as exc:
            raise InvalidRequestError(f"A filter value has the wrong type: {exc}") from exc

    def records(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        names, rows = self.fetch(sql, params)
        return [{n: to_json_scalar(v) for n, v in zip(names, row, strict=True)} for row in rows]

    def one(self, sql: str, params: Sequence[Any] = ()) -> tuple:
        return self.fetch(sql, params)[1][0]


@contextmanager
def open_source(source: DataSource) -> Iterator[Query]:
    cursor = _database().cursor()
    try:
        if isinstance(source, pd.DataFrame):
            cursor.register(SOURCE, source)
        else:
            if not source.is_file():
                raise FileNotFoundError(source)
            # The path is server-generated (never user input); quoted as a literal
            # because DDL cannot take bound parameters.
            path = _quote_literal(source.resolve().as_posix())
            cursor.execute(f"CREATE TEMP VIEW {SOURCE} AS SELECT * FROM read_parquet({path})")
        rows = cursor.execute(f"DESCRIBE {SOURCE}").fetchall()
        yield Query(cursor, [Column(r[0], r[1], kind_for_type(r[1])) for r in rows])
    finally:
        cursor.close()


def describe_parquet(path: Path) -> list[Column]:
    with open_source(path) as query:
        return list(query.columns.values())
