"""Parsing uploaded files into clean DataFrames and persisting them as Parquet."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.exceptions import InvalidRequestError
from app.services.analytics import column_kind

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".json", ".parquet"}
MAX_COLUMNS = 2000


@dataclass(frozen=True)
class IngestResult:
    row_count: int
    column_count: int
    columns: list[dict[str, Any]]


def read_tabular_file(path: Path, extension: str) -> pd.DataFrame:
    try:
        match extension:
            case ".csv" | ".tsv":
                sep = "\t" if extension == ".tsv" else ","
                try:
                    return pd.read_csv(path, sep=sep, low_memory=False)
                except UnicodeDecodeError:
                    return pd.read_csv(path, sep=sep, low_memory=False, encoding="latin-1")
            case ".xlsx":
                return pd.read_excel(path, engine="openpyxl")
            case ".json":
                try:
                    return pd.read_json(path)
                except ValueError:
                    return pd.read_json(path, lines=True)
            case ".parquet":
                return pd.read_parquet(path)
    except InvalidRequestError:
        raise
    except Exception as exc:
        logger.info("Failed to parse uploaded %s file: %s", extension, exc)
        raise InvalidRequestError(f"Could not parse file as {extension.lstrip('.')}") from exc
    raise InvalidRequestError(f"Unsupported file extension: {extension}")


def _unique_column_names(columns: pd.Index) -> list[str]:
    seen: dict[str, int] = {}
    names = []
    for raw in columns:
        name = str(raw).strip() or "unnamed"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        seen.setdefault(name, 0)
        names.append(name)
    return names


def _normalize_object_column(series: pd.Series) -> pd.Series:
    """Coerce mixed-type object columns into something Parquet can store losslessly enough."""
    inferred = pd.api.types.infer_dtype(series, skipna=True)
    if inferred in {"string", "empty", "boolean"}:
        return series
    if inferred in {"integer", "floating", "mixed-integer-float", "decimal"}:
        return pd.to_numeric(series, errors="coerce")
    if inferred in {"date", "datetime", "datetime64"}:
        return pd.to_datetime(series, errors="coerce", utc=True)
    # Mixed types, nested lists/dicts, bytes, etc. are stored as text.
    return series.where(series.isna(), series.astype(str))


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.shape[1] == 0:
        raise InvalidRequestError("File contains no columns")
    if df.shape[1] > MAX_COLUMNS:
        raise InvalidRequestError(f"File has more than {MAX_COLUMNS} columns")
    if df.shape[0] == 0:
        raise InvalidRequestError("File contains no data rows")

    df = df.reset_index(drop=True)
    df.columns = _unique_column_names(df.columns)
    for name in df.columns:
        if df[name].dtype == object:
            df[name] = _normalize_object_column(df[name])
    return df


def describe_columns(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {"name": name, "dtype": str(df[name].dtype), "kind": column_kind(df[name])}
        for name in df.columns
    ]


def ingest_file(source: Path, extension: str, destination: Path) -> IngestResult:
    """Parse `source`, normalize it and write it to `destination` as Parquet."""
    df = normalize_dataframe(read_tabular_file(source, extension))
    try:
        df.to_parquet(destination, index=False)
    except Exception as exc:
        destination.unlink(missing_ok=True)
        logger.warning("Failed to convert dataset to parquet: %s", exc)
        raise InvalidRequestError("File contains data that could not be stored") from exc
    return IngestResult(row_count=len(df), column_count=df.shape[1], columns=describe_columns(df))
