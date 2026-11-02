"""DuckDB engine: analytics run against Parquet files in place, with pandas-equivalent results."""

import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from httpx import AsyncClient

from app.schemas.analytics import (
    AggregateRequest,
    CorrelationRequest,
    Filter,
    TimeSeriesRequest,
)
from app.services import analytics
from app.services.duckdb_engine import open_source


def write_parquet(tmp_path: Path, df: pd.DataFrame, name: str = "data.parquet") -> Path:
    path = tmp_path / name
    df.to_parquet(path, index=False)
    return path


def count_matching(path: Path, *filters: dict) -> int:
    request = AggregateRequest(filters=[Filter(**f) for f in filters])
    return analytics.aggregate(path, request, 10)["rows"][0]["count"]


@pytest.fixture
def people(tmp_path: Path) -> Path:
    return write_parquet(
        tmp_path,
        pd.DataFrame(
            {
                "name": ["Ann", "Bob", None, "50% off", "a_b"],
                "age": [30, None, 41, 25, 30],
                "joined": pd.to_datetime(
                    ["2024-01-05", "2024-02-01", None, "2024-03-10", "2024-03-11"]
                ),
                "active": [True, False, True, None, True],
            }
        ),
    )


# --------------------------------------------------------------------------- engine


def test_parquet_schema_and_kinds(people: Path) -> None:
    with open_source(people) as query:
        columns = query.columns
    assert {name: c.kind for name, c in columns.items()} == {
        "name": "categorical",
        "age": "numeric",
        "joined": "datetime",
        "active": "boolean",
    }
    assert (columns["name"].type, columns["age"].type) == ("VARCHAR", "DOUBLE")
    # Timestamp precision (TIMESTAMP vs TIMESTAMP_NS) depends on the pandas version.
    assert columns["joined"].type.startswith("TIMESTAMP")


def test_missing_parquet_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError), open_source(tmp_path / "gone.parquet"):
        pass


# --------------------------------------------------------------------------- filters


def test_null_semantics_match_previous_pandas_behaviour(people: Path) -> None:
    assert count_matching(people, {"column": "age", "op": "eq", "value": 30}) == 2
    # Nulls are "not equal" and "not in", as with the pandas implementation.
    assert count_matching(people, {"column": "age", "op": "ne", "value": 30}) == 3
    assert count_matching(people, {"column": "age", "op": "not_in", "value": [30, 41]}) == 2
    assert count_matching(people, {"column": "age", "op": "gt", "value": 26}) == 3
    assert count_matching(people, {"column": "active", "op": "is_null"}) == 1


def test_contains_is_literal_and_case_insensitive(people: Path) -> None:
    assert count_matching(people, {"column": "name", "op": "contains", "value": "%"}) == 1
    assert count_matching(people, {"column": "name", "op": "contains", "value": "_"}) == 1
    assert count_matching(people, {"column": "name", "op": "contains", "value": "BO"}) == 1


def test_datetime_filters_accept_iso_strings(people: Path) -> None:
    assert count_matching(people, {"column": "joined", "op": "gte", "value": "2024-03-01"}) == 2


def test_hostile_identifiers_and_values_are_data(tmp_path: Path) -> None:
    column = 'x"; DROP TABLE src; --'
    value = "b'); DROP TABLE src; --"
    path = write_parquet(tmp_path, pd.DataFrame({column: [1, 2, 3], "name": ["a", value, "c"]}))
    request = AggregateRequest(
        filters=[Filter(column="name", op="eq", value=value)],
        metrics=[{"func": "sum", "column": column, "alias": 'total"; --'}],
    )
    assert analytics.aggregate(path, request, 10)["rows"] == [{'total"; --': 2}]


# --------------------------------------------------------------------------- aggregate


def test_aggregate_types_and_default_ordering(tmp_path: Path) -> None:
    path = write_parquet(
        tmp_path,
        pd.DataFrame(
            {"g": ["b", "a", "b", None], "n": [1, 2, 3, 4], "flag": [True, True, False, True]}
        ),
    )
    request = AggregateRequest(
        group_by=["g"],
        metrics=[{"func": "sum", "column": "n"}, {"func": "sum", "column": "flag"}],
    )
    rows = analytics.aggregate(path, request, 10)["rows"]
    # Ordered by group key (nulls last); integer sums stay integers; booleans sum as 0/1.
    assert rows == [
        {"g": "a", "sum_n": 2, "sum_flag": 1},
        {"g": "b", "sum_n": 4, "sum_flag": 1},
        {"g": None, "sum_n": 4, "sum_flag": 1},
    ]
    assert isinstance(rows[0]["sum_n"], int)


# --------------------------------------------------------------------------- time series


def test_timeseries_fills_gaps_and_parses_text_dates(tmp_path: Path) -> None:
    path = write_parquet(
        tmp_path,
        pd.DataFrame(
            {
                "when": ["2024-01-31T23:30:00-02:00", "2024-03-05", "not a date", None],
                "v": [5, 7, 100, 100],
            }
        ),
    )

    def run(agg: str) -> list[tuple]:
        request = TimeSeriesRequest(
            date_column="when", value_columns=["v"], frequency="month", agg=agg
        )
        return [(r["period"][:7], r["v"]) for r in analytics.timeseries(path, request, 100)["rows"]]

    # 23:30 at -02:00 is 01:30 UTC on Feb 1; unparseable and missing dates are skipped.
    assert run("sum") == [("2024-02", 5), ("2024-03", 7)]
    assert run("count") == [("2024-02", 1), ("2024-03", 1)]


def test_timeseries_empty_buckets(tmp_path: Path) -> None:
    path = write_parquet(
        tmp_path,
        pd.DataFrame({"d": pd.to_datetime(["2024-01-10", "2024-04-02"]), "v": [2.0, 4.0]}),
    )
    for agg, gap in (("sum", 0), ("count", 0), ("mean", None), ("max", None)):
        request = TimeSeriesRequest(
            date_column="d", value_columns=["v"], frequency="month", agg=agg
        )
        values = [r["v"] for r in analytics.timeseries(path, request, 100)["rows"]]
        assert values[1:3] == [gap, gap], agg


def test_weekly_buckets_start_on_monday(tmp_path: Path) -> None:
    path = write_parquet(
        tmp_path,
        pd.DataFrame({"d": pd.to_datetime(["2024-05-08", "2024-05-12"]), "v": [1, 1]}),  # Wed, Sun
    )
    request = TimeSeriesRequest(date_column="d", value_columns=["v"], frequency="week")
    assert [
        (r["period"][:10], r["v"]) for r in analytics.timeseries(path, request, 10)["rows"]
    ] == [("2024-05-06", 2)]


# --------------------------------------------------------------------------- correlation


@pytest.fixture
def numeric_frame() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "a": rng.normal(size=400),
            "b": rng.integers(0, 6, 400).astype(float),  # many ties
            "c": rng.normal(size=400),
            "constant": 1.0,
        }
    )
    df["c"] += df["a"] * 0.7
    return df


@pytest.mark.parametrize("method", ["pearson", "spearman"])
@pytest.mark.parametrize("with_missing", [False, True])
def test_correlation_matches_pandas(
    tmp_path: Path, numeric_frame: pd.DataFrame, method: str, with_missing: bool
) -> None:
    df = numeric_frame.copy()
    if with_missing:
        df.loc[::7, "a"] = np.nan
        df.loc[::11, "b"] = np.nan
    path = write_parquet(tmp_path, df)

    got = np.array(
        analytics.correlation(path, CorrelationRequest(method=method))["matrix"], dtype=float
    )
    want = df.corr(method=method).to_numpy()

    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12, equal_nan=True)


# --------------------------------------------------------------------------- profile & preview


def test_profile_matches_pandas(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    df = pd.DataFrame(
        {
            "x": np.where(rng.random(300) < 0.1, np.nan, rng.normal(size=300).round(2)),
            "city": rng.choice(["Oslo", "Lima", None], 300),
        }
    )
    df = pd.concat([df, df.iloc[:12]], ignore_index=True)
    path = write_parquet(tmp_path, df)

    result = analytics.profile(path)
    x, city = result["columns"]
    numeric = df["x"].dropna()

    assert result["duplicate_rows"] == int(df.duplicated().sum())
    assert result["storage_bytes"] == path.stat().st_size
    assert result["memory_bytes"] is None
    assert x["unique_count"] == numeric.nunique()
    assert x["null_count"] == int(df["x"].isna().sum())
    for key, expected in {
        "mean": numeric.mean(),
        "std": numeric.std(),
        "p25": numeric.quantile(0.25),
        "median": numeric.median(),
        "p75": numeric.quantile(0.75),
        "sum": numeric.sum(),
    }.items():
        assert math.isclose(x["numeric_stats"][key], expected, rel_tol=1e-9), key
    assert city["unique_count"] == 2
    assert {v["value"]: v["count"] for v in city["top_values"]} == df[
        "city"
    ].value_counts().to_dict()


def test_preview_pages_in_file_order(tmp_path: Path) -> None:
    path = write_parquet(tmp_path, pd.DataFrame({"i": range(1000)}))
    page = analytics.preview(path, offset=990, limit=20)
    assert page["total_rows"] == 1000
    assert [r["i"] for r in page["rows"]] == list(range(990, 1000))


# --------------------------------------------------------------------------- concurrency


def test_concurrent_queries_are_isolated(tmp_path: Path) -> None:
    paths = [
        write_parquet(
            tmp_path, pd.DataFrame({"g": ["a", "b"] * 500, "v": [float(i)] * 1000}), f"{i}.parquet"
        )
        for i in range(4)
    ]
    request = AggregateRequest(group_by=["g"], metrics=[{"func": "mean", "column": "v"}])

    def mean_for(index: int) -> tuple[int, float]:
        rows = analytics.aggregate(paths[index % 4], request, 10)["rows"]
        return index % 4, rows[0]["mean_v"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(mean_for, range(64)))
    assert all(mean == float(source) for source, mean in results)


# --------------------------------------------------------------------------- ingestion


async def test_stored_metadata_uses_engine_types(sales_dataset: dict) -> None:
    columns = {c["name"]: (c["dtype"], c["kind"]) for c in sales_dataset["columns"]}
    assert columns["revenue"] == ("DOUBLE", "numeric")
    assert columns["region"] == ("VARCHAR", "categorical")
    assert columns["returned"] == ("BOOLEAN", "boolean")


async def test_profile_endpoint_reports_storage_size(
    client: AsyncClient, auth_headers: dict, sales_dataset: dict
) -> None:
    response = await client.get(
        f"/api/v1/datasets/{sales_dataset['id']}/profile", headers=auth_headers
    )
    body = response.json()
    assert body["storage_bytes"] > 0
    assert body["memory_bytes"] is None
    assert body["duplicate_rows"] == 0
