import math

import pandas as pd
import pytest
from httpx import AsyncClient

from app.core.exceptions import InvalidRequestError
from app.schemas.analytics import AggregateRequest, Filter
from app.services import analytics


def _url(dataset: dict, op: str) -> str:
    return f"/api/v1/datasets/{dataset['id']}/analytics/{op}"


async def test_aggregate_group_filter_sort(
    client: AsyncClient, auth_headers: dict, sales_dataset: dict
) -> None:
    response = await client.post(
        _url(sales_dataset, "aggregate"),
        headers=auth_headers,
        json={
            "filters": [{"column": "product", "op": "eq", "value": "Widget"}],
            "group_by": ["region"],
            "metrics": [
                {"func": "sum", "column": "revenue", "alias": "total_revenue"},
                {"func": "count"},
            ],
            "sort": [{"column": "total_revenue", "direction": "desc"}],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["columns"] == ["region", "total_revenue", "count"]
    assert body["rows"] == [
        {"region": "North", "total_revenue": 220.0, "count": 2},
        {"region": "East", "total_revenue": 80.0, "count": 1},
        {"region": "South", "total_revenue": 50.0, "count": 1},
    ]
    assert body["truncated"] is False


async def test_aggregate_without_grouping(
    client: AsyncClient, auth_headers: dict, sales_dataset: dict
) -> None:
    response = await client.post(
        _url(sales_dataset, "aggregate"),
        headers=auth_headers,
        json={
            "filters": [{"column": "date", "op": "gte", "value": "2024-02-01"}],
            "metrics": [
                {"func": "mean", "column": "units"},
                {"func": "nunique", "column": "region"},
            ],
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["rows"] == [{"mean_units": 10.0, "nunique_region": 3}]


async def test_aggregate_limit_truncates(
    client: AsyncClient, auth_headers: dict, sales_dataset: dict
) -> None:
    response = await client.post(
        _url(sales_dataset, "aggregate"),
        headers=auth_headers,
        json={"group_by": ["date"], "limit": 2},
    )
    body = response.json()
    assert len(body["rows"]) == 2
    assert body["row_count"] == 6
    assert body["truncated"] is True


@pytest.mark.parametrize(
    "payload",
    [
        {"group_by": ["missing"]},
        {"metrics": [{"func": "sum", "column": "region"}]},
        {"filters": [{"column": "units", "op": "gt", "value": "abc"}]},
        {"metrics": [{"func": "count"}, {"func": "count"}]},
        {"sort": [{"column": "nope"}]},
    ],
)
async def test_aggregate_invalid_requests(
    client: AsyncClient, auth_headers: dict, sales_dataset: dict, payload: dict
) -> None:
    response = await client.post(
        _url(sales_dataset, "aggregate"), headers=auth_headers, json=payload
    )
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "invalid_request"


@pytest.mark.parametrize(
    "payload",
    [
        {"metrics": [{"func": "sum"}]},
        {"filters": [{"column": "units", "op": "in", "value": 3}]},
        {"filters": [{"column": "units", "op": "drop_table"}]},
    ],
)
async def test_aggregate_schema_validation(
    client: AsyncClient, auth_headers: dict, sales_dataset: dict, payload: dict
) -> None:
    response = await client.post(
        _url(sales_dataset, "aggregate"), headers=auth_headers, json=payload
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


async def test_correlation(client: AsyncClient, auth_headers: dict, sales_dataset: dict) -> None:
    response = await client.post(
        _url(sales_dataset, "correlation"),
        headers=auth_headers,
        json={"columns": ["units", "revenue"], "method": "spearman"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["columns"] == ["units", "revenue"]
    assert body["matrix"][0][0] == pytest.approx(1.0)
    assert -1.0 <= body["matrix"][0][1] <= 1.0

    response = await client.post(_url(sales_dataset, "correlation"), headers=auth_headers, json={})
    assert response.status_code == 200
    assert response.json()["columns"] == ["units", "revenue"]

    response = await client.post(
        _url(sales_dataset, "correlation"),
        headers=auth_headers,
        json={"columns": ["units", "region"]},
    )
    assert response.status_code == 422


async def test_timeseries_monthly(
    client: AsyncClient, auth_headers: dict, sales_dataset: dict
) -> None:
    response = await client.post(
        _url(sales_dataset, "timeseries"),
        headers=auth_headers,
        json={
            "date_column": "date",
            "value_columns": ["revenue", "units"],
            "frequency": "month",
            "agg": "sum",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["columns"] == ["period", "revenue", "units"]
    assert [row["revenue"] for row in body["rows"]] == [240.0, 125.0, 120.0]
    assert [row["units"] for row in body["rows"]] == [18.0, 8.0, 12.0]
    assert body["rows"][0]["period"].startswith("2024-01-01")


async def test_timeseries_rejects_numeric_date_column(
    client: AsyncClient, auth_headers: dict, sales_dataset: dict
) -> None:
    response = await client.post(
        _url(sales_dataset, "timeseries"),
        headers=auth_headers,
        json={"date_column": "units", "value_columns": ["revenue"]},
    )
    assert response.status_code == 422


# ---------------------------------------------------------------- service-level unit tests


def test_filters_on_datetime_and_null_ops() -> None:
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2024-01-01", "2024-06-01", None]),
            "name": ["Alpha", None, "beta"],
        }
    )
    result = analytics.apply_filters(df, [Filter(column="ts", op="lt", value="2024-03-01")])
    assert len(result) == 1

    result = analytics.apply_filters(df, [Filter(column="name", op="contains", value="A")])
    assert list(result["name"]) == ["Alpha", "beta"]

    result = analytics.apply_filters(df, [Filter(column="name", op="is_null")])
    assert len(result) == 1


def test_aggregate_does_not_mutate_input() -> None:
    df = pd.DataFrame({"g": ["a", "b", "a"], "v": [1, 2, 3]})
    snapshot = df.copy()
    analytics.aggregate(
        df, AggregateRequest(group_by=["g"], metrics=[{"func": "sum", "column": "v"}]), 100
    )
    pd.testing.assert_frame_equal(df, snapshot)


def test_unknown_filter_column_raises() -> None:
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(InvalidRequestError):
        analytics.apply_filters(df, [Filter(column="b", op="eq", value=1)])


def test_to_json_scalar_handles_special_values() -> None:
    assert analytics.to_json_scalar(float("nan")) is None
    assert analytics.to_json_scalar(math.inf) is None
    assert analytics.to_json_scalar(pd.NaT) is None
    assert analytics.to_json_scalar(pd.Timestamp("2024-01-01")) == "2024-01-01T00:00:00"
