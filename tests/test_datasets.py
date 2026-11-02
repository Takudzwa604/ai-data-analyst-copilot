import io
import json

import pandas as pd
from httpx import AsyncClient

from tests.conftest import create_user_headers


async def test_upload_csv_infers_schema(sales_dataset: dict) -> None:
    assert sales_dataset["name"] == "Sales"
    assert sales_dataset["file_format"] == "csv"
    assert sales_dataset["row_count"] == 6
    assert sales_dataset["column_count"] == 6
    kinds = {c["name"]: c["kind"] for c in sales_dataset["columns"]}
    assert kinds["units"] == "numeric"
    assert kinds["region"] == "categorical"
    assert kinds["returned"] == "boolean"


async def test_upload_json_and_excel(client: AsyncClient, auth_headers: dict) -> None:
    records = [{"a": 1, "b": "x", "nested": {"k": 1}}, {"a": 2, "b": "y", "nested": [1, 2]}]
    response = await client.post(
        "/api/v1/datasets",
        headers=auth_headers,
        files={"file": ("data.json", json.dumps(records).encode(), "application/json")},
    )
    assert response.status_code == 201, response.text
    assert response.json()["name"] == "data"
    assert response.json()["row_count"] == 2

    buffer = io.BytesIO()
    pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]}).to_excel(buffer, index=False)
    response = await client.post(
        "/api/v1/datasets",
        headers=auth_headers,
        files={"file": ("sheet.xlsx", buffer.getvalue(), "application/octet-stream")},
    )
    assert response.status_code == 201, response.text
    assert response.json()["row_count"] == 3


async def test_upload_rejects_unsupported_and_invalid_files(
    client: AsyncClient, auth_headers: dict
) -> None:
    response = await client.post(
        "/api/v1/datasets", headers=auth_headers, files={"file": ("evil.exe", b"MZ", "x")}
    )
    assert response.status_code == 415

    response = await client.post(
        "/api/v1/datasets", headers=auth_headers, files={"file": ("empty.csv", b"a,b\n", "x")}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"

    response = await client.post(
        "/api/v1/datasets", headers=auth_headers, files={"file": ("bad.parquet", b"nope", "x")}
    )
    assert response.status_code == 422


async def test_upload_size_limit(client: AsyncClient, auth_headers: dict) -> None:
    big = b"a\n" + b"1\n" * (2 * 1024 * 1024)  # > 2 MB test limit, < limit + overhead
    response = await client.post(
        "/api/v1/datasets", headers=auth_headers, files={"file": ("big.csv", big, "text/csv")}
    )
    assert response.status_code == 413

    huge = b"a\n" + b"1\n" * (4 * 1024 * 1024)  # rejected by Content-Length before parsing
    response = await client.post(
        "/api/v1/datasets", headers=auth_headers, files={"file": ("huge.csv", huge, "text/csv")}
    )
    assert response.status_code == 413


async def test_list_get_update_delete(
    client: AsyncClient, auth_headers: dict, sales_dataset: dict
) -> None:
    dataset_id = sales_dataset["id"]

    listing = (await client.get("/api/v1/datasets", headers=auth_headers)).json()
    assert listing["total"] == 1
    assert listing["items"][0]["id"] == dataset_id

    search = await client.get("/api/v1/datasets?search=nomatch", headers=auth_headers)
    assert search.json()["total"] == 0

    response = await client.patch(
        f"/api/v1/datasets/{dataset_id}", headers=auth_headers, json={"name": "Renamed"}
    )
    assert response.status_code == 200
    assert response.json()["name"] == "Renamed"
    assert response.json()["description"] == "Q1 sales"

    response = await client.delete(f"/api/v1/datasets/{dataset_id}", headers=auth_headers)
    assert response.status_code == 204
    response = await client.get(f"/api/v1/datasets/{dataset_id}", headers=auth_headers)
    assert response.status_code == 404


async def test_datasets_are_isolated_between_users(
    client: AsyncClient, sales_dataset: dict
) -> None:
    other = await create_user_headers(client, email="mallory@example.com")
    dataset_id = sales_dataset["id"]

    assert (await client.get(f"/api/v1/datasets/{dataset_id}", headers=other)).status_code == 404
    response = await client.post(
        f"/api/v1/datasets/{dataset_id}/analytics/aggregate", headers=other, json={}
    )
    assert response.status_code == 404
    assert (await client.delete(f"/api/v1/datasets/{dataset_id}", headers=other)).status_code == 404
    assert (await client.get("/api/v1/datasets", headers=other)).json()["total"] == 0


async def test_preview_pagination(
    client: AsyncClient, auth_headers: dict, sales_dataset: dict
) -> None:
    response = await client.get(
        f"/api/v1/datasets/{sales_dataset['id']}/preview?offset=4&limit=10", headers=auth_headers
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total_rows"] == 6
    assert len(body["rows"]) == 2
    assert body["rows"][0]["units"] is None  # NaN serialized as null
    assert body["rows"][0]["region"] == "South"


async def test_profile(client: AsyncClient, auth_headers: dict, sales_dataset: dict) -> None:
    response = await client.get(
        f"/api/v1/datasets/{sales_dataset['id']}/profile", headers=auth_headers
    )
    assert response.status_code == 200
    columns = {c["name"]: c for c in response.json()["columns"]}

    units = columns["units"]
    assert units["null_count"] == 1
    assert units["numeric_stats"]["sum"] == 38
    assert units["numeric_stats"]["max"] == 12

    region = columns["region"]
    assert region["unique_count"] == 3
    assert region["top_values"][0] == {"value": "North", "count": 3}
