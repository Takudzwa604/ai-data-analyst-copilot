import os
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

# Configure the environment before any app module reads settings.
_TMP = Path(tempfile.mkdtemp(prefix="dap-tests-"))
os.environ.update(
    {
        "ENVIRONMENT": "test",
        "DATABASE_URL": f"sqlite+aiosqlite:///{(_TMP / 'test.db').as_posix()}",
        "STORAGE_DIR": str(_TMP / "storage"),
        "SECRET_KEY": "test-secret-key-with-plenty-of-entropy-0123456789",
        "MAX_UPLOAD_SIZE_MB": "2",
        "MAX_RESULT_ROWS": "100",
        "LOG_LEVEL": "WARNING",
        "AUTO_CREATE_TABLES": "false",
        "CORS_ORIGINS": "",
    }
)

import pytest  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

from app.db.base import Base  # noqa: E402
from app.db.session import engine  # noqa: E402
from app.main import app  # noqa: E402

SALES_CSV = b"""date,region,product,units,revenue,returned
2024-01-01,North,Widget,10,100.0,false
2024-01-02,South,Widget,5,50.0,false
2024-01-15,North,Gadget,3,90.0,true
2024-02-01,East,Widget,8,80.0,false
2024-02-10,South,Gadget,,45.0,false
2024-03-05,North,Widget,12,120.0,false
"""


@pytest.fixture(autouse=True)
async def _database() -> AsyncIterator[None]:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def create_user_headers(
    client: AsyncClient, email: str = "analyst@example.com", password: str = "s3cure-pass!"
) -> dict[str, str]:
    response = await client.post(
        "/api/v1/auth/register", json={"email": email, "password": password}
    )
    assert response.status_code == 201, response.text
    response = await client.post(
        "/api/v1/auth/login", data={"username": email, "password": password}
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture
async def auth_headers(client: AsyncClient) -> dict[str, str]:
    return await create_user_headers(client)


@pytest.fixture
async def sales_dataset(client: AsyncClient, auth_headers: dict[str, str]) -> dict:
    response = await client.post(
        "/api/v1/datasets",
        headers=auth_headers,
        files={"file": ("sales.csv", SALES_CSV, "text/csv")},
        data={"name": "Sales", "description": "Q1 sales"},
    )
    assert response.status_code == 201, response.text
    return response.json()
