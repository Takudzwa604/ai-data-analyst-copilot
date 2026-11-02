from httpx import AsyncClient

from tests.conftest import create_user_headers


async def test_health(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "X-Request-ID" in response.headers
    assert response.headers["X-Content-Type-Options"] == "nosniff"


async def test_readiness(client: AsyncClient) -> None:
    response = await client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["checks"] == {"database": "ok", "storage": "ok"}


async def test_request_id_is_propagated(client: AsyncClient) -> None:
    response = await client.get("/health", headers={"X-Request-ID": "abc-123"})
    assert response.headers["X-Request-ID"] == "abc-123"


async def test_register_login_and_me(client: AsyncClient) -> None:
    headers = await create_user_headers(client, email="Alice@Example.com")
    response = await client.get("/api/v1/users/me", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["email"] == "alice@example.com"
    assert "hashed_password" not in body


async def test_duplicate_registration_conflicts(client: AsyncClient) -> None:
    payload = {"email": "dup@example.com", "password": "long-enough-pw"}
    assert (await client.post("/api/v1/auth/register", json=payload)).status_code == 201
    response = await client.post("/api/v1/auth/register", json=payload)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "conflict"


async def test_register_validates_input(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/auth/register", json={"email": "not-an-email", "password": "short"}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


async def test_login_rejects_bad_credentials(client: AsyncClient) -> None:
    await create_user_headers(client, email="bob@example.com", password="correct-horse")
    for username, password in [("bob@example.com", "wrong-pass"), ("nobody@example.com", "x")]:
        response = await client.post(
            "/api/v1/auth/login", data={"username": username, "password": password}
        )
        assert response.status_code == 401
        assert response.headers["WWW-Authenticate"] == "Bearer"


async def test_protected_route_requires_token(client: AsyncClient) -> None:
    assert (await client.get("/api/v1/users/me")).status_code == 401
    response = await client.get("/api/v1/users/me", headers={"Authorization": "Bearer garbage"})
    assert response.status_code == 401


async def test_refresh_flow_and_token_type_enforcement(client: AsyncClient) -> None:
    await client.post(
        "/api/v1/auth/register", json={"email": "carol@example.com", "password": "password123"}
    )
    tokens = (
        await client.post(
            "/api/v1/auth/login", data={"username": "carol@example.com", "password": "password123"}
        )
    ).json()

    # A refresh token must not work as an access token, and vice versa.
    me = await client.get(
        "/api/v1/users/me", headers={"Authorization": f"Bearer {tokens['refresh_token']}"}
    )
    assert me.status_code == 401
    bad = await client.post("/api/v1/auth/refresh", json={"refresh_token": tokens["access_token"]})
    assert bad.status_code == 401

    response = await client.post(
        "/api/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]}
    )
    assert response.status_code == 200
    new_access = response.json()["access_token"]
    me = await client.get("/api/v1/users/me", headers={"Authorization": f"Bearer {new_access}"})
    assert me.status_code == 200


async def test_change_password(client: AsyncClient) -> None:
    headers = await create_user_headers(client, email="dave@example.com", password="old-password")

    response = await client.patch(
        "/api/v1/users/me", headers=headers, json={"password": "new-password"}
    )
    assert response.status_code == 422

    response = await client.patch(
        "/api/v1/users/me",
        headers=headers,
        json={"password": "new-password", "current_password": "wrong"},
    )
    assert response.status_code == 403

    response = await client.patch(
        "/api/v1/users/me",
        headers=headers,
        json={"password": "new-password", "current_password": "old-password", "full_name": "D"},
    )
    assert response.status_code == 200
    assert response.json()["full_name"] == "D"

    login = await client.post(
        "/api/v1/auth/login", data={"username": "dave@example.com", "password": "new-password"}
    )
    assert login.status_code == 200
