# Data Analytics Platform — API

A FastAPI backend for uploading tabular datasets and running analytics on them: profiling,
paginated previews, filtered group-by aggregations, correlation matrices and time-series
resampling.

## Features

- **Auth**: registration, OAuth2 password login, JWT access + refresh tokens, Argon2 password hashing
- **Datasets**: upload CSV, TSV, XLSX, JSON or Parquet; normalized and stored as Parquet; per-user isolation
- **Analytics**: profile, preview, aggregate (filters, group-by, metrics, sort), correlation, time series
- **Safe querying**: columns, operators and functions are whitelisted. User input never reaches `eval` or `DataFrame.query`
- **Operations**: async SQLAlchemy (Postgres or SQLite), Alembic migrations, structured JSON logs with request IDs,
  liveness/readiness probes, consistent error envelope, security headers, gzip, Docker image (non-root)

## Quick start (local)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env               # then set SECRET_KEY
alembic upgrade head
uvicorn app.main:app --reload
```

Interactive docs: http://localhost:8000/docs

## Quick start (Docker + Postgres)

```bash
cp .env.example .env               # set SECRET_KEY, POSTGRES_PASSWORD, ENVIRONMENT=production
docker compose up --build
```

Migrations run automatically on container start (set `RUN_MIGRATIONS=false` to disable, e.g. when
you run them as a separate release step).

## API overview

All endpoints except `/health*` and `/api/v1/auth/*` need `Authorization: Bearer <access_token>`.

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness |
| GET | `/health/ready` | Readiness (DB + storage) |
| POST | `/api/v1/auth/register` | Create account |
| POST | `/api/v1/auth/login` | Form fields `username` (email), `password` → token pair |
| POST | `/api/v1/auth/refresh` | Exchange a refresh token for a new pair |
| GET / PATCH | `/api/v1/users/me` | Current user / update name or password |
| POST | `/api/v1/datasets` | Multipart upload: `file`, optional `name`, `description` |
| GET | `/api/v1/datasets` | List (`limit`, `offset`, `search`) |
| GET / PATCH / DELETE | `/api/v1/datasets/{id}` | Read / rename / delete |
| GET | `/api/v1/datasets/{id}/preview` | Rows (`offset`, `limit` ≤ 1000) |
| GET | `/api/v1/datasets/{id}/profile` | Per-column stats, nulls, top values |
| POST | `/api/v1/datasets/{id}/analytics/aggregate` | Filter / group / aggregate |
| POST | `/api/v1/datasets/{id}/analytics/correlation` | Pearson or Spearman matrix |
| POST | `/api/v1/datasets/{id}/analytics/timeseries` | Resample by hour…year |

### Example: aggregate

```bash
curl -X POST localhost:8000/api/v1/datasets/$ID/analytics/aggregate \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{
        "filters":  [{"column": "date", "op": "gte", "value": "2024-01-01"}],
        "group_by": ["region"],
        "metrics":  [{"func": "sum", "column": "revenue", "alias": "revenue"}, {"func": "count"}],
        "sort":     [{"column": "revenue", "direction": "desc"}],
        "limit": 10
      }'
```

Filter ops: `eq ne gt gte lt lte in not_in contains is_null not_null`.
Metric funcs: `count nunique sum mean median min max std var`.

### Errors

Every error uses the same shape:

```json
{"error": {"code": "invalid_request", "message": "Unknown column(s): foo", "request_id": "…", "details": {}}}
```

## Configuration

All settings come from environment variables or `.env`. See [.env.example](.env.example).
In `ENVIRONMENT=production` the app won't start with the default or a short `SECRET_KEY`,
with `DEBUG=true`, or with wildcard CORS.

## Project layout

```
app/
  api/            routers, dependencies (auth, ownership, dataframe loading)
  core/           config, security, logging, middleware, error handling
  db/             SQLAlchemy base, engine, session
  models/         ORM models
  schemas/        Pydantic request/response models
  services/       ingestion, storage + cache, analytics (pure pandas), dataset CRUD
alembic/          migrations
tests/            pytest suite (API + service-level)
```

## Development

```bash
pytest                 # run tests
ruff check . && ruff format --check .
alembic revision --autogenerate -m "describe change"
```

## Production notes and limits

- **Body size**: requests with a `Content-Length` above the limit are rejected early, but chunked
  uploads are only checked while they're copied. Also set a limit at your reverse proxy
  (e.g. nginx `client_max_body_size`).
- **Storage**: datasets live on the local filesystem (`STORAGE_DIR`). With several hosts, mount
  shared storage or replace `LocalStorage` with an object-store backend that has the same interface.
- **Memory**: each worker caches up to `DATAFRAME_CACHE_SIZE` parsed datasets. Size workers and
  the cache to fit your largest datasets.
- **Uploads are processed synchronously** (parsed in a worker thread). For very large files,
  move ingestion to a task queue (Celery, RQ, Arq).
- **Tokens are stateless**: logout or revocation needs a denylist or a per-user token version.
- **Rate limiting** is not built in. Apply it at the gateway or proxy, especially on `/auth/login`.
