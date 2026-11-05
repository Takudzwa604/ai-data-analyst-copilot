# Data Analytics Platform — API

A FastAPI backend for uploading tabular datasets, analyzing them, and asking questions about
them in plain English: profiling, paginated previews, filtered group-by aggregations,
correlation matrices and time-series resampling, plus an AI analyst built on the Claude API
that chooses from validated, read-only tools instead of writing SQL.

You can run all of it for free: set `AI_DEMO_MODE=true` in `.env` to try plain-English
questions without an API key ([details](#ai-analyst-post-apiv1chat)).

New to the project? [WALKTHROUGH.md](WALKTHROUGH.md) is a guided tour: a live demo, one
request followed through the code, and the main design decisions.

## Features

- **Auth**: registration, OAuth2 password login, JWT access + refresh tokens, Argon2 password hashing
- **Datasets**: upload CSV, TSV, XLSX, JSON or Parquet; normalized and stored as Parquet; per-user isolation
- **Analytics**: profile, preview, aggregate (filters, group-by, metrics, sort), correlation, time series,
  executed by DuckDB directly on the stored Parquet files (only the needed columns and row groups are read)
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
| POST | `/api/v1/chat` | Ask the AI analyst a question about a dataset |

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

### AI analyst: `POST /api/v1/chat`

Ask a question in plain language. Claude reads the dataset's metadata (name, row count,
column names and types), picks from five read-only tools (`profile_dataset`,
`preview_dataset`, `aggregate_dataset`, `correlation_analysis`, `time_series_analysis`),
then summarizes the results. It never writes SQL. Tools run through the same analytics
service as the endpoints above and can only touch the dataset you own.

```bash
curl -X POST localhost:8000/api/v1/chat \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"dataset_id": "'$ID'", "question": "Which products generate the most revenue?"}'
```

```json
{
  "answer": "Widget generates the most revenue (350.0), followed by Gadget (135.0).",
  "tool_used": "aggregate_dataset",
  "results": [{"product": "Widget", "revenue": 350.0}, {"product": "Gadget", "revenue": 135.0}],
  "tool_calls": [{"tool": "aggregate_dataset", "input": {"...": "..."}, "is_error": false, "error": null}],
  "model": "claude-opus-5-5"
}
```

`tool_used` is the last tool that ran successfully and `results` holds its rows. `tool_calls`
lists every attempt, including rejected ones: invalid tool calls are returned to Claude as
errors so it can correct itself. Set `ANTHROPIC_API_KEY` (or sign in with `ant auth login`)
to enable it. Without credentials the endpoint returns `503 ai_unavailable`. Code lives in
`app/ai/`: `schemas.py` (tool inputs), `tools.py` (registry), `prompts.py`, `service.py`
(dataset context), `analyst.py` (the Claude loop) and `demo.py` (demo mode).

**Trying it for free:** set `AI_DEMO_MODE=true` to answer without Claude, with no API key or
cost. A keyword matcher picks a tool from the same registry and runs it on your real data, then
fills in a template answer. Answers start with `[Demo mode]` and `model` is `demo-mode (no AI)`.
It only understands simple questions, such as "which townships have the most sales?", "average
revenue by region", "any missing values?" or "revenue trend by month", and it can't apply
filters: if a question seems to ask about a subset, the answer says it covers every row.
The app won't start with demo mode on in `ENVIRONMENT=production`.

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
  ai/             AI analyst: tool registry, Claude loop, dataset context, demo mode
  api/            routers, dependencies (auth, ownership, dataset files)
  core/           config, security, logging, middleware, error handling
  db/             SQLAlchemy base, engine, session
  models/         ORM models
  schemas/        Pydantic request/response models
  services/       ingestion (pandas), storage, DuckDB engine + analytics, dataset CRUD
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
- **Memory**: analytics stream over Parquet instead of loading datasets, so memory follows the query,
  not the dataset size. Each worker process has its own DuckDB instance that defaults to 80% of RAM
  and all cores: with several workers, set `DUCKDB_MEMORY_LIMIT` (it spills to `DUCKDB_TEMP_DIR`
  beyond that) and `DUCKDB_THREADS` so their total fits the machine. The most memory-hungry
  operations are exact distinct counts and quartiles in `profile`, which run one column at a time.
- **Spearman correlation** on columns with missing values ranks each column pair separately
  (exactly as pandas does), so its cost grows with the square of the column count.
- **Ingestion still uses pandas** (needed for Excel and type cleanup), so an upload is held in memory
  while it is converted to Parquet.
- **Uploads are processed synchronously** (parsed in a worker thread). For very large files,
  move ingestion to a task queue (Celery, RQ, Arq).
- **Tokens are stateless**: logout or revocation needs a denylist or a per-user token version.
- **Rate limiting** is not built in. Apply it at the gateway or proxy, especially on `/auth/login`.
