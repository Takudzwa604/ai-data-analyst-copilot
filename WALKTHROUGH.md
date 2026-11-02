# Project walkthrough

A guided tour of the project for a 30–45 minute session: what it does, a live demo,
one request followed through the code, how it's tested, and the main design decisions.

## 1. What it is

> A backend for a data analytics platform. You upload a spreadsheet, get statistics out of
> it, and can ask questions about it in plain English. An AI picks the right analysis and
> explains the answer, but it never writes database queries itself.

Three layers, bottom to top:

| Layer | Technology | Role |
|---|---|---|
| Storage | Parquet files | Every upload is cleaned and stored as Parquet |
| Analytics engine | DuckDB | Runs analytics directly on the Parquet files |
| AI analyst | Claude API | Turns a question into one of five analytics tools and explains the result |

Around them: FastAPI for the HTTP API, a SQL database (SQLite locally, Postgres in
production) for users and dataset records, and GitHub Actions for automated checks.

## 2. Live demo

### Setup (once)

```powershell
copy .env.example .env
.venv\Scripts\alembic upgrade head
```

Save this as `sales.csv` to upload during the demo:

```
date,region,product,units,revenue
2024-01-05,North,Widget,10,100
2024-01-20,South,Gadget,4,80
2024-02-03,North,Widget,7,70
2024-03-11,East,Gadget,5,100
```

### Run it

```powershell
.venv\Scripts\uvicorn app.main:app --reload
```

Open http://localhost:8000. It redirects to the interactive API docs, where every endpoint
can be called with "Try it out".

### Demo script

1. **`POST /api/v1/auth/register`**: create an account (email + password of 8+ characters).
2. **Authorize** (top right): email as *username*, plus the password. This is the login system.
3. **`POST /api/v1/datasets`**: upload `sales.csv`. The file is cleaned and stored as Parquet.
   Copy the `id` from the response.
4. **`GET /api/v1/datasets/{id}/profile`**: statistics for every column (nulls, distinct
   values, mean, quartiles, most common values).
5. **`POST /api/v1/datasets/{id}/analytics/aggregate`**: answer a business question,
   "which products generate the most revenue?":
   ```json
   {
     "group_by": ["product"],
     "metrics": [{"func": "sum", "column": "revenue", "alias": "revenue"}],
     "sort": [{"column": "revenue", "direction": "desc"}]
   }
   ```
   Expected: Gadget 180, Widget 170.
6. **`POST /api/v1/datasets/{id}/analytics/timeseries`**: revenue by month:
   ```json
   {"date_column": "date", "value_columns": ["revenue"], "frequency": "month"}
   ```
   Expected: January 180, February 70, March 100.
7. **`POST /api/v1/chat`**: ask the same question in plain English:
   ```json
   {"dataset_id": "PASTE-ID", "question": "Which products generate the most revenue?"}
   ```
   Without an Anthropic API key this returns `503 ai_unavailable`, which is expected. With a
   key in `.env` (`ANTHROPIC_API_KEY=...`), Claude picks `aggregate_dataset`, the server runs
   it, and Claude explains the result. Each question costs a few cents.

## 3. Follow one request through the code

### An analytics request (step 5 of the demo)

| Step | File | What to point out |
|---|---|---|
| 1 | `app/api/routes/analytics.py` | The endpoint is a few lines; it hands off to the service |
| 2 | `app/api/deps.py` | Login check, then "does this user own the dataset?", then the path to its Parquet file. Someone else's dataset returns 404, so IDs aren't revealed |
| 3 | `app/schemas/analytics.py` | The request is validated first: only known operators (`eq`, `gt`, `in`, …) and functions (`sum`, `mean`, …) are accepted |
| 4 | `app/services/analytics.py` → `aggregate` | The request becomes SQL. Column names are quoted and values are passed separately as parameters, so SQL injection is not possible |
| 5 | `app/services/duckdb_engine.py` | DuckDB queries the Parquet file in place, reading only the columns the query needs |

### An AI request (step 7)

It ends at the same analytics functions:

| Step | File | What to point out |
|---|---|---|
| 1 | `app/api/routes/chat.py` | Same login and ownership checks |
| 2 | `app/ai/service.py` | Claude is given the dataset's name, row count and column names/types, not the data |
| 3 | `app/ai/tools.py` | The five tools Claude can choose from: `profile_dataset`, `preview_dataset`, `aggregate_dataset`, `correlation_analysis`, `time_series_analysis`. Every tool call is validated before it runs |
| 4 | `app/ai/analyst.py` | The loop: send the question to Claude, validate the tool it picks, run it, send back the result, and return Claude's explanation |
| 5 | `app/ai/prompts.py` | The instructions Claude follows, including treating dataset contents as data, never as instructions |

A good test to show: `test_chat_recovers_from_invalid_tool_calls` in `tests/test_ai.py`.
A scripted Claude tries a made-up `run_sql` tool, then an unknown column; both are rejected
with an error it can read, and it succeeds on the third try.

## 4. How we know it works

```powershell
.venv\Scripts\python -m pytest -q
```

94 automated tests cover logins, uploads, every analytics operation, the AI layer (with a
fake Claude client, so no API key or cost), and the DuckDB engine checked against pandas'
results.

On GitHub, every pull request runs the **CI** workflow (`.github/workflows/ci.yml`):

- **lint**: code style and common bugs (ruff)
- **test**: the test suite on Python 3.12, 3.13 and 3.14, plus a check that database
  migrations match the models
- **docker**: builds the container image, starts it and waits for it to report healthy

`main` is protected: changes arrive through pull requests that pass all checks. Dependabot
opens weekly pull requests for dependency updates, which go through the same checks.

## 5. Design decisions

**Why DuckDB instead of pandas?**
pandas loads a whole dataset into memory before computing anything. DuckDB reads only the
columns it needs, straight from the file. On a 10-million-row test file, aggregation went
from 3.9 s and 4.3 GB of memory to 1.7 s and 0.24 GB, with identical results.

**Why doesn't the AI write SQL?**
Safety and predictability. Claude chooses from a fixed menu of validated tools, and the
tools only ever see the dataset the user is allowed to access. Even if the model were
tricked by text inside a dataset, it couldn't reach other users' data or run arbitrary
queries.

**Why store uploads as Parquet?**
It's a compressed, column-oriented format that DuckDB can query directly, so the original
upload format (CSV, Excel, JSON) doesn't matter after ingestion.

## 6. What's not built yet

- Rate limiting (best handled at a gateway or reverse proxy)
- Logging out / revoking tokens before they expire
- A background queue for very large uploads (they're processed while the request waits)
- Shared file storage for running on several servers
- Hosting and deployment

These are also listed under "Production notes and limits" in the [README](README.md).
