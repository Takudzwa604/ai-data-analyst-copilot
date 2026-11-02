import copy
import json
import uuid
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pandas as pd
import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.ai.analyst import echo_content
from app.ai.prompts import REFUSAL_ANSWER, SYSTEM_PROMPT
from app.ai.service import build_dataset_context, get_dataset_context
from app.ai.tools import TOOL_REGISTRY, AnalystTool, ToolContext, ToolError, strict_json_schema
from app.api import deps
from app.core.config import get_settings
from app.core.exceptions import NotFoundError
from app.db.session import SessionLocal
from app.main import app
from app.models import User
from tests.conftest import SALES_CSV, create_user_headers

EXPECTED_TOOLS = [
    "aggregate_dataset",
    "correlation_analysis",
    "preview_dataset",
    "profile_dataset",
    "time_series_analysis",
]
REVENUE_BY_PRODUCT = {
    "group_by": ["product"],
    "metrics": [{"func": "sum", "column": "revenue", "alias": "revenue"}],
    "sort": [{"column": "revenue", "direction": "desc"}],
}

# --------------------------------------------------------------------------- fake Claude


def text(value: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=value)


def tool_use(name: str, tool_input: Any, block_id: str = "toolu_1") -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=tool_input)


def message(*blocks: SimpleNamespace, stop_reason: str | None = None) -> SimpleNamespace:
    if stop_reason is None:
        stop_reason = "tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn"
    return SimpleNamespace(
        content=list(blocks), stop_reason=stop_reason, stop_details=None, model="claude-opus-5-5"
    )


class FakeMessages:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    async def create(self, **params: Any) -> Any:
        self.calls.append(copy.deepcopy(params))  # the service keeps appending to messages
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeAnthropic:
    def __init__(self, responses: list[Any]) -> None:
        self.beta = SimpleNamespace(messages=FakeMessages(responses))

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.beta.messages.calls


@pytest.fixture
def fake_claude() -> Iterator[Any]:
    def install(*responses: Any, **settings_overrides: Any) -> FakeAnthropic:
        fake = FakeAnthropic(list(responses))
        app.dependency_overrides[deps.get_anthropic_client] = lambda: fake
        if settings_overrides:
            settings = get_settings().model_copy(update=settings_overrides)
            app.dependency_overrides[get_settings] = lambda: settings
        return fake

    yield install
    app.dependency_overrides.clear()


async def ask(client: AsyncClient, headers: dict, dataset: dict, question: str) -> Any:
    return await client.post(
        "/api/v1/chat",
        headers=headers,
        json={"dataset_id": dataset["id"], "question": question},
    )


def _sales_df() -> pd.DataFrame:
    import io

    return pd.read_csv(io.BytesIO(SALES_CSV))


# --------------------------------------------------------------------------- dataset context


async def test_dataset_context_generation(sales_dataset: dict) -> None:
    async with SessionLocal() as db:
        owner = await db.scalar(select(User).where(User.email == "analyst@example.com"))
        context = await get_dataset_context(db, owner, uuid.UUID(sales_dataset["id"]))

    assert context.model_dump() == {
        "dataset_name": "Sales",
        "row_count": 6,
        "columns": [
            {"name": "date", "type": "categorical", "description": None},
            {"name": "region", "type": "categorical", "description": None},
            {"name": "product", "type": "categorical", "description": None},
            {"name": "units", "type": "numeric", "description": None},
            {"name": "revenue", "type": "numeric", "description": None},
            {"name": "returned", "type": "boolean", "description": None},
        ],
    }


async def test_dataset_context_enforces_ownership(sales_dataset: dict) -> None:
    stranger = User(id=uuid.uuid4(), email="x@example.com", hashed_password="x", is_superuser=False)
    async with SessionLocal() as db:
        with pytest.raises(NotFoundError):
            await get_dataset_context(db, stranger, uuid.UUID(sales_dataset["id"]))


def test_build_dataset_context_uses_stored_metadata_only() -> None:
    dataset = SimpleNamespace(
        name="Orders",
        row_count=42,
        columns=[{"name": "placed_at", "dtype": "datetime64[ns]", "kind": "datetime"}],
    )
    context = build_dataset_context(dataset)
    assert context.columns[0].type == "datetime"
    assert context.row_count == 42


# --------------------------------------------------------------------------- tool registry


def _schema_nodes(node: Any) -> Iterator[dict]:
    """Yield every schema object, skipping the field-name level of properties/$defs."""
    if isinstance(node, list):
        for item in node:
            yield from _schema_nodes(item)
    elif isinstance(node, dict):
        yield node
        for key, value in node.items():
            if key in ("properties", "$defs"):
                for sub in value.values():
                    yield from _schema_nodes(sub)
            else:
                yield from _schema_nodes(value)


def test_registry_exposes_strict_tool_definitions() -> None:
    definitions = TOOL_REGISTRY.definitions(strict=True)
    assert [d["name"] for d in definitions] == EXPECTED_TOOLS
    for definition in definitions:
        assert definition["strict"] is True
        assert definition["description"]
        for node in _schema_nodes(definition["input_schema"]):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
            assert not {"minimum", "maximum", "maxLength", "minItems", "default"} & node.keys()

    assert "strict" not in TOOL_REGISTRY.definitions(strict=False)[0]


def test_strict_schema_keeps_fields_named_like_keywords() -> None:
    from pydantic import BaseModel, Field

    class Odd(BaseModel):
        title: str
        default: int = Field(default=1, ge=0)

    schema = strict_json_schema(Odd)
    assert set(schema["properties"]) == {"title", "default"}
    assert "minimum" not in schema["properties"]["default"]


def test_registry_rejects_duplicate_registration() -> None:
    tool = TOOL_REGISTRY.parse("profile_dataset", {})[0]
    with pytest.raises(ValueError, match="already registered"):
        TOOL_REGISTRY.register(AnalystTool(tool.name, "dup", tool.input_model, tool.handler))


def test_registry_rejects_unknown_tool() -> None:
    with pytest.raises(ToolError, match="Unknown tool 'run_sql'"):
        TOOL_REGISTRY.parse("run_sql", {"query": "SELECT * FROM users"})


@pytest.mark.parametrize(
    ("name", "tool_input"),
    [
        ("aggregate_dataset", {}),  # metrics required
        ("aggregate_dataset", {"metrics": [{"func": "count"}], "sql": "DROP TABLE x"}),
        ("aggregate_dataset", {"metrics": [{"func": "sum"}]}),  # sum needs a column
        ("aggregate_dataset", {"metrics": [{"func": "median_ish", "column": "units"}]}),
        (
            "aggregate_dataset",
            {"metrics": [{"func": "count"}], "filters": [{"column": "units", "op": "in"}]},
        ),
        ("preview_dataset", {"limit": 5000}),
        ("time_series_analysis", {"date_column": "date", "value_columns": []}),
        ("correlation_analysis", {"columns": ["units"]}),  # needs two columns
        ("profile_dataset", "not an object"),
    ],
)
def test_registry_rejects_invalid_tool_input(name: str, tool_input: Any) -> None:
    context = ToolContext(source=_sales_df(), max_rows=100)
    with pytest.raises(ToolError, match="Invalid input"):
        TOOL_REGISTRY.execute(name, tool_input, context)


def test_registry_surfaces_analytics_errors() -> None:
    context = ToolContext(source=_sales_df(), max_rows=100)
    with pytest.raises(ToolError, match="Unknown column"):
        TOOL_REGISTRY.execute(
            "aggregate_dataset", {"group_by": ["sku"], "metrics": [{"func": "count"}]}, context
        )


def test_registry_executes_each_tool() -> None:
    context = ToolContext(source=_sales_df(), max_rows=100)
    output = TOOL_REGISTRY.execute("aggregate_dataset", REVENUE_BY_PRODUCT, context)
    assert output.rows == [
        {"product": "Widget", "revenue": 350.0},
        {"product": "Gadget", "revenue": 135.0},
    ]

    correlation = TOOL_REGISTRY.execute("correlation_analysis", {}, context)
    assert [row["column"] for row in correlation.rows] == ["units", "revenue"]

    series = TOOL_REGISTRY.execute(
        "time_series_analysis", {"date_column": "date", "value_columns": ["revenue"]}, context
    )
    assert [row["revenue"] for row in series.rows] == [240.0, 125.0, 120.0]

    assert len(TOOL_REGISTRY.execute("preview_dataset", {"limit": 2}, context).rows) == 2
    assert len(TOOL_REGISTRY.execute("profile_dataset", {}, context).rows) == 6


def test_preview_respects_server_row_cap() -> None:
    context = ToolContext(source=_sales_df(), max_rows=3)
    assert len(TOOL_REGISTRY.execute("preview_dataset", {"limit": 50}, context).rows) == 3


def test_echo_content_drops_declined_attempt_before_fallback() -> None:
    content = [
        SimpleNamespace(type="thinking"),
        text("partial"),
        tool_use("profile_dataset", {}, "toolu_old"),
        SimpleNamespace(type="fallback"),
        SimpleNamespace(type="thinking"),
        tool_use("aggregate_dataset", {}, "toolu_new"),
    ]
    kept = echo_content(content)
    assert [b.type for b in kept] == ["text", "fallback", "thinking", "tool_use"]
    assert kept[-1].id == "toolu_new"
    assert echo_content(content[:3]) == content[:3]  # no fallback: unchanged


# --------------------------------------------------------------------------- /chat endpoint


async def test_chat_selects_tool_and_summarises(
    client: AsyncClient, auth_headers: dict, sales_dataset: dict, fake_claude: Any
) -> None:
    fake = fake_claude(
        message(
            text("Let me total revenue by product."),
            tool_use("aggregate_dataset", REVENUE_BY_PRODUCT),
        ),
        message(text("Widget generates the most revenue: 350.0, versus 135.0 for Gadget.")),
    )

    response = await ask(
        client, auth_headers, sales_dataset, "Which products generate the most revenue?"
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"].startswith("Widget generates the most revenue")
    assert body["tool_used"] == "aggregate_dataset"
    assert body["results"] == [
        {"product": "Widget", "revenue": 350.0},
        {"product": "Gadget", "revenue": 135.0},
    ]
    assert body["tool_calls"] == [
        {"tool": "aggregate_dataset", "input": REVENUE_BY_PRODUCT, "is_error": False, "error": None}
    ]

    first, second = fake.calls
    assert first["model"] == "claude-opus-5-5"
    assert first["system"] == SYSTEM_PROMPT
    assert [t["name"] for t in first["tools"]] == EXPECTED_TOOLS
    assert all(t["strict"] for t in first["tools"])
    assert first["thinking"] == {"type": "adaptive"}
    assert first["output_config"] == {"effort": "medium"}
    assert first["fallbacks"] == "default"
    assert "tool_choice" not in first
    user_message = first["messages"][0]["content"]
    assert '"dataset_name": "Sales"' in user_message
    assert "Which products generate the most revenue?" in user_message

    assistant_turn, tool_turn = second["messages"][1:]
    assert assistant_turn["role"] == "assistant"
    [tool_result] = tool_turn["content"]
    assert tool_result["tool_use_id"] == "toolu_1"
    assert "is_error" not in tool_result
    assert json.loads(tool_result["content"])["rows"][0] == {"product": "Widget", "revenue": 350.0}


async def test_chat_recovers_from_invalid_tool_calls(
    client: AsyncClient, auth_headers: dict, sales_dataset: dict, fake_claude: Any
) -> None:
    fake = fake_claude(
        message(tool_use("run_sql", {"query": "SELECT 1"}, "toolu_a")),
        message(
            tool_use(
                "aggregate_dataset",
                {"group_by": ["sku"], "metrics": [{"func": "count"}]},
                "toolu_b",
            )
        ),
        message(tool_use("aggregate_dataset", REVENUE_BY_PRODUCT, "toolu_c")),
        message(text("Widget leads with 350.0 in revenue.")),
    )

    body = (await ask(client, auth_headers, sales_dataset, "Top products by revenue?")).json()

    assert body["tool_used"] == "aggregate_dataset"
    assert body["results"][0] == {"product": "Widget", "revenue": 350.0}
    assert [(c["tool"], c["is_error"]) for c in body["tool_calls"]] == [
        ("run_sql", True),
        ("aggregate_dataset", True),
        ("aggregate_dataset", False),
    ]
    assert "Unknown tool 'run_sql'" in body["tool_calls"][0]["error"]
    assert "Unknown column(s): sku" in body["tool_calls"][1]["error"]

    # Each rejected call went back to Claude as an error result it could act on.
    rejected = fake.calls[1]["messages"][-1]["content"][0]
    assert rejected == {
        "type": "tool_result",
        "tool_use_id": "toolu_a",
        "content": rejected["content"],
        "is_error": True,
    }
    assert rejected["content"].startswith("Error: Unknown tool")


async def test_chat_runs_parallel_tool_calls_in_one_turn(
    client: AsyncClient, auth_headers: dict, sales_dataset: dict, fake_claude: Any
) -> None:
    fake = fake_claude(
        message(
            tool_use("profile_dataset", {}, "toolu_1"),
            tool_use("aggregate_dataset", REVENUE_BY_PRODUCT, "toolu_2"),
        ),
        message(text("Done.")),
    )

    body = (await ask(client, auth_headers, sales_dataset, "Overview and top product?")).json()

    results_turn = fake.calls[1]["messages"][-1]
    assert results_turn["role"] == "user"
    assert [r["tool_use_id"] for r in results_turn["content"]] == ["toolu_1", "toolu_2"]
    assert body["tool_used"] == "aggregate_dataset"


async def test_chat_answers_without_tools(
    client: AsyncClient, auth_headers: dict, sales_dataset: dict, fake_claude: Any
) -> None:
    fake_claude(message(text("This dataset has no customer column, so I can't rank customers.")))

    body = (await ask(client, auth_headers, sales_dataset, "Who is our best customer?")).json()

    assert body["tool_used"] is None
    assert body["results"] == []
    assert body["tool_calls"] == []
    assert "no customer column" in body["answer"]


async def test_chat_round_limit_forces_text_answer(
    client: AsyncClient, auth_headers: dict, sales_dataset: dict, fake_claude: Any
) -> None:
    fake = fake_claude(
        message(tool_use("profile_dataset", {})),
        message(text("Based on the profile, revenue ranges from 45 to 120.")),
        ai_max_tool_rounds=1,
    )

    body = (await ask(client, auth_headers, sales_dataset, "Describe revenue")).json()

    assert body["tool_used"] == "profile_dataset"
    assert fake.calls[1]["tool_choice"] == {"type": "none"}


async def test_chat_handles_refusal(
    client: AsyncClient, auth_headers: dict, sales_dataset: dict, fake_claude: Any
) -> None:
    fake_claude(message(text(""), stop_reason="refusal"))

    body = (await ask(client, auth_headers, sales_dataset, "Something disallowed")).json()

    assert body["answer"] == REFUSAL_ANSWER
    assert body["tool_used"] is None


async def test_chat_never_runs_tools_from_truncated_response(
    client: AsyncClient, auth_headers: dict, sales_dataset: dict, fake_claude: Any
) -> None:
    fake_claude(message(tool_use("aggregate_dataset", {"metrics": []}), stop_reason="max_tokens"))

    response = await ask(client, auth_headers, sales_dataset, "Totals?")

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "ai_service_error"


async def test_chat_fallbacks_can_be_disabled(
    client: AsyncClient, auth_headers: dict, sales_dataset: dict, fake_claude: Any
) -> None:
    fake = fake_claude(message(text("ok")), ai_fallbacks_enabled=False)
    await ask(client, auth_headers, sales_dataset, "Hi")
    assert "fallbacks" not in fake.calls[0]
    assert "betas" not in fake.calls[0]


async def test_chat_requires_auth_and_ownership(
    client: AsyncClient, sales_dataset: dict, fake_claude: Any
) -> None:
    fake = fake_claude()
    unauthenticated = await client.post(
        "/api/v1/chat", json={"dataset_id": sales_dataset["id"], "question": "Hi"}
    )
    assert unauthenticated.status_code == 401

    other = await create_user_headers(client, email="eve@example.com")
    response = await ask(client, other, sales_dataset, "Which products generate the most revenue?")
    assert response.status_code == 404
    assert fake.calls == []  # Claude is never called for a dataset the user can't access


@pytest.mark.parametrize(
    "payload",
    [
        {"dataset_id": "not-a-uuid", "question": "Hi"},
        {"dataset_id": str(uuid.uuid4()), "question": ""},
        {"dataset_id": str(uuid.uuid4()), "question": "x" * 2001},
        {"question": "Hi"},
    ],
)
async def test_chat_validates_request(
    client: AsyncClient, auth_headers: dict, payload: dict
) -> None:
    response = await client.post("/api/v1/chat", headers=auth_headers, json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


_REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (anthropic.APIConnectionError(request=_REQUEST), 502, "ai_service_error"),
        (
            anthropic.AuthenticationError(
                "invalid x-api-key", response=httpx2.Response(401, request=_REQUEST), body=None
            ),
            503,
            "ai_unavailable",
        ),
        (
            anthropic.RateLimitError(
                "rate limited", response=httpx2.Response(429, request=_REQUEST), body=None
            ),
            503,
            "ai_unavailable",
        ),
        (
            anthropic.InternalServerError(
                "overloaded", response=httpx2.Response(500, request=_REQUEST), body=None
            ),
            502,
            "ai_service_error",
        ),
    ],
)
async def test_chat_maps_claude_api_errors(
    client: AsyncClient,
    auth_headers: dict,
    sales_dataset: dict,
    fake_claude: Any,
    error: Exception,
    status: int,
    code: str,
) -> None:
    fake_claude(error)
    response = await ask(client, auth_headers, sales_dataset, "Totals?")
    assert response.status_code == status
    assert response.json()["error"]["code"] == code


async def test_chat_reports_unconfigured_client(
    client: AsyncClient, auth_headers: dict, sales_dataset: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_credentials() -> None:
        raise anthropic.AnthropicError("no credentials")

    monkeypatch.setattr(deps, "_build_anthropic_client", no_credentials)
    response = await ask(client, auth_headers, sales_dataset, "Totals?")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "ai_unavailable"
