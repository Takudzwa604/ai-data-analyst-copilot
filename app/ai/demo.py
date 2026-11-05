"""Demo mode: answer plain-English questions without calling Claude, at no cost.

A keyword matcher stands in for the model. It picks a tool from the same registry and runs
it on the real data through the same validation, then fills in a template answer. It only
understands simple phrasings and cannot apply filters; answers say that they came from demo
mode. Enabled with AI_DEMO_MODE=true.
"""

import logging
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fastapi.concurrency import run_in_threadpool

from app.ai.schemas import ChatResponse, ColumnContext, DatasetContext, ToolCallRecord
from app.ai.tools import TOOL_REGISTRY, ToolContext, ToolError, ToolOutput, ToolRegistry
from app.core.config import Settings
from app.schemas.analytics import AggFunc, TimeFrequency
from app.services.duckdb_engine import DataSource

logger = logging.getLogger(__name__)

DEMO_MODEL = "demo-mode (no AI)"
_PREFIX = "[Demo mode] "
_TOP_IN_ANSWER = 5
_DEFAULT_LIMIT = 10
# A token shared by more columns than this ("sale" in sale_price, sale_date, ...) says too
# little to identify a column by itself.
_MAX_COLUMNS_FOR_WEAK_MATCH = 3

_STOPWORDS = frozenset(
    "a an and are at be by can do does did each every for from had has have how i in is it its "  # noqa: SIM905
    "less many me more most much my of on or our per same show than that the their there they "
    "this to top was were what when where which who will with within".split()
)
_GROUP_WORDS = frozenset({"by", "per", "each", "every", "across"})
_FILLER_WORDS = frozenset({"the", "each", "every", "a"})

# First match wins, so "average" beats "how many" in "how many parcels on average?".
_FUNC_PATTERNS = [
    (AggFunc.MEAN, r"\b(average|avg|mean)\b"),
    (AggFunc.MEDIAN, r"\bmedian\b"),
    (AggFunc.SUM, r"\b(total|sum)\b"),
    (AggFunc.NUNIQUE, r"\b(distinct|unique)\b"),
    (AggFunc.COUNT, r"\b(how many|number of|count)\b"),
    (AggFunc.MAX, r"\b(max|maximum)\b"),
    (AggFunc.MIN, r"\b(min|minimum)\b"),
]
_NUMERIC_FUNCS = {AggFunc.MEAN, AggFunc.MEDIAN, AggFunc.SUM, AggFunc.MAX, AggFunc.MIN}
_FUNC_LABELS = {
    AggFunc.MEAN: "average",
    AggFunc.MEDIAN: "median",
    AggFunc.SUM: "total",
    AggFunc.NUNIQUE: "distinct count",
    AggFunc.COUNT: "count",
    AggFunc.MAX: "maximum",
    AggFunc.MIN: "minimum",
}

_OVERVIEW_RE = re.compile(
    r"\b(missing|nulls?|blanks?|empty|quality|overview|summari[sz]e|summary|describe|profile|"
    r"what columns|which columns|about)\b"
)
_CORRELATION_RE = re.compile(r"\b(correlat\w*|relationship|related)\b")
_TREND_RE = re.compile(
    r"\b(trends?|over time|growth|seasonal\w*|hourly|daily|weekly|monthly|quarterly|yearly|"
    r"annual\w*)\b"
)
_PERIOD_RE = re.compile(r"\b(?:by|per|each|every)\s+(hour|day|week|month|quarter|year)\b")
_FREQUENCIES = [
    (TimeFrequency.HOUR, r"\b(hour|hourly)\b"),
    (TimeFrequency.DAY, r"\b(day|daily)\b"),
    (TimeFrequency.WEEK, r"\b(week|weekly)\b"),
    (TimeFrequency.MONTH, r"\b(month|monthly)\b"),
    (TimeFrequency.QUARTER, r"\b(quarter|quarterly)\b"),
    (TimeFrequency.YEAR, r"\b(year|yearly|annual\w*)\b"),
]
_RANKING_RE = re.compile(
    r"\b(most|top|least|fewest|highest|lowest|best|worst|biggest|largest|smallest|bottom|"
    r"common|frequent|popular|rarest)\b"
)
_LOW_END_RE = re.compile(r"\b(least|fewest|lowest|worst|smallest|bottom|rarest)\b")
_EXTREME_RE = re.compile(r"\b(highest|lowest|biggest|largest|smallest)\b")
_ASKS_WHICH_RE = re.compile(r"\b(which|what|who)\b")
_LIMIT_RE = re.compile(
    r"\b(?:top|bottom|first|last)\s+(\d{1,3})\b|\b(\d{1,3})\s+(?:rows|records)\b"
)
_PREVIEW_RE = re.compile(r"\b(show|examples?|samples?|preview|rows|records|look like)\b")
_ROWS_RE = re.compile(r"\b(rows|records|entries)\b")
_SUBSET_RE = re.compile(
    r"\b(where|only|excluding|except|between|before|after|during|since|above|below|"
    r"(?:greater|less|more|fewer) than|(?:over|under) \d+\w*|(?:19|20)\d\d)\b"
    r"|\b(?:in|for|from|at|on)\s+"
    r"(?!each\b|every\b|all\b|total\b|average\b|the dataset\b|the data\b|this\b|my\b|our\b)\w+"
)

_HELP = (
    "I couldn't match that to an analysis. Demo mode understands simple questions such as "
    "'which <column> is most common', 'average <column> by <column>', 'how many rows', "
    "'trend over time', 'how are the numeric columns correlated', 'any missing values' and "
    "'show some rows'. Refer to columns by name. This dataset's columns are: {columns}."
)
_NO_FILTER_NOTE = (
    " Note: demo mode can't filter, so this covers every row even if your question was "
    "about a subset."
)
_TEXT_COLUMN_HINT = (
    " Columns stored as text, such as prices written like '$1,000', need cleaning before "
    "they can be used in calculations."
)

Summarize = Callable[[ToolOutput], str]


@dataclass(frozen=True)
class Plan:
    tool: str
    input: dict[str, Any]
    summarize: Summarize


@dataclass(frozen=True)
class Mention:
    column: ColumnContext
    start: int  # position of the first matching word in the question


# --------------------------------------------------------------------------- matching


def _tokens(text: str) -> list[str]:
    """Lower-case words without plurals, so 'Townships' matches the column 'township_code'."""
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    return [_singular(word) for word in re.findall(r"[a-z0-9]+", spaced.lower())]


def _singular(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return f"{word[:-3]}y"
    if word.endswith(("sses", "xes", "ches", "shes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _find_mentions(columns: list[ColumnContext], words: list[str]) -> list[Mention]:
    """Columns the question refers to, in question order. Each word counts for one column."""
    column_words = {
        c.name: [w for w in dict.fromkeys(_tokens(c.name)) if w not in _STOPWORDS] for c in columns
    }
    shared_by = Counter(w for names in column_words.values() for w in names)
    available = {i: w for i, w in enumerate(words) if w not in _STOPWORDS}
    remaining = list(columns)
    mentions = []

    while True:
        best = None
        best_key = (0, 0.0)
        for column in remaining:
            names = column_words[column.name]
            hits = [next((i for i, w in available.items() if w == name), None) for name in names]
            hits = [i for i in hits if i is not None]
            distinctive = len(hits) == len(names) or any(
                shared_by[words[i]] <= _MAX_COLUMNS_FOR_WEAK_MATCH for i in hits
            )
            if not hits or not distinctive:
                continue
            key = (len(hits), len(hits) / len(names))
            if key > best_key:
                best, best_key, best_hits = column, key, hits
        if best is None:
            return sorted(mentions, key=lambda m: m.start)
        mentions.append(Mention(best, min(best_hits)))
        remaining.remove(best)
        for i in best_hits:
            del available[i]


def _follows_group_word(mention: Mention, words: list[str]) -> bool:
    before = words[max(mention.start - 2, 0) : mention.start]
    if not before:
        return False
    if before[-1] in _GROUP_WORDS:
        return True
    return len(before) == 2 and before[-1] in _FILLER_WORDS and before[0] in _GROUP_WORDS


def _explicit_func(question: str) -> AggFunc | None:
    return next((func for func, pattern in _FUNC_PATTERNS if re.search(pattern, question)), None)


def _limit(question: str, default: int, maximum: int) -> int:
    match = _LIMIT_RE.search(question)
    return max(1, min(int(match.group(1) or match.group(2)), maximum)) if match else default


# --------------------------------------------------------------------------- planning


def plan_question(context: DatasetContext, question: str) -> Plan | str:
    """The tool call for a question, or the text to reply with when there isn't one."""
    lowered = question.lower()
    words = _tokens(question)
    mentions = _find_mentions(context.columns, words)
    column_names = ", ".join(c.name for c in context.columns)

    if _OVERVIEW_RE.search(lowered):
        return Plan("profile_dataset", {}, _summarize_profile)
    if _CORRELATION_RE.search(lowered):
        return _plan_correlation(mentions)
    if _wants_time_series(lowered, mentions):
        return _plan_time_series(context, lowered, mentions)

    aggregate = _plan_aggregate(lowered, words, mentions)
    if aggregate is not None:
        return aggregate
    if _PREVIEW_RE.search(lowered):
        limit = _limit(lowered, default=5, maximum=50)
        return Plan("preview_dataset", {"limit": limit}, _summarize_preview)
    return _HELP.format(columns=column_names)


def _plan_correlation(mentions: list[Mention]) -> Plan:
    numeric = [m.column.name for m in mentions if m.column.type == "numeric"]
    columns = numeric if len(numeric) >= 2 else []
    return Plan("correlation_analysis", {"columns": columns}, _summarize_correlation)


def _wants_time_series(question: str, mentions: list[Mention]) -> bool:
    if _TREND_RE.search(question):
        return True
    period = _PERIOD_RE.search(question)
    # "sales by year" on a dataset with a `year` column is an ordinary group-by.
    return period is not None and not any(
        period.group(1) in _tokens(m.column.name) for m in mentions
    )


def _is_date_column(column: ColumnContext) -> bool:
    if column.type == "datetime":
        return True
    words = _tokens(column.name)
    is_text = column.type == "categorical"
    return is_text and (
        bool({"date", "time", "timestamp", "datetime"} & set(words))
        or column.name.lower().endswith("_at")
    )


def _plan_time_series(
    context: DatasetContext, question: str, mentions: list[Mention]
) -> Plan | str:
    mentioned_dates = [m.column for m in mentions if _is_date_column(m.column)]
    candidates = sorted(
        mentioned_dates or [c for c in context.columns if _is_date_column(c)],
        key=lambda c: c.type != "datetime",
    )
    if not candidates:
        return "This dataset has no date or time column, so I can't show a trend."
    date_column = candidates[0]

    frequency = next(
        (freq for freq, pattern in _FREQUENCIES if re.search(pattern, question)),
        TimeFrequency.MONTH,
    )
    values = [
        m.column.name for m in mentions if m.column.type == "numeric" and m.column != date_column
    ]
    func = _explicit_func(question)
    if values:
        agg = func if func in _NUMERIC_FUNCS else AggFunc.SUM
    else:
        # No measure named: count how many rows fall in each period.
        others = [c for c in context.columns if c != date_column]
        if not others:
            return "This dataset has nothing but its date column, so there is nothing to trend."
        values, agg = [others[0].name], AggFunc.COUNT

    tool_input = {
        "date_column": date_column.name,
        "value_columns": values,
        "frequency": frequency.value,
        "agg": agg.value,
    }
    return Plan("time_series_analysis", tool_input, _time_series_summary(tool_input))


def _plan_aggregate(question: str, words: list[str], mentions: list[Mention]) -> Plan | str | None:
    func = _explicit_func(question)
    group_hint = next((m for m in mentions if _follows_group_word(m, words)), None)
    others = [m for m in mentions if m is not group_hint]
    group, value = group_hint, others[0] if others else None

    # "How many rows?" means the whole table, even if a column is called row_id.
    if group is None and func == AggFunc.COUNT and _ROWS_RE.search(question):
        return _aggregate_plan(question, None, AggFunc.COUNT, None)

    if group is None and len(others) >= 2:
        # "Which products generate the most revenue?" names the group first; "average revenue
        # of products" names the value first.
        first_is_value = func is not None and func != AggFunc.COUNT
        value, group = (others[0], others[1]) if first_is_value else (others[1], others[0])
    elif group is None and value is not None and func is None:
        if _EXTREME_RE.search(question) and value.column.type == "numeric":
            func = AggFunc.MIN if _LOW_END_RE.search(question) else AggFunc.MAX
        elif _RANKING_RE.search(question) or _ASKS_WHICH_RE.search(question):
            group, value = value, None
        else:
            return None

    # A true/false column is answered by counting each side, not by counting 2 distinct values.
    is_flag = value is not None and value.column.type == "boolean"
    if group is None and is_flag and func in (None, AggFunc.COUNT):
        group, value = value, None

    if group is None and value is None:
        if func == AggFunc.COUNT or (_ROWS_RE.search(question) and func is not None):
            return _aggregate_plan(question, None, AggFunc.COUNT, None)
        return None

    if value is None:
        if func in (AggFunc.MEAN, AggFunc.MEDIAN, AggFunc.MAX, AggFunc.MIN, AggFunc.NUNIQUE):
            return f"Which column should I use for the {_FUNC_LABELS[func]}? Please name it."
        return _aggregate_plan(question, group, AggFunc.COUNT, None)

    if func == AggFunc.COUNT or (func is None and value.column.type != "numeric"):
        func = AggFunc.NUNIQUE  # "how many buyers per township"
    elif func is None:
        func = AggFunc.SUM  # "which products generate the most revenue"
    return _aggregate_plan(question, group, func, value)


def _aggregate_plan(
    question: str, group: Mention | None, func: AggFunc, value: Mention | None
) -> Plan:
    metric: dict[str, Any] = {"func": func.value}
    if value is not None:
        metric["column"] = value.column.name
    output = f"{func.value}_{value.column.name}" if value is not None else "count"
    tool_input: dict[str, Any] = {"metrics": [metric]}
    label = f"{_FUNC_LABELS[func]} of {value.column.name}" if value is not None else "row count"

    if group is None:
        return Plan("aggregate_dataset", tool_input, _overall_summary(output, label))

    direction = "asc" if _LOW_END_RE.search(question) else "desc"
    tool_input.update(
        group_by=[group.column.name],
        sort=[{"column": output, "direction": direction}],
        limit=_limit(question, default=_DEFAULT_LIMIT, maximum=100),
    )
    return Plan(
        "aggregate_dataset",
        tool_input,
        _grouped_summary(group.column.name, output, label, direction),
    )


# --------------------------------------------------------------------------- answers


def _format(value: Any) -> str:
    if value is None:
        return "(empty)"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.0f}" if value.is_integer() else f"{value:,.2f}"
    return str(value)


def _label(value: Any) -> str:
    """A group's name: identifiers such as the year 2023 get no thousands separator."""
    return str(value).lower() if isinstance(value, bool) else str(value)


def _overall_summary(output: str, label: str) -> Summarize:
    def summarize(result: ToolOutput) -> str:
        return f"The {label} across the dataset is {_format(result.rows[0][output])}."

    return summarize


def _grouped_summary(group: str, output: str, label: str, direction: str) -> Summarize:
    def summarize(result: ToolOutput) -> str:
        named = [row for row in result.rows if row[group] is not None]
        if not named:
            return f"No rows to group by {group}."
        listed = ", ".join(
            f"{_label(row[group])} ({_format(row[output])})" for row in named[:_TOP_IN_ANSWER]
        )
        end = "lowest" if direction == "asc" else "highest"
        empty = next((row for row in result.rows if row[group] is None), None)
        note = f" Rows with no {group} are left out of this list." if empty else ""
        return (
            f"{group} with the {end} {label}: {listed}. "
            f"There are {_format(result.payload['row_count'])} different {group} values in total."
            f"{note}"
        )

    return summarize


def _time_series_summary(tool_input: dict[str, Any]) -> Summarize:
    date_column = tool_input["date_column"]
    value = tool_input["value_columns"][0]
    agg = AggFunc(tool_input["agg"])
    label = "rows with a value" if agg == AggFunc.COUNT else f"{_FUNC_LABELS[agg]} {value}"

    def summarize(result: ToolOutput) -> str:
        if not result.rows:
            return (
                f"No usable dates found in {date_column}. Text dates must look like "
                "'2024-01-31'; a format like 'January 12, 2023' isn't recognised yet."
            )
        rows = [r for r in result.rows if r[value] is not None]
        peak = max(rows, key=lambda r: r[value])
        return (
            f"Using {date_column}, grouped by {tool_input['frequency']}: {len(result.rows)} "
            f"periods from {str(result.rows[0]['period'])[:10]} to "
            f"{str(result.rows[-1]['period'])[:10]}. The peak was {str(peak['period'])[:10]}, "
            f"with {_format(peak[value])} ({label})."
        )

    return summarize


def _summarize_profile(result: ToolOutput) -> str:
    payload = result.payload
    missing = sorted(
        (c for c in payload["columns"] if c["null_count"]),
        key=lambda c: c["null_count"],
        reverse=True,
    )
    listed = ", ".join(f"{c['name']} ({c['null_pct']}%)" for c in missing[:_TOP_IN_ANSWER])
    gaps = f"Most missing values: {listed}." if missing else "No column has missing values."
    return (
        f"The dataset has {_format(payload['row_count'])} rows and {payload['column_count']} "
        f"columns, with {_format(payload['duplicate_rows'])} duplicate rows. {gaps}"
    )


def _summarize_preview(result: ToolOutput) -> str:
    total = _format(result.payload["total_rows"])
    return f"Here are the first {len(result.rows)} of {total} rows."


def _summarize_correlation(result: ToolOutput) -> str:
    columns = result.payload["columns"]
    matrix = result.payload["matrix"]
    pairs = [
        (matrix[i][j], columns[i], columns[j])
        for i in range(len(columns))
        for j in range(i + 1, len(columns))
        if matrix[i][j] is not None
    ]
    if not pairs:
        return "None of these columns vary enough to compute a correlation."
    r, a, b = max(pairs, key=lambda p: abs(p[0]))
    strength = "weak" if abs(r) < 0.3 else "moderate" if abs(r) < 0.7 else "strong"
    return f"The strongest correlation is between {a} and {b}: r = {r:.2f}, which is {strength}."


# --------------------------------------------------------------------------- service


class DemoAnalystService:
    """Same interface as AnalystService, without any calls to the Claude API."""

    def __init__(self, settings: Settings, registry: ToolRegistry = TOOL_REGISTRY) -> None:
        self.settings = settings
        self.registry = registry

    async def answer(
        self, *, context: DatasetContext, question: str, source: DataSource
    ) -> ChatResponse:
        plan = plan_question(context, question)
        if isinstance(plan, str):
            return _response(plan)

        tool_context = ToolContext(
            source=source,
            max_rows=min(self.settings.max_result_rows, self.settings.ai_max_tool_rows),
        )
        try:
            output = await run_in_threadpool(
                self.registry.execute, plan.tool, plan.input, tool_context
            )
        except ToolError as exc:
            logger.info("Demo analyst tool call rejected", extra={"tool": plan.tool})
            record = ToolCallRecord(tool=plan.tool, input=plan.input, is_error=True, error=str(exc))
            return _response(f"I can't answer that: {exc}.{_TEXT_COLUMN_HINT}", calls=[record])

        answer = plan.summarize(output)
        if plan.tool != "preview_dataset" and _SUBSET_RE.search(question.lower()):
            answer += _NO_FILTER_NOTE
        record = ToolCallRecord(tool=plan.tool, input=plan.input, is_error=False)
        return _response(answer, tool=plan.tool, rows=output.rows, calls=[record])


def _response(
    answer: str,
    *,
    tool: str | None = None,
    rows: list[dict[str, Any]] | None = None,
    calls: list[ToolCallRecord] | None = None,
) -> ChatResponse:
    return ChatResponse(
        answer=f"{_PREFIX}{answer}",
        tool_used=tool,
        results=rows or [],
        tool_calls=calls or [],
        model=DEMO_MODEL,
    )
