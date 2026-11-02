"""AnalystService: question -> dataset context -> Claude -> tool calls -> Claude summary.

A small manual tool-use loop (no agent framework): Claude picks tools from the registry,
we validate and execute them against the authorized DataFrame, and feed results back until
Claude answers or the round limit is reached.
"""

import json
import logging
from typing import Any

import anthropic
import pandas as pd
from fastapi.concurrency import run_in_threadpool

from app.ai.prompts import REFUSAL_ANSWER, SYSTEM_PROMPT, build_user_message
from app.ai.schemas import ChatResponse, DatasetContext, ToolCallRecord
from app.ai.tools import TOOL_REGISTRY, ToolContext, ToolError, ToolOutput, ToolRegistry
from app.core.config import Settings
from app.core.exceptions import AIServiceError, AIUnavailableError

logger = logging.getLogger(__name__)

_FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Blocks that must not be echoed back when they precede a mid-output refusal fallback.
_PRE_FALLBACK_DROP = {"thinking", "redacted_thinking", "tool_use"}


def echo_content(content: list[Any]) -> list[Any]:
    """Assistant content to send back on the next turn.

    Thinking blocks are passed back unchanged. If a refusal fallback happened mid-output,
    thinking and tool_use blocks before the last `fallback` marker belong to the declined
    attempt and are dropped.
    """
    last_fallback = max((i for i, b in enumerate(content) if b.type == "fallback"), default=-1)
    return [
        block
        for i, block in enumerate(content)
        if i > last_fallback or block.type not in _PRE_FALLBACK_DROP
    ]


def _text_of(content: list[Any]) -> str:
    return "\n\n".join(b.text for b in content if b.type == "text" and b.text.strip()).strip()


class AnalystService:
    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        settings: Settings,
        registry: ToolRegistry = TOOL_REGISTRY,
    ) -> None:
        self.client = client
        self.settings = settings
        self.registry = registry
        self._tools = registry.definitions(strict=settings.ai_strict_tools)

    async def answer(
        self, *, context: DatasetContext, question: str, df: pd.DataFrame
    ) -> ChatResponse:
        tool_context = ToolContext(
            df=df,
            max_rows=min(self.settings.max_result_rows, self.settings.ai_max_tool_rows),
        )
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": build_user_message(context, question)}
        ]
        calls: list[ToolCallRecord] = []
        last_success: tuple[str, ToolOutput] | None = None
        max_rounds = self.settings.ai_max_tool_rounds

        for round_number in range(max_rounds + 1):
            # After the last tool round, tools stay defined but Claude must answer in text.
            allow_tools = round_number < max_rounds
            response = await self._create(messages, allow_tools=allow_tools)

            if response.stop_reason == "refusal":
                logger.info(
                    "AI analyst request refused", extra={"stop_details": str(response.stop_details)}
                )
                answer = REFUSAL_ANSWER
                break
            if response.stop_reason == "max_tokens":
                # A truncated tool_use input can still parse; never execute it.
                raise AIServiceError("The AI response was cut off before it finished")

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason != "tool_use" or not tool_uses or not allow_tools:
                answer = _text_of(response.content) or "I couldn't produce an answer."
                break

            messages.append({"role": "assistant", "content": echo_content(response.content)})
            results = []
            for block in tool_uses:
                record, output, tool_result = await self._run_tool(block, tool_context)
                calls.append(record)
                results.append(tool_result)
                if output is not None:
                    last_success = (block.name, output)
            # All results for one turn go back in a single user message.
            messages.append({"role": "user", "content": results})

        logger.info(
            "AI analyst answered",
            extra={"tool_calls": len(calls), "tool_used": last_success and last_success[0]},
        )
        return ChatResponse(
            answer=answer,
            tool_used=last_success[0] if last_success else None,
            results=last_success[1].rows if last_success else [],
            tool_calls=calls,
            model=response.model,
        )

    async def _run_tool(
        self, block: Any, context: ToolContext
    ) -> tuple[ToolCallRecord, ToolOutput | None, dict[str, Any]]:
        raw_input = block.input if isinstance(block.input, dict) else {}
        try:
            output = await run_in_threadpool(
                self.registry.execute, block.name, block.input, context
            )
        except ToolError as exc:
            logger.info("Rejected AI tool call", extra={"tool": block.name, "error": str(exc)})
            return (
                ToolCallRecord(tool=block.name, input=raw_input, is_error=True, error=str(exc)),
                None,
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": f"Error: {exc}",
                    "is_error": True,
                },
            )
        return (
            ToolCallRecord(tool=block.name, input=raw_input, is_error=False),
            output,
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(output.payload, default=str),
            },
        )

    async def _create(self, messages: list[dict[str, Any]], *, allow_tools: bool) -> Any:
        params: dict[str, Any] = {
            "model": self.settings.ai_model,
            "max_tokens": self.settings.ai_max_tokens,
            "system": SYSTEM_PROMPT,
            "tools": self._tools,
            "messages": messages,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.settings.ai_effort},
            "cache_control": {"type": "ephemeral"},
        }
        if not allow_tools:
            params["tool_choice"] = {"type": "none"}
        if self.settings.ai_fallbacks_enabled:
            params["betas"] = [_FALLBACK_BETA]
            params["fallbacks"] = "default"

        try:
            return await self.client.beta.messages.create(**params)
        except anthropic.AuthenticationError as exc:
            logger.error("Anthropic authentication failed: %s", exc.message)
            raise AIUnavailableError("The AI analyst is not configured correctly") from exc
        except anthropic.RateLimitError as exc:
            raise AIUnavailableError("The AI analyst is busy; please retry shortly") from exc
        except anthropic.APIStatusError as exc:
            logger.error(
                "Anthropic API error %s (request %s): %s",
                exc.status_code,
                exc.request_id,
                exc.message,
            )
            raise AIServiceError() from exc
        except anthropic.APIConnectionError as exc:
            logger.error("Could not reach the Anthropic API: %s", exc)
            raise AIServiceError("The AI analyst could not be reached") from exc
