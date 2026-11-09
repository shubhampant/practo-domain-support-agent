
"""Deterministic, offline AutoGen review stage for the Practo capstone."""

from __future__ import annotations

import sys
import asyncio
import json
import re
from typing import Any, AsyncGenerator, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from autogen_core import CancellationToken
from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    ModelInfo,
    RequestUsage,
)
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.teams import RoundRobinGroupChat


class ReviewVerdict(BaseModel):
    """Structured governance verdict returned by the Final-Editor."""

    model_config = ConfigDict(extra="forbid")

    approved: bool = Field(
        description="True only when the original draft is approved unchanged."
    )
    final_answer: str = Field(
        description="Approved original draft or its corrected version."
    )
    reason: str = Field(
        description="Brief explanation of the approval or revision."
    )


def _message_text(messages: Sequence[Any]) -> str:
    """Combine textual message contents for deterministic mock processing."""
    parts: list[str] = []

    for message in messages:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(
                item for item in content if isinstance(item, str)
            )

    return "\n".join(parts)


def _extract_task_payload(text: str) -> dict[str, Any]:
    """Extract the JSON payload supplied to the review team."""
    marker = "TASK_PAYLOAD_JSON:"
    marker_position = text.rfind(marker)

    if marker_position < 0:
        return {}

    payload_text = text[marker_position + len(marker):].lstrip()

    try:
        payload, _ = json.JSONDecoder().raw_decode(payload_text)
    except json.JSONDecodeError:
        return {}

    return payload if isinstance(payload, dict) else {}


def _unsupported_claim(draft: str, context: str) -> bool:
    """Detect the deliberately unsupported refund guarantee in MOCK_LLM mode."""
    draft_lower = draft.casefold()
    context_lower = context.casefold()

    unsupported_phrases = (
        "guarantees a full refund",
        "guaranteed full refund",
        "guaranteed refund for every cancellation",
    )

    return any(
        phrase in draft_lower and phrase not in context_lower
        for phrase in unsupported_phrases
    )


def _remove_unsupported_claim(draft: str) -> str:
    """Remove the deliberately unsupported refund-guarantee sentence."""
    sentences = re.split(r"(?<=[.!?])\s+", draft.strip())

    retained = [
        sentence
        for sentence in sentences
        if not any(
            phrase in sentence.casefold()
            for phrase in (
                "guarantees a full refund",
                "guaranteed full refund",
                "guaranteed refund for every cancellation",
            )
        )
    ]

    return " ".join(retained).strip()


class DeterministicReviewClient(ChatCompletionClient):
    """Small mock client; performs no network calls and requires no API key."""

    def __init__(self, role: str) -> None:
        super().__init__()
        if role not in {"reviewer", "editor"}:
            raise ValueError("role must be 'reviewer' or 'editor'")

        self.role = role
        self._prompt_tokens = 0
        self._completion_tokens = 0

    @property
    def model_info(self) -> ModelInfo:
        return {
            "vision": False,
            "function_calling": False,
            "json_output": True,
            "family": "unknown",
            "structured_output": True,
            "multiple_system_messages": True,
        }

    @property
    def capabilities(self) -> Any:
        return {
            "vision": False,
            "function_calling": False,
            "json_output": True,
        }

    def actual_usage(self) -> RequestUsage:
        return RequestUsage(
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
        )

    def total_usage(self) -> RequestUsage:
        return self.actual_usage()

    def count_tokens(
        self,
        messages: Sequence[Any],
        *,
        tools: Sequence[Any] = (),
    ) -> int:
        return max(1, len(_message_text(messages)) // 4)

    def remaining_tokens(
        self,
        messages: Sequence[Any],
        *,
        tools: Sequence[Any] = (),
    ) -> int:
        return max(0, 8192 - self.count_tokens(messages, tools=tools))

    async def create(
        self,
        messages: Sequence[Any],
        *,
        tools: Sequence[Any] = (),
        tool_choice: Any = "auto",
        json_output: Any = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: CancellationToken | None = None,
    ) -> CreateResult:
        text = _message_text(messages)
        prompt_tokens = self.count_tokens(messages, tools=tools)
        payload = _extract_task_payload(text)

        draft = str(payload.get("draft", ""))
        context = str(payload.get("context", ""))
        unsupported = _unsupported_claim(draft, context)

        if self.role == "reviewer":
            if unsupported:
                response = (
                    "Policy review: the draft contains a refund guarantee "
                    "that is not supported by the supplied context. "
                    "The Final-Editor must remove that claim."
                )
            else:
                response = (
                    "Policy review: no configured unsupported refund "
                    "guarantee was found. The draft can be approved unchanged."
                )
        else:
            if unsupported:
                verdict = ReviewVerdict(
                    approved=False,
                    final_answer=_remove_unsupported_claim(draft),
                    reason=(
                        "Revised the draft because the supplied context "
                        "does not substantiate its refund guarantee."
                    ),
                )
            else:
                verdict = ReviewVerdict(
                    approved=True,
                    final_answer=draft,
                    reason=(
                        "The draft contains no configured unsupported "
                        "refund guarantee and is approved unchanged."
                    ),
                )

            response = verdict.model_dump_json()

        completion_tokens = max(1, len(response) // 4)
        self._prompt_tokens += prompt_tokens
        self._completion_tokens += completion_tokens

        return CreateResult(
            finish_reason="stop",
            content=response,
            usage=RequestUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ),
            cached=False,
        )

    async def create_stream(
        self,
        messages: Sequence[Any],
        *,
        tools: Sequence[Any] = (),
        tool_choice: Any = "auto",
        json_output: Any = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: CancellationToken | None = None,
    ) -> AsyncGenerator[str | CreateResult, None]:
        result = await self.create(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            json_output=json_output,
            extra_create_args=extra_create_args,
            cancellation_token=cancellation_token,
        )
        yield result

    async def close(self) -> None:
        return None


def build_review_team() -> RoundRobinGroupChat:
    """Build the two-agent review team with structured-message registration."""

    reviewer = AssistantAgent(
        name="Policy_Compliance_Reviewer",
        model_client=DeterministicReviewClient("reviewer"),
        system_message=(
            "You are the Policy-Compliance-Reviewer. Compare the draft "
            "with the supplied retrieved context. Flag unsupported claims. "
            "Do not invent policy facts. Hand the review to Final-Editor."
        ),
    )

    editor = AssistantAgent(
        name="Final_Editor",
        model_client=DeterministicReviewClient("editor"),
        system_message=(
            "You are the Final-Editor. Return the required structured verdict. "
            "Approve the draft unchanged when no configured unsupported "
            "refund guarantee is present. Otherwise remove that claim. "
            "Use only the provided draft and context."
        ),
        output_content_type=ReviewVerdict,
    )

    return RoundRobinGroupChat(
        participants=[reviewer, editor],
        max_turns=2,
        custom_message_types=[StructuredMessage[ReviewVerdict]],
    )


async def review_draft(
    *,
    query: str,
    draft: str,
    context: str,
) -> dict[str, Any]:
    """Review a draft and return its structured verdict and execution trace."""

    payload = {
        "query": query,
        "draft": draft,
        "context": context,
    }
    task = (
        "Review this CrewAI draft against the original retrieved context.\n"
        "TASK_PAYLOAD_JSON:"
        + json.dumps(payload, ensure_ascii=False)
    )

    team = build_review_team()
    result = await team.run(task=task)

    verdict: ReviewVerdict | None = None
    trace: list[dict[str, str]] = []

    for message in result.messages:
        source = str(getattr(message, "source", "unknown"))
        content = getattr(message, "content", "")
        trace.append(
            {
                "source": source,
                "message_type": type(message).__name__,
                "content": (
                    content.model_dump_json()
                    if isinstance(content, BaseModel)
                    else str(content)
                ),
            }
        )

        if isinstance(content, ReviewVerdict):
            verdict = content
        elif isinstance(message, StructuredMessage):
            if isinstance(message.content, ReviewVerdict):
                verdict = message.content

    if verdict is None:
        raise RuntimeError(
            "AutoGen completed without returning a ReviewVerdict."
        )

    return {
        "query": query,
        "verdict": verdict.model_dump(),
        "trace": trace,
        "turn_count": sum(
            1 for message in result.messages
            if getattr(message, "source", None) in {
                "Policy_Compliance_Reviewer",
                "Final_Editor",
            }
        ),
    }


async def _demo() -> None:
    """Run the two required deterministic acceptance examples."""

    context = (
        "Cancellation or rescheduling requests should be submitted at least "
        "24 hours before the appointment. This policy does not promise refunds."
    )

    examples = [
        {
            "query": "What is the cancellation window?",
            "draft": (
                "Cancellation or rescheduling requests should be submitted "
                "at least 24 hours before the appointment."
            ),
            "context": context,
        },
        {
            "query": "Can I get a refund for cancellation?",
            "draft": (
                "Cancellation or rescheduling requests should be submitted "
                "at least 24 hours before the appointment. "
                "Our clinic guarantees a full refund for every cancellation."
            ),
            "context": context,
        },
    ]

    for index, example in enumerate(examples, start=1):
        result = await review_draft(**example)
        print(f"\n=== Demo {index} ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))


async def _run_json_cli() -> None:
    """Read one JSON payload from stdin and emit one JSON result."""

    payload = json.load(sys.stdin)

    result = await review_draft(
        query=payload["query"],
        draft=payload["draft"],
        context=payload["context"],
    )

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--json-stdin":
        asyncio.run(_run_json_cli())
    else:
        asyncio.run(_demo())
