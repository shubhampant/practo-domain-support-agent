import asyncio

from autogen_review import review_draft


CONTEXT = (
    "Cancellation or rescheduling requests should be submitted at least "
    "24 hours before the appointment. This policy does not promise refunds."
)


def run_review(*, query: str, draft: str, context: str = CONTEXT):
    return asyncio.run(
        review_draft(query=query, draft=draft, context=context)
    )


def test_supported_draft_is_approved_unchanged():
    draft = (
        "Cancellation or rescheduling requests should be submitted at least "
        "24 hours before the appointment."
    )

    result = run_review(
        query="What is the cancellation window?",
        draft=draft,
    )

    verdict = result["verdict"]
    assert isinstance(verdict, dict)
    assert verdict["approved"] is True
    assert verdict["final_answer"] == draft
    assert verdict["reason"]
    assert result["turn_count"] == 2


def test_unsupported_refund_guarantee_is_removed():
    draft = (
        "Cancellation or rescheduling requests should be submitted at least "
        "24 hours before the appointment. "
        "Our clinic guarantees a full refund for every cancellation."
    )

    result = run_review(
        query="Can I get a refund for cancellation?",
        draft=draft,
    )

    verdict = result["verdict"]
    assert isinstance(verdict, dict)
    assert verdict["approved"] is False
    assert "guarantees a full refund" not in verdict["final_answer"]
    assert "24 hours before the appointment" in verdict["final_answer"]
    assert verdict["final_answer"] != draft
    assert verdict["reason"]
    assert result["turn_count"] == 2


def test_review_returns_both_agent_messages():
    draft = (
        "Cancellation or rescheduling requests should be submitted at least "
        "24 hours before the appointment."
    )

    result = run_review(
        query="What is the cancellation window?",
        draft=draft,
    )

    agent_sources = {
        item["source"]
        for item in result["trace"]
        if item["source"] != "user"
    }

    assert agent_sources == {
        "Policy_Compliance_Reviewer",
        "Final_Editor",
    }


def test_mock_review_works_without_an_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    draft = (
        "Cancellation or rescheduling requests should be submitted at least "
        "24 hours before the appointment."
    )

    result = run_review(
        query="What is the cancellation window?",
        draft=draft,
    )

    assert result["verdict"]["approved"] is True
    assert result["verdict"]["final_answer"] == draft
