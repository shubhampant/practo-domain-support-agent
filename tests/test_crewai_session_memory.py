
import pytest

from crewai_support import SESSION_MEMORY, run_support_crew


@pytest.fixture(autouse=True)
def clear_session_memory():
    SESSION_MEMORY.clear_all()
    yield
    SESSION_MEMORY.clear_all()


def test_follow_up_uses_previous_cancellation_context():
    session_id = "cancellation-follow-up-test"

    first = run_support_crew(
        "rag",
        query="What is the cancellation policy?",
        session_id=session_id,
    )

    second = run_support_crew(
        "rag",
        query="What if I miss that window?",
        session_id=session_id,
    )

    assert first["session_id"] == session_id
    assert second["session_id"] == session_id

    assert len(SESSION_MEMORY.get_history(session_id)) == 4

    second_tool_call = second["tool_invocations"][0]
    contextual_query = second_tool_call["arguments"]["query"]

    assert "cancellation" in contextual_query.lower()
    assert "What if I miss that window?" in contextual_query


def test_new_session_does_not_reuse_another_sessions_context():
    run_support_crew(
        "rag",
        query="What is the cancellation policy?",
        session_id="session-A",
    )

    second = run_support_crew(
        "rag",
        query="What if I miss that window?",
        session_id="session-B",
    )

    contextual_query = second["tool_invocations"][0]["arguments"]["query"]

    assert contextual_query == "What if I miss that window?"
    assert SESSION_MEMORY.get_history("session-B")[0]["content"] == (
        "What if I miss that window?"
    )


def test_missing_session_id_creates_a_one_shot_session():
    first = run_support_crew(
        "rag",
        query="What is the cancellation policy?",
    )

    second = run_support_crew(
        "rag",
        query="What is the cancellation policy?",
    )

    assert first["session_id"] != second["session_id"]
    assert first["session_id"].startswith("one-shot-")
    assert second["session_id"].startswith("one-shot-")


def test_blank_session_id_is_rejected():
    with pytest.raises(ValueError, match="session_id"):
        run_support_crew(
            "rag",
            query="What is the cancellation policy?",
            session_id="   ",
        )
