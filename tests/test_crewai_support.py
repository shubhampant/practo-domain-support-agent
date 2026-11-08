
import os

# Keep CrewAI execution tracing disabled during tests.
os.environ["CREWAI_TRACING_ENABLED"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"

from crewai_support import build_support_crew, run_support_crew


def test_crew_has_three_agents_and_tracing_disabled():
    crew = build_support_crew("rag")

    assert len(crew.agents) == 3
    assert crew.tracing is False


def test_rag_mode_invokes_retrieval_tool():
    result = run_support_crew(
        mode="rag",
        query="What is the cancellation policy?",
    )

    assert result["mode"] == "rag"
    assert len(result["tool_invocations"]) == 1
    assert result["tool_invocations"][0]["tool"] == "rag_lookup"
    assert (
        result["tool_invocations"][0]["arguments"]["query"]
        == "What is the cancellation policy?"
    )

    rag_result = result["tool_results"][0]["result"]

    assert rag_result["status"] == "grounded_answer"
    assert rag_result["similarity"] >= rag_result["threshold"]
    assert "cancellation_rescheduling.md" in rag_result["sources"]
    assert "24 hours" in result["output"]


def test_lookup_mode_invokes_appointment_tool():
    result = run_support_crew(
        mode="lookup",
        record_id="APT-0010",
    )

    assert result["mode"] == "lookup"
    assert len(result["tool_invocations"]) == 1

    invocation = result["tool_invocations"][0]
    assert invocation["tool"] == "appointment_status_lookup"
    assert invocation["arguments"]["record_id"] == "APT-0010"

    appointment = result["tool_results"][0]["result"]

    assert appointment["found"] is True
    assert appointment["record_id"] == "APT-0010"
    assert appointment["status"] == "No-Show"
    assert appointment["consultation_fee_inr"] == 913
    assert appointment["escalation_score"] == 0.8333
    assert appointment["escalation_recommended"] is True

    assert "No-Show" in result["output"]
    assert "913" in result["output"]


def test_invalid_mode_is_rejected():
    try:
        run_support_crew(mode="invalid")
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid mode should raise ValueError")


def test_rag_mode_requires_query():
    try:
        run_support_crew(mode="rag")
    except ValueError:
        pass
    else:
        raise AssertionError("Missing query should raise ValueError")


def test_lookup_mode_requires_record_id():
    try:
        run_support_crew(mode="lookup")
    except ValueError:
        pass
    else:
        raise AssertionError("Missing record_id should raise ValueError")
