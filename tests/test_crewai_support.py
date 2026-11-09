
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


def test_task14_passes_composer_output_and_original_context_to_autogen(monkeypatch):
    import crewai_support

    composer_draft = (
        "Your cancellation request should be submitted at least 24 hours "
        "before the appointment. Sources: cancellation_rescheduling.md"
    )
    retrieved_context = (
        "Cancellation or rescheduling requests must be submitted "
        "at least 24 hours before the appointment."
    )
    reviewed_answer = (
        "Please submit cancellation or rescheduling requests at least "
        "24 hours before the appointment."
    )
    captured = {}

    class FakeCrew:
        def kickoff(self, inputs):
            return composer_draft

    def fake_review(*, query, draft, context):
        captured["query"] = query
        captured["draft"] = draft
        captured["context"] = context

        return {
            "verdict": {
                "approved": True,
                "final_answer": reviewed_answer,
                "reason": "The response is supported by the retrieved context.",
            },
            "trace": [
                {"source": "Policy_Compliance_Reviewer"},
                {"source": "Final_Editor"},
            ],
            "turn_count": 2,
        }

    monkeypatch.setattr(
        crewai_support, "build_support_crew", lambda mode: FakeCrew()
    )
    monkeypatch.setattr(
        crewai_support,
        "_build_structured_response",
        lambda mode: {
            "status": "answered",
            "answer": "Retrieval answer that is not the Composer draft.",
        },
    )
    monkeypatch.setattr(
        crewai_support,
        "_latest_result",
        lambda mode: {"context": retrieved_context},
    )
    monkeypatch.setattr(
        crewai_support, "_run_autogen_review", fake_review
    )

    result = crewai_support.run_support_crew(
        mode="rag",
        query="What is the cancellation window?",
    )

    # AutoGen receives the actual CrewAI kickoff output.
    assert captured["draft"] == composer_draft
    assert captured["draft"] != (
        "Retrieval answer that is not the Composer draft."
    )

    # The original retrieved context and user query are preserved.
    assert captured["context"] == retrieved_context
    assert captured["query"] == "What is the cancellation window?"

    # The AutoGen verdict is reflected in the final response.
    assert result["output"] == reviewed_answer
    assert result["structured_response"]["answer"] == reviewed_answer

    # Review metadata confirms a completed, bounded review.
    assert result["review"]["status"] == "completed"
    assert result["review"]["verdict"]["approved"] is True
    assert result["review"]["turn_count"] == 2
