
import pytest

from crewai_support import run_support_crew
from response_schema import SupportResponse


def test_rag_run_returns_valid_structured_response():
    result = run_support_crew(
        "rag",
        query="What is the cancellation policy?",
    )

    response = SupportResponse.model_validate(
        result["structured_response"]
    )

    assert response.mode == "rag"
    assert response.status in {"answered", "refused"}
    assert response.answer
    assert isinstance(response.sources, list)
    assert result["output"]


def test_unknown_rag_question_returns_refused_status():
    result = run_support_crew(
        "rag",
        query="Who won the FIFA World Cup in 2018?",
    )

    response = SupportResponse.model_validate(
        result["structured_response"]
    )

    assert response.mode == "rag"
    assert response.status == "refused"


def test_found_appointment_returns_structured_fields():
    result = run_support_crew(
        "lookup",
        record_id="APT-0010",
    )

    response = SupportResponse.model_validate(
        result["structured_response"]
    )

    assert response.mode == "lookup"
    assert response.status == "answered"
    assert response.record_id == "APT-0010"
    assert response.consultation_fee_inr is not None
    assert 0 <= response.escalation_score <= 1
    assert isinstance(response.escalation_recommended, bool)


def test_missing_appointment_returns_not_found():
    result = run_support_crew(
        "lookup",
        record_id="APT-NOT-FOUND",
    )

    response = SupportResponse.model_validate(
        result["structured_response"]
    )

    assert response.mode == "lookup"
    assert response.status == "not_found"
    assert response.record_id == "APT-NOT-FOUND"


@pytest.mark.parametrize("mode", ["rag", "lookup"])
def test_structured_response_preserves_tool_audit(mode):
    if mode == "rag":
        result = run_support_crew(
            "rag",
            query="What is the cancellation policy?",
        )
        expected_tool = "rag_lookup"
    else:
        result = run_support_crew(
            "lookup",
            record_id="APT-0010",
        )
        expected_tool = "appointment_status_lookup"

    assert result["tool_invocations"]
    assert result["tool_invocations"][0]["tool"] == expected_tool
    assert result["tool_results"]
    assert result["session_id"]
