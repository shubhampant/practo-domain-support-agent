
import pytest
from pydantic import ValidationError

from response_schema import SupportResponse


def test_valid_rag_response():
    response = SupportResponse(
        mode="rag",
        status="answered",
        answer="You can reschedule your appointment.",
        sources=["cancellation_rescheduling.md"],
    )

    assert response.mode == "rag"
    assert response.status == "answered"
    assert response.sources == ["cancellation_rescheduling.md"]
    assert response.record_id is None


def test_valid_lookup_response():
    response = SupportResponse(
        mode="lookup",
        status="answered",
        answer="Your appointment is scheduled.",
        record_id="APT-001",
        consultation_fee_inr=500,
        escalation_score=0.5,
        escalation_recommended=False,
    )

    assert response.record_id == "APT-001"
    assert response.consultation_fee_inr == 500
    assert response.escalation_score == 0.5
    assert response.escalation_recommended is False


def test_successful_lookup_requires_record_id():
    with pytest.raises(ValidationError, match="record_id"):
        SupportResponse(
            mode="lookup",
            status="answered",
            answer="Appointment details retrieved.",
        )


def test_invalid_escalation_score_is_rejected():
    with pytest.raises(ValidationError):
        SupportResponse(
            mode="lookup",
            status="answered",
            answer="Appointment found.",
            record_id="APT-001",
            escalation_score=1.5,
        )


def test_negative_consultation_fee_is_rejected():
    with pytest.raises(ValidationError):
        SupportResponse(
            mode="lookup",
            status="answered",
            answer="Appointment found.",
            record_id="APT-001",
            consultation_fee_inr=-100,
        )


def test_invalid_mode_is_rejected():
    with pytest.raises(ValidationError):
        SupportResponse(
            mode="invalid",
            status="answered",
            answer="Test response.",
        )


def test_extra_fields_are_rejected():
    with pytest.raises(ValidationError):
        SupportResponse(
            mode="rag",
            status="answered",
            answer="Test response.",
            unexpected_field="not allowed",
        )


def test_blank_answer_is_rejected():
    with pytest.raises(ValidationError):
        SupportResponse(
            mode="rag",
            status="answered",
            answer="   ",
        )
