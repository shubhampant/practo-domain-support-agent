
import pytest

from appointment_lookup import (
    ESCALATION_THRESHOLD,
    calculate_escalation_score,
    check_appointment_status,
)
from dataset import APPOINTMENTS


def test_score_without_follow_up_and_zero_days():
    assert calculate_escalation_score(False, 0) == 0.0


def test_score_with_follow_up_and_zero_days():
    assert calculate_escalation_score(True, 0) == 0.5


def test_score_without_follow_up_and_maximum_age():
    assert calculate_escalation_score(False, 30) == 0.5


def test_maximum_score_is_one():
    assert calculate_escalation_score(True, 30) == 1.0


def test_score_stays_between_zero_and_one():
    for appointment in APPOINTMENTS:
        score = calculate_escalation_score(
            appointment["follow_up_required"],
            appointment["days_since_created"],
        )
        assert 0.0 <= score <= 1.0


def test_existing_appointment_returns_required_fields():
    record = APPOINTMENTS[0]
    result = check_appointment_status(record["record_id"])

    assert result["found"] is True
    assert result["record_id"] == record["record_id"]
    assert result["status"] == record["status"]
    assert result["consultation_fee_inr"] == record["consultation_fee_inr"]
    assert 0.0 <= result["escalation_score"] <= 1.0
    assert result["escalation_recommended"] == (
        result["escalation_score"] >= ESCALATION_THRESHOLD
    )


def test_lookup_accepts_lowercase_and_surrounding_spaces():
    record = APPOINTMENTS[0]
    result = check_appointment_status(f"  {record['record_id'].lower()}  ")

    assert result["found"] is True
    assert result["record_id"] == record["record_id"]


def test_unknown_appointment_returns_not_found():
    result = check_appointment_status("APT-9999")

    assert result["found"] is False
    assert result["record_id"] == "APT-9999"
    assert "error" in result


def test_empty_record_id_is_rejected():
    with pytest.raises(ValueError):
        check_appointment_status("   ")


def test_invalid_recency_is_rejected():
    with pytest.raises(ValueError):
        calculate_escalation_score(False, 31)


def test_invalid_follow_up_type_is_rejected():
    with pytest.raises(TypeError):
        calculate_escalation_score(1, 10)
