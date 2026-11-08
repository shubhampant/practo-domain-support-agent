
"""Appointment lookup tool and deterministic escalation scoring."""

from dataset import APPOINTMENTS

MAX_AGE_DAYS = 30
FOLLOW_UP_WEIGHT = 0.5
RECENCY_WEIGHT = 0.5
ESCALATION_THRESHOLD = 0.8


def calculate_escalation_score(
    follow_up_required: bool,
    days_since_created: int,
) -> float:
    """Calculate a deterministic operational escalation score in [0, 1]."""
    if not isinstance(follow_up_required, bool):
        raise TypeError("follow_up_required must be a bool.")

    if not isinstance(days_since_created, int) or isinstance(
        days_since_created, bool
    ):
        raise TypeError("days_since_created must be an integer.")

    if not 0 <= days_since_created <= MAX_AGE_DAYS:
        raise ValueError("days_since_created must be between 0 and 30.")

    follow_up_signal = float(follow_up_required)
    recency_signal = days_since_created / MAX_AGE_DAYS

    score = (
        FOLLOW_UP_WEIGHT * follow_up_signal
        + RECENCY_WEIGHT * recency_signal
    )

    return round(score, 4)


def check_appointment_status(record_id: str) -> dict:
    """Look up an appointment and return its status, fee, and escalation score."""
    if not isinstance(record_id, str) or not record_id.strip():
        raise ValueError("record_id must be a non-empty string.")

    normalized_id = record_id.strip().upper()

    appointment = next(
        (
            record
            for record in APPOINTMENTS
            if record["record_id"] == normalized_id
        ),
        None,
    )

    if appointment is None:
        return {
            "found": False,
            "record_id": normalized_id,
            "error": "Appointment record not found.",
        }

    score = calculate_escalation_score(
        follow_up_required=appointment["follow_up_required"],
        days_since_created=appointment["days_since_created"],
    )

    return {
        "found": True,
        "record_id": appointment["record_id"],
        "status": appointment["status"],
        "consultation_fee_inr": appointment["consultation_fee_inr"],
        "escalation_score": score,
        "escalation_recommended": score >= ESCALATION_THRESHOLD,
    }
