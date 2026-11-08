"""Generate and validate a deterministic synthetic appointment dataset."""

import random
from collections import Counter
from typing import TypedDict

SEED = 42
RECORD_COUNT = 50

CATEGORIES = [
    "General Medicine",
    "Cardiology",
    "Dermatology",
    "Pediatrics",
    "Orthopedics",
]

STATUSES = [
    "Scheduled",
    "Completed",
    "Cancelled",
    "No-Show",
    "Rescheduled",
]

# Synthetic fees in INR. These are project assumptions, not Practo's
# actual pricing. The range allows variation across specialties.
FEE_RANGES = {
    "General Medicine": (300, 800),
    "Cardiology": (600, 1800),
    "Dermatology": (400, 1200),
    "Pediatrics": (300, 900),
    "Orthopedics": (500, 1500),
}


class Appointment(TypedDict):
    record_id: str
    category: str
    status: str
    consultation_fee_inr: int
    days_since_created: int
    follow_up_required: bool


def generate_appointments() -> list[Appointment]:
    """Generate reproducible records using a fixed random seed."""
    rng = random.Random(SEED)
    appointments: list[Appointment] = []

    # Guarantee category and status coverage without hand-editing
    # individual records after generation.
    planned_categories = [CATEGORIES[i % len(CATEGORIES)] for i in range(RECORD_COUNT)]
    planned_statuses = [STATUSES[i % len(STATUSES)] for i in range(RECORD_COUNT)]

    rng.shuffle(planned_categories)
    rng.shuffle(planned_statuses)

    for i in range(RECORD_COUNT):
        category = planned_categories[i]
        fee_min, fee_max = FEE_RANGES[category]

        # A follow-up probability of 20% targets the required 10–30% band.
        appointments.append(
            {
                "record_id": f"APT-{i + 1:04d}",
                "category": category,
                "status": planned_statuses[i],
                "consultation_fee_inr": rng.randint(fee_min, fee_max),
                "days_since_created": rng.randint(0, 30),
                "follow_up_required": rng.random() < 0.20,
            }
        )

    return appointments


def validate_appointments(
    appointments: list[Appointment],
) -> None:
    """Raise an error if the dataset violates capstone requirements."""
    assert len(appointments) >= 40, "Need at least 40 appointments."

    record_ids = [item["record_id"] for item in appointments]
    assert len(record_ids) == len(set(record_ids)), "Duplicate record IDs."

    category_counts = Counter(item["category"] for item in appointments)
    status_counts = Counter(item["status"] for item in appointments)

    for category in CATEGORIES:
        assert (
            category_counts[category] >= 3
        ), f"Category {category} has fewer than 3 records."

    for status in STATUSES:
        assert status_counts[status] >= 1, f"Status {status} is missing."

    for item in appointments:
        assert item["category"] in CATEGORIES
        assert item["status"] in STATUSES
        assert 0 <= item["days_since_created"] <= 30
        assert isinstance(item["follow_up_required"], bool)

        fee_min, fee_max = FEE_RANGES[item["category"]]
        assert fee_min <= item["consultation_fee_inr"] <= fee_max

    follow_up_count = sum(item["follow_up_required"] for item in appointments)
    follow_up_percentage = 100 * follow_up_count / len(appointments)

    assert 10 <= follow_up_percentage <= 30, (
        f"Follow-up rate is {follow_up_percentage:.1f}%; "
        "required range is 10–30%. Change the seed or weights."
    )


def main() -> None:
    appointments = generate_appointments()
    validate_appointments(appointments)

    category_counts = Counter(item["category"] for item in appointments)
    status_counts = Counter(item["status"] for item in appointments)
    follow_up_count = sum(item["follow_up_required"] for item in appointments)
    follow_up_percentage = 100 * follow_up_count / len(appointments)

    print(f"Seed: {SEED}")
    print(f"Total appointments: {len(appointments)}")

    print("\nCounts per category:")
    for category in CATEGORIES:
        print(f"  {category}: {category_counts[category]}")

    print("\nCounts per status:")
    for status in STATUSES:
        print(f"  {status}: {status_counts[status]}")

    print(
        f"\nFollow-ups required: {follow_up_count}/{len(appointments)} "
        f"({follow_up_percentage:.1f}%)"
    )
    print("\nValidation: PASSED")
    print("\nFirst 5 appointment records:")
    for appointment in appointments[:5]:
        print(appointment)


APPOINTMENTS = generate_appointments()


if __name__ == "__main__":
    main()
