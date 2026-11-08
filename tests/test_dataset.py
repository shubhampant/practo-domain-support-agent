from collections import Counter

from dataset import (
    APPOINTMENTS,
    CATEGORIES,
    STATUSES,
    FEE_RANGES,
    generate_appointments,
    validate_appointments,
)


def test_dataset_has_at_least_40_records():
    assert len(APPOINTMENTS) >= 40


def test_dataset_is_deterministic():
    assert generate_appointments() == generate_appointments()


def test_record_ids_are_unique():
    ids = [item["record_id"] for item in APPOINTMENTS]
    assert len(ids) == len(set(ids))


def test_every_category_has_at_least_three_records():
    counts = Counter(item["category"] for item in APPOINTMENTS)

    for category in CATEGORIES:
        assert counts[category] >= 3


def test_every_required_status_is_present():
    counts = Counter(item["status"] for item in APPOINTMENTS)

    for status in STATUSES:
        assert counts[status] >= 1


def test_follow_up_percentage_is_between_10_and_30():
    percentage = (
        100
        * sum(item["follow_up_required"] for item in APPOINTMENTS)
        / len(APPOINTMENTS)
    )
    assert 10 <= percentage <= 30


def test_fees_match_specialty_ranges():
    for item in APPOINTMENTS:
        minimum, maximum = FEE_RANGES[item["category"]]
        assert minimum <= item["consultation_fee_inr"] <= maximum


def test_dataset_validation_passes():
    validate_appointments(APPOINTMENTS)


def test_creation_age_is_valid():
    for item in APPOINTMENTS:
        assert isinstance(item["days_since_created"], int)
        assert 0 <= item["days_since_created"] <= 30
