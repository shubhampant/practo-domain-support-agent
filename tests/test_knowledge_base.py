
from pathlib import Path
import re

KNOWLEDGE_DIR = Path("data/knowledge_base")

REQUIRED_DOCUMENTS = {
    "appointment_booking.md",
    "cancellation_rescheduling.md",
    "consultation_fees.md",
    "insurance_claims.md",
    "prescription_refills.md",
    "lab_test_turnaround.md",
    "telemedicine.md",
    "emergency_protocol.md",
    "patient_privacy.md",
    "follow_up_discount.md",
    "second_opinion.md",
    "home_visits.md",
}


def get_policy_sentences(text):
    """Count sentence-like units in the body, excluding the heading."""
    body = "\n".join(
        line for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", body)
        if sentence.strip()
    ]


def test_all_required_documents_exist():
    existing = {path.name for path in KNOWLEDGE_DIR.glob("*.md")}
    assert len(existing) >= 12
    assert REQUIRED_DOCUMENTS.issubset(existing)


def test_each_required_document_has_two_to_five_sentences():
    for filename in REQUIRED_DOCUMENTS:
        text = (KNOWLEDGE_DIR / filename).read_text(encoding="utf-8-sig")
        sentences = get_policy_sentences(text)

        assert 2 <= len(sentences) <= 5, (
            f"{filename} has {len(sentences)} sentences; expected 2–5"
        )


def test_every_required_document_has_content():
    for filename in REQUIRED_DOCUMENTS:
        text = (KNOWLEDGE_DIR / filename).read_text(encoding="utf-8-sig")
        assert len(text.strip()) > 0, f"{filename} is empty"


def test_documents_are_marked_as_synthetic_or_capstone_rules():
    for filename in REQUIRED_DOCUMENTS:
        text = (KNOWLEDGE_DIR / filename).read_text(
            encoding="utf-8-sig"
        ).lower()

        assert (
            "fictional" in text
            or "synthetic" in text
            or "capstone" in text
            or "illustrative" in text
        ), f"{filename} does not identify its simulated status"


def test_privacy_document_states_masker_scope():
    text = (
        KNOWLEDGE_DIR / "patient_privacy.md"
    ).read_text(encoding="utf-8-sig").lower()

    assert "contact numbers must be masked" in text
    assert "patient names" in text
    assert "insurance ids" in text
    assert "fabricated examples" in text
