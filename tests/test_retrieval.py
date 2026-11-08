
from retrieval import (
    SIMILARITY_THRESHOLD,
    build_retrieval_response,
    classify_similarity,
)


def test_similarity_above_threshold_is_accepted():
    assert classify_similarity(0.70) is True


def test_similarity_below_threshold_is_rejected():
    assert classify_similarity(0.20) is False


def test_similarity_equal_to_threshold_is_accepted():
    assert classify_similarity(SIMILARITY_THRESHOLD) is True


def test_empty_results_return_fallback():
    result = build_retrieval_response([])

    assert result["answerable"] is False
    assert result["status"] == "insufficient_evidence"
    assert result["similarity"] is None
    assert result["evidence"] == []


def test_relevant_results_return_source_evidence():
    results = [
        {
            "text": "Cancellation requests must be made 24 hours in advance.",
            "metadata": {
                "source": "cancellation_rescheduling.md"
            },
            "distance": 0.30,
        }
    ]

    result = build_retrieval_response(results)

    assert result["answerable"] is True
    assert result["status"] == "evidence_found"
    assert result["evidence"][0]["source"] == (
        "cancellation_rescheduling.md"
    )


def test_irrelevant_results_are_not_passed_as_evidence():
    results = [
        {
            "text": "A completely unrelated document.",
            "metadata": {
                "source": "lab_test_turnaround.md"
            },
            "distance": 0.90,
        }
    ]

    result = build_retrieval_response(results)

    assert result["answerable"] is False
    assert result["status"] == "insufficient_evidence"
    assert result["evidence"] == []
