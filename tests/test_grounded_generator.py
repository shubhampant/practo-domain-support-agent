
from grounded_generator import MockGroundedGenerator


def test_generator_uses_retrieved_evidence():
    generator = MockGroundedGenerator()
    evidence = [{
        "text": (
            "In this simulation, a user may request cancellation "
            "at least 24 hours before the appointment start time. "
            "Completed appointments cannot be rescheduled."
        ),
        "source": "cancellation_rescheduling.md",
    }]

    answer = generator.generate("When can I cancel?", evidence)

    assert "24 hours" in answer
    assert "24 hours" in evidence[0]["text"]


def test_generator_does_not_create_answer_without_evidence():
    generator = MockGroundedGenerator()

    answer = generator.generate("Who won the FIFA World Cup?", [])

    assert "I don't know" in answer


def test_generator_does_not_add_unretrieved_policy():
    generator = MockGroundedGenerator()
    evidence = [{
        "text": (
            "Cardiology consultation fees range from INR 600–1800. "
            "Dermatology fees range from INR 400–1200."
        ),
        "source": "consultation_fees.md",
    }]

    answer = generator.generate("What is the cardiology fee?", evidence)

    assert "INR 600–1800" in answer
    assert "General Medicine" not in answer


def test_generator_filters_unrelated_sentences():
    generator = MockGroundedGenerator()
    evidence = [{
        "text": (
            "A routine blood test has an estimated turnaround "
            "of 1–2 calendar days. "
            "Patients requesting second opinions may consult "
            "another qualified clinician."
        ),
        "source": "lab_test_turnaround.md",
    }]

    answer = generator.generate("How long does a routine blood test take?", evidence)

    assert "1–2 calendar days" in answer
    assert "second opinions" not in answer


def test_generator_handles_irrelevant_evidence():
    generator = MockGroundedGenerator()
    evidence = [{
        "text": "Orthopedics appointments use a synthetic fee range.",
        "source": "consultation_fees.md",
    }]

    answer = generator.generate("How long does a routine blood test take?", evidence)

    assert "I don't know" in answer


def test_telemedicine_answer_includes_eligibility_conditions():
    generator = MockGroundedGenerator()

    evidence = [{
        "text": (
            "In the simulation, a user is eligible to request a remote "
            "consultation when a provider offers the required specialty "
            "and a remote appointment slot is available for the requested time. "
            "Before joining, the user should verify the appointment confirmation, "
            "consultation time, and joining instructions. "
            "The agent must not guarantee clinical suitability for remote care; "
            "the clinician determines whether an in-person assessment is necessary."
        ),
        "source": "telemedicine.md",
    }]

    answer = generator.generate(
        "What conditions are required for telemedicine?",
        evidence,
    )

    assert "eligible to request a remote consultation" in answer
    assert "remote appointment slot is available" in answer


def test_cardiology_answer_excludes_other_specialty_fees():
    generator = MockGroundedGenerator()

    evidence = [{
        "text": (
            "General Medicine appointments use a fee range of INR 300–800, "
            "Cardiology INR 600–1800, Dermatology INR 400–1200, "
            "Pediatrics INR 300–900, and Orthopedics INR 500–1500."
        ),
        "source": "consultation_fees.md",
    }]

    answer = generator.generate(
        "What is the cardiology consultation fee range?",
        evidence,
    )

    assert "Cardiology INR 600–1800" in answer
    assert "Dermatology" not in answer
    assert "Orthopedics" not in answer


def test_unsupported_generated_answer_is_blocked_by_grounding_guardrail(
    monkeypatch,
):
    from grounded_generator import GroundedSupportAgent

    agent = GroundedSupportAgent()

    evidence = [
        {
            "source": "cancellation_rescheduling.md",
            "text": (
                "Patients can cancel an appointment before "
                "the scheduled consultation time."
            ),
        }
    ]

    retrieval_result = {
        "answerable": True,
        "similarity": 0.9,
        "evidence": evidence,
    }

    # Retrieval ko deterministic evidence denge.
    monkeypatch.setattr(
        agent.retriever,
        "retrieve",
        lambda query: retrieval_result,
    )

    # Generator jaan-boojhkar unsupported claim generate karega.
    monkeypatch.setattr(
        agent.generator,
        "generate",
        lambda query, evidence: (
            "Every patient receives a guaranteed 90 percent "
            "discount on all future medical procedures."
        ),
    )

    result = agent.ask("What is the cancellation policy?")

    assert result["status"] == "insufficient_evidence"
    assert result["answer"].startswith("I don't know.")
    assert result["sources"] == []
    assert result["guardrail"]["triggered"] is True
    assert (
        result["guardrail"]["reason"]
        == "unsupported_output_claim"
    )
