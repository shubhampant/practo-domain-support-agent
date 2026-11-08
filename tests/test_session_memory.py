
import pytest

from session_memory import SessionMemory


def test_new_session_has_empty_history():
    memory = SessionMemory()

    assert memory.get_history("session-1") == []
    assert memory.format_history("session-1") == "No previous conversation."


def test_add_and_retrieve_conversation_turn():
    memory = SessionMemory()

    memory.add_turn(
        "session-1",
        "What is the cancellation policy?",
        "Cancel at least 24 hours before the appointment.",
    )

    assert memory.get_history("session-1") == [
        {
            "role": "user",
            "content": "What is the cancellation policy?",
        },
        {
            "role": "assistant",
            "content": "Cancel at least 24 hours before the appointment.",
        },
    ]


def test_sessions_are_isolated():
    memory = SessionMemory()

    memory.add_turn("session-1", "My appointment is tomorrow.", "Understood.")
    memory.add_turn("session-2", "What are your consultation fees?", "Fees vary by specialty.")

    history_1 = memory.get_history("session-1")
    history_2 = memory.get_history("session-2")

    assert len(history_1) == 2
    assert len(history_2) == 2
    assert "appointment" in history_1[0]["content"].lower()
    assert "consultation fees" in history_2[0]["content"].lower()
    assert all(
        "appointment" not in message["content"].lower()
        for message in history_2
    )


def test_history_is_limited_to_max_turns():
    memory = SessionMemory(max_turns=2)

    memory.add_turn("session-1", "Question 1", "Answer 1")
    memory.add_turn("session-1", "Question 2", "Answer 2")
    memory.add_turn("session-1", "Question 3", "Answer 3")

    history = memory.get_history("session-1")

    assert len(history) == 4
    assert history[0]["content"] == "Question 2"
    assert history[-1]["content"] == "Answer 3"


def test_clear_one_session_does_not_clear_another():
    memory = SessionMemory()

    memory.add_turn("session-1", "Question A", "Answer A")
    memory.add_turn("session-2", "Question B", "Answer B")

    assert memory.clear_session("session-1") is True
    assert memory.get_history("session-1") == []
    assert len(memory.get_history("session-2")) == 2


def test_clear_nonexistent_session_returns_false():
    memory = SessionMemory()

    assert memory.clear_session("missing-session") is False


def test_clear_all_sessions():
    memory = SessionMemory()

    memory.add_turn("session-1", "Question A", "Answer A")
    memory.add_turn("session-2", "Question B", "Answer B")

    memory.clear_all()

    assert memory.get_history("session-1") == []
    assert memory.get_history("session-2") == []


@pytest.mark.parametrize("max_turns", [0, -1])
def test_invalid_max_turns_are_rejected(max_turns):
    with pytest.raises(ValueError):
        SessionMemory(max_turns=max_turns)


def test_empty_session_id_is_rejected():
    memory = SessionMemory()

    with pytest.raises(ValueError):
        memory.get_history("   ")


def test_empty_messages_are_rejected():
    memory = SessionMemory()

    with pytest.raises(ValueError):
        memory.add_turn("session-1", "", "Answer")

    with pytest.raises(ValueError):
        memory.add_turn("session-1", "Question", "  ")


def test_returned_history_cannot_mutate_internal_history():
    memory = SessionMemory()
    memory.add_turn("session-1", "Original question", "Original answer")

    history = memory.get_history("session-1")
    history[0]["content"] = "Modified externally"
    history.clear()

    saved_history = memory.get_history("session-1")

    assert len(saved_history) == 2
    assert saved_history[0]["content"] == "Original question"
