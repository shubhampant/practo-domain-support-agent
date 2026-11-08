
"""Generate Task 8 session-memory demonstration transcripts."""

import json
import os
from pathlib import Path

# Set telemetry controls before importing CrewAI project modules.
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from crewai_support import SESSION_MEMORY, run_support_crew


TRANSCRIPT_DIR = Path("transcripts")
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)


def format_result(result: dict) -> str:
    return json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
        default=str,
    )


def save_transcript(filename: str, content: str) -> None:
    path = TRANSCRIPT_DIR / filename
    path.write_text(content.rstrip() + "\n", encoding="utf-8")
    print(f"Saved transcript: {path}")


# --------------------------------------------------
# Demonstration A: Multiple turns in the same session
# --------------------------------------------------

same_session_id = "task8-demo-same-session"
SESSION_MEMORY.clear_session(same_session_id)

same_session_lines = [
    "TASK 8 — SAME-SESSION MULTI-TURN DEMONSTRATION",
    "=" * 55,
    f"Session ID: {same_session_id}",
    "",
    "TURN 1 — Initial question",
    "User: What is the cancellation policy?",
    "History before turn 1:",
    SESSION_MEMORY.format_history(same_session_id),
    "",
]

turn1 = run_support_crew(
    "rag",
    query="What is the cancellation policy?",
    session_id=same_session_id,
)

same_session_lines.extend(
    [
        "Assistant/tool result for turn 1:",
        format_result(turn1),
        "",
        "History after turn 1:",
        SESSION_MEMORY.format_history(same_session_id),
        "",
        "TURN 2 — Context-dependent follow-up",
        "User: What if I need to cancel it?",
        "History supplied before turn 2:",
        SESSION_MEMORY.format_history(same_session_id),
        "",
    ]
)

turn2 = run_support_crew(
    "rag",
    query="What if I need to cancel it?",
    session_id=same_session_id,
)

same_session_lines.extend(
    [
        "Assistant/tool result for turn 2:",
        format_result(turn2),
        "",
        "History after turn 2:",
        SESSION_MEMORY.format_history(same_session_id),
        "",
        "Verification:",
        (
            "PASS — the same session ID was used for both turns."
            if turn1["session_id"] == turn2["session_id"]
            else "FAIL — session IDs differ."
        ),
    ]
)

save_transcript(
    "task8_same_session.txt",
    "\n".join(same_session_lines),
)


# --------------------------------------------------
# Demonstration B: A separate fresh conversation
# --------------------------------------------------

fresh_session_id = "task8-demo-fresh-session"
SESSION_MEMORY.clear_session(fresh_session_id)

fresh_history_before = SESSION_MEMORY.format_history(
    fresh_session_id
)

fresh_session_lines = [
    "TASK 8 — FRESH-CONVERSATION DEMONSTRATION",
    "=" * 55,
    f"Session ID: {fresh_session_id}",
    "",
    "This is a separate session from the previous demonstration.",
    "",
    "TURN 1 — Follow-up-style question without prior context",
    "User: What if I need to cancel it?",
    "",
    "History before this conversation:",
    fresh_history_before,
    "",
]

fresh_result = run_support_crew(
    "rag",
    query="What if I need to cancel it?",
    session_id=fresh_session_id,
)

fresh_session_lines.extend(
    [
        "Assistant/tool result:",
        format_result(fresh_result),
        "",
        "History after this turn:",
        SESSION_MEMORY.format_history(fresh_session_id),
        "",
        "Verification:",
        (
            "PASS — the fresh session started without prior history."
            if fresh_history_before in (
                "",
                "No previous conversation.",
            )
            else (
                "CHECK — unexpected history was present: "
                + fresh_history_before
            )
        ),
    ]
)

save_transcript(
    "task8_fresh_session.txt",
    "\n".join(fresh_session_lines),
)

print("\nTask 8 transcript generation finished.")
