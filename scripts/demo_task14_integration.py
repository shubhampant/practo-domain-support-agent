
"""Demonstrate the integrated CrewAI + AutoGen Task 14 workflow."""

import json
import os
import sys
from pathlib import Path

# Keep the demonstration deterministic and disable tracing/telemetry.
os.environ["OPENAI_API_KEY"] = ""
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["CREWAI_TRACING_ENABLED"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crewai_support import run_support_crew  # noqa: E402


def main() -> int:
    query = "What is the cancellation window?"

    print("=" * 72)
    print("TASK 14 — INTEGRATED CREWAI + AUTOGEN DEMONSTRATION")
    print("=" * 72)
    print(f"Query: {query}")
    print("Running CrewAI retrieval and Composer, followed by AutoGen review...")
    print()

    result = run_support_crew(mode="rag", query=query)

    review = result.get("review") or {}
    verdict = review.get("verdict") or {}
    structured_response = result.get("structured_response") or {}
    trace = review.get("trace") or []

    checks = {
        "CrewAI RAG workflow executed": result.get("mode") == "rag",
        "AutoGen review completed": review.get("status") == "completed",
        "Review verdict contains an approval decision": isinstance(
            verdict.get("approved"), bool
        ),
        "Review is bounded to two turns": review.get("turn_count") == 2,
        "Reviewed answer matches final output": (
            result.get("output") == verdict.get("final_answer")
        ),
        "Structured response matches final output": (
            structured_response.get("answer") == result.get("output")
        ),
        "Both AutoGen agents appear in the trace": (
            "Policy_Compliance_Reviewer" in str(trace)
            and "Final_Editor" in str(trace)
        ),
    }

    print("CHECKS")
    for description, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {description}")

    print()
    print("REVIEW SUMMARY")
    print(
        json.dumps(
            {
                "query": query,
                "status": review.get("status"),
                "verdict": verdict,
                "turn_count": review.get("turn_count"),
                "trace": trace,
                "final_answer": result.get("output"),
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )

    failures = [name for name, passed in checks.items() if not passed]
    print()
    if failures:
        print("DEMONSTRATION FAILED:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("DEMONSTRATION PASSED: integrated workflow checks succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
