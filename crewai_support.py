"""CrewAI integration for the local Practo support capstone.

Runtime design:
- Uses the existing local RAG pipeline.
- Uses the existing appointment lookup function.
- Uses CrewAI's Agent, Task, Crew and kickoff workflow.
- Uses a deterministic local mock LLM; no external inference API.
- Maintains isolated, in-memory conversation history per session.
"""

from __future__ import annotations

import json
import re
from typing import Any, ClassVar
from uuid import uuid4

from crewai import Agent, Crew, Process, Task
from crewai.llms.base_llm import BaseLLM
from crewai.tools import tool

from appointment_lookup import check_appointment_status
from grounded_generator import GroundedSupportAgent
from session_memory import SessionMemory
from response_schema import SupportResponse
from guardrails import guard_input, mask_contact_numbers
from pathlib import Path
import subprocess

# Tool execution audit records.
TOOL_INVOCATIONS: list[dict[str, Any]] = []
TOOL_RESULTS: list[dict[str, Any]] = []

# In-memory conversation store. Each session has isolated history.
SESSION_MEMORY = SessionMemory(max_turns=20)


def reset_tool_audit() -> None:
    """Reset tool audit and deterministic mock state for a new run."""
    TOOL_INVOCATIONS.clear()
    TOOL_RESULTS.clear()
    DeterministicMockLLM._task_calls.clear()


@tool("rag_lookup")
def rag_lookup(query: str) -> str:
    """Search the local knowledge base and return a grounded support answer.

    Args:
        query: The customer's question about clinic support policies.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")

    clean_query = query.strip()
    result = GroundedSupportAgent().ask(clean_query)

    TOOL_INVOCATIONS.append(
        {
            "tool": "rag_lookup",
            "arguments": {"query": clean_query},
        }
    )
    TOOL_RESULTS.append({"kind": "rag", "result": result})

    return json.dumps(result, ensure_ascii=False)


@tool("appointment_status_lookup")
def appointment_status_lookup(record_id: str) -> str:
    """Look up an appointment by record ID.

    Returns its status, consultation fee, escalation score and recommendation.

    Args:
        record_id: The appointment identifier, such as APT-0010.
    """
    if not isinstance(record_id, str) or not record_id.strip():
        raise ValueError("record_id must be a non-empty string.")

    clean_record_id = record_id.strip()
    result = check_appointment_status(clean_record_id)

    TOOL_INVOCATIONS.append(
        {
            "tool": "appointment_status_lookup",
            "arguments": {"record_id": clean_record_id},
        }
    )
    TOOL_RESULTS.append({"kind": "appointment", "result": result})

    return json.dumps(result, ensure_ascii=False)


def _schema_fields(tool_object: Any) -> set[str]:
    """Return a tool's declared argument names from its schema."""
    schema = getattr(tool_object, "args_schema", None)
    fields = getattr(schema, "model_fields", None)

    if fields is None:
        return set()

    return set(fields.keys())


def _tool_for_argument(
    tools: list[Any],
    argument_name: str,
) -> Any | None:
    """Select a tool by its declared argument schema, not its name."""
    matches = [
        candidate for candidate in tools if argument_name in _schema_fields(candidate)
    ]

    if len(matches) != 1:
        return None

    return matches[0]


def _task_key(task: Any) -> str:
    """Create a stable key for per-task mock execution state."""
    task_id = getattr(task, "id", None)
    return str(task_id if task_id is not None else id(task))


def _extract_query(description: str) -> str:
    """Extract the customer question from the retrieval task description."""
    match = re.search(
        r"Customer question:\s*(.+)",
        description,
        flags=re.IGNORECASE,
    )

    if not match:
        raise ValueError("The retrieval task has no customer question.")

    query = match.group(1).strip()

    if not query:
        raise ValueError("The customer question is empty.")

    return query


def _extract_conversation_history(description: str) -> str:
    """Extract conversation history from a task description."""
    match = re.search(
        r"Conversation history:\s*(.*?)\nCustomer question:",
        description,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if not match:
        return ""

    return match.group(1).strip()


def _resolve_follow_up_query(query: str, history: str) -> str:
    """Add basic cancellation context to certain ambiguous follow-up queries.

    This is a deliberately simple heuristic, not a general-purpose
    conversational coreference resolver.
    """
    if not history or history == "No previous conversation.":
        return query

    query_lower = query.lower()

    references_previous_context = any(
        phrase in query_lower
        for phrase in (
            "that",
            "it",
            "this",
            "those",
            "there",
            "the window",
            "what if",
            "what about",
        )
    )

    if not references_previous_context:
        return query

    history_lower = history.lower()
    cancellation_terms = ("cancel", "cancellation", "reschedul")

    if any(term in history_lower for term in cancellation_terms):
        return (
            "Appointment cancellation and rescheduling policy. "
            f"Customer follow-up question: {query}"
        )

    return query


def _extract_record_id(description: str) -> str:
    """Extract the appointment record ID from a lookup task description."""
    match = re.search(
        r"Appointment record ID:\s*([A-Za-z0-9_-]+)",
        description,
        flags=re.IGNORECASE,
    )

    if not match:
        raise ValueError("The lookup task has no appointment record ID.")

    return match.group(1)


def _latest_result(kind: str) -> dict[str, Any]:
    """Get the most recent result of the requested kind."""
    for item in reversed(TOOL_RESULTS):
        if item["kind"] == kind:
            return item["result"]

    raise RuntimeError(f"No tool result has been recorded for {kind!r}.")


def _build_structured_response(mode: str) -> dict[str, Any]:
    """Convert the latest tool result into a validated response."""
    if mode == "rag":
        result = _latest_result("rag")

        response_status = (
            "answered" if result.get("status") == "grounded_answer" else "refused"
        )

        response = SupportResponse(
            mode="rag",
            status=response_status,
            answer=result["answer"],
            sources=result.get("sources", []),
        )

    elif mode == "lookup":
        result = _latest_result("appointment")

        if not result.get("found"):
            response = SupportResponse(
                mode="lookup",
                status="not_found",
                answer=(
                    "I could not find that appointment record. "
                    "Please verify the appointment ID."
                ),
                record_id=result.get("record_id"),
            )
        else:
            response = SupportResponse(
                mode="lookup",
                status="answered",
                answer=(
                    f"Appointment {result['record_id']} has status "
                    f"{result['status']}. Consultation fee: "
                    f"INR {result['consultation_fee_inr']}. "
                    f"Escalation score: {result['escalation_score']:.4f}. "
                    f"Escalation recommended: "
                    f"{result['escalation_recommended']}."
                ),
                record_id=result["record_id"],
                consultation_fee_inr=result["consultation_fee_inr"],
                escalation_score=result["escalation_score"],
                escalation_recommended=result["escalation_recommended"],
            )
    else:
        raise ValueError("mode must be either 'rag' or 'lookup'.")

    return response.model_dump()


def _run_autogen_review(
    *,
    query: str,
    draft: str,
    context: str,
) -> dict[str, Any]:
    """Run the isolated AutoGen reviewer and return its JSON result."""

    project_dir = Path(__file__).resolve().parent
    autogen_python = (
        project_dir / ".venv-autogen" / "Scripts" / "python.exe"
    )
    autogen_script = project_dir / "autogen_review.py"

    if not autogen_python.is_file():
        raise FileNotFoundError(
            f"AutoGen Python environment not found: {autogen_python}"
        )

    if not autogen_script.is_file():
        raise FileNotFoundError(
            f"AutoGen review script not found: {autogen_script}"
        )

    payload = {
        "query": query,
        "draft": draft,
        "context": context,
    }

    completed = subprocess.run(
        [
            str(autogen_python),
            str(autogen_script),
            "--json-stdin",
        ],
        input=json.dumps(payload, ensure_ascii=False),
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
        cwd=str(project_dir),
    )

    result = json.loads(completed.stdout)

    verdict = result.get("verdict")
    if not isinstance(verdict, dict):
        raise ValueError("AutoGen returned no structured verdict.")

    if not isinstance(verdict.get("approved"), bool):
        raise ValueError("AutoGen verdict has an invalid approved field.")

    if not isinstance(verdict.get("final_answer"), str):
        raise ValueError("AutoGen verdict has an invalid final_answer field.")

    if not isinstance(verdict.get("reason"), str):
        raise ValueError("AutoGen verdict has an invalid reason field.")

    return result


class DeterministicMockLLM(BaseLLM):
    """Minimal text-action LLM for offline CrewAI demonstrations."""

    model: str = "practo-deterministic-mock"

    # ClassVar prevents Pydantic from treating this dictionary as a model field.
    _task_calls: ClassVar[dict[str, int]] = {}

    def call(
        self,
        messages: Any,
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
    ) -> str:
        """Return deterministic CrewAI action or final-answer text."""
        del messages, tools, callbacks, available_functions, response_model

        if from_task is None:
            return "Final Answer: No task context was provided."

        key = _task_key(from_task)
        call_number = self._task_calls.get(key, 0)
        self._task_calls[key] = call_number + 1

        agent = from_agent or getattr(from_task, "agent", None)
        agent_tools = list(getattr(agent, "tools", None) or [])
        description = str(getattr(from_task, "description", ""))

        # Select tools using their declared input schemas.
        query_tool = _tool_for_argument(agent_tools, "query")
        record_tool = _tool_for_argument(agent_tools, "record_id")

        # Retrieval Agent.
        if query_tool is not None and record_tool is None:
            if call_number == 0:
                query = _extract_query(description)
                history = _extract_conversation_history(description)
                query = _resolve_follow_up_query(query, history)

                arguments = json.dumps(
                    {"query": query},
                    ensure_ascii=False,
                )

                return (
                    "Thought: I will search the local knowledge base.\n"
                    f"Action: {query_tool.name}\n"
                    f"Action Input: {arguments}"
                )

            result = _latest_result("rag")
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            status = result.get("status", "unknown")

            return (
                "Thought: I have received the retrieval tool result.\n"
                f"Final Answer: Status: {status}. "
                f"Answer: {answer} "
                f"Sources: {', '.join(sources) or 'None'}"
            )

        # Lookup Agent.
        if record_tool is not None and query_tool is None:
            if call_number == 0:
                record_id = _extract_record_id(description)
                arguments = json.dumps({"record_id": record_id})

                return (
                    "Thought: I will look up the appointment record.\n"
                    f"Action: {record_tool.name}\n"
                    f"Action Input: {arguments}"
                )

            result = _latest_result("appointment")

            return (
                "Thought: I have received the appointment lookup result.\n"
                "Final Answer: " + json.dumps(result, ensure_ascii=False)
            )

        # Response Composer.
        if agent is not None and getattr(agent, "role", "") == "Response Composer":
            if not TOOL_RESULTS:
                return (
                    "Thought: No tool result is available.\n"
                    "Final Answer: I could not prepare a response because "
                    "no support tool result was recorded."
                )

            latest = TOOL_RESULTS[-1]

            if latest["kind"] == "rag":
                result = latest["result"]
                sources = result.get("sources", [])

                response = result.get("answer", "")

                if sources:
                    response += " Sources: " + ", ".join(sources)
                else:
                    response += " No supporting sources were returned."

                return (
                    "Thought: I will compose a response from retrieved evidence.\n"
                    "Final Answer: " + response
                )

            result = latest["result"]

            if not result.get("found"):
                response = (
                    "I could not find that appointment record. "
                    "Please verify the appointment ID."
                )
            else:
                response = (
                    f"Appointment {result['record_id']} has status "
                    f"{result['status']}. Consultation fee: "
                    f"INR {result['consultation_fee_inr']}. "
                    f"Escalation score: {result['escalation_score']:.4f}. "
                    f"Escalation recommended: "
                    f"{result['escalation_recommended']}."
                )

            return (
                "Thought: I will summarize the lookup result.\n"
                "Final Answer: " + response
            )

        return (
            "Thought: No matching tool schema was found for this task.\n"
            "Final Answer: I could not identify a suitable support tool."
        )


def build_support_crew(mode: str) -> Crew:
    """Build a three-agent crew for either RAG or appointment lookup."""
    if mode not in {"rag", "lookup"}:
        raise ValueError("mode must be either 'rag' or 'lookup'.")

    llm = DeterministicMockLLM(model="practo-deterministic-mock")

    retrieval_agent = Agent(
        role="Retrieval Agent",
        goal=(
            "Retrieve relevant support policy evidence "
            "from the local knowledge base."
        ),
        backstory=(
            "You use the existing local retrieval pipeline " "and do not invent policy."
        ),
        tools=[rag_lookup],
        llm=llm,
        allow_delegation=False,
        max_iter=3,
        verbose=False,
    )

    lookup_agent = Agent(
        role="Lookup Agent",
        goal="Retrieve appointment details using the appointment lookup tool.",
        backstory=(
            "You report the returned record fields " "without inventing values."
        ),
        tools=[appointment_status_lookup],
        llm=llm,
        allow_delegation=False,
        max_iter=3,
        verbose=False,
    )

    composer_agent = Agent(
        role="Response Composer",
        goal="Compose a concise response based only on the tool result.",
        backstory=("You summarize tool outputs faithfully " "and do not invent facts."),
        llm=llm,
        allow_delegation=False,
        max_iter=2,
        verbose=False,
    )

    if mode == "rag":
        source_task = Task(
            description=(
                "Use the retrieval tool to answer the customer question.\n"
                "Conversation history:\n{conversation_history}\n"
                "Customer question: {query}"
            ),
            expected_output=(
                "A grounded answer with its evidence status " "and source filenames."
            ),
            agent=retrieval_agent,
        )
    else:
        source_task = Task(
            description=(
                "Look up the requested appointment using the lookup tool.\n"
                "Conversation history:\n{conversation_history}\n"
                "Appointment record ID: {record_id}"
            ),
            expected_output=(
                "The record status, consultation fee " "and escalation recommendation."
            ),
            agent=lookup_agent,
        )

    composer_task = Task(
        description=(
            "Compose the final customer-facing response from the preceding "
            "support task. Preserve the facts and do not invent information."
        ),
        expected_output=(
            "A concise customer-facing response grounded in the tool result."
        ),
        agent=composer_agent,
        context=[source_task],
    )

    return Crew(
        agents=[retrieval_agent, lookup_agent, composer_agent],
        tasks=[source_task, composer_task],
        process=Process.sequential,
        verbose=False,
        memory=False,
        tracing=False,
    )


def run_support_crew(
    mode: str,
    *,
    query: str | None = None,
    record_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Execute the crew with session memory and input guardrails."""
    if mode == "rag":
        if not isinstance(query, str) or not query.strip():
            raise ValueError("A non-empty query is required in rag mode.")

        current_message = query.strip()
        inputs: dict[str, str] = {"query": current_message}

    elif mode == "lookup":
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("A non-empty record_id is required in lookup mode.")

        clean_record_id = record_id.strip()
        current_message = f"Appointment status lookup for {clean_record_id}"
        inputs = {"record_id": clean_record_id}

    else:
        raise ValueError("mode must be either 'rag' or 'lookup'.")

    if session_id is None:
        effective_session_id = f"one-shot-{uuid4().hex}"
    elif not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be a non-empty string.")
    else:
        effective_session_id = session_id.strip()

    # Run input guardrails before constructing or executing the crew.
    if mode == "rag":
        guardrail_result = guard_input(current_message)

        if not guardrail_result.allowed:
            reset_tool_audit()

            refusal = SupportResponse(
                mode="rag",
                status="refused",
                answer=(
                    "I can help with clinic-support questions, but I cannot "
                    "follow instructions that attempt to override safety "
                    "instructions."
                ),
                sources=[],
            )

            refusal_data = refusal.model_dump()

            # Do not store the rejected raw input in session memory.
            return {
                "mode": mode,
                "session_id": effective_session_id,
                "output": refusal.answer,
                "structured_response": refusal_data,
                "tool_invocations": [],
                "tool_results": [],
                "guardrail": {
                    "allowed": False,
                    "reason": guardrail_result.reason,
                },
            }

        # Only the masked query proceeds to the CrewAI workflow.
        current_message = guardrail_result.text
        inputs["query"] = current_message
        guardrail_metadata = {
            "allowed": True,
            "reason": "allowed",
        }
    else:
        # The appointment ID is not a free-text customer question.
        guardrail_metadata = {
            "allowed": True,
            "reason": "not_applicable_to_record_id",
        }

    # Attach only the selected session's history.
    # Mask older stored history too, in case it contains a contact number.
    previous_history = SESSION_MEMORY.format_history(effective_session_id)
    inputs["conversation_history"] = mask_contact_numbers(previous_history)

    # Reset shared tool audit and mock state before starting this run.
    reset_tool_audit()

    crew = build_support_crew(mode)
    result = crew.kickoff(inputs=inputs)
    output = str(result)

    structured_response = _build_structured_response(mode)


    review_metadata: dict[str, Any] | None = None

    if mode == "rag" and structured_response["status"] == "answered":
        rag_result = _latest_result("rag")
        context = rag_result.get("context", "")

        if isinstance(context, str) and context.strip():
            try:
                review_result = _run_autogen_review(
                    query=current_message,
                    draft=output,
                    context=context,
                )

                verdict = review_result["verdict"]
                structured_response["answer"] = verdict["final_answer"]

                review_metadata = {
                    "status": "completed",
                    "verdict": verdict,
                    "turn_count": review_result.get("turn_count"),
                    "trace": review_result.get("trace", []),
                }

            except Exception as exc:
                # Preserve the original grounded answer if review fails.
                review_metadata = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        else:
            review_metadata = {
                "status": "skipped",
                "reason": "No original retrieved context was available.",
            }

    # Store the masked current message, not the original raw query.
    memory_answer = structured_response["answer"]

    SESSION_MEMORY.add_turn(
        effective_session_id,
        current_message,
        memory_answer,
    )

    if review_metadata and review_metadata.get("status") == "completed":
        output = structured_response["answer"]

    return {
        "mode": mode,
        "session_id": effective_session_id,
        "output": output,
        "structured_response": structured_response,
        "tool_invocations": list(TOOL_INVOCATIONS),
        "tool_results": list(TOOL_RESULTS),
        "guardrail": guardrail_metadata,
        "review": review_metadata,
    }


if __name__ == "__main__":
    print("=== RAG sample ===")
    print(
        json.dumps(
            run_support_crew(
                "rag",
                query="What is the cancellation policy?",
            ),
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\n=== Appointment lookup sample ===")
    print(
        json.dumps(
            run_support_crew(
                "lookup",
                record_id="APT-0010",
            ),
            indent=2,
            ensure_ascii=False,
        )
    )
