
"""API acceptance tests for the Practo support agent."""

from fastapi.testclient import TestClient

import api
import time
import json
import logging

def test_health_endpoint():
    client = TestClient(api.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_appointment_status_endpoint():
    client = TestClient(api.app)

    response = client.post(
        "/appointment-status",
        json={"record_id": "APT-0010"},
    )

    assert response.status_code == 200
    assert response.json()["found"] is True
    assert response.json()["record_id"] == "APT-0010"


def test_unknown_appointment_returns_not_found_result():
    client = TestClient(api.app)

    response = client.post(
        "/appointment-status",
        json={"record_id": "INVALID-ID"},
    )

    assert response.status_code == 200
    assert response.json()["found"] is False


def test_extra_request_fields_are_rejected():
    client = TestClient(api.app)

    response = client.post(
        "/appointment-status",
        json={
            "record_id": "APT-0010",
            "extra": "unexpected",
        },
    )

    assert response.status_code == 422


def test_ask_endpoint_returns_structured_response():
    client = TestClient(api.app)

    response = client.post(
        "/ask",
        json={"query": "What is the cancellation policy?"},
    )

    assert response.status_code == 200

    body = response.json()
    assert body["request_id"]
    assert body["session_id"]
    assert body["response"]["mode"] == "rag"
    assert body["response"]["status"] in {
        "answered",
        "not_found",
        "refused",
    }
    assert body["response"]["answer"]
    assert isinstance(body["response"]["sources"], list)


def test_websocket_supports_multiple_turns():
    client = TestClient(api.app)

    with client.websocket_connect("/ws/chat") as websocket:
        websocket.send_json(
            {"query": "What is the cancellation policy?"}
        )
        first = websocket.receive_json()

        assert first["type"] == "message"
        assert first["session_id"].startswith("ws-")
        assert first["response"]["answer"]

        websocket.send_json(
            {"query": "Can you clarify the policy?"}
        )
        second = websocket.receive_json()

        assert second["type"] == "message"
        assert second["session_id"] == first["session_id"]
        assert second["response"]["answer"]


def test_websocket_rejects_client_controlled_session_id():
    client = TestClient(api.app)

    with client.websocket_connect("/ws/chat") as websocket:
        websocket.send_json(
            {
                "query": "What is the cancellation policy?",
                "session_id": "client-selected-session",
            }
        )

        response = websocket.receive_json()

        assert response["type"] == "error"
        assert "managed by the server" in response["message"]


def test_websocket_disconnect_does_not_break_server():
    client = TestClient(api.app)

    # Opening and immediately closing simulates a client disconnect.
    with client.websocket_connect("/ws/chat"):
        pass

    # A fresh request must still be served after that disconnect.
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ask_rejects_empty_query():
    client = TestClient(api.app)

    response = client.post("/ask", json={"query": ""})

    assert response.status_code == 422


def test_websocket_invalid_message_does_not_end_connection():
    client = TestClient(api.app)

    with client.websocket_connect("/ws/chat") as websocket:
        # Empty query must produce an error event.
        websocket.send_json({"query": ""})
        invalid_response = websocket.receive_json()

        assert invalid_response["type"] == "error"
        assert invalid_response["session_id"].startswith("ws-")

        # The same connection should still accept a valid message.
        websocket.send_json(
            {"query": "What is the cancellation policy?"}
        )
        valid_response = websocket.receive_json()

        assert valid_response["type"] == "message"
        assert valid_response["response"]["answer"]


def test_websocket_disconnect_during_processing_keeps_server_healthy(
    monkeypatch,
):
    client = TestClient(api.app)
    original_run_support_crew = api.run_support_crew

    def slow_run_support_crew(*args, **kwargs):
        # Give the client time to disconnect while processing is underway.
        time.sleep(0.5)
        return original_run_support_crew(*args, **kwargs)

    monkeypatch.setattr(
        api,
        "run_support_crew",
        slow_run_support_crew,
    )

    with client.websocket_connect("/ws/chat") as websocket:
        websocket.send_json(
            {"query": "What is the cancellation policy?"}
        )

        # Close without waiting for the response.
        websocket.close()

    # A fresh HTTP request must still work after the disconnect.
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_http_request_creates_one_jsonl_log_with_trace_and_timing(caplog):
    client = TestClient(api.app)

    with caplog.at_level(logging.INFO, logger="practo_support_api"):
        response = client.get("/health")

    assert response.status_code == 200
    assert response.headers.get("X-Trace-ID")

    records = [
        record
        for record in caplog.records
        if record.name == "practo_support_api"
        and record.getMessage().find('"event":"http_request"') >= 0
    ]

    assert len(records) == 1

    entry = json.loads(records[0].getMessage())

    assert entry["trace_id"] == response.headers["X-Trace-ID"]
    assert entry["method"] == "GET"
    assert entry["path"] == "/health"
    assert entry["status_code"] == 200
    assert isinstance(entry["duration_ms"], (int, float))
    assert entry["duration_ms"] >= 0


def test_ask_log_masks_contact_number(monkeypatch, caplog):
    client = TestClient(api.app)
    raw_number = "9876543210"

    def fake_run_support_crew(mode, query, session_id=None):
        # Simulate the existing Task 10 input guardrail.
        from guardrails import guard_input

        guarded = guard_input(query)
        assert guarded.allowed is True
        assert raw_number not in guarded.text

        return {
            "mode": mode,
            "session_id": session_id or "test-session",
            "structured_response": {
                "mode": "rag",
                "status": "answered",
                "answer": "Please refer to the cancellation policy.",
                "sources": ["cancellation_rescheduling.md"],
            },
        }

    monkeypatch.setattr(api, "run_support_crew", fake_run_support_crew)

    with caplog.at_level(logging.INFO, logger="practo_support_api"):
        response = client.post(
            "/ask",
            json={"query": f"Please contact me at {raw_number}"},
        )

    assert response.status_code == 200

    records = [
        record
        for record in caplog.records
        if record.name == "practo_support_api"
        and record.getMessage().find('"event":"http_request"') >= 0
    ]

    assert len(records) == 1

    entry = json.loads(records[0].getMessage())
    serialized_entry = json.dumps(entry)

    assert entry["trace_id"] == response.headers["X-Trace-ID"]
    assert raw_number not in serialized_entry
    assert "[MASKED_CONTACT_NUMBER]" in entry["request_text"]


def test_websocket_log_has_jsonl_trace_timing_and_masked_query(
    monkeypatch,
    caplog,
):
    client = TestClient(api.app)
    raw_number = "9876543210"

    def fake_run_support_crew(mode, query, session_id=None):
        from guardrails import guard_input

        guarded = guard_input(query)
        assert guarded.allowed is True
        assert raw_number not in guarded.text

        return {
            "mode": mode,
            "session_id": session_id,
            "structured_response": {
                "mode": "rag",
                "status": "answered",
                "answer": "Please refer to the support policy.",
                "sources": ["cancellation_rescheduling.md"],
            },
        }

    monkeypatch.setattr(api, "run_support_crew", fake_run_support_crew)

    with caplog.at_level(logging.INFO, logger="practo_support_api"):
        with client.websocket_connect("/ws/chat") as websocket:
            websocket.send_json(
                {"query": f"My contact number is {raw_number}"}
            )
            response = websocket.receive_json()

            assert response["type"] == "message"

    records = [
        record
        for record in caplog.records
        if record.name == "practo_support_api"
        and '"event":"websocket_message_completed"' in record.getMessage()
    ]

    assert len(records) == 1

    entry = json.loads(records[0].getMessage())
    serialized_entry = json.dumps(entry)

    assert entry["trace_id"]
    assert isinstance(entry["duration_ms"], (int, float))
    assert entry["duration_ms"] >= 0
    assert raw_number not in serialized_entry
    assert entry["request_text"] == (
        "My contact number is [MASKED_CONTACT_NUMBER]"
    )


def test_invalid_http_request_still_creates_one_jsonl_log(caplog):
    client = TestClient(api.app)

    with caplog.at_level(logging.INFO, logger="practo_support_api"):
        response = client.post(
            "/appointment-status",
            json={"record_id": "APT-0010", "unexpected": "field"},
        )

    assert response.status_code == 422

    records = [
        record
        for record in caplog.records
        if record.name == "practo_support_api"
        and '"event":"http_request"' in record.getMessage()
    ]

    assert len(records) == 1

    entry = json.loads(records[0].getMessage())

    assert entry["trace_id"] == response.headers["X-Trace-ID"]
    assert entry["status_code"] == 422
    assert isinstance(entry["duration_ms"], (int, float))
    assert entry["duration_ms"] >= 0
