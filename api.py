"""FastAPI deployment for the Practo Domain Support Agent."""

# Configure telemetry before importing CrewAI or its application module.
import os

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import asyncio
import json
import logging
import os
import time
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from appointment_lookup import check_appointment_status
from crewai_support import run_support_crew
from guardrails import mask_contact_numbers
from response_schema import SupportResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("practo_support_api")

app = FastAPI(
    title="Practo Domain Support Agent",
    description="Local-first support API for the capstone project.",
    version="1.0.0",
)

# The existing CrewAI implementation has shared tool-audit/mock state.
# Serialize CrewAI executions to avoid cross-request state contamination.
crew_lock = asyncio.Lock()


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)


class AppointmentStatusRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    record_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
    )


class SupportAPIResponse(BaseModel):
    request_id: str
    session_id: str
    response: SupportResponse


class AppointmentStatusResponse(BaseModel):
    found: bool
    record_id: str
    status: str | None = None
    consultation_fee_inr: float | None = None
    escalation_score: float | None = None
    escalation_recommended: bool | None = None
    error: str | None = None


class ChatMessage(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)


class ChatEvent(BaseModel):
    type: Literal["message", "error"]
    request_id: str
    session_id: str | None = None
    response: SupportResponse | None = None
    message: str | None = None


def safe_request_id() -> str:
    """Generate a server-controlled correlation ID."""
    return uuid4().hex


def safe_log(event: str, request_id: str, **fields: object) -> None:
    """Write one JSON-serializable, privacy-safe structured log event."""
    allowed_fields = {
        "mode",
        "status",
        "duration_ms",
        "outcome",
        "method",
        "path",
        "status_code",
        "request_text",
    }

    safe_fields = {key: value for key, value in fields.items() if key in allowed_fields}

    # Defense in depth: never serialize an unmasked contact number.
    if isinstance(safe_fields.get("request_text"), str):
        safe_fields["request_text"] = mask_contact_numbers(safe_fields["request_text"])

    entry = {
        "event": event,
        "trace_id": request_id,
        **safe_fields,
    }

    logger.info(
        "%s",
        json.dumps(entry, ensure_ascii=False, separators=(",", ":")),
    )


class RequestLoggingMiddleware:
    """Log each HTTP request once, with timing and a server-generated trace ID."""

    MAX_CAPTURE_BYTES = 16 * 1024

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        trace_id = uuid4().hex
        started_at = time.perf_counter()
        status_code = 500

        path = scope.get("path", "")
        method = scope.get("method", "")
        captured_chunks: list[bytes] = []
        captured_size = 0
        capture_allowed = path == "/ask"

        async def receive_wrapper() -> Message:
            nonlocal captured_size, capture_allowed

            message = await receive()

            if capture_allowed and message["type"] == "http.request":
                chunk = message.get("body", b"")
                captured_size += len(chunk)

                if captured_size <= self.MAX_CAPTURE_BYTES:
                    captured_chunks.append(chunk)
                else:
                    # Do not retain oversized request bodies for logging.
                    captured_chunks.clear()
                    capture_allowed = False

            return message

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code

            if message["type"] == "http.response.start":
                status_code = message["status"]

                headers = list(message.get("headers", []))
                headers.append((b"x-trace-id", trace_id.encode("ascii")))
                message["headers"] = headers

            await send(message)

        try:
            await self.app(scope, receive_wrapper, send_wrapper)
        finally:
            duration_ms = round(
                (time.perf_counter() - started_at) * 1000,
                3,
            )

            fields: dict[str, object] = {
                "method": method,
                "path": path,
                "status_code": status_code,
                "duration_ms": duration_ms,
            }

            # Only capture the /ask query, never the entire request body.
            # Oversized or invalid JSON bodies are omitted.
            if capture_allowed and captured_chunks:
                try:
                    body = b"".join(captured_chunks)
                    payload = json.loads(body)
                    query = payload.get("query")

                    if isinstance(query, str):
                        fields["request_text"] = mask_contact_numbers(query)
                except (ValueError, UnicodeDecodeError, AttributeError):
                    pass

            safe_log("http_request", trace_id, **fields)


app.add_middleware(RequestLoggingMiddleware)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "practo-support-agent"}


@app.post("/ask", response_model=SupportAPIResponse)
async def ask(request: AskRequest) -> SupportAPIResponse:
    request_id = safe_request_id()

    try:
        async with crew_lock:
            result = await asyncio.to_thread(
                run_support_crew,
                "rag",
                query=request.query,
                session_id=request.session_id,
            )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="The support request could not be completed.",
        ) from exc

    structured = SupportResponse.model_validate(result["structured_response"])

    return SupportAPIResponse(
        request_id=request_id,
        session_id=result["session_id"],
        response=structured,
    )


@app.post(
    "/appointment-status",
    response_model=AppointmentStatusResponse,
)
async def appointment_status(
    request: AppointmentStatusRequest,
) -> AppointmentStatusResponse:
    request_id = safe_request_id()

    try:
        # Do not log the supplied record ID; it may be sensitive.
        result = await asyncio.to_thread(
            check_appointment_status,
            request.record_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="The appointment lookup could not be completed.",
        ) from exc

    return AppointmentStatusResponse.model_validate(result)


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket) -> None:
    await websocket.accept()

    connection_started_at = time.perf_counter()

# One session ID is retained throughout this connection.
    session_id = f"ws-{uuid4().hex}"

    try:
        while True:
            raw_message = await websocket.receive_json()
            message_started_at = time.perf_counter()

            try:
                message = ChatMessage.model_validate(raw_message)
            except Exception:
                request_id = safe_request_id()

                safe_log(
                    "websocket_message_rejected",
                    request_id,
                    duration_ms=round(
                        (time.perf_counter() - message_started_at) * 1000,
                        3,
                    ),
                    outcome="validation_error",
                )

                await websocket.send_json(
                    ChatEvent(
                        type="error",
                        request_id=request_id,
                        session_id=session_id,
                        message="Invalid message. Send a non-empty query.",
                    ).model_dump()
                )
                continue

            request_id = safe_request_id()
            safe_request_text = mask_contact_numbers(message.query)

            # Do not allow a client to change session identity mid-connection.
            if message.session_id and message.session_id != session_id:
                safe_log(
                    "websocket_message_rejected",
                    request_id,
                    duration_ms=round(
                        (time.perf_counter() - message_started_at) * 1000,
                        3,
                    ),
                    request_text=safe_request_text,
                    outcome="session_id_rejected",
                )
                await websocket.send_json(
                    ChatEvent(
                        type="error",
                        request_id=request_id,
                        session_id=session_id,
                        message="The session ID is managed by the server.",
                    ).model_dump()
                )
                continue


            try:
                async with crew_lock:
                    result = await asyncio.to_thread(
                        run_support_crew,
                        "rag",
                        query=message.query,
                        session_id=session_id,
                    )

                structured = SupportResponse.model_validate(
                    result["structured_response"]
                )

                await websocket.send_json(
                    ChatEvent(
                        type="message",
                        request_id=request_id,
                        session_id=session_id,
                        response=structured,
                    ).model_dump()
                )

                safe_log(
                    "websocket_message_completed",
                    request_id,
                    mode="rag",
                    status=structured.status,
                    duration_ms=round(
                        (time.perf_counter() - message_started_at) * 1000,
                        3,
                    ),
                    request_text=safe_request_text,
                )

            except WebSocketDisconnect:
                raise

            except Exception:
                safe_log(
                    "websocket_message_failed",
                    request_id,
                    duration_ms=round(
                        (time.perf_counter() - message_started_at) * 1000,
                        3,
                    ),
                    request_text=safe_request_text,
                    outcome="internal_error",
                )

                await websocket.send_json(
                    ChatEvent(
                        type="error",
                        request_id=request_id,
                        session_id=session_id,
                        message="The message could not be processed.",
                    ).model_dump()
                )


    except WebSocketDisconnect:
        # Log connection lifecycle metadata without client message content.
        safe_log(
            "websocket_disconnected",
            safe_request_id(),
            duration_ms=round(
                (time.perf_counter() - connection_started_at) * 1000,
                3,
            ),
            outcome="client_disconnected",
        )
    except Exception:
        # Avoid logging raw exception details that could contain user data.
        safe_log(
            "websocket_handler_failed",
            safe_request_id(),
            duration_ms=round(
                (time.perf_counter() - connection_started_at) * 1000,
                3,
            ),
            outcome="internal_error",
        )
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
