
"""Pydantic schema for structured Practo support responses."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SupportResponse(BaseModel):
    """Validate the structure of every support-agent response."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    mode: Literal["rag", "lookup"]
    status: Literal["answered", "not_found", "refused"]

    answer: str = Field(min_length=1)
    sources: list[str] = Field(default_factory=list)

    record_id: str | None = None
    consultation_fee_inr: float | None = Field(default=None, ge=0)
    escalation_score: float | None = Field(default=None, ge=0, le=1)
    escalation_recommended: bool | None = None

    @model_validator(mode="after")
    def validate_response_fields(self) -> "SupportResponse":
        """Enforce additional rules across related fields."""

        if self.mode == "lookup" and self.status == "answered":
            if not self.record_id:
                raise ValueError(
                    "A successful lookup response must include record_id."
                )

        return self
