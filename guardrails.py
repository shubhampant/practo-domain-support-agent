
"""Input and output guardrails for the Practo support capstone."""

import re
from dataclasses import dataclass


# Matches Indian mobile numbers in a 10-digit format, optionally
# preceded by the +91 country code.
CONTACT_NUMBER_PATTERN = re.compile(
    r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)"
)

MASKED_CONTACT_NUMBER = "[MASKED_CONTACT_NUMBER]"


@dataclass(frozen=True)
class GuardrailResult:
    allowed: bool
    text: str
    reason: str


def mask_contact_numbers(text: str) -> str:
    """Mask matching Indian mobile numbers in user-provided text."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    return CONTACT_NUMBER_PATTERN.sub(MASKED_CONTACT_NUMBER, text)


def detect_prompt_injection(text: str) -> GuardrailResult:
    """Reject common, explicit prompt-injection instructions."""
    if not isinstance(text, str) or not text.strip():
        return GuardrailResult(
            allowed=False,
            text="",
            reason="empty_input",
        )

    patterns = (
        r"\bignore\s+(?:all\s+)?previous\s+instructions\b",
        r"\bignore\s+(?:all\s+)?prior\s+instructions\b",
        r"\breveal\s+(?:the\s+)?system\s+prompt\b",
        r"\bshow\s+(?:me\s+)?your\s+system\s+prompt\b",
        r"\bdisregard\s+(?:all\s+)?(?:previous|prior)\s+instructions\b",
        r"\b(?:bypass|disable|override)\s+(?:the\s+)?(?:safety|security)\s+guardrails\b",
    )

    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return GuardrailResult(
                allowed=False,
                text=text,
                reason="prompt_injection_detected",
            )

    return GuardrailResult(
        allowed=True,
        text=text,
        reason="allowed",
    )


def guard_input(text: str) -> GuardrailResult:
    """Detect injection first, then mask contact numbers."""
    injection_result = detect_prompt_injection(text)

    if not injection_result.allowed:
        return injection_result

    return GuardrailResult(
        allowed=True,
        text=mask_contact_numbers(text),
        reason="allowed",
    )

def check_output_groundedness(answer: str, evidence) -> GuardrailResult:
    """Reject an answer whose factual content lacks retrieved support."""
    if not isinstance(answer, str) or not answer.strip():
        return GuardrailResult(
            allowed=False,
            text="",
            reason="empty_output",
        )

    if not evidence:
        return GuardrailResult(
            allowed=False,
            text=answer,
            reason="missing_evidence",
        )

    evidence_parts = []
    for item in evidence:
        if isinstance(item, dict):
            evidence_parts.append(str(item.get("text", "")))
        elif isinstance(item, str):
            evidence_parts.append(item)

    evidence_text = " ".join(evidence_parts).lower()
    evidence_tokens = set(re.findall(r"[a-zA-Z0-9]+", evidence_text))

    # Remove generator framing, not the answer's factual claims.
    content = re.sub(
        r"^\s*Based only on the available knowledge base:\s*",
        "",
        answer.strip(),
        flags=re.IGNORECASE,
    )

    sentences = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", content)
        if part.strip()
    ]

    checked_sentences = 0

    for sentence in sentences:
        # This is an explicit capstone disclaimer, not a clinic-policy claim.
        if sentence.lower().rstrip(".") == (
            "these are synthetic capstone fee values, not official practo prices"
        ):
            continue

        tokens = {
            token
            for token in re.findall(r"[a-zA-Z0-9]+", sentence.lower())
            if len(token) > 2
        }

        if not tokens:
            continue

        checked_sentences += 1
        overlap = len(tokens & evidence_tokens) / len(tokens)

        if overlap < 0.72:
            return GuardrailResult(
                allowed=False,
                text=answer,
                reason="unsupported_output_claim",
            )

    if checked_sentences == 0:
        return GuardrailResult(
            allowed=False,
            text=answer,
            reason="no_verifiable_claims",
        )

    return GuardrailResult(
        allowed=True,
        text=answer,
        reason="grounded",
    )