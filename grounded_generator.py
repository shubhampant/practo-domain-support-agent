import re

from retrieval import LocalRetriever, SIMILARITY_THRESHOLD
from guardrails import check_output_groundedness


class MockGroundedGenerator:
    """
    Deterministic, local-only grounded answer generator.

    Selects sentences exclusively from retrieved evidence.
    No external LLM or network call is used.
    """

    STOP_WORDS = {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "can",
        "could",
        "would",
        "should",
        "i",
        "we",
        "you",
        "they",
        "it",
        "to",
        "of",
        "for",
        "in",
        "on",
        "at",
        "and",
        "or",
        "how",
        "what",
        "when",
        "where",
        "who",
        "why",
        "many",
        "much",
        "long",
        "before",
        "after",
        "please",
        "tell",
        "me",
        "about",
        "my",
        "this",
        "that",
        "patient",
        "user",
        "required",
        "conditions",
        "a",
        "an",
        "the",
        "be",
        "been",
        "being",
    }

    # Explicit vocabulary normalization for common support questions.
    SYNONYMS = {
        "cancel": "cancellation",
        "cancelling": "cancellation",
        "cancelled": "cancellation",
        "canceled": "cancellation",
        "eligible": "eligibility",
        "eligibility": "eligibility",
        "conditions": "eligibility",
        "condition": "eligibility",
        "requirements": "eligibility",
        "required": "eligibility",
        "remote": "telemedicine",
        "online": "telemedicine",
        "video": "telemedicine",
        "consultation": "consultation",
        "consultations": "consultation",
        "fees": "fee",
        "prices": "fee",
        "price": "fee",
        "cost": "fee",
        "costs": "fee",
        "hours": "time",
        "days": "time",
        "duration": "time",
        "take": "turnaround",
        "takes": "turnaround",
        "turnaround": "turnaround",
        "opinion": "second_opinion",
    }

    def _tokens(self, text):
        words = re.findall(r"[a-zA-Z0-9]+", text.lower())
        normalized = set()

        for word in words:
            if word in self.STOP_WORDS or len(word) <= 1:
                continue

            if word in self.SYNONYMS:
                word = self.SYNONYMS[word]
            elif word.startswith("reschedul"):
                word = "reschedule"
            elif word.endswith("ies") and len(word) > 4:
                word = word[:-3] + "y"
            elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
                word = word[:-1]

            normalized.add(word)

        return normalized

    def _sentences(self, text):
        """Split text into sentences after removing known Markdown headings."""
        heading_titles = [
            "Appointment Booking",
            "Cancellation and Rescheduling",
            "Consultation Fees",
            "Insurance Claims",
            "Prescription Refills",
            "Laboratory Test Turnaround Times",
            "Telemedicine Eligibility",
            "Emergency Protocol",
            "Patient Privacy",
            "Follow-Up Discount",
            "Second-Opinion Process",
            "Home Visits",
        ]

        for heading in heading_titles:
            pattern = rf"#+\s*{re.escape(heading)}"
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

        return [
            part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()
        ]

    def generate(self, query, evidence):
        fallback = (
            "I don't know. The available knowledge base does not "
            "contain sufficient supporting information."
        )

        if not evidence:
            return fallback

        query_lower = query.lower()
        query_tokens = self._tokens(query)

        if not query_tokens:
            return fallback

        # Handle specialty-specific fee questions precisely.
        # Extract only the requested specialty's clause from the
        # retrieved fee sentence; never invent a fee.
        specialty_names = [
            "General Medicine",
            "Cardiology",
            "Dermatology",
            "Pediatrics",
            "Orthopedics",
        ]

        if "fee" in query_lower or "price" in query_lower or "cost" in query_lower:
            requested_specialty = next(
                (
                    specialty
                    for specialty in specialty_names
                    if specialty.lower() in query_lower
                ),
                None,
            )

            if requested_specialty:
                for item in evidence:
                    if item.get("source") != "consultation_fees.md":
                        continue

                    text = item.get("text", "")

                    # Find the requested specialty and its fee range.
                    pattern = (
                        rf"\b{re.escape(requested_specialty)}\b"
                        rf".{{0,100}}?"
                        rf"\bINR\s*[\d,]+\s*[\u2013-]\s*[\d,]+"
                    )

                    match = re.search(pattern, text, flags=re.IGNORECASE)

                    if match:
                        extracted = match.group(0).strip().rstrip(" ,.;:")
                        return (
                            "Based only on the available knowledge base: "
                            + extracted
                            + ". These are synthetic capstone fee values, "
                            "not official Practo prices."
                        )

        # Prefer the explicit eligibility rule when the query asks
        # about telemedicine requirements or eligibility.
        asks_telemedicine_eligibility = (
            "telemedicine" in query_lower
            or "remote consultation" in query_lower
            or "remote care" in query_lower
        ) and any(
            word in query_lower
            for word in (
                "condition",
                "conditions",
                "requirement",
                "requirements",
                "required",
                "eligible",
                "eligibility",
            )
        )

        if asks_telemedicine_eligibility:
            for item in evidence:
                if item.get("source") != "telemedicine.md":
                    continue

                for sentence in self._sentences(item.get("text", "")):
                    sentence_lower = sentence.lower()

                    if (
                        "eligible to request a remote consultation" in sentence_lower
                        and "appointment slot is available" in sentence_lower
                    ):
                        return (
                            "Based only on the available knowledge base: "
                            + sentence.strip()
                        )

        candidates = []
        seen = set()

        for item in evidence:
            for sentence in self._sentences(item.get("text", "")):
                normalized_sentence = sentence.lower().strip()

                if normalized_sentence in seen:
                    continue

                seen.add(normalized_sentence)

                sentence_tokens = self._tokens(sentence)
                overlap = query_tokens.intersection(sentence_tokens)

                if not overlap:
                    continue

                score = len(overlap) / len(query_tokens)

                candidates.append(
                    {
                        "score": score,
                        "sentence": sentence,
                    }
                )

        if not candidates:
            return (
                "I don't know. The retrieved evidence does not "
                "contain a sufficiently relevant answer."
            )

        candidates.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        best_score = candidates[0]["score"]

        selected = [
            item["sentence"]
            for item in candidates
            if item["score"] >= max(best_score * 0.5, 0.15)
        ][:3]

        if not selected:
            return (
                "I don't know. The retrieved evidence does not "
                "contain a sufficiently relevant answer."
            )

        return "Based only on the available knowledge base: " + " ".join(selected)


class GroundedSupportAgent:
    """Combines retrieval thresholding and grounded answer generation."""

    def __init__(self):
        self.retriever = LocalRetriever()
        self.generator = MockGroundedGenerator()

    def ask(self, query):
        retrieval_result = self.retriever.retrieve(query)

        if not retrieval_result["answerable"]:
            return {
                "query": query,
                "status": "insufficient_evidence",
                "answer": (
                    "I don't know. The available knowledge base does not "
                    "contain sufficient supporting information."
                ),
                "similarity": retrieval_result["similarity"],
                "threshold": SIMILARITY_THRESHOLD,
                "sources": [],
            }

        # Verify generated content before returning it to the caller.
        evidence = retrieval_result["evidence"]
        answer = self.generator.generate(query, evidence)

        if not answer.startswith("I don't know."):
            grounding_check = check_output_groundedness(answer, evidence)

            if not grounding_check.allowed:
                return {
                    "query": query,
                    "status": "insufficient_evidence",
                    "answer": (
                        "I don't know. The generated answer failed the "
                        "output-groundedness check."
                    ),
                    "similarity": retrieval_result["similarity"],
                    "threshold": SIMILARITY_THRESHOLD,
                    "sources": [],
                    "guardrail": {
                        "triggered": True,
                        "reason": grounding_check.reason,
                    },
                }

        if answer.startswith("I don't know."):
            return {
                "query": query,
                "status": "insufficient_evidence",
                "answer": answer,
                "similarity": retrieval_result["similarity"],
                "threshold": SIMILARITY_THRESHOLD,
                "sources": [],
            }

        return {
            "query": query,
            "status": "grounded_answer",
            "answer": answer,
            "similarity": retrieval_result["similarity"],
            "threshold": SIMILARITY_THRESHOLD,
            "sources": sorted(
                {item["source"] for item in evidence if item.get("source")}
            ),
            "context": "\n\n".join(
                item["text"] for item in evidence if item.get("text")
            ),
        }


def main():
    agent = GroundedSupportAgent()

    queries = [
        "How many hours before an appointment can I cancel?",
        "What is the cardiology consultation fee range?",
        "How long does a routine blood test take?",
        "What conditions are required for telemedicine?",
        "How can a patient request a second opinion?",
        "Who won the FIFA World Cup in 2018?",
    ]

    for index, query in enumerate(queries, start=1):
        result = agent.ask(query)

        print("\n" + "=" * 75)
        print(f"DEMONSTRATION {index}")
        print("=" * 75)
        print("Query:", result["query"])
        print("Status:", result["status"])
        print("Similarity:", result["similarity"])
        print("Threshold:", result["threshold"])
        print("Answer:", result["answer"])
        print("Sources:", result["sources"])


if __name__ == "__main__":
    main()
