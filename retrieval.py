
from vector_store import (
    COLLECTIONS,
    get_client,
    load_local_model,
    search_collection,
)


# Initial threshold derived from our seven-query experiment.
SIMILARITY_THRESHOLD = 0.4026

DEFAULT_TOP_K = 3

FALLBACK_MESSAGE = (
    "I could not find sufficiently relevant information "
    "in the available knowledge base. Please contact "
    "human support for further assistance."
)


def classify_similarity(similarity, threshold=SIMILARITY_THRESHOLD):
    """Decide whether the retrieved evidence is sufficiently similar."""
    return similarity >= threshold


def build_retrieval_response(
    results,
    threshold=SIMILARITY_THRESHOLD,
):
    """
    Convert ChromaDB search results into a structured response.

    The highest-ranked result determines whether the query
    passes the similarity threshold.
    """
    if not results:
        return {
            "status": "insufficient_evidence",
            "answerable": False,
            "message": FALLBACK_MESSAGE,
            "similarity": None,
            "threshold": threshold,
            "evidence": [],
        }

    top_result = results[0]

    similarity = 1.0 - top_result["distance"]

    evidence = [
        {
            "text": result["text"],
            "source": result["metadata"]["source"],
            "similarity": round(1.0 - result["distance"], 4),
        }
        for result in results
    ]

    if not classify_similarity(similarity, threshold):
        return {
            "status": "insufficient_evidence",
            "answerable": False,
            "message": FALLBACK_MESSAGE,
            "similarity": round(similarity, 4),
            "threshold": threshold,
            "evidence": [],
        }

    return {
        "status": "evidence_found",
        "answerable": True,
        "message": "Relevant knowledge-base evidence found.",
        "similarity": round(similarity, 4),
        "threshold": threshold,
        "evidence": evidence,
    }


class LocalRetriever:
    """Semantic retrieval using the local model and ChromaDB."""

    def __init__(self):
        self.model = load_local_model()
        self.client = get_client()

        self.collection = self.client.get_collection(
            name=COLLECTIONS["sentence"]
        )

    def retrieve(self, query, top_k=DEFAULT_TOP_K):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        results = search_collection(
            model=self.model,
            collection=self.collection,
            query=query.strip(),
            top_k=top_k,
        )

        return build_retrieval_response(results)


if __name__ == "__main__":
    retriever = LocalRetriever()

    test_queries = [
        "How many hours before an appointment can I cancel?",
        "What is the cardiology consultation fee?",
        "Who won the FIFA World Cup in 2018?",
    ]

    for query in test_queries:
        result = retriever.retrieve(query)

        print("\n" + "=" * 60)
        print("QUERY:", query)
        print("STATUS:", result["status"])
        print("ANSWERABLE:", result["answerable"])
        print("SIMILARITY:", result["similarity"])
        print("MESSAGE:", result["message"])

        for evidence in result["evidence"]:
            print("\nSOURCE:", evidence["source"])
            print("SIMILARITY:", evidence["similarity"])
            print("TEXT:", evidence["text"])
