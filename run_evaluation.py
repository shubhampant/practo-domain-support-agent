
from evaluation import EVALUATION_QUERIES, OUT_OF_SCOPE_QUERIES

from vector_store import (
    COLLECTIONS,
    get_client,
    load_local_model,
    search_collection,
)


TOP_K = 3


def calculate_precision_recall(retrieved_sources, relevant_sources):
    """
    Calculate document-level precision and recall.

    Chunks are mapped to parent documents and deduplicated
    before scoring.
    """
    retrieved = set(retrieved_sources)
    relevant = set(relevant_sources)

    true_positives = len(retrieved.intersection(relevant))

    precision = (
        true_positives / len(retrieved)
        if retrieved else 0.0
    )

    recall = (
        true_positives / len(relevant)
        if relevant else 0.0
    )

    return precision, recall, true_positives, len(retrieved)


def evaluate_collection(model, collection, strategy):
    precision_scores = []
    recall_scores = []

    print("\n" + "=" * 70)
    print(f"EVALUATION: {strategy.upper()}")
    print("=" * 70)

    for item in EVALUATION_QUERIES:
        query = item["query"]
        relevant = item["relevant_documents"]

        results = search_collection(
            model,
            collection,
            query,
            top_k=TOP_K,
        )

        retrieved_sources = [
            result["metadata"]["source"]
            for result in results
        ]

        precision, recall, tp, retrieved_count = (
            calculate_precision_recall(
                retrieved_sources,
                relevant,
            )
        )

        precision_scores.append(precision)
        recall_scores.append(recall)

        print(f"\nQuery: {query}")
        print("Relevant documents:", relevant)
        print("Retrieved documents:", sorted(set(retrieved_sources)))
        print(
            f"Precision = {tp}/{retrieved_count} = {precision:.3f}"
        )
        print(
            f"Recall = {tp}/{len(set(relevant))} = {recall:.3f}"
        )

    average_precision = sum(precision_scores) / len(precision_scores)
    average_recall = sum(recall_scores) / len(recall_scores)

    print("\n--- COLLECTION SUMMARY ---")
    print(f"Mean Precision: {average_precision:.3f}")
    print(f"Mean Recall:    {average_recall:.3f}")

    return {
        "strategy": strategy,
        "precision": average_precision,
        "recall": average_recall,
    }


def measure_similarity(model, collection, queries, label):
    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)

    measurements = []

    for query in queries:
        results = search_collection(
            model,
            collection,
            query,
            top_k=1,
        )

        if not results:
            print("No indexed results:", query)
            continue

        distance = results[0]["distance"]

        # ChromaDB uses cosine distance for our collections.
        similarity = 1.0 - distance

        measurements.append({
            "query": query,
            "distance": distance,
            "similarity": similarity,
            "source": results[0]["metadata"]["source"],
        })

        print(f"\nQuery: {query}")
        print(f"Top source: {results[0]['metadata']['source']}")
        print(f"Cosine distance: {distance:.4f}")
        print(f"Cosine similarity: {similarity:.4f}")

    return measurements


def main():
    model = load_local_model()
    client = get_client()

    collections = {
        strategy: client.get_collection(name=name)
        for strategy, name in COLLECTIONS.items()
    }

    summaries = []

    for strategy, collection in collections.items():
        summaries.append(
            evaluate_collection(model, collection, strategy)
        )

    print("\n\nCOMPARISON")
    print("-" * 50)

    for result in summaries:
        print(
            f"{result['strategy']}: "
            f"Precision={result['precision']:.3f}, "
            f"Recall={result['recall']:.3f}"
        )

    # Calibrate the fallback using the sentence-based collection
    # as an initial candidate. Final choice depends on measured data.
    candidate = collections["sentence"]

    in_scope_measurements = measure_similarity(
        model,
        candidate,
        [item["query"] for item in EVALUATION_QUERIES],
        "IN-SCOPE SIMILARITY MEASUREMENTS",
    )

    out_scope_measurements = measure_similarity(
        model,
        candidate,
        OUT_OF_SCOPE_QUERIES,
        "OUT-OF-SCOPE SIMILARITY MEASUREMENTS",
    )

    print("\n--- THRESHOLD CALIBRATION DATA ---")

    in_scores = [
        item["similarity"]
        for item in in_scope_measurements
    ]

    out_scores = [
        item["similarity"]
        for item in out_scope_measurements
    ]

    print("In-scope similarities:", [round(x, 4) for x in in_scores])
    print("Out-of-scope similarities:", [round(x, 4) for x in out_scores])

    if in_scores and out_scores:
        lowest_in = min(in_scores)
        highest_out = max(out_scores)

        if highest_out < lowest_in:
            threshold = (highest_out + lowest_in) / 2

            print(f"Observed separation: YES")
            print(f"Lowest in-scope similarity: {lowest_in:.4f}")
            print(f"Highest out-of-scope similarity: {highest_out:.4f}")
            print(f"Candidate threshold: {threshold:.4f}")
        else:
            print("Observed separation: NO")
            print(
                "The similarity clusters overlap. Do not select "
                "a threshold from these measurements yet."
            )


if __name__ == "__main__":
    main()
