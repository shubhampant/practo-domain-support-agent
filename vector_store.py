
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from chunking import load_documents, build_chunk_records


MODEL_PATH = Path("models/all-MiniLM-L6-v2")
CHROMA_PATH = Path("data/chroma_db")

COLLECTIONS = {
    "fixed": "knowledge_fixed_chunks",
    "sentence": "knowledge_sentence_chunks",
}

CHUNK_SIZE = 350
FIXED_OVERLAP = 60
SENTENCE_OVERLAP = 1


def load_local_model():
    """Load only the locally saved embedding model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Local embedding model not found: {MODEL_PATH}. "
            "Run python download_model.py first."
        )

    return SentenceTransformer(
        str(MODEL_PATH.resolve()),
        local_files_only=True,
    )


def get_client():
    """Create a persistent ChromaDB client."""
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_PATH))


def build_records(documents):
    """Create records for both retrieval experiments."""
    fixed_records = build_chunk_records(
        documents,
        strategy="fixed",
        chunk_size=CHUNK_SIZE,
        overlap=FIXED_OVERLAP,
    )

    sentence_records = build_chunk_records(
        documents,
        strategy="sentence",
        chunk_size=CHUNK_SIZE,
        overlap_sentences=SENTENCE_OVERLAP,
    )

    return {
        "fixed": fixed_records,
        "sentence": sentence_records,
    }


def rebuild_collection(client, collection_name, records, model):
    """Replace a collection with a fresh index."""
    try:
        client.delete_collection(name=collection_name)
    except Exception as exc:
        # Ignore only the expected case where the collection is absent.
        if "does not exist" not in str(exc).lower():
            raise

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    if not records:
        return collection

    texts = [record["text"] for record in records]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    collection.upsert(
    ids=[record["id"] for record in records],
    documents=texts,
    metadatas=[record["metadata"] for record in records],
    embeddings=embeddings.tolist(),
)

    return collection


def build_indexes():
    documents = load_documents()

    if len(documents) < 12:
        raise ValueError(
            f"Expected at least 12 knowledge documents; found {len(documents)}."
        )

    model = load_local_model()
    client = get_client()
    records_by_strategy = build_records(documents)

    collections = {}

    for strategy, records in records_by_strategy.items():
        name = COLLECTIONS[strategy]

        collection = rebuild_collection(
            client=client,
            collection_name=name,
            records=records,
            model=model,
        )

        collections[strategy] = collection

        print(
            f"{strategy}: indexed {collection.count()} chunks "
            f"into collection '{name}'"
        )

    return model, collections


def search_collection(model, collection, query, top_k=3):
    """Return the most similar chunks and their source metadata."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if collection.count() == 0:
        return []

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )[0]

    result = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    matches = []

    for i, text in enumerate(result["documents"][0]):
        matches.append({
            "text": text,
            "metadata": result["metadatas"][0][i],
            "distance": result["distances"][0][i],
        })

    return matches


def main():
    model, collections = build_indexes()

    query = "How long before an appointment can I cancel?"

    for strategy, collection in collections.items():
        print(f"\n--- {strategy.upper()} SEARCH ---")
        print("Query:", query)

        for rank, match in enumerate(
            search_collection(model, collection, query, top_k=3),
            start=1,
        ):
            print(f"\nResult {rank}")
            print("Source:", match["metadata"]["source"])
            print("Distance:", round(match["distance"], 4))
            print("Text:", match["text"])


if __name__ == "__main__":
    build_model_and_indexes = main
    build_model_and_indexes()
