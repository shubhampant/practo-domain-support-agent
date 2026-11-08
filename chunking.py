
from pathlib import Path
import re


KNOWLEDGE_DIR = Path("data/knowledge_base")


def load_documents(directory=KNOWLEDGE_DIR):
    """Load Markdown documents and retain their source filenames."""
    directory = Path(directory)

    documents = []

    for path in sorted(directory.glob("*.md")):
        text = path.read_text(encoding="utf-8-sig").strip()

        if text:
            documents.append({
                "source": path.name,
                "text": text,
            })

    return documents


def fixed_size_chunks(text, chunk_size=350, overlap=60):
    """
    Split text into fixed-character chunks with overlapping characters.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size.")

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def split_into_sentences(text):
    """Split text at sentence-ending punctuation."""
    text = re.sub(r"\s+", " ", text.strip())

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)

    return [sentence.strip() for sentence in sentences if sentence.strip()]


def sentence_based_chunks(
    text,
    max_chars=350,
    overlap_sentences=1,
):
    """
    Group complete sentences up to a target character count.

    Adjacent chunks may repeat the final sentence from the
    previous chunk to preserve context.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero.")

    if overlap_sentences < 0:
        raise ValueError("overlap_sentences cannot be negative.")

    sentences = split_into_sentences(text)

    if not sentences:
        return []

    chunks = []
    current = []

    for sentence in sentences:
        candidate = " ".join(current + [sentence])

        if current and len(candidate) > max_chars:
            chunks.append(" ".join(current))

            overlap = current[-overlap_sentences:] if overlap_sentences else []
            current = list(overlap)

            # A single long sentence can exceed max_chars.
            # Keep the complete sentence rather than silently truncating it.
            current.append(sentence)
        else:
            current.append(sentence)

    if current:
        chunks.append(" ".join(current))

    return chunks


def build_chunk_records(
    documents,
    strategy,
    chunk_size=350,
    overlap=60,
    overlap_sentences=1,
):
    """Build text chunks and attach source-tracking metadata."""
    if strategy not in {"fixed", "sentence"}:
        raise ValueError("strategy must be 'fixed' or 'sentence'.")

    records = []

    for document in documents:
        if strategy == "fixed":
            pieces = fixed_size_chunks(
                document["text"],
                chunk_size=chunk_size,
                overlap=overlap,
            )
        else:
            pieces = sentence_based_chunks(
                document["text"],
                max_chars=chunk_size,
                overlap_sentences=overlap_sentences,
            )

        for index, piece in enumerate(pieces):
            records.append({
                "id": f"{strategy}:{document['source']}:{index:03d}",
                "text": piece,
                "metadata": {
                    "source": document["source"],
                    "strategy": strategy,
                    "chunk_index": index,
                },
            })

    return records


def main():
    documents = load_documents()

    fixed_records = build_chunk_records(
        documents,
        strategy="fixed",
        chunk_size=350,
        overlap=60,
    )

    sentence_records = build_chunk_records(
        documents,
        strategy="sentence",
        chunk_size=350,
        overlap_sentences=1,
    )

    print(f"Documents loaded: {len(documents)}")
    print(f"Fixed-size chunks: {len(fixed_records)}")
    print(f"Sentence-based chunks: {len(sentence_records)}")

    for label, records in [
        ("FIXED-SIZE EXAMPLE", fixed_records),
        ("SENTENCE-BASED EXAMPLE", sentence_records),
    ]:
        print(f"\n--- {label} ---")

        if records:
            record = records[0]
            print("ID:", record["id"])
            print("Source:", record["metadata"]["source"])
            print("Text:", record["text"])


if __name__ == "__main__":
    main()
