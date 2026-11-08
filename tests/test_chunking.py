
from chunking import (
    build_chunk_records,
    fixed_size_chunks,
    load_documents,
    sentence_based_chunks,
    split_into_sentences,
)


def test_all_knowledge_documents_load():
    documents = load_documents()
    assert len(documents) >= 12


def test_fixed_size_chunks_respect_size_limit():
    text = "A" * 900
    chunks = fixed_size_chunks(text, chunk_size=350, overlap=60)

    assert len(chunks) > 1
    assert all(len(chunk) <= 350 for chunk in chunks)


def test_fixed_size_chunks_overlap():
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 30
    chunks = fixed_size_chunks(text, chunk_size=100, overlap=20)

    assert len(chunks) > 1
    assert chunks[0][-20:] == chunks[1][:20]


def test_sentence_splitter_preserves_sentences():
    text = "First sentence. Second sentence! Third sentence?"
    sentences = split_into_sentences(text)

    assert sentences == [
        "First sentence.",
        "Second sentence!",
        "Third sentence?",
    ]


def test_sentence_chunks_preserve_complete_short_sentences():
    text = (
        "First sentence. "
        "Second sentence. "
        "Third sentence. "
        "Fourth sentence."
    )

    chunks = sentence_based_chunks(
        text,
        max_chars=35,
        overlap_sentences=1,
    )

    assert len(chunks) > 1

    for sentence in split_into_sentences(text):
        assert any(sentence in chunk for chunk in chunks)


def test_chunk_records_have_source_metadata():
    documents = [
        {
            "source": "example.md",
            "text": "First sentence. Second sentence. Third sentence.",
        }
    ]

    records = build_chunk_records(documents, strategy="fixed")

    assert records
    assert len({record["id"] for record in records}) == len(records)

    for record in records:
        assert record["metadata"]["source"] == "example.md"
        assert record["metadata"]["strategy"] == "fixed"
        assert record["text"]


def test_invalid_chunk_settings_are_rejected():
    try:
        fixed_size_chunks("example text", chunk_size=10, overlap=10)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid overlap should raise ValueError.")
