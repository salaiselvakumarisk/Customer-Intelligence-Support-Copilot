
"""
diagnose_zero_overlap.py

Investigates factual failures from the zero-overlap RAG run.
Prints retrieved chunks and identifies policy sentences
that cross chunk boundaries.
"""

import re
import pandas as pd

from policy_rag import (
    EVAL_FILE,
    load_policy_documents,
    chunk_documents,
    build_retrieval_index,
    retrieve_chunks,
)


RESULT_FILE = "outputs/policy_eval_zero_overlap.csv"


def normalize_words(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def find_sequence(sequence, words):
    """Return the starting index of a word sequence."""

    length = len(sequence)

    for index in range(len(words) - length + 1):
        if words[index:index + length] == sequence:
            return index

    return -1


def diagnose_failures(documents, chunks, vectorizer, vectors):

    results = pd.read_csv(RESULT_FILE)

    failures = results[
        results["factual_correct"] == False
    ]

    print("\n========== FAILED QUESTION DIAGNOSIS ==========")

    for _, row in failures.iterrows():

        question = row["question"]
        expected_doc = row["expected_doc"]

        print("\n" + "=" * 65)
        print(f"QUESTION: {question}")
        print(f"EXPECTED DOCUMENT: {expected_doc}")
        print(f"EXPECTED FACT: {row['expected_fact']}")
        print(f"GENERATED ANSWER: {row['answer']}")

        retrieved = retrieve_chunks(
            question,
            chunks,
            vectorizer,
            vectors,
            top_k=3
        )

        print("\nTOP RETRIEVED CHUNKS:")

        for item in retrieved:
            print(
                f"\nChunk {item['chunk_id']} | "
                f"Source: {item['source']} | "
                f"Similarity: {item['similarity']:.4f}"
            )
            print(item["text"])

        print("\nALL CHUNKS FROM EXPECTED DOCUMENT:")

        for chunk in chunks:
            if chunk["source"] == expected_doc:
                print(
                    f"\nChunk {chunk['chunk_id']}: "
                    f"{chunk['text']}"
                )


def find_sentence_boundary_breaks(documents, chunks):

    print("\n========== SENTENCE BOUNDARY ANALYSIS ==========")

    for filename, content in documents.items():

        document_words = normalize_words(content)

        document_chunks = [
            chunk
            for chunk in chunks
            if chunk["source"] == filename
        ]

        chunk_ranges = []

        for chunk in document_chunks:

            chunk_words = normalize_words(chunk["text"])

            start = chunk["chunk_id"] * 40

            chunk_ranges.append({
                "id": chunk["chunk_id"],
                "start": start,
                "end": start + len(chunk_words)
            })

        sentences = re.split(
            r"(?<=[.!?])\s+",
            content
        )

        for sentence in sentences:

            sentence_words = normalize_words(sentence)

            if len(sentence_words) < 8:
                continue

            start = find_sequence(
                sentence_words,
                document_words
            )

            if start == -1:
                continue

            end = start + len(sentence_words)

            containing_chunk = any(
                chunk["start"] <= start
                and end <= chunk["end"]
                for chunk in chunk_ranges
            )

            if containing_chunk:
                continue

            overlapping_chunks = [
                chunk["id"]
                for chunk in chunk_ranges
                if chunk["start"] < end
                and chunk["end"] > start
            ]

            if len(overlapping_chunks) > 1:

                print(f"\nDocument: {filename}")
                print(
                    f"Sentence crosses chunks: "
                    f"{overlapping_chunks}"
                )
                print(f"Sentence: {sentence}")


def main():

    print("Loading policy documents...")

    documents = load_policy_documents()

    print("Creating zero-overlap chunks...")

    chunks = chunk_documents(
        documents,
        overlap=0
    )

    print(f"Total chunks: {len(chunks)}")

    vectorizer, vectors = build_retrieval_index(chunks)

    diagnose_failures(
        documents,
        chunks,
        vectorizer,
        vectors
    )

    find_sentence_boundary_breaks(
        documents,
        chunks
    )

    print("\nDiagnosis completed.")


if __name__ == "__main__":
    main()
