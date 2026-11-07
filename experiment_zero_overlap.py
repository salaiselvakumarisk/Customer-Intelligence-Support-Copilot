
"""
experiment_zero_overlap.py

Part D - Policy RAG
Zero-overlap experiment.

Compares retrieval and factual accuracy using:
- Chunk size: 40 words
- Chunk overlap: 0 words

Does not overwrite the original evaluation output.
"""

import re
import pandas as pd

from policy_rag import (
    CHUNK_SIZE,
    EVAL_FILE,
    load_policy_documents,
    chunk_documents,
    build_retrieval_index,
    evaluate_policy_rag,
)


# ==================================================
# 1. NORMALIZE TEXT FOR BOUNDARY ANALYSIS
# ==================================================

def normalize_words(text):
    """Convert text into comparable lowercase word tokens."""

    return re.findall(
        r"[a-z0-9]+",
        str(text).lower()
    )


# ==================================================
# 2. CHECK WHETHER FACTS CROSS CHUNK BOUNDARIES
# ==================================================

def check_chunk_boundaries(documents, chunks, evaluation_df):
    """
    Find expected facts that exist in the original
    document but do not exist completely in any
    individual chunk.
    """

    print("\n========== CHUNK BOUNDARY ANALYSIS ==========")

    boundary_breaks = []

    for _, row in evaluation_df.iterrows():

        expected_doc = str(row["expected_doc"]).strip()
        expected_fact = str(row["expected_fact"]).strip()
        question = str(row["question"])

        if expected_doc.upper() == "NONE":
            continue

        if expected_doc not in documents:
            continue

        fact_words = normalize_words(expected_fact)

        if not fact_words:
            continue

        document_words = normalize_words(
            documents[expected_doc]
        )

        chunk_word_lists = [
            normalize_words(chunk["text"])
            for chunk in chunks
            if chunk["source"] == expected_doc
        ]

        # Check whether the complete expected fact
        # exists in the original document.
        fact_exists_in_document = any(
            document_words[i:i + len(fact_words)] == fact_words
            for i in range(
                len(document_words) - len(fact_words) + 1
            )
        )

        if not fact_exists_in_document:
            continue

        # Check whether the complete expected fact
        # exists inside one individual chunk.
        fact_exists_in_chunk = any(
            any(
                chunk_words[i:i + len(fact_words)] == fact_words
                for i in range(
                    len(chunk_words) - len(fact_words) + 1
                )
            )
            for chunk_words in chunk_word_lists
        )

        if not fact_exists_in_chunk:

            boundary_breaks.append({
                "question": question,
                "document": expected_doc,
                "expected_fact": expected_fact
            })

            print("\nBOUNDARY-SPLIT FACT FOUND")

            print(f"Document: {expected_doc}")
            print(f"Question: {question}")
            print(f"Expected fact: {expected_fact}")

    if not boundary_breaks:

        print(
            "\nNo exact expected facts were found "
            "split across chunks."
        )

        print(
            "The zero-overlap evaluation results will "
            "still show whether retrieval or answering "
            "was affected."
        )

    else:

        print(
            f"\nTotal boundary-split expected facts: "
            f"{len(boundary_breaks)}"
        )

    return boundary_breaks


# ==================================================
# 3. RUN ZERO-OVERLAP EXPERIMENT
# ==================================================

def main():

    print("\n============================================")
    print("       ZERO-OVERLAP RAG EXPERIMENT")
    print("============================================")

    # Load documents

    print("\nLoading policy documents...")

    documents = load_policy_documents()

    print(
        f"Documents loaded: {len(documents)}"
    )

    # Create zero-overlap chunks

    print("\nCreating zero-overlap chunks...")

    chunks = chunk_documents(
        documents,
        overlap=0
    )

    print(
        f"Total chunks created: {len(chunks)}"
    )

    print(
        f"Chunk size: {CHUNK_SIZE} words"
    )

    print("Chunk overlap: 0 words")

    # Build retrieval index

    print("\nBuilding TF-IDF retrieval index...")

    vectorizer, vectors = build_retrieval_index(
        chunks
    )

    print(
        "Zero-overlap retrieval index created."
    )

    # Boundary analysis

    evaluation_df = pd.read_csv(EVAL_FILE)

    boundary_breaks = check_chunk_boundaries(
        documents,
        chunks,
        evaluation_df
    )

    # Run evaluation separately

    print("\nRunning zero-overlap evaluation...")

    results_df = evaluate_policy_rag(
        documents,
        chunks,
        vectorizer,
        vectors,
        output_filename="policy_eval_zero_overlap.csv"
    )

    # Display failures

    print("\n========== ZERO-OVERLAP FAILURES ==========")

    factual_failures = results_df[
        results_df["factual_correct"] == False
    ]

    retrieval_failures = results_df[
        results_df["retrieval_correct"] == False
    ]

    if factual_failures.empty:
        print("No factual answer failures.")
    else:
        print("\nFactual answer failures:")
        print(
            factual_failures[
                ["question", "expected_fact", "answer"]
            ].to_string(index=False)
        )

    if retrieval_failures.empty:
        print("\nNo retrieval failures.")
    else:
        print("\nRetrieval failures:")
        print(
            retrieval_failures[
                ["question", "expected_doc", "retrieved_sources"]
            ].to_string(index=False)
        )

    # Final summary

    print("\n========== EXPERIMENT SUMMARY ==========")

    print(f"Zero-overlap chunks: {len(chunks)}")

    print(
        f"Boundary-split expected facts: "
        f"{len(boundary_breaks)}"
    )

    print(
        f"Factual failures: {len(factual_failures)}"
    )

    print(
        f"Retrieval failures: {len(retrieval_failures)}"
    )

    print(
        "\nOutput: outputs/policy_eval_zero_overlap.csv"
    )

    print("\nZero-overlap experiment completed.")


if __name__ == "__main__":
    main()
