
"""
policy_rag.py
Part D - Policy RAG

Features:
- Load five policy documents.
- Create 40-word chunks with configurable overlap.
- Build a TF-IDF retrieval index.
- Retrieve top 3 chunks using cosine similarity.
- Generate answers using mock_llm.
- Evaluate retrieval, factual accuracy and fallback.
"""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import mock_llm


# ==================================================
# CONFIGURATION
# ==================================================

POLICY_DIR = Path("data/policies")
EVAL_FILE = Path("data/raw/policy_eval.csv")
OUTPUT_DIR = Path("outputs")

CHUNK_SIZE = 40
CHUNK_OVERLAP = 10
TOP_K = 3
MIN_SIMILARITY = 0.05

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 1. LOAD POLICY DOCUMENTS
# ==================================================

def load_policy_documents():

    documents = {}

    for file_path in sorted(POLICY_DIR.glob("*.txt")):

        documents[file_path.name] = file_path.read_text(
            encoding="utf-8"
        )

    if not documents:
        raise FileNotFoundError(
            f"No policy documents found in {POLICY_DIR}"
        )

    return documents


# ==================================================
# 2. CHUNK POLICY DOCUMENTS
# ==================================================

def chunk_documents(documents, overlap=CHUNK_OVERLAP):
    """
    Split documents into chunks of 40 words.

    Default overlap = 10 words.
    For zero-overlap experiment, pass overlap=0.
    """

    if not isinstance(overlap, int):
        raise TypeError("Overlap must be an integer.")

    if overlap < 0 or overlap >= CHUNK_SIZE:
        raise ValueError(
            f"Overlap must be between 0 and {CHUNK_SIZE - 1}"
        )

    chunks = []

    stride = CHUNK_SIZE - overlap

    for filename, content in documents.items():

        words = content.split()

        start = 0
        chunk_number = 0

        while start < len(words):

            remaining_words = len(words) - start

            chunk_words = words[
                start:start + CHUNK_SIZE
            ]

            if not chunk_words:
                break

            chunks.append({
                "source": filename,
                "chunk_id": chunk_number,
                "text": " ".join(chunk_words)
            })

            chunk_number += 1

            # Stop if the current chunk already contains
            # all remaining words.
            if remaining_words <= CHUNK_SIZE:
                break

            start += stride

    return chunks


# ==================================================
# 3. BUILD TF-IDF RETRIEVAL INDEX
# ==================================================

def build_retrieval_index(chunks):

    if not chunks:
        raise ValueError(
            "Cannot build retrieval index with zero chunks."
        )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    vectors = vectorizer.fit_transform(texts)

    return vectorizer, vectors


# ==================================================
# 4. RETRIEVE RELEVANT CHUNKS
# ==================================================

def retrieve_chunks(
    question,
    chunks,
    vectorizer,
    vectors,
    top_k=TOP_K
):

    query_vector = vectorizer.transform(
        [question]
    )

    similarities = cosine_similarity(
        query_vector,
        vectors
    ).flatten()

    ranked_indices = similarities.argsort()[::-1]

    results = []

    for index in ranked_indices:

        score = float(similarities[index])

        if score < MIN_SIMILARITY:
            continue

        result = chunks[index].copy()

        result["similarity"] = score

        results.append(result)

        if len(results) >= top_k:
            break

    return results


# ==================================================
# 5. GENERATE POLICY ANSWER
# ==================================================

def answer(
    question,
    chunks,
    vectorizer,
    vectors,
    top_k=TOP_K
):

    retrieved = retrieve_chunks(
        question,
        chunks,
        vectorizer,
        vectors,
        top_k
    )

    if not retrieved:

        return {
            "answer": (
                "I could not find that information "
                "in the policy documents."
            ),
            "sources": [],
            "scores": [],
            "fallback": True,
            "tokens_used": 0
        }

    context_parts = []

    for item in retrieved:

        context_parts.append(
            f'[{item["source"]}] {item["text"]}'
        )

    context = "\n".join(context_parts)

    system_prompt = """
POLICY ASSISTANT

Answer the user's question using only the supplied
policy context.

Rules:
1. Do not invent policy information.
2. Do not use outside knowledge.
3. If the answer is unavailable in the context,
   clearly state that the information could not
   be found in the policy documents.
4. Give a concise and direct answer.
"""

    user_prompt = (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}"
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    response = mock_llm.chat(messages)

    answer_text = (
        response.choices[0].message.content
    )

    return {
        "answer": answer_text,
        "sources": [
            item["source"]
            for item in retrieved
        ],
        "scores": [
            item["similarity"]
            for item in retrieved
        ],
        "fallback": False,
        "tokens_used": response.usage.total_tokens
    }


# ==================================================
# 6. EVALUATE POLICY RAG
# ==================================================

def evaluate_policy_rag(
    documents,
    chunks,
    vectorizer,
    vectors,
    output_filename="policy_eval_results.csv"
):

    evaluation_df = pd.read_csv(EVAL_FILE)

    results = []

    retrieval_correct = 0
    factual_correct = 0
    fallback_correct = 0

    total_in_scope = 0
    total_out_of_scope = 0

    for _, row in evaluation_df.iterrows():

        question = str(row["question"])

        expected_doc = str(
            row["expected_doc"]
        ).strip()

        expected_fact = str(
            row["expected_fact"]
        ).strip()

        retrieved = retrieve_chunks(
            question,
            chunks,
            vectorizer,
            vectors,
            top_k=TOP_K
        )

        retrieved_sources = [
            item["source"]
            for item in retrieved
        ]

        result = answer(
            question,
            chunks,
            vectorizer,
            vectors
        )

        generated_answer = result["answer"]

        if expected_doc.upper() == "NONE":

            total_out_of_scope += 1

            is_fallback = (
                result["fallback"]
                or "could not find"
                in generated_answer.lower()
            )

            if is_fallback:
                fallback_correct += 1

            retrieval_pass = None
            factual_pass = None

        else:

            total_in_scope += 1

            retrieval_pass = (
                expected_doc in retrieved_sources
            )

            if retrieval_pass:
                retrieval_correct += 1

            factual_pass = (
                expected_fact.lower()
                in generated_answer.lower()
            )

            if factual_pass:
                factual_correct += 1

            is_fallback = False

        results.append({

            "question": question,

            "expected_doc": expected_doc,

            "expected_fact": expected_fact,

            "retrieved_sources": ", ".join(
                retrieved_sources
            ),

            "answer": generated_answer,

            "retrieval_correct": retrieval_pass,

            "factual_correct": factual_pass,

            "fallback_correct": (
                is_fallback
                if expected_doc.upper() == "NONE"
                else None
            ),

            "tokens_used": result["tokens_used"]
        })

    results_df = pd.DataFrame(results)

    output_path = OUTPUT_DIR / output_filename

    results_df.to_csv(
        output_path,
        index=False
    )

    print("\n========== POLICY RAG EVALUATION ==========")

    print(
        f"Total questions: {len(evaluation_df)}"
    )

    print(
        f"Retrieval accuracy: "
        f"{retrieval_correct}/{total_in_scope}"
    )

    print(
        f"Factual answer accuracy: "
        f"{factual_correct}/{total_in_scope}"
    )

    print(
        f"Out-of-scope fallback: "
        f"{fallback_correct}/{total_out_of_scope}"
    )

    print(
        f"\nEvaluation output: {output_path}"
    )

    return results_df


# ==================================================
# 7. MAIN EXECUTION
# ==================================================

def main():

    print("Loading policy documents...")

    documents = load_policy_documents()

    print(
        f"Documents loaded: {len(documents)}"
    )

    print("\nCreating policy chunks...")

    chunks = chunk_documents(documents)

    print(
        f"Total chunks created: {len(chunks)}"
    )

    print(
        f"Chunk size: {CHUNK_SIZE} words"
    )

    print(
        f"Chunk overlap: {CHUNK_OVERLAP} words"
    )

    print("\nBuilding TF-IDF retrieval index...")

    vectorizer, vectors = build_retrieval_index(
        chunks
    )

    print(
        "Retrieval index created successfully."
    )

    # Sample retrieval

    sample_question = (
        "refund policy late delivery"
    )

    print("\nTesting sample query...")

    retrieved = retrieve_chunks(
        sample_question,
        chunks,
        vectorizer,
        vectors
    )

    for item in retrieved:

        print(
            f"\nSource: {item['source']}"
        )

        print(
            f"Chunk ID: {item['chunk_id']}"
        )

        print(
            f"Similarity: {item['similarity']:.4f}"
        )

        print(
            f"Text: {item['text']}"
        )

    # Sample answer

    print("\nTesting policy answer...")

    sample_answer = answer(
        "What delivery time makes a completed order eligible for a full refund?",
        chunks,
        vectorizer,
        vectors
    )

    print(
        f"\nAnswer: {sample_answer['answer']}"
    )

    print(
        f"Sources: {sample_answer['sources']}"
    )

    print(
        f"Tokens: {sample_answer['tokens_used']}"
    )

    # Evaluation

    print("\nRunning evaluation...")

    evaluate_policy_rag(
        documents,
        chunks,
        vectorizer,
        vectors
    )


if __name__ == "__main__":
    main()
