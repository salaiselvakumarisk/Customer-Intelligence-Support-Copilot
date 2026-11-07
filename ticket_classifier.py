
import json
import random
import re
import time
from pathlib import Path

import pandas as pd

import mock_llm
from mock_llm import TransientLLMError


BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "outputs" / "tickets_preprocessed.csv"

ZERO_SHOT_OUTPUT = BASE_DIR / "outputs" / "triage_zero_shot.csv"
FEW_SHOT_OUTPUT = BASE_DIR / "outputs" / "triage_few_shot.csv"

CATEGORIES = {"billing", "delivery", "technical", "fraud"}

MAX_RETRIES = 3

ZERO_SHOT_PROMPT = """
CLASSIFY

You are a customer support ticket classification assistant.

Classify each customer ticket into exactly one of these categories:
billing, delivery, technical, fraud.

Return JSON only in this format:
{"category": "category_name", "reason": "short explanation"}

Do not include markdown or additional text.
"""

FEW_SHOT_PROMPT = """
CLASSIFY

You are a customer support ticket classification assistant.

Classify each customer ticket into exactly one of these categories:
billing, delivery, technical, fraud.

EXAMPLES:

Ticket: I noticed an unauthorized transaction on my account.
{"category": "fraud", "reason": "Unauthorized transaction"}

Ticket: My invoice shows an incorrect amount.
{"category": "billing", "reason": "Incorrect invoice amount"}

Ticket: The application crashes whenever I try to log in.
{"category": "technical", "reason": "Application malfunction"}

Return JSON only in this format:
{"category": "category_name", "reason": "short explanation"}

Do not include markdown or additional text.
"""


def build_messages(system_prompt, ticket_text):
    # The user message must contain ONLY the redacted ticket text.
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": ticket_text},
    ]


def call_llm_with_retry(messages, ticket_id):
    """Call mock LLM with three retries and exponential backoff."""

    for retry_number in range(MAX_RETRIES + 1):

        try:
            response = mock_llm.chat(messages)
            return response

        except TransientLLMError:

            if retry_number == MAX_RETRIES:
                print(
                    f"[ERROR] Ticket {ticket_id}: "
                    "All retries exhausted."
                )
                return None

            wait_time = (
                0.05
                * (2 ** retry_number)
                * random.uniform(0.9, 1.15)
            )

            print(
                f"[RETRY] Ticket {ticket_id} | "
                f"Retry {retry_number + 1}/{MAX_RETRIES} | "
                f"Waiting {wait_time:.4f}s"
            )

            time.sleep(wait_time)

    return None


def validate_response(raw_output):
    """Validate JSON first, then attempt keyword fallback."""

    try:
        parsed = json.loads(raw_output)

        if (
            isinstance(parsed, dict)
            and parsed.get("category") in CATEGORIES
        ):
            return (
                "classified_json",
                parsed["category"],
                parsed.get("reason", ""),
            )

        # Valid JSON with an invalid category is rejected.
        return ("flagged_invalid_output", None, "")

    except (json.JSONDecodeError, TypeError):

        words = re.findall(r"[a-z]+", (raw_output or "").lower())

        matches = [
            word for word in words
            if word in CATEGORIES
        ]

        if len(matches) == 1:
            return (
                "classified_keyword",
                matches[0],
                "Recovered using keyword fallback",
            )

        return ("flagged_invalid_output", None, "")


def run_classification(df, mode):

    if mode == "zero_shot":
        system_prompt = ZERO_SHOT_PROMPT
        output_file = ZERO_SHOT_OUTPUT

    elif mode == "few_shot":
        system_prompt = FEW_SHOT_PROMPT
        output_file = FEW_SHOT_OUTPUT

    else:
        raise ValueError(f"Unknown classification mode: {mode}")

    # Reset mock counters before each independent full run.
    mock_llm.reset()

    results = []
    total_tokens = 0

    for _, row in df.iterrows():

        result = {
            "ticket_id": row["ticket_id"],
            "true_category": row["true_category"],
            "status": row["status"],
            "category": None,
            "reason": "",
            "tokens_used": 0,
        }

        # Preserve preprocessing guardrails.
        if row["status"] != "ready_for_classification":
            results.append(result)
            continue

        messages = build_messages(
            system_prompt,
            row["redacted_text"],
        )

        response = call_llm_with_retry(
            messages,
            row["ticket_id"],
        )

        if response is None:
            result["status"] = "flagged_invalid_output"
            result["reason"] = "Transient retries exhausted"
            results.append(result)
            continue

        raw_output = response.choices[0].message.content

        tokens_used = response.usage.total_tokens

        total_tokens += tokens_used

        status, category, reason = validate_response(raw_output)

        result["status"] = status
        result["category"] = category
        result["reason"] = reason
        result["tokens_used"] = tokens_used

        results.append(result)

    result_df = pd.DataFrame(results)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_file, index=False)

    return result_df, total_tokens


def evaluate_results(result_df, mode, total_tokens):

    classified = result_df[
        result_df["status"].isin(
            ["classified_json", "classified_keyword"]
        )
    ].copy()

    correct = (
        classified["category"] == classified["true_category"]
    ).sum()

    classified_count = len(classified)

    accuracy = (
        correct / classified_count * 100
        if classified_count else 0
    )

    print(f"\n========== {mode.upper()} RESULTS ==========")

    print(f"Total classified: {classified_count}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Total tokens: {total_tokens}")

    print(
        "Tokens per classified ticket:",
        round(total_tokens / classified_count, 2)
        if classified_count else 0,
    )

    print("\nSTATUS COUNTS:")
    print(result_df["status"].value_counts().to_string())

    return classified


def main():

    df = pd.read_csv(INPUT_FILE)

    zero_df, zero_tokens = run_classification(
        df,
        "zero_shot",
    )

    few_df, few_tokens = run_classification(
        df,
        "few_shot",
    )

    zero_classified = evaluate_results(
        zero_df,
        "zero_shot",
        zero_tokens,
    )

    few_classified = evaluate_results(
        few_df,
        "few_shot",
        few_tokens,
    )

    print("\n========== CLASSIFICATION COMPARISON ==========")

    zero_accuracy = (
        (zero_classified["category"] ==
         zero_classified["true_category"]).mean() * 100
    )

    few_accuracy = (
        (few_classified["category"] ==
         few_classified["true_category"]).mean() * 100
    )

    print(f"Zero-shot accuracy: {zero_accuracy:.1f}%")
    print(f"Few-shot accuracy: {few_accuracy:.1f}%")

    print(f"Zero-shot tokens: {zero_tokens}")
    print(f"Few-shot tokens: {few_tokens}")

    print("\nZero-shot confusion matrix:")

    print(
        pd.crosstab(
            zero_classified["true_category"],
            zero_classified["category"],
            rownames=["Actual"],
            colnames=["Predicted"],
        )
    )

    print("\nOutput files:")
    print(ZERO_SHOT_OUTPUT)
    print(FEW_SHOT_OUTPUT)


if __name__ == "__main__":
    main()
