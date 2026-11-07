
import re
from pathlib import Path

import pandas as pd
from mock_llm import est_tokens


# Project paths
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / "raw" / "tickets.csv"
OUTPUT_FILE = BASE_DIR / "outputs" / "tickets_preprocessed.csv"

MAX_TOKENS = 400


# Required PII patterns — keep this exact order
PII_PATTERNS = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
    ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]"),
]


def redact_pii(text):
    """Redact PII and return replacement counts."""

    counts = {
        "email": 0,
        "phone": 0,
        "order_id": 0,
    }

    redacted_text = text

    for name, pattern, replacement in PII_PATTERNS:
        redacted_text, count = pattern.subn(
            replacement, redacted_text
        )
        counts[name] = count

    return redacted_text, counts


def process_tickets():

    # Read input data
    df = pd.read_csv(INPUT_FILE)

    # Handle missing ticket text
    df["text"] = df["text"].fillna("").astype(str)

    # Token estimation
    df["est_tokens"] = df["text"].apply(est_tokens)

    # Initialise output columns
    df["redacted_text"] = ""
    df["email_redactions"] = 0
    df["phone_redactions"] = 0
    df["order_id_redactions"] = 0
    df["status"] = ""

    total_counts = {
        "email": 0,
        "phone": 0,
        "order_id": 0,
    }

    for index, row in df.iterrows():

        text = row["text"]

        # Empty ticket guardrail
        if not text.strip():
            df.at[index, "status"] = "skipped_empty"
            continue

        # Redact PII
        redacted_text, counts = redact_pii(text)

        df.at[index, "redacted_text"] = redacted_text
        df.at[index, "email_redactions"] = counts["email"]
        df.at[index, "phone_redactions"] = counts["phone"]
        df.at[index, "order_id_redactions"] = counts["order_id"]

        for key in total_counts:
            total_counts[key] += counts[key]

        # Long ticket guardrail
        if row["est_tokens"] > MAX_TOKENS:
            df.at[index, "status"] = "flagged_long"
        else:
            df.at[index, "status"] = "ready_for_classification"

    # Save preprocessed tickets
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    # Summary
    print("\n========== PREPROCESSING SUMMARY ==========")

    print(f"Total tickets: {len(df)}")
    print(f"Empty tickets: {(df['status'] == 'skipped_empty').sum()}")
    print(f"Long tickets: {(df['status'] == 'flagged_long').sum()}")
    print(
        "Eligible tickets:",
        (df["status"] == "ready_for_classification").sum()
    )

    print("\nPII REDACTION COUNTS")
    print(f"Emails: {total_counts['email']}")
    print(f"Phones: {total_counts['phone']}")
    print(f"Order IDs: {total_counts['order_id']}")

    print("\nOutput saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    process_tickets()
