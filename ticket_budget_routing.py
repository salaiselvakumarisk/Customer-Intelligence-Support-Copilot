
import random
import time
from pathlib import Path

import pandas as pd

import mock_llm
from mock_llm import TransientLLMError

from ticket_classifier import (
    FEW_SHOT_PROMPT,
    ZERO_SHOT_PROMPT,
    build_messages,
    validate_response,
)


BASE_DIR = Path(__file__).resolve().parent

TICKETS_FILE = BASE_DIR / "outputs" / "tickets_preprocessed.csv"
SEGMENTS_FILE = BASE_DIR / "outputs" / "customer_segments.csv"

BUDGET_DEMO_FILE = BASE_DIR / "outputs" / "triage_budget_demo.csv"
FINAL_FILE = BASE_DIR / "outputs" / "triage_results.csv"

MAX_RETRIES = 3


def run_budgeted_classification(df, system_prompt, budget, mode):

    mock_llm.reset()

    results = []
    total_tokens = 0

    for _, row in df.iterrows():

        result = {
            "ticket_id": row["ticket_id"],
            "customer_id": row["customer_id"],
            "true_category": row["true_category"],
            "status": row["status"],
            "category": None,
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

        response = None
        budget_blocked = False

        # Three retries means up to four total attempts.
        for retry_number in range(MAX_RETRIES + 1):

            # Check budget before EVERY model call.
            if total_tokens >= budget:
                budget_blocked = True
                break

            try:
                response = mock_llm.chat(messages)
                break

            except TransientLLMError:

                if retry_number == MAX_RETRIES:
                    print(
                        f"[ERROR] {row['ticket_id']}: "
                        "Retries exhausted."
                    )
                    break

                wait_time = (
                    0.05
                    * (2 ** retry_number)
                    * random.uniform(0.9, 1.15)
                )

                print(
                    f"[RETRY] {row['ticket_id']} | "
                    f"Retry {retry_number + 1}/{MAX_RETRIES} | "
                    f"Waiting {wait_time:.4f}s"
                )

                time.sleep(wait_time)

        if budget_blocked:
            result["status"] = "budget_exhausted"

        elif response is None:
            result["status"] = "flagged_invalid_output"

        else:
            tokens_used = response.usage.total_tokens
            total_tokens += tokens_used

            raw_output = response.choices[0].message.content

            status, category, reason = validate_response(
                raw_output
            )

            result["status"] = status
            result["category"] = category
            result["tokens_used"] = tokens_used

        results.append(result)

    result_df = pd.DataFrame(results)

    print(f"\n========== {mode.upper()} BUDGET RESULTS ==========")
    print(f"Budget: {budget}")
    print(f"Tokens consumed: {total_tokens}")
    print(f"Budget remaining: {max(0, budget - total_tokens)}")
    print("\nStatus counts:")
    print(result_df["status"].value_counts().to_string())

    return result_df, total_tokens


def route_tickets(result_df, segments_df):

    # Prepare customer segment lookup.
    segment_lookup = segments_df[
        ["customer_id", "segment"]
    ].drop_duplicates(subset=["customer_id"])

    segment_lookup = segment_lookup.rename(
        columns={"segment": "segment_name"}
    )

    # Join customer segments.
    result_df = result_df.merge(
        segment_lookup,
        on="customer_id",
        how="left",
    )

    # Missing customers are treated as non-premium.
    result_df["segment_name"] = (
        result_df["segment_name"].fillna("Unknown")
    )

    is_premium = result_df["segment_name"].eq("Premium")

    human_review_statuses = {
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted",
    }

    result_df["route"] = "support_queue"
    result_df["priority"] = "NORMAL"

    # Human review first.
    human_review = result_df["status"].isin(
        human_review_statuses
    )

    result_df.loc[
        human_review, "route"
    ] = "human_review"

    # Fraud cases go to security.
    fraud = (
        ~human_review
        & result_df["category"].eq("fraud")
    )

    result_df.loc[
        fraud, "route"
    ] = "security_team"

    result_df.loc[
        fraud, "priority"
    ] = "HIGH"

    # Premium customers go to the premium desk,
    # unless already assigned to security.
    premium = (
        ~human_review
        & ~fraud
        & is_premium
    )

    result_df.loc[
        premium, "route"
    ] = "premium_desk"

    result_df.loc[
        premium, "priority"
    ] = "HIGH"

    # Keep only required final columns.
    final_df = result_df[
        [
            "ticket_id",
            "customer_id",
            "status",
            "category",
            "segment_name",
            "priority",
            "route",
            "tokens_used",
        ]
    ].copy()

    return final_df


def main():

    tickets_df = pd.read_csv(TICKETS_FILE)

    segments_df = pd.read_csv(SEGMENTS_FILE)

    # Run zero-shot budget demonstration.
    zero_budget_df, zero_tokens = run_budgeted_classification(
        tickets_df,
        ZERO_SHOT_PROMPT,
        budget=6000,
        mode="zero_shot_budget_demo",
    )

    BUDGET_DEMO_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    zero_budget_df.to_csv(
        BUDGET_DEMO_FILE,
        index=False,
    )

    # Run final few-shot classification.
    few_budget_df, few_tokens = run_budgeted_classification(
        tickets_df,
        FEW_SHOT_PROMPT,
        budget=100000,
        mode="few_shot_final",
    )

    # Route final few-shot results.
    final_df = route_tickets(
        few_budget_df,
        segments_df,
    )

    final_df.to_csv(
        FINAL_FILE,
        index=False,
    )

    print("\n========== FINAL VALIDATION ==========")

    print(f"Final rows: {len(final_df)}")
    print(f"Final columns: {list(final_df.columns)}")

    print("\nRouting summary:")
    print(
        final_df.groupby(
            ["route", "priority"]
        ).size().to_string()
    )

    print("\nFinal output:", FINAL_FILE)
    print("Budget demo:", BUDGET_DEMO_FILE)


if __name__ == "__main__":
    main()
