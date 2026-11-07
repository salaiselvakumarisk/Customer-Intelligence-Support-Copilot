
"""
agent_tools.py
Part D - Refund Agent Tools

Implements:
1. get_order
2. search_policy
3. calculate_refund
4. issue_refund
5. escalate_to_human

Uses the existing policy_rag.py retrieval implementation.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from policy_rag import (
    load_policy_documents,
    chunk_documents,
    build_retrieval_index,
    retrieve_chunks,
)


DB_PATH = Path("data/instamart.db")
LEDGER_PATH = Path("outputs/refund_ledger.jsonl")

LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)

# Build the retrieval index once and reuse it.
_documents = load_policy_documents()
_chunks = chunk_documents(_documents)
_vectorizer, _vectors = build_retrieval_index(_chunks)


def get_order(order_id):
    """Retrieve order status, amount and delivery time from SQLite."""

    if not DB_PATH.exists():
        return {"error": "database not found"}

    try:
        with sqlite3.connect(DB_PATH) as connection:
            connection.row_factory = sqlite3.Row

            row = connection.execute(
                """
                SELECT order_id, status, order_amount, delivery_minutes
                FROM orders
                WHERE order_id = ?
                """,
                (order_id,)
            ).fetchone()

        if row is None:
            return {"error": "order not found"}

        return {
            "order_id": row["order_id"],
            "status": row["status"],
            "order_amount": float(row["order_amount"]),
            "delivery_minutes": (
                float(row["delivery_minutes"])
                if row["delivery_minutes"] is not None
                else None
            )
        }

    except sqlite3.Error as error:
        return {"error": f"database error: {error}"}


def search_policy(query):
    """Retrieve the three most relevant policy chunks."""

    results = retrieve_chunks(
        question=query,
        chunks=_chunks,
        vectorizer=_vectorizer,
        vectors=_vectors,
        top_k=3
    )

    return {
        "results": [
            {
                "source": item["source"],
                "score": item["similarity"],
                "text": item["text"]
            }
            for item in results
        ]
    }


def calculate_refund(order_amount, delivery_minutes):
    """
    Calculate refund amount.
    More than 45 minutes: 100% refund.
    45 minutes or less: zero refund.
    """

    try:
        amount = float(order_amount)
        delivery_time = float(delivery_minutes)
    except (TypeError, ValueError):
        return {"error": "invalid order amount or delivery time"}

    if amount < 0 or delivery_time < 0:
        return {"error": "amount and delivery time must be non-negative"}

    refund = round(amount, 2) if delivery_time > 45 else 0.0

    # Keep the exact return format required by the assignment.
    # The agent checks whether refund_inr exceeds INR 1000.
    return {"refund_inr": refund}


def _write_ledger(order_id, refund_inr, tool_name):
    """Append one simulated transaction to the refund ledger."""

    entry = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "tool": tool_name,
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2)
    }

    with LEDGER_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry) + "\n")

    return {"refund_inr": entry["refund_inr"]}


def issue_refund(order_id, refund_inr):
    """Record an automatically approved refund of INR 1000 or less."""

    try:
        amount = float(refund_inr)
    except (TypeError, ValueError):
        return {"error": "invalid refund amount"}

    if amount <= 0:
        return {"error": "refund amount must be positive"}

    if amount > 1000:
        return {"error": "refund above INR 1000 requires human approval"}

    return _write_ledger(order_id, amount, "issue_refund")


def escalate_to_human(order_id, refund_inr):
    """Record a refund above INR 1000 for human approval."""

    try:
        amount = float(refund_inr)
    except (TypeError, ValueError):
        return {"error": "invalid refund amount"}

    if amount <= 1000:
        return {"error": "human escalation is required only above INR 1000"}

    return _write_ledger(order_id, amount, "escalate_to_human")


TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Retrieve an order's status, amount and delivery time.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"}
            },
            "required": ["order_id"]
        }
    },
    {
        "name": "search_policy",
        "description": "Search policy documents for relevant information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculate_refund",
        "description": "Calculate refund amount using order amount and delivery time.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {"type": "number"},
                "delivery_minutes": {"type": "number"}
            },
            "required": ["order_amount", "delivery_minutes"]
        }
    },
    {
        "name": "issue_refund",
        "description": "Record an automatically approved refund of INR 1000 or less.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    },
    {
        "name": "escalate_to_human",
        "description": "Escalate a refund above INR 1000 for human approval.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    }
]


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}
