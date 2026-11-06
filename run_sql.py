
import re
import sqlite3
import pandas as pd
from pathlib import Path

DB = Path("data/instamart.db")
SQL_FILE = Path("sql/queries.sql")

conn = sqlite3.connect(DB)

sql_text = SQL_FILE.read_text(encoding="utf-8")
queries = re.split(r"-- QUERY: (Q\d+)", sql_text)

results = {}

for i in range(1, len(queries), 2):
    query_name = queries[i]
    query = queries[i + 1].strip()

    df = pd.read_sql_query(query, conn)
    results[query_name] = df

    print(f"\n========== {query_name} ==========")
    print("Rows:", len(df))

    if query_name == "Q7":
        print("\nFirst 5 rows:")
        print(df.head().to_string(index=False))
        print("\nLast row:")
        print(df.tail(1).to_string(index=False))
    elif query_name == "Q6":
        print("\nTop customer in each city:")
        print(
            df[df["row_number"] == 1].to_string(index=False)
        )
        print(
            "\nRows where RANK differs from ROW_NUMBER:",
            (df["spend_rank"] != df["row_number"]).sum()
        )
    else:
        print(df.head(10).to_string(index=False))

print("\n========== SQL VALIDATION ==========")

checks = {
    "Q1": 30,
    "Q2": 30,
    "Q3": 11,
    "Q4": 25,
    "Q5": 550,
    "Q7": 183
}

for query_name, expected in checks.items():
    actual = len(results[query_name])
    print(
        f"{query_name}: actual={actual}, expected={expected}, "
        f"{'PASS' if actual == expected else 'CHECK'}"
    )

print(
    "\nQ2 total revenue:",
    round(results["Q2"]["revenue"].sum(), 2)
)

overall_average = pd.read_sql_query("""
    SELECT AVG(total_spend) AS average_spend
    FROM (
        SELECT customer_id, SUM(order_amount) AS total_spend
        FROM orders
        WHERE status = 'completed'
        GROUP BY customer_id
    )
""", conn).iloc[0]["average_spend"]

print("\nOverall average customer spend:", round(overall_average, 2))

conn.close()
