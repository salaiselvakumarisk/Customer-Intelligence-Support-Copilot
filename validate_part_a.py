
import sqlite3
import pandas as pd
import re
from pathlib import Path

DB = Path("data/instamart.db")
CLEAN = Path("data/clean")
SQL_FILE = Path("sql/queries.sql")

checks = []

def check(name, condition):
    checks.append((name, bool(condition)))
    print(f"{'PASS' if condition else 'FAIL'}: {name}")

# 1. Clean CSV validation
customers = pd.read_csv(CLEAN / "customers.csv")
products = pd.read_csv(CLEAN / "products.csv")
orders = pd.read_csv(CLEAN / "orders.csv")

check("Customer rows = 1200", len(customers) == 1200)
check("Product rows = 40", len(products) == 40)
check("Order rows = 12309", len(orders) == 12309)
check("Customer ages have no missing values", customers["age"].isna().sum() == 0)
check("Customer ages between 10 and 100", customers["age"].between(10, 100).all())

# 2. Database validation
conn = sqlite3.connect(DB)

for table, expected in [
    ("customers", 1200),
    ("products", 40),
    ("orders", 12309)
]:
    actual = conn.execute(
        f"SELECT COUNT(*) FROM {table}"
    ).fetchone()[0]
    check(f"SQLite {table} row count = {expected}", actual == expected)

completed = conn.execute("""
    SELECT COUNT(*) FROM orders WHERE status='completed'
""").fetchone()[0]

cancelled = conn.execute("""
    SELECT COUNT(*) FROM orders WHERE status='cancelled'
""").fetchone()[0]

revenue = conn.execute("""
    SELECT ROUND(SUM(order_amount), 2)
    FROM orders WHERE status='completed'
""").fetchone()[0]

check("Completed orders = 11110", completed == 11110)
check("Cancelled orders = 1199", cancelled == 1199)
check("Completed revenue = 2998668.80", revenue == 2998668.80)

# 3. SQL query validation
sql_text = SQL_FILE.read_text(encoding="utf-8")
parts = re.split(r"-- QUERY: (Q\d+)", sql_text)

expected_rows = {
    "Q1": 30,
    "Q2": 30,
    "Q3": 11,
    "Q4": 25,
    "Q5": 550,
    "Q6": 1175,
    "Q7": 183
}

sql_results = {}

for i in range(1, len(parts), 2):
    name = parts[i]
    query = parts[i + 1].strip()
    result = pd.read_sql_query(query, conn)
    sql_results[name] = result

    check(
        f"{name} expected row count = {expected_rows[name]}",
        len(result) == expected_rows[name]
    )

check(
    "Q2 revenue total matches",
    round(sql_results["Q2"]["revenue"].sum(), 2) == 2998668.80
)

check(
    "Q6 has 8 RANK and ROW_NUMBER differences",
    (
        sql_results["Q6"]["spend_rank"]
        != sql_results["Q6"]["row_number"]
    ).sum() == 8
)

check(
    "Q7 final cumulative revenue matches",
    round(sql_results["Q7"].iloc[-1]["cumulative_revenue"], 2)
    == 2998668.80
)

# 4. Pandas analysis validation
pivot = pd.read_csv(
    "outputs/revenue_pivot.csv", index_col=0
)

city_stats = pd.read_csv(
    "outputs/city_statistics.csv", index_col=0
)

check(
    "Pandas pivot total matches",
    round(pivot.loc["Grand Total", "Grand Total"], 2)
    == 2998668.80
)

check(
    "Bengaluru revenue matches",
    round(city_stats.loc["Bengaluru", "total_revenue"], 2)
    == 523643.40
)

check(
    "Mumbai revenue matches",
    round(city_stats.loc["Mumbai", "total_revenue"], 2)
    == 479058.50
)

conn.close()

# 5. Final result
print("\n========== FINAL VALIDATION ==========")

failed = [name for name, passed in checks if not passed]

print(f"Checks passed: {len(checks) - len(failed)}/{len(checks)}")

if failed:
    print("Failed checks:")
    for name in failed:
        print("-", name)
    raise SystemExit(1)

print("ALL CHECKPOINTS PASSED")
