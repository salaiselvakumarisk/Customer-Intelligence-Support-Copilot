
import csv
import re
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "instamart.db"
SCENARIOS_PATH = BASE_DIR / "data" / "raw" / "agent_scenarios.csv"


def main():
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        return

    if not SCENARIOS_PATH.exists():
        print(f"Scenario file not found: {SCENARIOS_PATH}")
        return

    # 1. Read agent scenarios
    print("\n" + "=" * 70)
    print("AGENT SCENARIOS")
    print("=" * 70)

    with SCENARIOS_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        scenarios = list(csv.DictReader(file))

    for scenario in scenarios:
        print(scenario)

    # Extract order IDs from all scenario fields
    scenario_text = " ".join(
        str(value)
        for row in scenarios
        for value in row.values()
    )

    order_ids = sorted(set(re.findall(r"ORD-\d{6}", scenario_text)))

    print("\nOrder IDs found:", order_ids)

    # 2. Inspect database schema
    print("\n" + "=" * 70)
    print("DATABASE TABLES AND SCHEMAS")
    print("=" * 70)

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    tables = cursor.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        AND name NOT LIKE 'sqlite_%'
        ORDER BY name
    """).fetchall()

    for table_row in tables:
        table_name = table_row["name"]
        print(f"\nTABLE: {table_name}")

        columns = cursor.execute(
            f'PRAGMA table_info("{table_name}")'
        ).fetchall()

        for column in columns:
            print(
                f"  {column['name']} "
                f"({column['type']})"
            )

    # 3. Find scenario orders in the database
    print("\n" + "=" * 70)
    print("MATCHING ORDER RECORDS")
    print("=" * 70)

    for table_row in tables:
        table_name = table_row["name"]

        columns = cursor.execute(
            f'PRAGMA table_info("{table_name}")'
        ).fetchall()

        column_names = [column["name"] for column in columns]

        for column_name in column_names:
            for order_id in order_ids:
                sql = (
                    f'SELECT * FROM "{table_name}" '
                    f'WHERE CAST("{column_name}" AS TEXT) = ?'
                )

                try:
                    rows = cursor.execute(sql, (order_id,)).fetchall()

                    if rows:
                        print(
                            f"\nFound {order_id} "
                            f"in {table_name}.{column_name}"
                        )

                        for row in rows:
                            print(dict(row))

                except sqlite3.Error:
                    continue

    connection.close()

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
