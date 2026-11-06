
import sqlite3
import pandas as pd
from pathlib import Path

CLEAN = Path("data/clean")
DB_PATH = Path("data/instamart.db")

DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Read cleaned CSV files
customers = pd.read_csv(CLEAN / "customers.csv")
products = pd.read_csv(CLEAN / "products.csv")
orders = pd.read_csv(CLEAN / "orders.csv")

# Connect to SQLite
conn = sqlite3.connect(DB_PATH)

try:
    # Load tables
    customers.to_sql("customers", conn, if_exists="replace", index=False)
    products.to_sql("products", conn, if_exists="replace", index=False)
    orders.to_sql("orders", conn, if_exists="replace", index=False)

    # Create useful indexes
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_orders_customer "
        "ON orders(customer_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_orders_product "
        "ON orders(product_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_orders_status "
        "ON orders(status)"
    )

    conn.commit()

    print("SQLite database created successfully.")

    # Verify tables
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
        conn
    )

    print("\nDatabase tables:")
    print(tables.to_string(index=False))

    # Verify row counts
    for table in ["customers", "products", "orders"]:
        count = conn.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0]
        print(f"{table}: {count} rows")

finally:
    conn.close()
