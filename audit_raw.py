
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")

customers = pd.read_csv(RAW / "customers.csv", keep_default_na=False)
orders = pd.read_csv(RAW / "orders.csv", keep_default_na=False)
products = pd.read_csv(RAW / "products.csv", keep_default_na=False)
tickets = pd.read_csv(RAW / "tickets.csv", keep_default_na=False)

print("\n========== RAW DATA AUDIT ==========\n")

print("1. ROW COUNTS")
for name, df in [
    ("Customers", customers),
    ("Products", products),
    ("Orders", orders),
    ("Tickets", tickets),
]:
    print(f"{name}: {len(df)}")

print("\n2. CUSTOMER DATA QUALITY")
print("Blank ages:", (customers["age"].astype(str).str.strip() == "").sum())
print("Blank genders:", (customers["gender"].astype(str).str.strip() == "").sum())
print("Distinct raw city spellings:", customers["city"].nunique())
print(
    "Distinct nonblank gender spellings:",
    customers.loc[
        customers["gender"].astype(str).str.strip() != "", "gender"
    ].nunique()
)

ages = pd.to_numeric(customers["age"], errors="coerce")
print("Ages outside 10-100:", ((ages < 10) | (ages > 100)).sum())

print("\n3. ORDER DATA QUALITY")
print("Exact duplicate rows:", orders.duplicated().sum())
print("Distinct order IDs:", orders["order_id"].nunique())
print("\nOrder statuses:")
print(orders["status"].value_counts().to_string())

amounts = pd.to_numeric(orders["order_amount"], errors="coerce")
print(
    "\nNonnumeric nonblank amounts:",
    (
        (orders["order_amount"].astype(str).str.strip() != "")
        & amounts.isna()
    ).sum()
)

delivery = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
print("Negative delivery times:", (delivery < 0).sum())
print(
    "Blank delivery times:",
    (orders["delivery_minutes"].astype(str).str.strip() == "").sum()
)

print("\n4. PRODUCT DATA")
print("Products:", len(products))
print("Categories:", products["category"].nunique())

print("\n5. TICKET DATA")
print("Tickets:", len(tickets))
print("Blank ticket texts:", (tickets["text"].str.strip() == "").sum())

print("\n========== AUDIT COMPLETED ==========\n")
