
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
CLEAN = Path("data/clean")
CLEAN.mkdir(parents=True, exist_ok=True)

# --------------------------------
# 1. CLEAN CUSTOMERS
# --------------------------------

customers = pd.read_csv(RAW / "customers.csv", keep_default_na=False)

customers["city"] = customers["city"].astype(str).str.strip().str.title()

customers["gender"] = (
    customers["gender"].astype(str).str.strip().str.lower()
    .map({
        "male": "male", "m": "male",
        "female": "female", "f": "female"
    })
    .fillna("unknown")
)

customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
customers.loc[
    ~customers["age"].between(10, 100), "age"
] = float("nan")

customers["age"] = customers["age"].fillna(
    customers.groupby("city")["age"].transform("median")
)

customers["age"] = customers["age"].round().astype(int)

customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="coerce"
)

customers.to_csv(CLEAN / "customers.csv", index=False)

# --------------------------------
# 2. CLEAN PRODUCTS
# --------------------------------

products = pd.read_csv(RAW / "products.csv")

products.to_csv(CLEAN / "products.csv", index=False)

# --------------------------------
# 3. CLEAN ORDERS
# --------------------------------

orders = pd.read_csv(RAW / "orders.csv", keep_default_na=False)

orders["order_ts"] = pd.to_datetime(
    orders["order_ts"], errors="coerce"
)

# Remove exact duplicate rows
orders = orders.drop_duplicates()

# Retain latest record per order ID
orders = orders.sort_values("order_ts")
orders = orders.drop_duplicates(
    subset="order_id", keep="last"
).copy()

# Convert amount to numeric
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

# Join catalogue prices for missing amount repair
orders = orders.merge(
    products[["product_id", "price_inr"]],
    on="product_id",
    how="left",
    validate="many_to_one"
)

missing_amount = orders["order_amount"].isna()

orders.loc[missing_amount, "order_amount"] = (
    orders.loc[missing_amount, "price_inr"]
    * orders.loc[missing_amount, "quantity"]
    * (
        1 - orders.loc[missing_amount, "discount_pct"] / 100
    )
).round(2)

orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

orders.loc[
    orders["delivery_minutes"] < 0, "delivery_minutes"
] = float("nan")

# Completed-order delivery imputation by product category
category_map = products.set_index("product_id")["category"]
orders["category"] = orders["product_id"].map(category_map)

completed_mask = orders["status"].eq("completed")

category_medians = (
    orders.loc[completed_mask]
    .groupby("category")["delivery_minutes"]
    .median()
)

missing_completed = (
    completed_mask & orders["delivery_minutes"].isna()
)

orders.loc[missing_completed, "delivery_minutes"] = (
    orders.loc[missing_completed, "category"]
    .map(category_medians)
)

# Remove temporary columns
orders = orders.drop(columns=["price_inr", "category"])

orders["order_ts"] = orders["order_ts"].dt.strftime(
    "%Y-%m-%d %H:%M:%S"
)

orders.to_csv(CLEAN / "orders.csv", index=False)

# --------------------------------
# 4. CLEANING SUMMARY
# --------------------------------

print("\n========== CLEANING SUMMARY ==========")

print("Customers:", len(customers))
print("Products:", len(products))
print("Orders:", len(orders))

print("\nOrder status counts:")
print(orders["status"].value_counts().to_string())

print("\nMissing customer ages:", customers["age"].isna().sum())
print("Missing order amounts:", orders["order_amount"].isna().sum())

print(
    "Missing completed delivery times:",
    orders.loc[orders["status"] == "completed",
               "delivery_minutes"].isna().sum()
)

print(
    "Missing cancelled delivery times:",
    orders.loc[orders["status"] == "cancelled",
               "delivery_minutes"].isna().sum()
)

print("\nCompleted order amount total:")
print(
    round(
        orders.loc[orders["status"] == "completed",
                   "order_amount"].sum(), 2
    )
)

print("\nCleaned CSV files saved successfully.")
