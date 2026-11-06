
import pandas as pd
from pathlib import Path

# --------------------------------
# CONFIGURATION
# --------------------------------

DATA = Path("data/clean")
OUTPUT = Path("outputs")
OUTPUT.mkdir(exist_ok=True)

SNAPSHOT_DATE = pd.Timestamp("2025-10-01 00:00:00")

# --------------------------------
# LOAD DATA
# --------------------------------

customers = pd.read_csv(DATA / "customers.csv")
orders = pd.read_csv(DATA / "orders.csv")

orders["order_ts"] = pd.to_datetime(orders["order_ts"])
orders["order_amount"] = pd.to_numeric(orders["order_amount"])

# --------------------------------
# FILTER COMPLETED ORDERS
# --------------------------------

completed = orders[
    orders["status"] == "completed"
].copy()

# --------------------------------
# CALCULATE RFM
# --------------------------------

rfm = (
    completed.groupby("customer_id")
    .agg(
        last_order_date=("order_ts", "max"),
        frequency=("order_id", "count"),
        monetary=("order_amount", "sum")
    )
    .reset_index()
)

rfm["recency"] = (
    SNAPSHOT_DATE - rfm["last_order_date"]
).dt.days

# --------------------------------
# ADD CUSTOMER DETAILS
# --------------------------------

rfm = rfm.merge(
    customers,
    on="customer_id",
    how="left"
)

# Arrange columns
rfm = rfm[
    [
        "customer_id",
        "city",
        "age",
        "gender",
        "signup_date",
        "last_order_date",
        "recency",
        "frequency",
        "monetary"
    ]
]

rfm["monetary"] = rfm["monetary"].round(2)

# --------------------------------
# SAVE OUTPUT
# --------------------------------

rfm.to_csv(
    OUTPUT / "rfm_features.csv",
    index=False
)

# --------------------------------
# VALIDATION SUMMARY
# --------------------------------

print("\n========== RFM ANALYSIS ==========")

print("Snapshot date:", SNAPSHOT_DATE)
print("Completed orders:", len(completed))
print("Customers with completed orders:", len(rfm))

print("\nRFM summary:")
print(
    rfm[["recency", "frequency", "monetary"]]
    .describe()
    .round(2)
)

print("\nMissing values:")
print(rfm.isnull().sum())

print("\nFirst five customers:")
print(rfm.head().to_string(index=False))

print("\nRFM dataset saved: outputs/rfm_features.csv")
