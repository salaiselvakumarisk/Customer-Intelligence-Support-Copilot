
import pandas as pd
import sqlite3
from pathlib import Path

DB = Path("data/instamart.db")
OUTPUT = Path("outputs")
OUTPUT.mkdir(exist_ok=True)

conn = sqlite3.connect(DB)

# Load completed orders with customer and product details
df = pd.read_sql_query("""
    SELECT
        o.order_id,
        o.customer_id,
        o.order_ts,
        o.order_amount,
        o.quantity,
        o.discount_pct,
        c.city,
        p.category
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    JOIN products p ON o.product_id = p.product_id
    WHERE o.status = 'completed'
""", conn)

conn.close()

# --------------------------------
# 1. PIVOT TABLE: CITY x CATEGORY
# --------------------------------

print("\n========== REVENUE PIVOT ==========")

pivot = pd.pivot_table(
    df,
    index="city",
    columns="category",
    values="order_amount",
    aggfunc="sum",
    fill_value=0,
    margins=True,
    margins_name="Grand Total"
)

print(pivot.round(2).to_string())

print(
    "\nPivot grand total:",
    round(pivot.loc["Grand Total", "Grand Total"], 2)
)

# --------------------------------
# 2. CITY-WISE AGGREGATION
# --------------------------------

print("\n========== CITY-WISE STATISTICS ==========")

city_stats = (
    df.groupby("city")
    .agg(
        total_revenue=("order_amount", "sum"),
        average_order_value=("order_amount", "mean"),
        order_count=("order_amount", "count")
    )
    .round(2)
    .sort_values("total_revenue", ascending=False)
)

print(city_stats.to_string())

# --------------------------------
# 3. ORDER AMOUNT DISTRIBUTION
# --------------------------------

print("\n========== ORDER AMOUNT STATISTICS ==========")

amount = df["order_amount"]

mean_value = amount.mean()
median_value = amount.median()
skew_value = amount.skew()
q1 = amount.quantile(0.25)
q3 = amount.quantile(0.75)
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

above_outliers = (amount > upper_fence).sum()
below_outliers = (amount < lower_fence).sum()

print("Mean:", round(mean_value, 2))
print("Median:", round(median_value, 2))
print("Skewness:", round(skew_value, 2))
print("Q1:", round(q1, 2))
print("Q3:", round(q3, 2))
print("IQR:", round(iqr, 2))
print("Lower fence:", round(lower_fence, 2))
print("Upper fence:", round(upper_fence, 2))
print("Outliers above upper fence:", above_outliers)
print("Outliers below lower fence:", below_outliers)

# --------------------------------
# 4. CORRELATION ANALYSIS
# --------------------------------

print("\n========== CORRELATION ANALYSIS ==========")

correlations = df[
    ["order_amount", "discount_pct", "quantity"]
].corr()

print(correlations.round(3).to_string())

amount_correlations = correlations["order_amount"].drop(
    "order_amount"
)

strongest = amount_correlations.abs().idxmax()

print(
    "\nLargest absolute correlation with order amount:",
    strongest,
    round(amount_correlations[strongest], 3)
)

# --------------------------------
# 5. SAVE ANALYSIS OUTPUTS
# --------------------------------

pivot.to_csv(OUTPUT / "revenue_pivot.csv")
city_stats.to_csv(OUTPUT / "city_statistics.csv")

print("\nAnalysis outputs saved successfully.")
