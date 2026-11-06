
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. CONFIGURATION
# ==========================================

DATA = Path("data/clean")
OUTPUT = Path("outputs")
OUTPUT.mkdir(exist_ok=True)

CUTOFF_DATE = pd.Timestamp("2025-09-01")
PREDICTION_END = pd.Timestamp("2025-10-01")

RANDOM_STATE = 42

# ==========================================
# 2. LOAD DATA
# ==========================================

customers = pd.read_csv(DATA / "customers.csv")
orders = pd.read_csv(DATA / "orders.csv")
products = pd.read_csv(DATA / "products.csv")

orders["order_ts"] = pd.to_datetime(orders["order_ts"])
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"]
)

orders["order_amount"] = pd.to_numeric(
    orders["order_amount"]
)

orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"],
    errors="coerce"
)

# ==========================================
# 3. CREATE PRE-CUTOFF COHORT
# ==========================================

pre_orders = orders[
    (orders["order_ts"] < CUTOFF_DATE) &
    (orders["status"].isin(["completed", "cancelled"]))
].copy()

completed_pre = pre_orders[
    pre_orders["status"] == "completed"
].copy()

# Customers must have at least one completed
# order before September 1.

cohort_ids = completed_pre["customer_id"].unique()

# ==========================================
# 4. BUILD PRE-CUTOFF FEATURES
# ==========================================

# RFM features from completed orders

rfm = (
    completed_pre.groupby("customer_id")
    .agg(
        last_order_date=("order_ts", "max"),
        frequency=("order_id", "count"),
        monetary=("order_amount", "sum")
    )
    .reset_index()
)

rfm["recency"] = (
    CUTOFF_DATE - rfm["last_order_date"]
).dt.days

# Average discount from all final orders
# before cutoff (completed + cancelled)

discount_features = (
    pre_orders.groupby("customer_id")
    .agg(
        avg_discount=("discount_pct", "mean")
    )
    .reset_index()
)

# Cancellation rate:
# cancelled / all final orders before cutoff

status_counts = (
    pre_orders.groupby("customer_id")["status"]
    .value_counts()
    .unstack(fill_value=0)
)

for status in ["completed", "cancelled"]:
    if status not in status_counts.columns:
        status_counts[status] = 0

status_counts["cancellation_rate"] = (
    status_counts["cancelled"] /
    (
        status_counts["completed"] +
        status_counts["cancelled"]
    )
)

cancellation_features = (
    status_counts[["cancellation_rate"]]
    .reset_index()
)

# Average delivery time from completed orders

delivery_features = (
    completed_pre.groupby("customer_id")
    .agg(
        avg_delivery_time=("delivery_minutes", "mean")
    )
    .reset_index()
)

# Favourite product category:
# Most frequently ordered completed category

completed_products = completed_pre.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

favourite_category = (
    completed_products.groupby("customer_id")["category"]
    .agg(lambda x: x.mode().iloc[0])
    .reset_index(name="favourite_category")
)

# ==========================================
# 5. COMBINE FEATURES
# ==========================================

features = rfm.merge(
    customers,
    on="customer_id",
    how="left"
)

for feature_df in [
    discount_features,
    cancellation_features,
    delivery_features,
    favourite_category
]:
    features = features.merge(
        feature_df,
        on="customer_id",
        how="left"
    )

features["tenure_days"] = (
    CUTOFF_DATE - features["signup_date"]
).dt.days

# Keep only the cohort

features = features[
    features["customer_id"].isin(cohort_ids)
].copy()

# ==========================================
# 6. CREATE LAPSE TARGET
# ==========================================

september_completed = orders[
    (orders["status"] == "completed") &
    (orders["order_ts"] >= CUTOFF_DATE) &
    (orders["order_ts"] < PREDICTION_END)
]

active_september_ids = set(
    september_completed["customer_id"]
)

features["lapsed"] = (
    ~features["customer_id"].isin(active_september_ids)
).astype(int)

print("\n========== LAPSE DATASET ==========")

print("Cohort customers:", len(features))
print("Lapsed customers:", int(features["lapsed"].sum()))
print(
    "Active customers:",
    int((features["lapsed"] == 0).sum())
)

print("\nTarget distribution:")
print(features["lapsed"].value_counts())

# ==========================================
# 7. SELECT MODEL FEATURES
# ==========================================

numeric_features = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount",
    "cancellation_rate",
    "avg_delivery_time",
    "age",
    "tenure_days"
]

categorical_features = [
    "favourite_category",
    "city",
    "gender"
]

feature_columns = numeric_features + categorical_features

X = features[feature_columns].copy()
y = features["lapsed"].copy()

# ==========================================
# 8. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\n========== TRAIN / TEST SPLIT ==========")

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

print("\nOriginal training class counts:")
print(y_train.value_counts())

# ==========================================
# 9. OVERSAMPLE TRAINING DATA ONLY
# ==========================================

train_data = X_train.copy()
train_data["lapsed"] = y_train.values

majority_count = train_data["lapsed"].value_counts().max()

balanced_parts = []

rng = np.random.default_rng(RANDOM_STATE)

for label in sorted(train_data["lapsed"].unique()):
    class_data = train_data[
        train_data["lapsed"] == label
    ]

    if len(class_data) < majority_count:
        extra_indices = rng.choice(
            class_data.index.to_numpy(),
            size=majority_count - len(class_data),
            replace=True
        )

        class_data = pd.concat([
            class_data,
            train_data.loc[extra_indices]
        ])

    balanced_parts.append(class_data)

balanced_train = pd.concat(
    balanced_parts
).sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

X_train_balanced = balanced_train[feature_columns]
y_train_balanced = balanced_train["lapsed"]

print("\nBalanced training class counts:")
print(y_train_balanced.value_counts())

# Test set remains unchanged.

# ==========================================
# 10. PREPROCESSING PIPELINE
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)

# ==========================================
# 11. LOGISTIC REGRESSION
# ==========================================

print("\n========== LOGISTIC REGRESSION ==========")

lr_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        LogisticRegression(max_iter=1000)
    )
])

lr_grid = GridSearchCV(
    estimator=lr_pipeline,
    param_grid={
        "classifier__C": [0.01, 0.1, 1, 10, 100]
    },
    scoring="f1",
    cv=5,
    n_jobs=-1
)

lr_grid.fit(X_train_balanced, y_train_balanced)

print("Best parameters:", lr_grid.best_params_)
print("Best CV F1:", round(lr_grid.best_score_, 4))

# ==========================================
# 12. RANDOM FOREST
# ==========================================

print("\n========== RANDOM FOREST ==========")

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE
        )
    )
])

rf_grid = GridSearchCV(
    estimator=rf_pipeline,
    param_grid={
        "classifier__max_depth": [3, 5, 8],
        "classifier__min_samples_leaf": [5, 10, 20]
    },
    scoring="f1",
    cv=5,
    n_jobs=-1
)

rf_grid.fit(X_train_balanced, y_train_balanced)

print("Best parameters:", rf_grid.best_params_)
print("Best CV F1:", round(rf_grid.best_score_, 4))

# ==========================================
# 13. BASELINE AND MODEL COMPARISON
# ==========================================

print("\n========== MODEL EVALUATION ==========")

# Baseline predicts every test customer as active.

baseline_predictions = np.zeros(len(y_test), dtype=int)

print(
    "Baseline accuracy:",
    round(accuracy_score(y_test, baseline_predictions), 4)
)

print(
    "Baseline lapse F1:",
    round(f1_score(y_test, baseline_predictions), 4)
)

models = {
    "Logistic Regression": lr_grid.best_estimator_,
    "Random Forest": rf_grid.best_estimator_
}

evaluation_results = []

for name, model in models.items():

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    evaluation_results.append({
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

    print(f"\n{name}")
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1:", round(f1, 4))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

evaluation_df = pd.DataFrame(evaluation_results)

evaluation_df.to_csv(
    OUTPUT / "lapse_model_evaluation.csv",
    index=False
)

# Select model with the best CV F1.

best_name = max(
    models,
    key=lambda name: (
        lr_grid.best_score_
        if name == "Logistic Regression"
        else rf_grid.best_score_
    )
)

best_model = models[best_name]

print("\nSelected model:", best_name)

# ==========================================
# 14. SAVE AND RELOAD MODEL
# ==========================================

model_path = OUTPUT / "lapse_model.joblib"

joblib.dump(best_model, model_path)

reloaded_model = joblib.load(model_path)

original_predictions = best_model.predict(X_test)
reloaded_predictions = reloaded_model.predict(X_test)

assert np.array_equal(
    original_predictions,
    reloaded_predictions
), "Reloaded model predictions do not match!"

print("\nModel reload validation: PASSED")

# ==========================================
# 15. PREMIUM RETENTION LIST
# ==========================================

segments = pd.read_csv(
    OUTPUT / "customer_segments.csv"
)

test_customers = features.loc[
    X_test.index,
    ["customer_id", "lapsed"]
].copy()

test_customers["lapse_probability"] = (
    reloaded_model.predict_proba(X_test)[:, 1]
)

premium_ids = set(
    segments.loc[
        segments["segment"] == "Premium",
        "customer_id"
    ]
)

retention_list = test_customers[
    test_customers["customer_id"].isin(premium_ids)
].copy()

retention_list = retention_list.sort_values(
    "lapse_probability",
    ascending=False
).head(10)

retention_list.to_csv(
    OUTPUT / "retention_list.csv",
    index=False
)

print("\n========== PREMIUM RETENTION LIST ==========")
print(retention_list.to_string(index=False))

# ==========================================
# 16. FINAL SUMMARY
# ==========================================

print("\n========== PART B LAPSE MODEL COMPLETED ==========")

print("Selected model:", best_name)
print("Training customers:", len(X_train_balanced))
print("Testing customers:", len(X_test))
print("Retention list rows:", len(retention_list))
print("Saved: outputs/lapse_model.joblib")
print("Saved: outputs/retention_list.csv")
print("Saved: outputs/lapse_model_evaluation.csv")
