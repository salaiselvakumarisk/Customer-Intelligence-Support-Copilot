
"""
lapse_prediction.py

Part B: Customer Intelligence - Lapse Prediction

Workflow:
1. Load cleaned customer, order and product data.
2. Build customer features using pre-cutoff orders only.
3. Create the September lapse target.
4. Validate features and split into training and test sets.
5. Tune Logistic Regression and Random Forest on original training rows.
6. Evaluate both models on the untouched test set.
7. Run a separate training-only class-balancing experiment.
8. Interpret, save and reload the selected model.
9. Generate the Premium retention list.

The Premium retention list uses the whole-window segment label.
It is illustrative and is not a leakage-free evaluation.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import clone


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "clean"
OUTPUT_DIR = BASE_DIR / "outputs"

CUSTOMERS_FILE = DATA_DIR / "customers.csv"
ORDERS_FILE = DATA_DIR / "orders.csv"
PRODUCTS_FILE = DATA_DIR / "products.csv"

SEGMENTS_FILE = OUTPUT_DIR / "customer_segments.csv"

MODEL_FILE = OUTPUT_DIR / "lapse_model.joblib"
EVALUATION_FILE = OUTPUT_DIR / "lapse_model_evaluation.csv"
BALANCE_FILE = OUTPUT_DIR / "lapse_balance_comparison.csv"
RETENTION_FILE = OUTPUT_DIR / "retention_list.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CUTOFF_DATE = pd.Timestamp("2025-09-01 00:00:00")
PREDICTION_END = pd.Timestamp("2025-10-01 00:00:00")

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

NUMERIC_FEATURES = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount",
    "cancellation_rate",
    "avg_delivery_time",
    "age",
    "tenure_days",
]

CATEGORICAL_FEATURES = [
    "favourite_category",
    "city",
    "gender",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET_COLUMN = "lapsed"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(title):
    """Print a consistent section heading."""
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


def calculate_metrics(y_true, predictions):
    """Calculate metrics for the positive class: lapsed."""
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
    }


def print_metrics(metrics):
    """Print evaluation metrics."""
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")


# ============================================================
# 2. LOAD DATA
# ============================================================

print_section("2. LOADING DATA")

customers = pd.read_csv(CUSTOMERS_FILE)
orders = pd.read_csv(ORDERS_FILE)
products = pd.read_csv(PRODUCTS_FILE)

orders["order_ts"] = pd.to_datetime(
    orders["order_ts"],
    errors="coerce",
)

customers["signup_date"] = pd.to_datetime(
    customers["signup_date"],
    errors="coerce",
)

orders["order_amount"] = pd.to_numeric(
    orders["order_amount"],
    errors="coerce",
)

orders["discount_pct"] = pd.to_numeric(
    orders["discount_pct"],
    errors="coerce",
)

orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"],
    errors="coerce",
)

print("Customers:", len(customers))
print("Orders:", len(orders))
print("Products:", len(products))


# ============================================================
# 3. CREATE PRE-CUTOFF DATA
# ============================================================

print_section("3. PRE-CUTOFF DATA")

# Only final-status orders before the cutoff are used
# for pre-cutoff behavioural features.

pre_orders = orders.loc[
    (orders["order_ts"] < CUTOFF_DATE)
    & orders["status"].isin(["completed", "cancelled"])
].copy()

completed_pre = pre_orders.loc[
    pre_orders["status"] == "completed"
].copy()

print("Orders before cutoff:", len(pre_orders))
print("Completed orders before cutoff:", len(completed_pre))
print("Final-status orders before cutoff:", len(pre_orders))

# Customers must have at least one completed order
# before the cutoff to enter the lapse cohort.

cohort_ids = completed_pre["customer_id"].drop_duplicates()

print("Customers with pre-cutoff completed orders:", len(cohort_ids))


# ============================================================
# 4. BUILD PRE-CUTOFF CUSTOMER FEATURES
# ============================================================

print_section("4. FEATURE ENGINEERING")

# ------------------------------------------------------------
# 4.1 RFM FEATURES
# ------------------------------------------------------------

rfm = (
    completed_pre.groupby("customer_id")
    .agg(
        last_order_date=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    )
    .reset_index()
)

rfm["recency"] = (
    CUTOFF_DATE - rfm["last_order_date"]
).dt.days

# ------------------------------------------------------------
# 4.2 AVERAGE DISCOUNT
# ------------------------------------------------------------

# Assignment requirement: average discount is calculated
# from completed orders only.

discount_features = (
    completed_pre.groupby("customer_id")
    .agg(
        avg_discount=("discount_pct", "mean"),
    )
    .reset_index()
)

# ------------------------------------------------------------
# 4.3 CANCELLATION RATE
# ------------------------------------------------------------

# Cancellation rate uses all final-status orders:
# completed + cancelled.

status_counts = (
    pre_orders.groupby("customer_id")["status"]
    .value_counts()
    .unstack(fill_value=0)
)

for status in ["completed", "cancelled"]:
    if status not in status_counts.columns:
        status_counts[status] = 0

total_final_orders = (
    status_counts["completed"] + status_counts["cancelled"]
)

status_counts["cancellation_rate"] = (
    status_counts["cancelled"]
    / total_final_orders.replace(0, np.nan)
)

cancellation_features = (
    status_counts[["cancellation_rate"]]
    .reset_index()
)

# ------------------------------------------------------------
# 4.4 AVERAGE DELIVERY TIME
# ------------------------------------------------------------

# Delivery time is calculated from completed orders only.

delivery_features = (
    completed_pre.groupby("customer_id")
    .agg(
        avg_delivery_time=("delivery_minutes", "mean"),
    )
    .reset_index()
)

# ------------------------------------------------------------
# 4.5 FAVOURITE PRODUCT CATEGORY
# ------------------------------------------------------------

completed_products = completed_pre.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left",
    validate="many_to_one",
)

def most_frequent_category(values):
    """Return the first mode for deterministic tie handling."""
    modes = values.dropna().mode()

    if modes.empty:
        return np.nan

    return modes.iloc[0]


favourite_category = (
    completed_products.groupby("customer_id")["category"]
    .agg(most_frequent_category)
    .reset_index(name="favourite_category")
)

# ------------------------------------------------------------
# 4.6 CUSTOMER DETAILS AND TENURE
# ------------------------------------------------------------

features = rfm.merge(
    customers,
    on="customer_id",
    how="left",
    validate="one_to_one",
)

for feature_table in [
    discount_features,
    cancellation_features,
    delivery_features,
    favourite_category,
]:
    features = features.merge(
        feature_table,
        on="customer_id",
        how="left",
        validate="one_to_one",
    )

features["tenure_days"] = (
    CUTOFF_DATE - features["signup_date"]
).dt.days

# Retain only customers in the pre-cutoff completed-order cohort.

features = features.loc[
    features["customer_id"].isin(cohort_ids)
].copy()

# ------------------------------------------------------------
# 4.7 CREATE SEPTEMBER LAPSE TARGET
# ------------------------------------------------------------

# September orders are used ONLY to create the target.
# They are never included in the model features.

september_completed = orders.loc[
    (orders["status"] == "completed")
    & (orders["order_ts"] >= CUTOFF_DATE)
    & (orders["order_ts"] < PREDICTION_END)
].copy()

active_september_ids = set(
    september_completed["customer_id"]
)

features[TARGET_COLUMN] = (
    ~features["customer_id"].isin(active_september_ids)
).astype(int)

print("\nCohort customers:", len(features))
print("Lapsed customers:", int(features[TARGET_COLUMN].sum()))
print(
    "Active customers:",
    int((features[TARGET_COLUMN] == 0).sum()),
)

print("\nTarget distribution:")
print(features[TARGET_COLUMN].value_counts().sort_index())

lapsed_percentage = features[TARGET_COLUMN].mean() * 100

print(f"\nLapsed percentage: {lapsed_percentage:.2f}%")


# ============================================================
# 5. FEATURE VALIDATION
# ============================================================

print_section("5. FEATURE VALIDATION")

X = features[FEATURE_COLUMNS].copy()
y = features[TARGET_COLUMN].copy()

print("Feature columns:")
print(FEATURE_COLUMNS)

print("\nMissing values per feature:")
print(X.isnull().sum())

missing_values = X.isnull().sum()

assert missing_values.sum() == 0, (
    "Feature validation failed. Missing values detected:\n"
    f"{missing_values[missing_values > 0]}"
)

assert len(X) == len(y), (
    "Feature and target row counts do not match!"
)

assert X.index.is_unique, (
    "Feature DataFrame index is not unique!"
)

assert set(y.unique()).issubset({0, 1}), (
    "Unexpected target values found!"
)

print("\nFeature missing-value validation: PASSED")
print("Cohort row validation: PASSED")


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

print_section("6. TRAIN / TEST SPLIT")

# Keep customer IDs as the index so they can be used later
# to connect predictions with the segment file.

X.index = pd.Index(
    features["customer_id"].astype(str),
    name="customer_id",
)

y.index = X.index

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

print("\nOriginal training class distribution:")
print(y_train.value_counts().sort_index())

print("\nTest class distribution:")
print(y_test.value_counts().sort_index())


# ============================================================
# 7. BASELINE MODEL
# ============================================================

print_section("7. BASELINE MODEL")

# Baseline: predict every customer as active (not lapsed).

baseline_predictions = np.zeros(
    len(y_test),
    dtype=int,
)

baseline_metrics = calculate_metrics(
    y_test,
    baseline_predictions,
)

print("Baseline: always predict active (not lapsed)")
print_metrics(baseline_metrics)


# ============================================================
# 8. PREPROCESSING PIPELINE
# ============================================================

print_section("8. PREPROCESSING")

# Numeric columns are standardised.
# Categorical columns are one-hot encoded with k-1 columns.
#
# The preprocessing steps are inside the model Pipeline.
# GridSearchCV therefore fits them only on each training fold.
# The untouched test set is never used to fit an encoder or scaler.

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            NUMERIC_FEATURES,
        ),
        (
            "categorical",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore",
            ),
            CATEGORICAL_FEATURES,
        ),
    ],
    remainder="drop",
)


# ============================================================
# 9. LOGISTIC REGRESSION GRID SEARCH
# ============================================================

print_section("9. LOGISTIC REGRESSION GRID SEARCH")

lr_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

lr_grid = GridSearchCV(
    estimator=lr_pipeline,
    param_grid={
        "classifier__C": [0.01, 0.1, 1, 10, 100],
    },
    scoring="f1",
    cv=CV_FOLDS,
    n_jobs=-1,
    refit=True,
    verbose=1,
)

# IMPORTANT:
# Grid search uses the original, imbalanced training rows.
# No oversampling is performed before cross-validation.

lr_grid.fit(X_train, y_train)

print("\nLogistic Regression best parameters:")
print(lr_grid.best_params_)

print(
    "Logistic Regression best CV F1:",
    f"{lr_grid.best_score_:.4f}",
)


# ============================================================
# 10. RANDOM FOREST GRID SEARCH
# ============================================================

print_section("10. RANDOM FOREST GRID SEARCH")

rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

rf_grid = GridSearchCV(
    estimator=rf_pipeline,
    param_grid={
        "classifier__max_depth": [3, 5, 8],
        "classifier__min_samples_leaf": [5, 10, 20],
    },
    scoring="f1",
    cv=CV_FOLDS,
    n_jobs=-1,
    refit=True,
    verbose=1,
)

rf_grid.fit(X_train, y_train)

print("\nRandom Forest best parameters:")
print(rf_grid.best_params_)

print(
    "Random Forest best CV F1:",
    f"{rf_grid.best_score_:.4f}",
)


# ============================================================
# 11. TEST SET EVALUATION
# ============================================================

print_section("11. TEST SET EVALUATION")

models = {
    "Logistic Regression": lr_grid.best_estimator_,
    "Random Forest": rf_grid.best_estimator_,
}

evaluation_results = []
test_predictions = {}

for model_name, model in models.items():

    predictions = model.predict(X_test)
    test_predictions[model_name] = predictions

    metrics = calculate_metrics(
        y_test,
        predictions,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    evaluation_results.append({
        "model": model_name,
        **metrics,
    })

    print(f"\n{model_name}")
    print("Confusion matrix:")
    print(matrix)
    print_metrics(metrics)

evaluation_df = pd.DataFrame(evaluation_results)

evaluation_df.to_csv(
    EVALUATION_FILE,
    index=False,
)


# ============================================================
# 12. MODEL SELECTION
# ============================================================

print_section("12. MODEL SELECTION")

# Select using cross-validation F1, not the test-set score.
# This avoids selecting a model simply because it performed
# better on the one held-out test sample.

cv_scores = {
    "Logistic Regression": lr_grid.best_score_,
    "Random Forest": rf_grid.best_score_,
}

selected_model_name = max(
    cv_scores,
    key=cv_scores.get,
)

selected_model = models[selected_model_name]

selected_test_metrics = calculate_metrics(
    y_test,
    test_predictions[selected_model_name],
)

print("Selected model:", selected_model_name)
print(
    "Selected model CV F1:",
    f"{cv_scores[selected_model_name]:.4f}",
)
print(
    "Selected model test F1:",
    f"{selected_test_metrics['f1']:.4f}",
)

assert selected_test_metrics["f1"] >= 0.55, (
    "Selected model did not meet the required test F1 of 0.55."
)

print(
    "\nSelection rationale: the selected model has the higher "
    "cross-validation F1. Missing a genuinely lapsing customer "
    "can cost future revenue, while wrongly targeting a loyal "
    "customer wastes retention incentives. The retention team "
    "should consider this precision-recall trade-off when "
    "choosing campaign thresholds."
)


# ============================================================
# 13. CLASS-BALANCE EXPERIMENT
# ============================================================

print_section("13. CLASS-BALANCE EXPERIMENT")

# Oversample only the original training rows.
# The test set remains unchanged.
#
# Keep every original training row and add randomly selected
# duplicates of the minority class until both classes match.

training_data = X_train.copy()
training_data[TARGET_COLUMN] = y_train

class_counts = y_train.value_counts()

majority_count = int(class_counts.max())

balanced_parts = []

for class_label in sorted(class_counts.index):

    class_rows = training_data.loc[
        training_data[TARGET_COLUMN] == class_label
    ]

    if len(class_rows) < majority_count:

        additional_rows = class_rows.sample(
            n=majority_count - len(class_rows),
            replace=True,
            random_state=RANDOM_STATE,
        )

        balanced_class_rows = pd.concat(
            [class_rows, additional_rows],
            axis=0,
        )

    else:
        balanced_class_rows = class_rows.copy()

    balanced_parts.append(balanced_class_rows)

balanced_training_data = pd.concat(
    balanced_parts,
    axis=0,
).sample(
    frac=1,
    random_state=RANDOM_STATE,
)

X_train_balanced = balanced_training_data[
    FEATURE_COLUMNS
].copy()

y_train_balanced = balanced_training_data[
    TARGET_COLUMN
].copy()

print("Original training class counts:")
print(y_train.value_counts().sort_index())

print("\nBalanced training class counts:")
print(y_train_balanced.value_counts().sort_index())

# Refit the best Logistic Regression configuration
# on the balanced training data.
#
# The Pipeline fits a fresh scaler and encoder on this
# balanced training data only.

balanced_lr_model = clone(lr_grid.best_estimator_)

balanced_lr_model.fit(
    X_train_balanced,
    y_train_balanced,
)

# Both models are evaluated on the original untouched test set.

unbalanced_lr_predictions = models[
    "Logistic Regression"
].predict(X_test)

balanced_lr_predictions = balanced_lr_model.predict(X_test)

unbalanced_lr_metrics = calculate_metrics(
    y_test,
    unbalanced_lr_predictions,
)

balanced_lr_metrics = calculate_metrics(
    y_test,
    balanced_lr_predictions,
)

print("\nUnbalanced Logistic Regression:")
print_metrics(unbalanced_lr_metrics)

print("\nBalanced Logistic Regression:")
print_metrics(balanced_lr_metrics)

balance_comparison = pd.DataFrame([
    {
        "experiment": "Unbalanced Logistic Regression",
        **unbalanced_lr_metrics,
    },
    {
        "experiment": "Balanced Logistic Regression",
        **balanced_lr_metrics,
    },
])

balance_comparison.to_csv(
    BALANCE_FILE,
    index=False,
)

print("\nAbsolute metric changes:")

metric_changes = {}

for metric_name in ["precision", "recall", "f1"]:

    change = abs(
        balanced_lr_metrics[metric_name]
        - unbalanced_lr_metrics[metric_name]
    )

    metric_changes[metric_name] = change

    print(f"{metric_name}: {change:.4f}")

largest_change = max(
    metric_changes,
    key=metric_changes.get,
)

print("\nLargest metric change:", largest_change)

if largest_change == "recall":
    print(
        "Interpretation: balancing makes the model identify "
        "more genuinely lapsing customers, but it can also "
        "increase false-positive retention targeting."
    )


# ============================================================
# 14. TRAINING SCALER MEAN CHECK
# ============================================================

print_section("14. TRAINING SCALER MEAN CHECK")

# The selected model's scaler is fitted on the original
# training rows through GridSearchCV's refitted best pipeline.

selected_preprocessor = (
    selected_model.named_steps["preprocessor"]
)

selected_scaler = (
    selected_preprocessor.named_transformers_["numeric"]
)

frequency_position = NUMERIC_FEATURES.index("frequency")

training_frequency_mean = selected_scaler.mean_[
    frequency_position
]

full_cohort_frequency_mean = X["frequency"].mean()

print(
    "Training scaler mean for frequency:",
    f"{training_frequency_mean:.4f}",
)

print(
    "Full-cohort mean for frequency:",
    f"{full_cohort_frequency_mean:.4f}",
)

print(
    "These means differ because the scaler is fitted using "
    "training rows only, whereas the full-cohort mean includes "
    "both training and test customers."
)


# ============================================================
# 15. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

print_section("15. RANDOM FOREST FEATURE IMPORTANCE")

best_rf_pipeline = rf_grid.best_estimator_

rf_preprocessor = best_rf_pipeline.named_steps["preprocessor"]
rf_classifier = best_rf_pipeline.named_steps["classifier"]

transformed_feature_names = (
    rf_preprocessor.get_feature_names_out()
)

feature_importances = pd.DataFrame({
    "feature": transformed_feature_names,
    "importance": rf_classifier.feature_importances_,
})

feature_importances = feature_importances.sort_values(
    "importance",
    ascending=False,
)

print("\nTop five Random Forest features:")
print(
    feature_importances.head(5).to_string(index=False)
)


# ============================================================
# 16. LOGISTIC REGRESSION COEFFICIENTS
# ============================================================

print_section("16. LOGISTIC REGRESSION COEFFICIENTS")

best_lr_pipeline = lr_grid.best_estimator_

lr_preprocessor = best_lr_pipeline.named_steps["preprocessor"]
lr_classifier = best_lr_pipeline.named_steps["classifier"]

lr_feature_names = lr_preprocessor.get_feature_names_out()

coefficients = pd.DataFrame({
    "feature": lr_feature_names,
    "coefficient": lr_classifier.coef_[0],
})

# Show the three strongest coefficients by absolute magnitude.

coefficients["absolute_coefficient"] = (
    coefficients["coefficient"].abs()
)

strongest_coefficients = (
    coefficients.sort_values(
        "absolute_coefficient",
        ascending=False,
    )
    .head(3)
)

print("\nThree strongest Logistic Regression coefficients:")
print(
    strongest_coefficients[
        ["feature", "coefficient"]
    ].to_string(index=False)
)


# ============================================================
# 17. MODEL PERSISTENCE
# ============================================================

print_section("17. MODEL PERSISTENCE")

joblib.dump(
    selected_model,
    MODEL_FILE,
)

print("Model saved to:", MODEL_FILE)

reloaded_model = joblib.load(MODEL_FILE)

selected_predictions = selected_model.predict(X_test)
reloaded_predictions = reloaded_model.predict(X_test)

predictions_match = np.array_equal(
    selected_predictions,
    reloaded_predictions,
)

print(
    "Reloaded model predictions exactly match original:",
    predictions_match,
)

assert predictions_match, (
    "Model persistence validation failed!"
)

print("Model persistence validation: PASSED")


# ============================================================
# 18. PREMIUM RETENTION LIST
# ============================================================

print_section("18. PREMIUM RETENTION LIST")

# IMPORTANT:
# The assignment explicitly requires the Premium flag from
# the whole-window RFM segmentation in customer_segments.csv.
# This includes the September label window, so this is a
# post-hoc selection and NOT a leakage-free evaluation.

segments = pd.read_csv(SEGMENTS_FILE)

required_segment_columns = {
    "customer_id",
    "segment",
}

missing_segment_columns = (
    required_segment_columns - set(segments.columns)
)

assert not missing_segment_columns, (
    f"Missing segment columns: {missing_segment_columns}"
)

# Normalize IDs before joining the two datasets.

segments["customer_id"] = (
    segments["customer_id"].astype(str).str.strip()
)

premium_ids = set(
    segments.loc[
        segments["segment"].astype(str).str.strip().eq("Premium"),
        "customer_id",
    ]
)

assert len(premium_ids) > 0, (
    "No Premium customers found in customer_segments.csv!"
)

# Only test-set customers are eligible for this list.

test_customer_ids = X_test.index.astype(str)

premium_test_ids = [
    customer_id
    for customer_id in test_customer_ids
    if customer_id in premium_ids
]

print(
    "Premium customers in test set:",
    len(premium_test_ids),
)

# Get predicted lapse probabilities from the selected model.

selected_probabilities = reloaded_model.predict_proba(
    X_test
)

positive_class_index = list(
    reloaded_model.classes_
).index(1)

test_probabilities = selected_probabilities[
    :,
    positive_class_index,
]

probability_table = pd.DataFrame({
    "customer_id": test_customer_ids,
    "lapse_probability": test_probabilities,
    "actually_lapsed": y_test.to_numpy(),
})

retention_list = probability_table.loc[
    probability_table["customer_id"].isin(premium_test_ids)
].copy()

retention_list = retention_list.sort_values(
    "lapse_probability",
    ascending=False,
)

retention_list = retention_list.head(10)

retention_list = retention_list[
    [
        "customer_id",
        "lapse_probability",
        "actually_lapsed",
    ]
]

assert len(retention_list) > 0, (
    "No Premium customers from the test set were found "
    "for the retention list!"
)

assert retention_list["customer_id"].is_unique, (
    "Duplicate customer IDs found in retention list!"
)

retention_list.to_csv(
    RETENTION_FILE,
    index=False,
)

print("\nTop Premium retention candidates:")
print(retention_list.to_string(index=False))

actually_lapsed_count = int(
    retention_list["actually_lapsed"].sum()
)

print(
    "\nNumber of customers in the retention list "
    "who actually lapsed:",
    actually_lapsed_count,
)

print(
    "\nIMPORTANT: The retention list is illustrative and not "
    "a leakage-free evaluation because the Premium flag comes "
    "from the post-hoc whole-window RFM segmentation."
)


# ============================================================
# 19. FINAL OUTPUT SUMMARY
# ============================================================

print_section("19. OUTPUT SUMMARY")

print(f"Selected model: {selected_model_name}")
print(f"Test F1: {selected_test_metrics['f1']:.4f}")

print("\nGenerated files:")

for output_file in [
    MODEL_FILE,
    EVALUATION_FILE,
    BALANCE_FILE,
    RETENTION_FILE,
]:
    print(f"- {output_file.relative_to(BASE_DIR)}")

print("\nALL LAPSE PREDICTION CHECKS COMPLETED")
