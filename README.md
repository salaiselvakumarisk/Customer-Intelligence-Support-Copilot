# Customer Intelligence & Support Copilot

## Overview

This project builds a customer intelligence and support automation system using fictional Swiggy Instamart data.

The project covers data quality auditing, data cleaning, customer segmentation, lapse prediction, support ticket triage, policy retrieval, and a guarded refund agent.

The objective is to transform raw operational data into actionable customer insights and build reliable support automation workflows.

## Reproduce

Follow the steps below to reproduce the implemented project components from a fresh clone.

### 1. Create a Python virtual environment

```bash
python -m venv .venv
```

### 2. Activate the virtual environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Generate the raw datasets

```bash
python generate_data.py
```

### 5. Audit the raw datasets

```bash
python audit_raw.py
```

### 6. Clean the datasets

```bash
python clean_data.py
```

### 7. Build the SQLite database

```bash
python build_database.py
```

### 8. Execute SQL analysis

```bash
python run_sql.py
```

### 9. Perform pandas analysis

```bash
python pandas_analysis.py
```

### 10. Validate Part A

```bash
python validate_part_a.py
```

Expected validation result:

```text
ALL CHECKPOINTS PASSED
```

### 11. Perform RFM analysis

```bash
python rfm_analysis.py
```

### 12. Perform customer segmentation

```bash
python customer_segmentation.py
```

### 13. Train and evaluate lapse prediction models

```bash
python lapse_prediction.py
```

## Part A — Data Foundation

**Status: Completed**

Part A establishes the data foundation for the customer intelligence system.

### 1. Raw Data Quality Audit

Audited the generated datasets to identify:

- Duplicate order records.
- Inconsistent city and gender spellings.
- Missing and invalid customer ages.
- Non-numeric and missing order amounts.
- Negative and missing delivery durations.
- Pending, completed, and cancelled order statuses.
- Empty support ticket descriptions.

### 2. Data Cleaning

Applied the following cleaning operations:

- Standardised customer city names and gender values.
- Handled invalid and missing customer ages using city-wise median imputation.
- Removed exact duplicate order records.
- Retained the latest record for each order ID.
- Converted order amounts into numeric values.
- Replaced missing order amounts using product price, quantity, and discount.
- Handled negative and missing delivery durations.
- Imputed missing delivery times for completed orders using product-category medians.
- Retained cancelled orders with missing delivery durations.

### 3. SQLite Database

Created the SQLite database with three tables:

- `customers`
- `products`
- `orders`

Database location:

`data/instamart.db`

### 4. SQL Analysis

Implemented seven analytical SQL queries covering:

- Customer spending analysis.
- City-wise and category-wise revenue.
- Products priced above the average catalogue price.
- Customers without completed orders.
- Above-average customer spending.
- Customer spending rankings using window functions.
- Daily revenue and cumulative revenue.

### 5. Pandas Analysis

Performed:

- City-wise and category-wise revenue pivot analysis.
- City-level revenue, average order value, and order count aggregation.
- Order amount distribution analysis.
- Outlier detection using the IQR method.
- Correlation analysis between order amount, discount percentage, and quantity.

### 6. Part A Validation

Validation completed successfully.

- Customer records: 1,200
- Product records: 40
- Clean order records: 12,309
- Completed orders: 11,110
- Cancelled orders: 1,199
- Completed order revenue: INR 2,998,668.80
- Validation checks passed: 24/24

**Final validation output: `ALL CHECKPOINTS PASSED`**

## Part B — Customer Segmentation and Lapse Prediction

**Status: Completed**

Part B uses customer purchasing behaviour to identify customer segments and predict potential customer lapse.

### 1. RFM Analysis

Customer behaviour was analysed using Recency, Frequency, and Monetary (RFM) metrics.

- Snapshot date: 2025-10-01
- Customers with completed orders: 1,175
- Average recency: 33.56 days
- Average frequency: 9.46 orders
- Average monetary value: INR 2,552.06

Generated customer-level RFM features using completed order history.

### 2. Customer Segmentation

K-Means clustering was evaluated for K values from 2 to 8.

The selected configuration was K=4, achieving a silhouette score of 0.526.

| Segment | Customers |
|---|---:|
| Regular | 436 |
| Loyal | 419 |
| At Risk | 182 |
| Premium | 138 |
| **Total** | **1,175** |

Principal Component Analysis (PCA) was applied to reduce the RFM feature space to two dimensions.

- PC1 explained variance: 63.4%
- PC2 explained variance: 21.8%
- Total explained variance: 85.2%

### 3. Lapse Prediction

The prediction cohort included 1,156 customers.

- Lapsed customers: 443
- Active customers: 713
- Training customers: 924
- Testing customers: 232

The training dataset was balanced using minority-class oversampling.

Two machine learning models were evaluated:

| Model | Test Accuracy | Lapsed-class F1 |
|---|---:|---:|
| Logistic Regression | 0.7069 | 0.6909 |
| Random Forest | 0.6810 | 0.6606 |

Random Forest was selected based on its higher cross-validation F1 score of 0.8057, compared with 0.7823 for Logistic Regression.

The selected model achieved a test F1 score of 0.6606 for the lapsed customer class, exceeding the required threshold of 0.55.

The selected model was saved using Joblib and successfully reloaded for prediction validation.

### 4. Retention Output

The top 10 Premium customers from the test set were identified using predicted lapse probabilities to support targeted retention campaigns.

**Generated outputs:**

- `outputs/rfm_features.csv`
- `outputs/customer_segments.csv`
- `outputs/lapse_model.joblib`
- `outputs/retention_list.csv`
- `outputs/lapse_model_evaluation.csv`

**Limitation:** The retention list is an illustrative prioritisation output. Premium segment membership is derived from the full-window RFM analysis and is not an input feature in the lapse prediction model.

## Part C — Support Ticket Triage with Guardrails

**Status: Pending**

This component will implement support ticket classification and routing with guardrails.

Planned capabilities:

- Personally identifiable information (PII) redaction.
- Zero-shot and few-shot ticket classification.
- Retry handling for transient model failures.
- Invalid response handling.
- Token budget management.
- Customer-segment-aware ticket routing.
- Human review for flagged tickets.

## Part D — Policy RAG Assistant and Refund Agent

**Status: Pending**

This component will implement a policy-aware retrieval assistant and a guarded refund agent.

Planned capabilities:

- Policy document chunking and retrieval.
- TF-IDF-based document retrieval.
- Policy-grounded question answering.
- SQLite-based order lookup.
- Refund eligibility and amount calculation.
- Automatic and human approval workflows.
- Tool execution limits and safe fallback handling.
- Checkpointing and crash-resume functionality.

## Limitations and Next Steps

The current implementation covers the data foundation, customer segmentation, and lapse prediction components.

The following improvements and implementations remain:

1. Complete Part C support ticket triage and guardrails.
2. Complete Part D policy RAG assistant and refund agent.
3. Implement end-to-end validation across all four project parts.
4. Update the reproduction instructions with the remaining scripts.
5. Document final evaluation results, limitations, and future improvements.

The project uses synthetic data and an offline mock LLM environment. Results demonstrate the implemented workflows and should not be interpreted as production performance guarantees.