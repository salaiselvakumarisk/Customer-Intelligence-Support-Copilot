# Customer Intelligence Support Copilot

An offline customer analytics and support-automation project built with Python, SQL, Pandas, NumPy, scikit-learn, TF-IDF, and a deterministic mock LLM.

The project follows a customer-support workflow from raw transactional data to customer intelligence, ticket triage, policy retrieval, and controlled refund-agent execution. It is designed to be reproducible locally: the supplied `generate_data.py` and `mock_llm.py` are retained unchanged, and the mock LLM avoids API keys, hosted-model dependencies, and paid inference calls.

## Overview

Customer-support operations sit across several kinds of information: customer profiles, orders, support tickets, and business policies. This project brings those pieces together in four connected parts:

| Part | Focus | What it covers |
|---|---|---|
| **A — Customer Data Foundation** | Reliable data | Data-quality auditing, cleaning, SQLite, SQL analysis, Pandas analysis, and validation |
| **B — Customer Intelligence** | Customer behaviour | RFM analysis, K-Means segmentation, and customer-lapse prediction |
| **C — Support Ticket Intelligence** | Ticket operations | Text preprocessing, PII redaction, zero-shot and few-shot classification, budget controls, and routing |
| **D — Policy RAG and Agent Automation** | Policy-grounded support | TF-IDF retrieval, chunk-overlap evaluation, and a tool-using refund agent with checkpoints and recovery |

The implementation is intentionally offline. The mock LLM makes the workflow deterministic and suitable for repeatable experiments, but it is not a substitute for evaluating a production language model.

### Technology

| Area | Tools |
|---|---|
| Language | Python 3.11+ |
| Data work | Pandas, NumPy |
| Database and querying | SQLite, SQL |
| Machine learning | scikit-learn |
| Model persistence | Joblib |
| Text processing | Regular expressions, TF-IDF |
| Retrieval | Cosine similarity |
| LLM interface | Supplied deterministic mock LLM |
| Version control | Git |

## Project structure

```text
Customer-Intelligence-Support-Copilot/
|
|-- generate_data.py
|-- mock_llm.py
|-- requirements.txt
|-- .gitignore
|-- README.md
|
|-- audit_raw.py
|-- clean_data.py
|-- build_database.py
|-- run_sql.py
|-- validate_part_a.py
|-- pandas_analysis.py
|
|-- rfm_analysis.py
|-- customer_segmentation.py
|-- lapse_prediction.py
|
|-- ticket_preprocessing.py
|-- ticket_classifier.py
|-- ticket_budget_routing.py
|
|-- policy_rag.py
|-- experiment_zero_overlap.py
|-- diagnose_zero_overlap.py
|-- agent_tools.py
|-- test_agent_tools.py
|-- inspect_agent_data.py
|-- refund_agent.py
|
|-- data/
|   |-- raw/
|   |-- clean/
|   |-- policies/
|   |-- instamart.db
|
|-- sql/
|   |-- queries.sql
|
|-- outputs/
|   |-- city_statistics.csv
|   |-- revenue_pivot.csv
|   |-- rfm_features.csv
|   |-- customer_segments.csv
|   |-- kmeans_evaluation.csv
|   |-- lapse_model.joblib
|   |-- lapse_model_evaluation.csv
|   |-- lapse_balance_comparison.csv
|   |-- retention_list.csv
|   |-- tickets_preprocessed.csv
|   |-- triage_zero_shot.csv
|   |-- triage_few_shot.csv
|   |-- triage_budget_demo.csv
|   |-- triage_results.csv
|   |-- policy_eval_results.csv
|   |-- policy_eval_zero_overlap.csv
|   |-- agent_runs.jsonl
|   |-- refund_ledger.jsonl
|
|-- state/
    |-- Runtime agent checkpoints (created during execution)
```

`data/` contains the generated raw and cleaned datasets, policy documents, and SQLite database. `sql/` contains the analytical queries. `outputs/` stores generated results, evaluation files, model artifacts, and agent logs. `state/` is used for runtime checkpoints.

The tree shows the main project files and representative generated outputs; some files are created or refreshed when the workflow runs.

## Reproduce

Run commands from the repository root. The workflow uses a local Python environment and SQLite; no hosted LLM credentials are required.

### Prerequisites

- Python 3.11 or later
- SQLite 3.25 or later
- Git
- PowerShell on Windows, or an equivalent terminal

### 1. Clone the repository

```bash
git clone https://github.com/salaiselvakumarisk/Customer-Intelligence-Support-Copilot.git
cd Customer-Intelligence-Support-Copilot
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Generate the data

```bash
python generate_data.py
```

### 5. Run Part A — Customer Data Foundation

```bash
python audit_raw.py
python clean_data.py
python build_database.py
python run_sql.py
python validate_part_a.py
python pandas_analysis.py
```

### 6. Run Part B — Customer Intelligence

```bash
python rfm_analysis.py
python customer_segmentation.py
python lapse_prediction.py
```

### 7. Run Part C — Support Ticket Intelligence

```bash
python ticket_preprocessing.py
python ticket_classifier.py
python ticket_budget_routing.py
```

### 8. Run Part D — Policy RAG and Agent Automation

```bash
python policy_rag.py
python experiment_zero_overlap.py
python diagnose_zero_overlap.py
python test_agent_tools.py
python refund_agent.py
```

`generate_data.py` and `mock_llm.py` are supplied project components and are retained unchanged. Running the scripts may regenerate files in `data/` and `outputs/`, and agent execution may update the runtime state and audit logs.

## Part A — Customer Data Foundation

**Result: 24 of 24 validation checks passed.**

Part A establishes the data layer used by the later analysis. The workflow audits the generated source files, cleans customer/order/product records, builds a SQLite database, runs analytical SQL, and checks that key totals reconcile.

### Data preparation

Implemented in `audit_raw.py` and `clean_data.py`.

The cleaning workflow includes:

- Normalising city and gender values.
- Checking customer age values and handling invalid or missing information.
- Parsing order timestamps and removing duplicate order records.
- Validating quantities and order amounts.
- Deriving missing order amounts from product price, quantity, and discount information where possible.
- Handling missing delivery durations using product-category information for completed orders.
- Preserving missing delivery duration for cancelled orders, where a completed delivery duration does not exist.

### SQLite database

Implemented in `build_database.py`.

Database: `data/instamart.db`

| Table | Rows |
|---|---:|
| `customers` | 1,200 |
| `products` | 40 |
| `orders` | 12,309 |

### SQL analysis

Queries are maintained in `sql/queries.sql` and executed by `run_sql.py`.

Seven queries cover:

1. Completed orders above INR 500 from customers whose completed-order spend exceeds INR 8,000.
2. Revenue by city and product category.
3. Products priced above the average catalogue price.
4. Customers with no completed orders.
5. Customers whose completed-order spending exceeds average customer spending.
6. Customer ranking within each city using a window function.
7. Daily completed revenue and cumulative revenue using a window function.

| Query | Analysis | Rows returned |
|---|---|---:|
| Q1 | High-value order analysis | 30 |
| Q2 | Revenue by city and category | 30 |
| Q3 | Products above average catalogue price | 11 |
| Q4 | Customers without completed orders | 25 |
| Q5 | Customers above average completed spend | 550 |
| Q6 | Customer ranking by city | 1,175 |
| Q7 | Daily and cumulative revenue | 183 |

### Pandas analysis and validation

Pandas analysis adds city-level customer/revenue statistics and a revenue pivot by city and category. These results are reconciled against the SQL outputs.

The validation script checks row counts, identifiers, missing-value rules, age and order-value constraints, referential integrity, SQLite table counts, completed/cancelled order counts, query outputs, revenue totals, window-function results, and Pandas aggregations.

Key data totals:

| Measure | Result |
|---|---:|
| Customers | 1,200 |
| Products | 40 |
| Orders | 12,309 |
| Completed orders | 11,110 |
| Cancelled orders | 1,199 |
| Completed-order revenue | INR 2,998,668.80 |

## Part B — Customer Intelligence

Part B uses completed-order history to describe purchasing behaviour, group customers, and estimate which customers may lapse during a defined future period.

### RFM analysis

Implemented in `rfm_analysis.py`.

RFM summarises customer behaviour using:

- **Recency:** days since the customer's most recent completed order.
- **Frequency:** number of completed orders.
- **Monetary:** total completed-order spend.

The snapshot date is **2025-10-01**.

| Measure | Result |
|---|---:|
| Customers with completed-order history | 1,175 |
| Average recency | 33.56 days |
| Average frequency | 9.46 orders |
| Average monetary value | INR 2,552.06 |

### Customer segmentation

Implemented in `customer_segmentation.py`.

K-Means was used to group customers based on purchasing behaviour. PCA was used to inspect how much variance was represented by the reduced feature space.

| Measure | Result |
|---|---:|
| Selected clusters | 4 |
| Silhouette score | 0.526 |
| PCA explained variance | 85.2% |

| Segment | Customers |
|---|---:|
| Regular | 436 |
| Loyal | 419 |
| At Risk | 182 |
| Premium | 138 |

The Premium segment is the cluster with the highest mean monetary value.

### Customer-lapse prediction

Implemented in `lapse_prediction.py`.

The task is to identify customers who did not place a completed order during September 2025. Feature engineering uses a cutoff of **2025-09-01**; September completed-order activity defines the target.

The cohort contains customers with at least one completed order before the cutoff. The model uses 11 customer-level features, including historical recency, frequency and monetary value, discount and delivery measures, cancellation rate, tenure, and customer attributes.

| Cohort measure | Result |
|---|---:|
| Customers | 1,156 |
| Lapsed | 443 |
| Active | 713 |
| Lapsed share | 38.32% |
| Training rows | 924 |
| Test rows | 232 |

The data was split into stratified training and test sets. Numerical and categorical preprocessing was kept in scikit-learn pipelines, and hyperparameters were selected with five-fold cross-validation using F1.

| Model | Search space | Selected parameters | CV F1 |
|---|---|---|---:|
| Logistic Regression | `C`: 0.01, 0.1, 1, 10, 100 | `C=100` | 0.6506 |
| Random Forest | `max_depth`: 3, 5, 8; `min_samples_leaf`: 5, 10, 20 | `max_depth=8`, `min_samples_leaf=10` | 0.6866 |

Test-set results:

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 72.84% | 72.41% |
| Precision | 64.44% | 64.04% |
| Recall | 65.17% | 64.04% |
| Lapsed-class F1 | 0.6480 | 0.6404 |

Random Forest was selected because it had the higher cross-validation F1 (0.6866). Its held-out test F1 was 0.6404, above the project's 0.55 threshold.

### Class-balance experiment

A separate Logistic Regression experiment compared the original training class distribution with a balanced training sample. Oversampling was applied to training data only; the test set remained unchanged.

| Metric | Original distribution | Balanced training |
|---|---:|---:|
| Accuracy | 72.84% | 71.55% |
| Precision | 64.44% | 58.91% |
| Recall | 65.17% | 85.39% |
| F1 | 0.6480 | 0.6972 |

Balancing improved recall and F1 in this experiment, while precision decreased. In practical terms, the model identified more lapsed customers but would also flag more customers who were not actually lapsed. This is a trade-off to consider when designing a retention campaign.

This balanced comparison is an experiment on the training sample, not a separate leakage-free cross-validation estimate.

### Interpretation, persistence, and retention output

The leading Random Forest feature importances were:

| Feature | Importance |
|---|---:|
| Monetary | 0.277857 |
| Frequency | 0.276777 |
| Recency | 0.111264 |
| Average delivery time | 0.061677 |
| Average discount | 0.053177 |

The selected model is saved with Joblib and reloaded. The reloaded model's predictions matched the original predictions.

The script exports up to 10 Premium customers from the test set, ranked by predicted lapse probability. The current run produced eight candidates. The output includes customer ID, predicted lapse probability, and actual lapse outcome.

**Interpretation caveat:** the Premium label comes from the separate whole-window RFM segmentation, which includes the September target period. The retention list is therefore illustrative; it should not be treated as a leakage-free prospective targeting evaluation.

## Part C — Support Ticket Intelligence

Part C prepares ticket text for classification, compares zero-shot and few-shot approaches, and applies budget-aware processing and routing.

### Ticket preprocessing and privacy

Implemented in `ticket_preprocessing.py`.

The preprocessing stage cleans ticket text, detects and redacts email addresses, phone numbers, and order IDs, and separates empty or unusually long tickets from the eligible classification set.

| Measure | Result |
|---|---:|
| Total tickets | 164 |
| Eligible tickets | 156 |
| Empty tickets | 4 |
| Long tickets | 4 |
| Email addresses redacted | 55 |
| Phone numbers redacted | 91 |
| Order IDs redacted | 132 |
| Total PII instances redacted | 278 |

No remaining PII was detected by the project's post-redaction check.

### Classification comparison

Implemented in `ticket_classifier.py`.

The deterministic mock LLM was evaluated in zero-shot and few-shot modes.

| Measure | Zero-shot | Few-shot |
|---|---:|---:|
| Classified tickets | 147 | 147 |
| Correct classifications | 126 | 147 |
| Accuracy | 85.7% | 100% |
| Tokens used | 12,832 | 21,568 |
| Average tokens per classified ticket | 87.29 | 146.72 |

Few-shot classification performed better on the supplied evaluation set, with higher token consumption. The result is specific to this dataset and deterministic mock implementation; it should not be read as a guarantee of performance on unseen production tickets.

### Budget controls, retries, and routing

Implemented in `ticket_budget_routing.py`.

The pipeline tracks token use, checks the configured budget before calls, retries transient failures, falls back when necessary, assigns priority, and routes tickets to a support destination.

Nineteen transient failures were recovered in each classification run.

| Processing status | Tickets |
|---|---:|
| JSON | 135 |
| Keyword fallback | 12 |
| Invalid | 9 |
| Skipped (empty) | 4 |
| Long | 4 |

The zero-shot budget demonstration used a configured budget of 6,000 tokens and consumed 6,018 tokens; 83 tickets were marked budget-exhausted. The final few-shot run used a budget of 100,000 tokens and consumed 21,568 tokens.

The zero-shot run can finish slightly above its configured budget because the budget is checked before an LLM call; the tokens used by the last permitted call are counted afterward.

Final routing distribution:

| Route | Priority | Tickets |
|---|---|---:|
| `human_review` | NORMAL | 17 |
| `premium_desk` | HIGH | 11 |
| `security_team` | HIGH | 34 |
| `support_queue` | NORMAL | 102 |

## Part D — Policy RAG and Agent Automation

Part D contains two related components: a policy question-answering pipeline and a refund agent that uses explicit tools to carry out or escalate requests.

### Policy retrieval-augmented generation

Implemented in `policy_rag.py`.

The system searches policy documents for relevant passages and supplies the retrieved context to the mock LLM. The retrieval stage uses `TfidfVectorizer`, cosine similarity, top-K selection, and a minimum similarity threshold.

The source documents are:

| Document | Subject |
|---|---|
| `refund_policy.txt` | Refund eligibility and approval |
| `delivery_policy.txt` | Delivery times, fees, and conditions |
| `payment_policy.txt` | Payment methods and payment issues |
| `privacy_policy.txt` | Customer information and privacy |
| `escalation_policy.txt` | Escalation rules and support timelines |

Retrieval configuration:

| Setting | Value |
|---|---:|
| Policy documents | 5 |
| Chunk size | 40 words |
| Chunk overlap | 10 words |
| Chunks created | 15 |
| Top-K | 3 |
| Minimum similarity | 0.05 |

The retrieved passages, question, and system instructions are assembled into the prompt. If the retrieved content does not meet the relevance threshold, the system uses a fallback response.

Evaluation on `data/raw/policy_eval.csv`:

| Measure | Result |
|---|---:|
| Evaluation questions | 14 |
| In-scope retrieval accuracy | 12/12 |
| In-scope factual answer accuracy | 12/12 |
| Out-of-scope fallback accuracy | 2/2 |

### Chunk-overlap experiment

Implemented in `experiment_zero_overlap.py` and `diagnose_zero_overlap.py`.

The experiment compared the standard 10-word overlap with a zero-overlap configuration, keeping the 40-word chunk size.

| Measure | 10-word overlap | Zero overlap |
|---|---:|---:|
| Chunks | 15 | 13 |
| Retrieval accuracy | 12/12 | 12/12 |
| Factual answer accuracy | 12/12 | 9/12 |
| Out-of-scope fallback accuracy | 2/2 | 2/2 |

The zero-overlap run produced three factual-answer failures. The diagnosis traced them to policy sentences split across chunk boundaries, including the late-delivery refund outcome, the human-approval threshold for refunds above INR 1,000, and the failed-payment reversal timeline.

On this evaluation set, 10-word overlap raised factual answer accuracy from 9/12 (75%) to 12/12 (100%). This is evidence for the value of overlap in this particular corpus and test set, rather than a universal guarantee.

### ReAct refund support agent

Implemented in `agent_tools.py` and `refund_agent.py`.

The agent follows a ReAct-style loop: it interprets a request, selects a tool, observes the result, and continues until it can return an outcome or must stop safely.

Five tools are registered:

| Tool | Responsibility |
|---|---|
| `get_order` | Retrieve order information |
| `search_policy` | Retrieve relevant policy information |
| `calculate_refund` | Check eligibility and calculate the amount |
| `issue_refund` | Issue an eligible refund |
| `escalate_to_human` | Send cases requiring approval for human review |

The implemented refund rules include:

- Only completed orders can qualify for late-delivery refunds.
- Delivery delays greater than 45 minutes qualify for a full refund.
- Delays of 45 minutes or less do not qualify.
- Refunds above INR 1,000 require human support lead approval.
- Refund requests must fall within the seven-day policy window.
- Cancelled orders are not eligible for a late-delivery refund.

The agent also applies execution guardrails:

| Guardrail | Limit |
|---|---:|
| Maximum iterations | 6 |
| Token limit | 6,000 |
| Execution time limit | 10 seconds |

It records tool execution, creates checkpoints, and can resume from a saved checkpoint without repeating completed tool actions.

Scenario results:

| Scenario | Outcome |
|---|---|
| S1 | REFUND — INR 128.25 |
| S2 | REFUND — INR 135.00 |
| S3 | ESCALATE — INR 1,999 |
| S4 | DENIED |
| S5 | DENIED |
| S6 | NOT_FOUND |
| S7 | FALLBACK_MAX_ITERATIONS |

Separate stop-safely demonstrations returned `FALLBACK_TOKEN_LIMIT` and `FALLBACK_TIMEOUT`.

A crash-and-resume test interrupted the agent after its second tool execution. After resumption, the workflow completed a refund of INR 128.25. The recorded sequence showed `get_order`, `search_policy`, `calculate_refund`, and `issue_refund` each executing exactly once in that recovery run.

## Limitations and Next Steps

### Limitations

- **Synthetic data:** The customer, order, product, ticket, and policy data are generated for this project. Results may not reflect real customer behaviour.
- **Mock LLM:** The deterministic mock provides repeatable behaviour for testing. Its results do not establish the quality of a hosted or production LLM.
- **Lapse prediction:** The model is evaluated on one historical dataset and one prediction window. Performance may change with different customer populations or time periods.
- **Retention-list caveat:** The Premium segment label uses whole-window RFM information, including the target period. The resulting list is illustrative, not a leakage-free prospective targeting evaluation.
- **Ticket classification:** The reported few-shot accuracy is measured on the supplied evaluation data and should not be assumed to generalise to unseen tickets.
- **Policy retrieval:** TF-IDF is lexical retrieval, not semantic understanding. Results depend on document wording, chunking, overlap, and the similarity threshold.
- **Refund execution:** The workflow runs in a controlled offline environment. It is not connected to real payment systems, production order services, or live approval infrastructure.
- **Production readiness:** PII redaction and execution guardrails are demonstrated, but authentication, authorisation, monitoring, deployment, and operational security would need further engineering.

### Possible next steps

- Evaluate with larger, more varied, and independently held-out datasets.
- Compare TF-IDF retrieval with embedding-based retrieval.
- Add a persistent vector store if the policy corpus grows.
- Expose selected workflows through authenticated API endpoints.
- Add role-based access controls and human approval integration.
- Add production monitoring, structured observability, and deployment automation.
- Connect to real systems only after security, privacy, and failure-handling requirements are designed and tested.

## Project status

| Component | Status |
|---|---|
| Part A — Customer Data Foundation | Completed |
| Part B — Customer Intelligence | Completed |
| Part C — Support Ticket Intelligence | Completed |
| Part D — Policy RAG | Completed |
| Part D — Chunk-overlap experiment | Completed |
| Part D — ReAct refund agent | Completed |

## Concepts applied

Data quality and cleaning · Relational data and SQL · Window functions · Pandas analysis · RFM analysis · K-Means and PCA · Classification · Train/test separation · Preprocessing pipelines · Cross-validation · Class imbalance · Model evaluation · PII redaction · Zero-shot and few-shot classification · Token budgeting · Retry and fallback handling · TF-IDF · Cosine similarity · Retrieval-augmented generation · Chunk-overlap evaluation · Tool-using agents · Execution guardrails · Checkpointing and recovery

## Author

**Customer Intelligence Support Copilot**

GitHub repository: [Customer-Intelligence-Support-Copilot](https://github.com/salaiselvakumarisk/Customer-Intelligence-Support-Copilot)
