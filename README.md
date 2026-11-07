# Customer Intelligence Support Copilot

## Overview

An offline, AI-powered customer intelligence and support automation project built using Python, SQL, Pandas, SQLite, Scikit-learn, TF-IDF, and a deterministic mock LLM.

The project demonstrates an end-to-end customer intelligence and support workflow through four major components:

1. **Customer Data Foundation:** Data cleaning, validation, database creation, and SQL-based analysis.
2. **Customer Intelligence:** RFM analysis, customer segmentation, and customer lapse prediction.
3. **Support Ticket Intelligence:** Ticket preprocessing, PII masking, classification, prioritisation, budget management, and routing.
4. **Policy RAG and Agent Automation:** Policy-based retrieval-augmented generation and a tool-using refund support agent.

The project uses a deterministic mock LLM to simulate hosted LLM behaviour without requiring API keys, internet access, or paid API usage.

### Technology Stack

| Area | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Database | SQLite |
| Data Analysis | SQL, Pandas |
| Machine Learning | Scikit-learn |
| Text Processing | Regular Expressions |
| Retrieval | TF-IDF, Cosine Similarity |
| LLM Simulation | Custom Mock LLM |
| Testing | Python scripts and evaluation datasets |
| Version Control | Git and GitHub |

### Project Structure

```text
Customer-Intelligence-Support-Copilot/
│
├── data/
│   ├── raw/
│   │   ├── customers.csv
│   │   ├── orders.csv
│   │   ├── products.csv
│   │   ├── tickets.csv
│   │   ├── policy_eval.csv
│   │   └── agent_scenarios.csv
│   ├── clean/
│   ├── policies/
│   │   ├── delivery_policy.txt
│   │   ├── escalation_policy.txt
│   │   ├── payment_policy.txt
│   │   ├── privacy_policy.txt
│   │   └── refund_policy.txt
│   └── instamart.db
│
├── outputs/
│   ├── tickets_preprocessed.csv
│   ├── triage_zero_shot.csv
│   ├── triage_few_shot.csv
│   ├── triage_budget_demo.csv
│   ├── triage_results.csv
│   ├── policy_eval_results.csv
│   ├── policy_eval_zero_overlap.csv
│   ├── agent_runs.jsonl
│   └── refund_ledger.jsonl
│
├── state/
│   └── Agent checkpoint files
│
├── generate_data.py
├── mock_llm.py
├── ticket_preprocessing.py
├── ticket_classifier.py
├── ticket_budget_routing.py
├── policy_rag.py
├── experiment_zero_overlap.py
├── diagnose_zero_overlap.py
├── agent_tools.py
├── test_agent_tools.py
├── refund_agent.py
├── inspect_agent_data.py
├── requirements.txt
└── README.md
```

---

## Reproduce

### Requirements

- Python 3.11 or later
- SQLite 3.25 or later
- Git

### Installation

**1. Clone the repository**

```bash
git clone <your-github-repository-url>
```

**2. Navigate to the project**

```bash
cd Customer-Intelligence-Support-Copilot
```

**3. Create a virtual environment**

```bash
python -m venv .venv
```

**4. Activate the environment**

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

**5. Install dependencies**

```bash
pip install -r requirements.txt
```

### Offline Execution

Run the project from its root directory. The supplied `mock_llm.py` provides deterministic offline LLM behaviour.

**Part C — Support Ticket Intelligence**

```bash
python ticket_preprocessing.py
python ticket_classifier.py
python ticket_budget_routing.py
```

**Part D1 — Policy RAG**

```bash
python policy_rag.py
```

**Part D1 — Zero-Overlap Experiment**

```bash
python experiment_zero_overlap.py
```

**Part D1 — Chunk Boundary Diagnosis**

```bash
python diagnose_zero_overlap.py
```

**Part D2 — Agent Tool Tests**

```bash
python test_agent_tools.py
```

**Part D2 — Refund Agent**

```bash
python refund_agent.py
```

The refund agent generates:

- `outputs/agent_runs.jsonl`
- `outputs/refund_ledger.jsonl`
- Checkpoint files under `state/`

The agent execution script also runs the seven scenarios, token-limit demonstration, timeout demonstration, and crash-recovery demonstration.

---

## Part A — Customer Data Foundation

**Status: Completed**

### Objectives

- Load and inspect customer, order, and product datasets.
- Clean and preprocess raw datasets.
- Handle missing values and inconsistent data.
- Validate data quality and relationships.
- Store structured data in SQLite.
- Perform SQL-based customer and business analysis.
- Use Pandas for further data exploration.

### Key Deliverables

- Cleaned customer, order, and product datasets.
- SQLite database: `data/instamart.db`.
- Data validation and analysis.
- SQL and Pandas-based insights.

---

## Part B — Customer Intelligence

**Status: Completed**

### Objectives

Develop customer intelligence using transactional data.

### 1. RFM Analysis

Customer behaviour is analysed using:

- **Recency:** How recently the customer purchased.
- **Frequency:** How often the customer purchased.
- **Monetary:** How much the customer spent.

RFM analysis was performed using a snapshot date of 2025-10-01.

| Metric | Result |
|---|---:|
| Customers with completed orders | 1,175 |
| Average recency | 33.56 days |
| Average frequency | 9.46 orders |
| Average monetary value | INR 2,552.06 |

### 2. Customer Segmentation

K-Means clustering was used to group customers based on purchasing behaviour.

| Metric | Result |
|---|---:|
| Number of clusters | 4 |
| Silhouette score | 0.526 |
| PCA explained variance | 85.2% |

Customer segments:

| Segment | Customers |
|---|---:|
| Regular | 436 |
| Loyal | 419 |
| At Risk | 182 |
| Premium | 138 |

### 3. Customer Lapse Prediction

A customer lapse prediction workflow was developed using Logistic Regression and Random Forest.

| Metric | Result |
|---|---:|
| Lapse prediction cohort | 1,156 |
| Lapsed customers | 443 |
| Active customers | 713 |
| Training records | 924 |
| Testing records | 232 |

The training workflow included oversampling to address class imbalance.

Model comparison:

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 70.69% | 68.10% |
| Lapsed-class F1 score | 0.6909 | 0.6606 |

Random Forest was selected for the lapse prediction workflow.

### Key Deliverables

- RFM metrics.
- Customer segments.
- Lapse prediction models.
- Customer intelligence outputs.

---

## Part C — Support Ticket Intelligence

**Status: Completed**

### Objectives

Build an automated support-ticket processing and classification pipeline.

### 1. Ticket Preprocessing

Implemented in `ticket_preprocessing.py`.

Key operations:

- Load support tickets.
- Clean and normalise ticket text.
- Detect and redact personally identifiable information (PII).
- Handle empty tickets.
- Identify unusually long tickets.
- Prepare tickets for classification.

PII categories handled:

- Email addresses.
- Phone numbers.
- Order IDs.

### 2. Ticket Classification

Implemented in `ticket_classifier.py`.

Two classification approaches were evaluated:

- Zero-shot classification.
- Few-shot classification.

The classifier uses the deterministic mock LLM to generate structured classification responses.

### 3. Budget-Aware Classification and Routing

Implemented in `ticket_budget_routing.py`.

The pipeline includes:

- Token usage tracking.
- Budget checks.
- Retry handling for transient failures.
- Fallback classification.
- Priority assignment.
- Department-based ticket routing.

### Evaluation Results

**Dataset:** 164 support tickets.

#### Preprocessing

| Metric | Result |
|---|---:|
| Total tickets | 164 |
| Eligible tickets | 156 |
| Empty tickets | 4 |
| Long tickets | 4 |
| Emails redacted | 55 |
| Phone numbers redacted | 91 |
| Order IDs redacted | 132 |
| Total PII instances redacted | 278 |

No remaining PII was detected in the processed ticket text.

#### Classification Comparison

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Classified tickets | 147 | 147 |
| Correct classifications | 126 | 147 |
| Accuracy | 85.7% | 100% |
| Tokens used | 12,832 | 21,568 |
| Average tokens per classified ticket | 87.29 | 146.72 |

#### Additional Results

- 19 transient failures were recovered in each classification run.
- Final processing statuses:

| Status | Tickets |
|---|---:|
| JSON | 135 |
| Keyword fallback | 12 |
| Invalid | 9 |
| Skipped | 4 |
| Long | 4 |

#### Budget Demonstration

A zero-shot budget demonstration used a token budget of 6,000.

- Tokens consumed: 6,018.
- Tickets marked as budget exhausted: 83.

The final few-shot run used a budget of 100,000 tokens and consumed 21,568 tokens.

The budget check occurs before an LLM call, so the final reported usage can exceed the configured budget by the tokens consumed by the last permitted call.

#### Final Ticket Routing

| Route | Priority | Tickets |
|---|---|---:|
| human_review | NORMAL | 17 |
| premium_desk | HIGH | 11 |
| security_team | HIGH | 34 |
| support_queue | NORMAL | 102 |

### Part C Output Files

- `outputs/tickets_preprocessed.csv`
- `outputs/triage_zero_shot.csv`
- `outputs/triage_few_shot.csv`
- `outputs/triage_budget_demo.csv`
- `outputs/triage_results.csv`

---

## Part D — Policy RAG and Agent Automation

**Status: Completed**

### Part D1 — Policy Retrieval-Augmented Generation (RAG)

### Objective

Build a policy question-answering system that retrieves relevant information from company policy documents and uses the retrieved content to generate answers.

#### 1. Policy Documents

Five policy documents were used:

| Policy Document | Purpose |
|---|---|
| `refund_policy.txt` | Refund eligibility and approval rules |
| `delivery_policy.txt` | Delivery times, fees, and delivery conditions |
| `payment_policy.txt` | Payment methods and payment-related issues |
| `privacy_policy.txt` | Customer data handling and privacy |
| `escalation_policy.txt` | Escalation rules and support timelines |

#### 2. Text Chunking

Implemented in `policy_rag.py`.

| Parameter | Value |
|---|---:|
| Chunk size | 40 words |
| Chunk overlap | 10 words |
| Top-K retrieval | 3 |
| Policy documents | 5 |
| Total chunks | 15 |

Overlapping chunks help preserve important information when a sentence crosses a chunk boundary.

#### 3. Retrieval Implementation

The retrieval pipeline uses:

- `TfidfVectorizer` for text vectorisation.
- Cosine similarity for measuring relevance.
- Top-K retrieval to select the three most relevant chunks.
- A minimum similarity threshold of 0.05.

For the sample query `refund policy late delivery`, the top three retrieved chunks came from `refund_policy.txt`.

#### 4. Answer Generation

The retrieved policy chunks are passed to the mock LLM using a structured prompt containing:

- Policy context.
- User question.
- Policy assistant system instructions.

The assistant generates an answer using the retrieved policy information and provides a fallback response when relevant information cannot be found.

#### 5. Policy RAG Evaluation

Evaluation was performed using `data/raw/policy_eval.csv`.

| Metric | Result |
|---|---:|
| Total evaluation questions | 14 |
| Retrieval accuracy | 12/12 |
| Factual answer accuracy | 12/12 |
| Out-of-scope fallback accuracy | 2/2 |

The system successfully retrieved relevant policy information and generated answers matching all expected facts in the in-scope evaluation questions.

#### 6. Chunk Overlap Experiment

Implemented in:

- `experiment_zero_overlap.py`
- `diagnose_zero_overlap.py`

The same policy documents and evaluation dataset were tested using zero overlap.

**Experimental comparison**

| Metric | 10-word overlap | Zero overlap |
|---|---:|---:|
| Chunk size | 40 words | 40 words |
| Total chunks | 15 | 13 |
| Retrieval accuracy | 12/12 | 12/12 |
| Factual answer accuracy | 12/12 | 9/12 |
| Out-of-scope fallback accuracy | 2/2 | 2/2 |

The zero-overlap experiment produced three factual answer failures.

Investigation identified sentences split across chunk boundaries:

1. Late-delivery refund eligibility:
   - One chunk ended before the customer refund outcome.
   - The next chunk contained the remaining sentence.

2. Refund approval above INR 1,000:
   - One chunk ended with `A refund above INR`.
   - The next chunk began with `1000 must be approved by a human support lead`.

3. Failed payment reversal:
   - One chunk ended with `returned automatically`.
   - The next chunk began with `within 48 hours`.

These breaks affected the information available to the answer-generation step.

**Conclusion**

With 10-word overlap, factual answer accuracy improved from 9/12 (75%) to 12/12 (100%).

This experiment demonstrates the importance of overlapping chunks for preserving context and improving retrieval-based answer generation.

### Part D1 Output Files

- `outputs/policy_eval_results.csv`
- `outputs/policy_eval_zero_overlap.csv`

---

### Part D2 — ReAct Refund Support Agent

**Status: Completed**

### Objective

Build a tool-using refund support agent using the ReAct (Reasoning and Acting) pattern.

The agent investigates customer refund requests, retrieves order information, searches policy documents, calculates refund eligibility, and either issues a refund or escalates the request.

### 1. Agent Tools

Implemented in `agent_tools.py`.

Five tools are defined through `TOOLS_SPEC` and mapped through `TOOL_MAP`.

| Tool | Responsibility |
|---|---|
| `get_order` | Retrieve order information from SQLite |
| `search_policy` | Search relevant company policy documents |
| `calculate_refund` | Calculate refund eligibility and amount |
| `issue_refund` | Process eligible refunds and append to the refund ledger |
| `escalate_to_human` | Escalate requests requiring human approval |

### 2. Refund Business Rules

- Only completed orders are eligible for late-delivery refunds.
- Delivery delays greater than 45 minutes qualify for a full refund.
- Delays of 45 minutes or less do not qualify.
- Refunds above INR 1,000 require human support lead approval.
- Refunds within the automatic approval limit can be issued through the refund tool.
- Refund requests must fall within the seven-day policy window.
- Cancelled orders are not eligible for a refund under the late-delivery rule.

### 3. ReAct Agent Implementation

Implemented in `refund_agent.py`.

The agent follows a tool-using execution loop:

1. Receive a customer refund request.
2. Call the mock LLM.
3. Identify the requested tool and arguments.
4. Execute the tool using `TOOL_MAP`.
5. Append the tool observation to the conversation.
6. Continue until the agent produces a final answer or reaches a safety limit.

The implementation includes:

- Tool calling through `TOOLS_SPEC` and `TOOL_MAP`.
- ReAct execution loop.
- Maximum iteration limit.
- Token usage limit.
- Execution time limit.
- Fallback handling.
- Tool observation tracking.
- JSONL execution logging.
- Checkpoint creation after every tool observation.
- Agent resumption without repeating completed tool actions.

Default execution limits:

| Parameter | Default |
|---|---:|
| Maximum iterations | 6 |
| Token limit | 6,000 |
| Time limit | 10 seconds |

### 4. Scenario Evaluation

Seven scenarios were executed using `data/raw/agent_scenarios.csv`.

| Scenario | Expected Outcome | Actual Outcome | LLM Calls |
|---|---|---|---:|
| S1 | Refund INR 128.25 | REFUND | 5 |
| S2 | Refund INR 135.00 | REFUND | 5 |
| S3 | Escalate INR 1,999 | ESCALATE | 5 |
| S4 | Refund denied | DENIED | 4 |
| S5 | Refund denied | DENIED | 3 |
| S6 | Order not found | NOT_FOUND | 2 |
| S7 | Maximum-iteration fallback | FALLBACK_MAX_ITERATIONS | 6 |

All seven scenarios produced their expected outcomes.

#### Token Usage

| Scenario | Tokens Used |
|---|---:|
| S1 | 2,183 |
| S2 | 2,183 |
| S3 | 2,170 |
| S4 | 1,626 |
| S5 | 1,097 |
| S6 | 575 |
| S7 | 4,560 |

### 5. Safety Limit Demonstrations

The agent was tested against its execution limits.

| Test | Configuration | Result |
|---|---|---|
| Token limit | `token_limit=200` | FALLBACK_TOKEN_LIMIT |
| Timeout | `time_limit_seconds=0` | FALLBACK_TIMEOUT |
| Maximum iterations | `max_iterations=6` | FALLBACK_MAX_ITERATIONS |

**Token limit test**

The agent stopped after one model call because the token budget was exceeded.

- LLM calls: 1
- Tools executed: 0
- Status: `FALLBACK_TOKEN_LIMIT`

The requested tool was not executed after the budget check failed.

**Timeout test**

- LLM calls: 0
- Status: `FALLBACK_TIMEOUT`

**Maximum iteration test**

The loop scenario continued requesting policy searches until the configured iteration limit was reached.

- LLM calls: 6
- Tokens used: 4,560
- Status: `FALLBACK_MAX_ITERATIONS`

All three tests stopped safely and returned human-review fallback responses.

### 6. Checkpoint and Crash Recovery

The agent saves its execution state after every tool observation.

Checkpoint information includes:

- Run ID.
- Scenario ID.
- Conversation messages.
- Token usage.
- Step number.
- Tools already executed.

The crash-recovery demonstration simulated a crash after the second tool execution.

The agent was then resumed using its saved checkpoint.

**Recovered execution sequence**

```text
get_order
    ↓
search_policy
    ↓
calculate_refund
    ↓
issue_refund
```

Final result:

| Metric | Result |
|---|---|
| Simulated crash point | After second tool |
| Resumed status | REFUND |
| Final refund | INR 128.25 |
| Total LLM calls | 5 |
| Repeated completed tool actions | None |

The resumed agent continued from the saved state and did not repeat the completed tool executions.

### 7. ReAct vs Plan-and-Execute

**ReAct** combines reasoning and action in an iterative loop. The agent decides which tool to use based on the current conversation and previous tool observations. This allows it to respond dynamically to retrieved order information and policy results.

**Plan-and-Execute** separates planning from execution. It first creates a sequence of actions and then carries them out. This can be useful for predictable workflows, but the plan may need to be revised when an unexpected result occurs.

For this refund workflow, ReAct provides a straightforward approach because the next action depends on the previous tool result. For example, the agent checks the order before deciding whether to calculate a refund or report that the order cannot be found.

The measured execution examples demonstrate that different scenarios require different numbers of model calls: S1 used five calls, S5 used three, and S6 used two. The loop scenario was safely stopped by the maximum-iteration limit.

### 8. Part D2 Output Files

- `outputs/agent_runs.jsonl` — scenario execution results, tool arguments, observations, and final statuses.
- `outputs/refund_ledger.jsonl` — refund and escalation records.
- `state/` — agent checkpoint files.

### 9. Part D2 Test Files

- `agent_tools.py`
- `test_agent_tools.py`
- `refund_agent.py`

---

## Limitations and Next Steps

### Current Limitations

- The mock LLM is deterministic and does not represent the full behaviour of a production language model.
- TF-IDF retrieval is based on lexical similarity and may not capture semantic relationships as effectively as embedding-based retrieval.
- The refund workflow operates on a local SQLite database and simulated refund ledger.
- The project is designed for offline development and evaluation rather than live customer transactions.
- The evaluation results are based on the supplied datasets and scenarios.

### Future Enhancements

Potential improvements include:

- Integrate a hosted LLM with appropriate security controls.
- Replace TF-IDF with semantic embeddings.
- Add a persistent vector database.
- Build a customer support dashboard.
- Add API endpoints for the copilot.
- Deploy the application.
- Add production-grade logging and monitoring.
- Integrate real payment and order-management systems with appropriate approval safeguards.

### Key Learnings

Through this project, the following concepts were implemented and explored:

- Data cleaning and validation.
- Relational database design and SQL querying.
- Customer segmentation using RFM analysis.
- Customer lapse prediction.
- PII detection and redaction.
- Zero-shot and few-shot classification.
- Token usage and budget management.
- Retry and fallback mechanisms.
- TF-IDF text vectorisation.
- Cosine similarity-based retrieval.
- Retrieval-Augmented Generation.
- Chunk size and overlap experimentation.
- Policy-grounded answer generation.
- ReAct agents and tool execution.
- Tool orchestration and execution limits.
- Checkpointing and crash recovery.

### Project Status

| Component | Status |
|---|---|
| Part A — Customer Data Foundation | Completed |
| Part B — Customer Intelligence | Completed |
| Part C — Support Ticket Intelligence | Completed |
| Part D1 — Policy RAG | Completed |
| Part D1 — Zero-Overlap Experiment | Completed |
| Part D2 — ReAct Refund Agent | Completed |

### Author

**Customer Intelligence Support Copilot**

An end-to-end learning project exploring customer analytics, machine learning, retrieval-based question answering, and AI-powered support automation.