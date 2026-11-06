
# Customer Intelligence & Support Copilot

## Overview

This project builds a customer intelligence and support automation
system using fictional Swiggy Instamart data.

The project covers data cleaning, customer segmentation,
lapse prediction, support ticket triage, policy retrieval,
and a guarded refund agent.

## Reproduce

### 1. Create and activate the environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Generate the raw data

```bash
python generate_data.py
```

### 4. Audit raw data

```bash
python audit_raw.py
```

### 5. Clean the data

```bash
python clean_data.py
```

### 6. Build the SQLite database

```bash
python build_database.py
```

### 7. Execute SQL analysis

```bash
python run_sql.py
```

### 8. Execute pandas analysis

```bash
python pandas_analysis.py
```

### 9. Validate Part A

```bash
python validate_part_a.py
```

## Part A — Data Foundation

Status: Completed.

- Raw data quality audit
- Customer and order data cleaning
- SQLite database creation
- Seven SQL analytical queries
- Pandas aggregation and exploratory analysis
- Validation: 24/24 checks passed

## Part B — Customer Segmentation and Lapse Prediction

Status: Pending.

## Part C — Support Ticket Triage with Guardrails

Status: Pending.

## Part D — Policy RAG Assistant and Refund Agent

Status: Pending.

## Limitations and Next Steps

To be completed after all four project parts are implemented.
