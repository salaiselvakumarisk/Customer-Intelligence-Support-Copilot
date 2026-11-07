
from agent_tools import (
    get_order,
    search_policy,
    calculate_refund,
    issue_refund,
    escalate_to_human,
    TOOLS_SPEC,
    TOOL_MAP
)

import json
from pathlib import Path


print("\n========== AGENT TOOL TESTS ==========")

# 1. Existing order
print("\n1. GET ORDER")
print(get_order("ORD-100048"))

# 2. Missing order
print("\n2. MISSING ORDER")
print(get_order("ORD-999999"))

# 3. Policy search
print("\n3. SEARCH POLICY")
policy_result = search_policy("refund policy late delivery")

for item in policy_result["results"]:
    print(item["source"], item["score"])
    print(item["text"])

# 4. Refund calculation
print("\n4. CALCULATE REFUND")

print("S1:", calculate_refund(128.25, 65))
print("S2:", calculate_refund(135.00, 59))
print("S3:", calculate_refund(1999.00, 54))
print("S4:", calculate_refund(109.65, 20))

# 5. Validate automatic refund protection
print("\n5. ISSUE REFUND")

print("S1:", issue_refund("ORD-100048", 128.25))

print(
    "S3 protection:",
    issue_refund("ORD-100145", 1999.00)
)

# 6. Human escalation
print("\n6. ESCALATE TO HUMAN")

print(
    escalate_to_human("ORD-100145", 1999.00)
)

# 7. Tool registration
print("\n7. TOOL REGISTRATION")

print("Tools specified:", len(TOOLS_SPEC))
print("Tools mapped:", len(TOOL_MAP))

assert len(TOOLS_SPEC) == 5
assert len(TOOL_MAP) == 5

# 8. Ledger verification
print("\n8. REFUND LEDGER")

ledger_path = Path("outputs/refund_ledger.jsonl")

if ledger_path.exists():
    with ledger_path.open(encoding="utf-8") as file:
        for line in file:
            print(json.loads(line))

print("\n========== TESTS COMPLETED ==========")
