
import csv
import json
import time
import uuid
from pathlib import Path

from mock_llm import chat
from agent_tools import TOOLS_SPEC, TOOL_MAP


SCENARIO_FILE = Path("data/raw/agent_scenarios.csv")
OUTPUT_DIR = Path("outputs")
STATE_DIR = Path("state")

AGENT_SYSTEM_PROMPT = (
    "You are a careful and reliable customer support refund agent. "
    "Your responsibility is to investigate refund requests and make "
    "decisions strictly according to the available order information "
    "and company policies. Always verify the order before making a "
    "decision. Retrieve the relevant policy and inspect the retrieved "
    "policy information before calculating or approving a refund. "
    "Only orders with completed status are eligible for refunds. "
    "Cancelled orders must never receive refunds. For completed orders, "
    "delivery taking more than 45 minutes qualifies for a full refund "
    "of the order amount. Delivery taking 45 minutes or less does not "
    "qualify for a late delivery refund. Refunds up to INR 1000 can be "
    "issued automatically when eligible. Refunds above INR 1000 must "
    "always be escalated to a human support lead and must never be "
    "issued automatically. Never invent order information, refund "
    "amounts, policy rules, or tool results. Use the available tools "
    "in the appropriate sequence and rely on their observations. "
    "If an order cannot be found, report NOT_FOUND. If a request is "
    "not eligible, report DENIED. If human approval is required, "
    "report ESCALATE. If a refund is successfully issued, report "
    "REFUND. Keep the final answer concise and begin it with the "
    "appropriate status followed by a colon."
)


class SimulatedCrash(Exception):
    """Used only to demonstrate checkpoint recovery."""


def _json_safe(value):
    """Convert values into JSON-compatible data."""
    return json.loads(json.dumps(value, default=str))


def _save_checkpoint(state):
    """Save the complete agent state after each tool observation."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = STATE_DIR / f"{state['run_id']}.json"

    with checkpoint_path.open("w", encoding="utf-8") as file:
        json.dump(state, file, indent=2, default=str)

    return checkpoint_path


def _load_scenarios():
    """Load the seven assignment scenarios."""
    with SCENARIO_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _fallback(status, reason, state):
    """Return a safe human-handoff result."""
    return {
        "status": status,
        "final_answer": (
            f"{status}: Agent stopped safely. "
            f"Reason: {reason}. Human review required."
        ),
        "steps": state["steps"],
        "llm_calls": state["llm_calls"],
        "tokens_used": state["tokens_used"],
        "run_id": state["run_id"],
        "executed_tools": state["executed_tools"],
    }


def _get_status(final_answer):
    """Extract the decision label from the final answer."""
    first_word = final_answer.strip().split(":", 1)[0].upper()

    allowed = {"REFUND", "ESCALATE", "DENIED", "NOT_FOUND"}

    if first_word in allowed:
        return first_word

    return "DENIED"


def _run_loop(state, crash_after_tools=None):
    """Continue the ReAct loop from the current state."""
    started_at = time.monotonic()

    max_iterations = state["max_iterations"]
    token_limit = state["token_limit"]
    time_limit_seconds = state["time_limit_seconds"]

    while state["llm_calls"] < max_iterations:

        # Check timeout before making another model call.
        if time.monotonic() - started_at >= time_limit_seconds:
            result = _fallback(
                "FALLBACK_TIMEOUT",
                "Time limit reached",
                state,
            )
            state["result"] = result
            _save_checkpoint(state)
            return result

        response = chat(
            state["messages"],
            tools=TOOLS_SPEC,
        )

        state["llm_calls"] += 1

        usage = getattr(response, "usage", None)
        call_tokens = getattr(usage, "total_tokens", 0) if usage else 0

        state["tokens_used"] += call_tokens

        # Do not execute a tool if the token budget is exceeded.
        if state["tokens_used"] > token_limit:
            result = _fallback(
                "FALLBACK_TOKEN_LIMIT",
                "Token budget exceeded",
                state,
            )
            state["result"] = result
            _save_checkpoint(state)
            return result

        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None)

        # No tool calls means the model has produced its final answer.
        if not tool_calls:
            final_answer = message.content or "DENIED: No final answer provided."

            result = {
                "status": _get_status(final_answer),
                "final_answer": final_answer,
                "steps": state["steps"],
                "llm_calls": state["llm_calls"],
                "tokens_used": state["tokens_used"],
                "run_id": state["run_id"],
                "executed_tools": state["executed_tools"],
            }

            state["result"] = result
            _save_checkpoint(state)
            return result

        assistant_tool_calls = []

        for call in tool_calls:
            arguments = call.arguments

            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            assistant_tool_calls.append({
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(arguments),
                },
            })

        state["messages"].append({
            "role": "assistant",
            "content": None,
            "tool_calls": assistant_tool_calls,
        })

        for call in tool_calls:
            tool_name = call.name
            arguments = call.arguments

            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            if tool_name not in TOOL_MAP:
                observation = {"error": f"Unknown tool: {tool_name}"}
            else:
                try:
                    observation = TOOL_MAP[tool_name](**arguments)
                except Exception as error:
                    observation = {
                        "error": f"Tool execution failed: {str(error)}"
                    }

            observation = _json_safe(observation)

            state["steps"].append({
                "tool": tool_name,
                "arguments": _json_safe(arguments),
                "observation": observation,
            })

            state["executed_tools"].append(tool_name)

            state["messages"].append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": tool_name,
                "content": json.dumps(observation),
            })

            # Critical: checkpoint only after the tool result is recorded.
            _save_checkpoint(state)

            # Simulate a crash after the requested number of tool executions.
            if (
                crash_after_tools is not None
                and len(state["executed_tools"]) >= crash_after_tools
            ):
                raise SimulatedCrash(
                    f"Simulated crash after {crash_after_tools} tools"
                )

    result = _fallback(
        "FALLBACK_MAX_ITERATIONS",
        "Maximum model iterations reached",
        state,
    )

    state["result"] = result
    _save_checkpoint(state)
    return result


def _new_state(
    user_message,
    scenario_id,
    max_iterations,
    token_limit,
    time_limit_seconds,
):
    """Create a fresh agent run."""
    run_id = f"{scenario_id}_{uuid.uuid4().hex[:8]}"

    return {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "user_message": user_message,
        "max_iterations": max_iterations,
        "token_limit": token_limit,
        "time_limit_seconds": time_limit_seconds,
        "messages": [
            {
                "role": "system",
                "content": AGENT_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        "tokens_used": 0,
        "llm_calls": 0,
        "steps": [],
        "executed_tools": [],
    }


def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
):
    """Run a new ReAct refund-agent session."""
    state = _new_state(
        user_message,
        scenario_id,
        max_iterations,
        token_limit,
        time_limit_seconds,
    )

    _save_checkpoint(state)

    return _run_loop(state)


def resume_agent(run_id):
    """Resume an interrupted run without repeating completed tools."""
    checkpoint_path = STATE_DIR / f"{run_id}.json"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    with checkpoint_path.open("r", encoding="utf-8") as file:
        state = json.load(file)

    if state.get("result"):
        return state["result"]

    print(f"Resuming run: {run_id}")

    return _run_loop(state)


def run_crash_recovery_demo(user_message, scenario_id="S1"):
    """Simulate a crash after two tools and resume from the checkpoint."""
    state = _new_state(
        user_message,
        scenario_id,
        max_iterations=6,
        token_limit=6000,
        time_limit_seconds=10,
    )

    _save_checkpoint(state)

    print("\n--- SIMULATING CRASH AFTER SECOND TOOL ---")

    try:
        _run_loop(state, crash_after_tools=2)
    except SimulatedCrash as error:
        print(error)

    print("\n--- RESUMING FROM CHECKPOINT ---")

    resumed_result = resume_agent(state["run_id"])

    return resumed_result


def _append_jsonl(path, record):
    """Append one JSON record per line."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, default=str) + "\n")


def run_all_scenarios():
    """Execute and save all seven assignment scenarios."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_path = OUTPUT_DIR / "agent_runs.jsonl"

    # Start a clean results file for this full demonstration.
    results_path.write_text("", encoding="utf-8")

    scenarios = _load_scenarios()
    results = []

    print("\n========== REFUND AGENT SCENARIOS ==========")

    for scenario in scenarios:
        scenario_id = scenario["scenario_id"]
        message = scenario["message"]

        print(f"\nRunning {scenario_id}: {message}")

        result = run_agent(
            user_message=message,
            scenario_id=scenario_id,
        )

        record = {
            "scenario_id": scenario_id,
            "expected_outcome": scenario["expected_outcome"],
            "expected_refund_inr": scenario["expected_refund_inr"],
            "status": result["status"],
            "final_answer": result["final_answer"],
            "llm_calls": result["llm_calls"],
            "tokens_used": result["tokens_used"],
            "steps": result["steps"],
        }

        _append_jsonl(results_path, record)
        results.append(record)

        print("Status:", result["status"])
        print("Answer:", result["final_answer"])
        print("LLM calls:", result["llm_calls"])
        print("Tokens:", result["tokens_used"])
        print("Tools:", result["executed_tools"])

    print("\n========== SCENARIO SUMMARY ==========")

    for result in results:
        print(
            result["scenario_id"],
            "|",
            result["status"],
            "|",
            result["llm_calls"],
            "LLM calls",
            "|",
            result["tokens_used"],
            "tokens",
        )

    return results


def run_limit_demos():
    """Demonstrate token and timeout fallbacks."""
    scenarios = _load_scenarios()
    s1 = next(item for item in scenarios if item["scenario_id"] == "S1")

    print("\n========== TOKEN LIMIT DEMO ==========")

    token_result = run_agent(
        s1["message"],
        "S1_TOKEN_LIMIT",
        token_limit=200,
    )

    print(token_result["status"])
    print(token_result["final_answer"])
    print("Tools executed:", token_result["executed_tools"])

    print("\n========== TIMEOUT DEMO ==========")

    timeout_result = run_agent(
        s1["message"],
        "S1_TIMEOUT",
        time_limit_seconds=0,
    )

    print(timeout_result["status"])
    print(timeout_result["final_answer"])
    print("LLM calls:", timeout_result["llm_calls"])


def main():
    # Clear the ledger before the complete demonstration.
    ledger_path = OUTPUT_DIR / "refund_ledger.jsonl"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text("", encoding="utf-8")

    # 1. Run all seven required scenarios.
    run_all_scenarios()

    # 2. Demonstrate token and timeout safeguards.
    run_limit_demos()

    # 3. Demonstrate checkpoint and resume behaviour.
    scenarios = _load_scenarios()
    s1 = next(item for item in scenarios if item["scenario_id"] == "S1")

    print("\n========== CRASH RECOVERY DEMO ==========")

    recovery_result = run_crash_recovery_demo(
        s1["message"],
        scenario_id="S1_CRASH_RECOVERY",
    )

    print("\nFinal status:", recovery_result["status"])
    print("Final answer:", recovery_result["final_answer"])
    print("Tools executed:", recovery_result["executed_tools"])
    print("LLM calls:", recovery_result["llm_calls"])

    print("\n========== AGENT EXECUTION COMPLETE ==========")
    print("Results:", OUTPUT_DIR / "agent_runs.jsonl")
    print("Ledger:", ledger_path)
    print("Checkpoints:", STATE_DIR)


if __name__ == "__main__":
    main()
