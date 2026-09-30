"""
impact.py - Agent 3: the Impact agent (a tool-calling agent)

The LLM does not touch the database directly. It is given TOOLS (function schemas).
It decides which tool to call and with what arguments; OUR code runs the tool and
returns the result; the loop repeats until the LLM writes its final answer.

    LLM -> "call estimate_impact(line_id=L1, ...)" -> code runs it -> result -> LLM -> ...
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.erp import estimate_impact, run_readonly_sql  # noqa: E402
from tools.llm import chat  # noqa: E402

MAX_STEPS = 5          # hard stop so the agent can never loop forever
DEFAULT_DOWNTIME = 2.0  # hours, if the Diagnoser gave no estimate

# Tool schemas: this is what the LLM "sees" (name, purpose, arguments)
TOOLS = [
    {"type": "function", "function": {
        "name": "estimate_impact",
        "description": "Compute lost production and customer orders put at risk if a line stops. "
                       "Returns exact numbers from the ERP.",
        "parameters": {"type": "object", "properties": {
            "line_id": {"type": "string", "description": "e.g. L1"},
            "start_time": {"type": "string", "description": "ISO timestamp when the line stops"},
            "downtime_hours": {"type": "number"}},
            "required": ["line_id", "start_time", "downtime_hours"]}}},
    {"type": "function", "function": {
        "name": "run_readonly_sql",
        "description": "Run ONE read-only SELECT on the ERP (SQLite). Tables: lines, products, inventory, "
                       "production_schedule, orders, tickets. Use only for extra context.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
]
TOOL_FUNCTIONS = {"estimate_impact": estimate_impact, "run_readonly_sql": run_readonly_sql}

SYSTEM_PROMPT = """You are the Impact agent for a beverage bottling plant.
Given an equipment incident and its diagnosis, report the business impact.
Rules:
- Always call estimate_impact first, using the incident time and the estimated repair hours.
- Never calculate or invent numbers yourself. Quote numbers exactly as the tools return them.
- Mention every order at risk by order_id, customer and short_cases.
- If no orders are at risk, say so clearly.
- Final answer: 2-4 plain sentences for a plant manager."""


def assess_impact(anomaly: dict, diagnosis: dict) -> dict:
    downtime = diagnosis.get("estimated_repair_hours") or DEFAULT_DOWNTIME
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Incident on line {anomaly['line_id']} detected at {anomaly['detected_at']}.\n"
            f"Diagnosis: {diagnosis['likely_error_code']} - {diagnosis['root_cause']}\n"
            f"Estimated repair time: {downtime} hours.")},
    ]
    trace, impact = [], None

    for _ in range(MAX_STEPS):
        message = chat(messages, tools=TOOLS).choices[0].message
        if not message.tool_calls:                      # no tool requested -> final answer
            return {"downtime_hours": downtime, "impact": impact,
                    "summary": message.content, "tool_trace": trace}

        # Record the assistant's tool request, then run each tool and return the result
        messages.append({"role": "assistant", "content": message.content, "tool_calls": [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in message.tool_calls]})
        for tc in message.tool_calls:
            args = json.loads(tc.function.arguments)
            func = TOOL_FUNCTIONS.get(tc.function.name)
            result = func(**args) if func else {"error": f"unknown tool {tc.function.name}"}
            if tc.function.name == "estimate_impact":
                impact = result                         # keep the exact numbers for later agents
            trace.append({"tool": tc.function.name, "args": args})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})

    return {"downtime_hours": downtime, "impact": impact,
            "summary": "Stopped: step limit reached. Escalate to a planner.", "tool_trace": trace}


if __name__ == "__main__":
    diagnoses = json.loads((ROOT / "data" / "diagnoses.json").read_text())
    results = []
    for item in diagnoses:
        a, d = item["anomaly"], item["diagnosis"]
        report = assess_impact(a, d)
        results.append({**item, "impact_report": report})
        print(f"\n{a['line_id']} {d['likely_error_code']} ({report['downtime_hours']}h downtime)")
        for step in report["tool_trace"]:
            print(f"  tool: {step['tool']}({step['args']})")
        print(f"  -> {report['summary']}")
    (ROOT / "data" / "impacts.json").write_text(json.dumps(results, indent=2))
    print("\nSaved to data/impacts.json")
