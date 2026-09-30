"""
graph.py - OpsPilot orchestration with LangGraph

Watcher (trigger) -> for each anomaly, run this graph:

    diagnose ──(confident?)──> assess_impact ──> plan_action ──> human_approval ──(approved?)──> execute
        │ no                                                              │ no
        └──> escalate                                                     └──> reject

- State: one shared IncidentState dict that every node reads and adds to
- Checkpointer: saves state after every node, so the graph can PAUSE for a human and resume
- interrupt(): the human-in-the-loop gate; nothing is written to the ERP before it
"""

import argparse
import json
from dataclasses import asdict
from typing import Optional, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agents.action import plan_action
from agents.diagnoser import diagnose
from agents.impact import assess_impact
from agents.watcher import watch
from tools.tickets import create_ticket

MIN_CONFIDENCE = 0.5


class IncidentState(TypedDict, total=False):
    anomaly: dict
    diagnosis: dict
    impact_report: dict
    action_plan: dict
    approval: dict
    ticket_id: Optional[int]
    outcome: str


# ---------------- Nodes: each takes the state and returns the fields it adds ----------------
def diagnose_node(state: IncidentState) -> dict:
    return {"diagnosis": diagnose(state["anomaly"]).model_dump()}


def impact_node(state: IncidentState) -> dict:
    return {"impact_report": assess_impact(state["anomaly"], state["diagnosis"])}


def action_node(state: IncidentState) -> dict:
    plan = plan_action(state["anomaly"], state["diagnosis"], state["impact_report"])
    return {"action_plan": plan.model_dump()}


def approval_node(state: IncidentState) -> dict:
    # Pauses the graph here. The value passed to interrupt() is shown to the human;
    # the graph resumes when someone calls it again with Command(resume=decision).
    decision = interrupt({"line": state["anomaly"]["line_id"], "plan": state["action_plan"]})
    return {"approval": decision}


def execute_node(state: IncidentState) -> dict:
    plan, diag = state["action_plan"], state["diagnosis"]
    ticket_id = create_ticket(
        line_id=state["anomaly"]["line_id"],
        fault_summary=f"{plan['priority']} {plan['ticket_title']}",
        root_cause=f"{diag['likely_error_code']}: {diag['root_cause']}",
        proposed_action="; ".join(plan["maintenance_steps"]),
        approved_by=state["approval"].get("by", "unknown"))
    return {"ticket_id": ticket_id, "outcome": "approved - ticket created"}


def reject_node(state: IncidentState) -> dict:
    return {"outcome": f"rejected - {state['approval'].get('reason', 'no reason given')}"}


def escalate_node(state: IncidentState) -> dict:
    return {"outcome": "escalated - diagnosis confidence too low, needs an engineer"}


# ---------------- Routing (conditional edges) ----------------
def route_after_diagnosis(state: IncidentState) -> str:
    d = state["diagnosis"]
    confident = d["likely_error_code"] != "UNKNOWN" and d["confidence"] >= MIN_CONFIDENCE
    return "assess_impact" if confident else "escalate"


def route_after_approval(state: IncidentState) -> str:
    return "execute" if state["approval"].get("approved") else "reject"


def build_graph():
    g = StateGraph(IncidentState)
    g.add_node("diagnose", diagnose_node)
    g.add_node("assess_impact", impact_node)
    g.add_node("plan_action", action_node)
    g.add_node("human_approval", approval_node)
    g.add_node("execute", execute_node)
    g.add_node("reject", reject_node)
    g.add_node("escalate", escalate_node)

    g.add_edge(START, "diagnose")
    g.add_conditional_edges("diagnose", route_after_diagnosis, ["assess_impact", "escalate"])
    g.add_edge("assess_impact", "plan_action")
    g.add_edge("plan_action", "human_approval")
    g.add_conditional_edges("human_approval", route_after_approval, ["execute", "reject"])
    for node in ("execute", "reject", "escalate"):
        g.add_edge(node, END)
    return g.compile(checkpointer=InMemorySaver())


def ask_human(payload: dict, auto: bool) -> dict:
    plan = payload["plan"]
    print(f"\n  ┌─ APPROVAL NEEDED ({payload['line']}) ─────────────────────")
    print(f"  │ {plan['priority']}  {plan['ticket_title']}")
    for step in plan["maintenance_steps"]:
        print(f"  │  - {step}")
    print(f"  │ Part: {plan['spare_part']}")
    print(f"  │ Planner: {plan['planner_recommendation']}")
    print(f"  │ Notify customers for: {', '.join(plan['customers_to_notify']) or 'none'}")
    print("  └──────────────────────────────────────────────────")
    if auto:
        print("  (auto-approved)")
        return {"approved": True, "by": "auto"}
    answer = input("  Approve? [y/n]: ").strip().lower()
    if answer == "y":
        return {"approved": True, "by": input("  Your name: ").strip() or "unknown"}
    return {"approved": False, "reason": input("  Reason for rejecting: ").strip()}


def run(limit: int | None, auto: bool):
    graph = build_graph()
    anomalies = [asdict(a) for a in watch()][:limit]
    print(f"Watcher raised {len(anomalies)} anomalies\n")
    summary = []
    for a in anomalies:
        config = {"configurable": {"thread_id": f"{a['line_id']}-{a['sensor']}-{a['detected_at']}"}}
        print(f"=== {a['line_id']} {a['sensor']} at {a['detected_at']} ===")
        next_input = {"anomaly": a}
        while True:
            for update in graph.stream(next_input, config, stream_mode="updates"):
                for node, value in update.items():
                    if node == "__interrupt__":
                        continue
                    print(f"  ✓ {node}")
            snapshot = graph.get_state(config)
            if not snapshot.interrupts:            # graph finished
                break
            decision = ask_human(snapshot.interrupts[0].value, auto)
            next_input = Command(resume=decision)  # resume from the checkpoint
        final = graph.get_state(config).values
        print(f"  => {final['outcome']}" + (f" (ticket #{final['ticket_id']})" if final.get("ticket_id") else ""))
        summary.append({k: final.get(k) for k in ("anomaly", "diagnosis", "action_plan", "outcome", "ticket_id")})
    with open("data/run_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\nSaved data/run_summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the OpsPilot incident pipeline")
    parser.add_argument("--limit", type=int, help="only process the first N anomalies")
    parser.add_argument("--auto-approve", action="store_true", help="skip the human prompt (for testing)")
    args = parser.parse_args()
    run(args.limit, args.auto_approve)
