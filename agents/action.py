"""
action.py - Agent 4: the Action agent

Turns diagnosis + impact into a concrete, reviewable plan:
  - a maintenance ticket (priority, what to do, which part)
  - a production/customer recommendation for the planner
It only PROPOSES. Nothing is written until a human approves in the graph.
"""

import json
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.llm import chat_structured  # noqa: E402


class ActionPlan(BaseModel):
    priority: Literal["P1", "P2", "P3"] = Field(description="P1 = customer orders at risk, P2 = production loss only, P3 = monitor")
    ticket_title: str
    maintenance_steps: list[str]
    spare_part: str | None
    planner_recommendation: str = Field(description="What production planning should do about affected orders")
    customers_to_notify: list[str] = Field(description="order_ids whose customers should be informed")


SYSTEM_PROMPT = """You are the Action agent for a beverage bottling plant.
Using the diagnosis and the impact report, propose a clear action plan for human approval.
Rules:
- priority P1 if any customer order is at risk, P2 if only production is lost, P3 if it is only a watch item.
- maintenance_steps: short, concrete steps taken from the diagnosis recommendations.
- customers_to_notify: exactly the order_ids listed as at risk in the impact report (empty if none).
- Do not invent parts, orders or numbers that are not in the inputs."""


def plan_action(anomaly: dict, diagnosis: dict, impact_report: dict) -> ActionPlan:
    user_prompt = (
        f"Line: {anomaly['line_id']}  detected: {anomaly['detected_at']}\n\n"
        f"Diagnosis:\n{json.dumps(diagnosis, indent=2)}\n\n"
        f"Impact report:\n{json.dumps(impact_report.get('impact'), indent=2)}\n"
        f"Impact summary: {impact_report.get('summary')}")
    plan, raw = chat_structured(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
        ActionPlan)
    if plan is None:
        raise ValueError(f"Action agent returned no valid plan: {raw}")

    # Guardrail: customers_to_notify must match the deterministic impact result
    at_risk = [o["order_id"] for o in (impact_report.get("impact") or {}).get("orders_at_risk", [])]
    plan.customers_to_notify = at_risk
    if at_risk:
        plan.priority = "P1"
    return plan
