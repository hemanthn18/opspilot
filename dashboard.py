"""
dashboard.py - OpsPilot web dashboard (Streamlit)

Talks to the API only over HTTP (it never imports the agents), so UI and backend
can be deployed and scaled separately.

Run:  streamlit run dashboard.py
"""

import os

import pandas as pd
import requests
import streamlit as st

API = os.environ.get("OPSPILOT_API_URL", "http://localhost:8000")

st.set_page_config(page_title="OpsPilot", page_icon="🏭", layout="wide")
st.title("OpsPilot · Incident Command Center")
st.caption("Watcher → Diagnoser (RAG) → Impact (ERP tools) → Action plan → **human approval** → ticket")


def api(method: str, path: str, **kwargs):
    try:
        r = requests.request(method, f"{API}{path}", timeout=60, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as err:
        st.error(f"API error on {path}: {err}")
        return None


# ---------------- Sidebar ----------------
with st.sidebar:
    st.subheader("Controls")
    if st.button("▶ Run Watcher", type="primary", width="stretch"):
        api("POST", "/incidents/run")
        st.success("Watcher started. Incidents appear as the agents finish (about 10–20 s each).")
    if st.button("⟳ Refresh", width="stretch"):
        st.rerun()
    health = api("GET", "/health")
    st.caption(f"API: {API} · {'🟢 healthy' if health else '🔴 unreachable'}")

incidents = api("GET", "/incidents") or []

# ---------------- KPI row ----------------
counts = {s: sum(i["status"] == s for i in incidents)
          for s in ("pending_approval", "approved", "rejected", "escalated")}
c = st.columns(5)
c[0].metric("Incidents", len(incidents))
c[1].metric("Awaiting approval", counts["pending_approval"])
c[2].metric("Approved", counts["approved"])
c[3].metric("Rejected", counts["rejected"])
c[4].metric("Escalated", counts["escalated"])

if not incidents:
    st.info("No incidents yet. Click **Run Watcher** in the sidebar.")
    st.stop()

BADGE = {"pending_approval": "🟠 Awaiting approval", "approved": "🟢 Approved", "rejected": "⚪ Rejected",
         "escalated": "🔴 Escalated", "processing": "🔵 Processing", "error": "❌ Error"}
SENSOR_LABEL = {"temperature_c": "Temperature (°C)", "vibration_mm_s": "Vibration (mm/s)",
                "pressure_bar": "CO₂ pressure (bar)", "filler_speed_bpm": "Filler speed (bottles/min)"}

# Pending first, then newest
incidents.sort(key=lambda i: (i["status"] != "pending_approval", i["detected_at"]))

for inc in incidents:
    title = (f"{BADGE.get(inc['status'], inc['status'])} · {inc['priority'] or ''} · Line {inc['line_id']} · "
             f"{SENSOR_LABEL.get(inc['sensor'], inc['sensor'])} · {inc['error_code'] or '…'} · {inc['detected_at']}")
    with st.expander(title, expanded=inc["status"] == "pending_approval"):
        detail = api("GET", f"/incidents/{inc['id']}")
        if not detail:
            continue
        a, d = detail.get("anomaly", {}), detail.get("diagnosis") or {}
        imp_rep, plan = detail.get("impact_report") or {}, detail.get("action_plan") or {}

        left, right = st.columns([3, 2])
        with left:
            st.markdown("**1 · Watcher: what was detected**")
            trend = api("GET", f"/incidents/{inc['id']}/trend")
            if trend and trend["points"]:
                df = pd.DataFrame(trend["points"])
                df["ts"] = pd.to_datetime(df["ts"])
                df = df.set_index("ts")
                st.line_chart(df, y="value", y_label=SENSOR_LABEL.get(a.get("sensor"), "value"), height=220)
            st.caption(f"Value {a.get('value')} vs baseline {a.get('baseline_mean')} · z-score {a.get('z_score')} · "
                       f"machine alarm at detection: {a.get('error_code_seen') or 'none (caught early)'}")
        with right:
            st.markdown("**2 · Diagnoser: why (RAG)**")
            st.markdown(f"**{d.get('likely_error_code', '…')}** — {d.get('root_cause', '')}")
            st.progress(float(d.get("confidence") or 0), text=f"Confidence {d.get('confidence')}")
            st.caption("Sources: " + " · ".join(d.get("sources") or ["none"]))

        impact = imp_rep.get("impact") or {}
        st.markdown("**3 · Impact agent: business effect (ERP)**")
        i1, i2 = st.columns(2)
        i1.dataframe(pd.DataFrame([{"SKU": k, "Lost cases": v} for k, v in (impact.get("lost_cases_by_sku") or {}).items()]),
                     hide_index=True, width="stretch")
        orders = impact.get("orders_at_risk") or []
        if orders:
            i2.dataframe(pd.DataFrame(orders)[["order_id", "customer", "short_cases", "due_date"]],
                         hide_index=True, width="stretch")
        else:
            i2.success("No customer orders at risk")
        if imp_rep.get("summary"):
            st.caption(imp_rep["summary"])

        if plan:
            st.markdown(f"**4 · Action plan — {plan.get('priority')}: {plan.get('ticket_title')}**")
            st.markdown("\n".join(f"- {s}" for s in plan.get("maintenance_steps", [])))
            st.caption(f"Part: {plan.get('spare_part') or '—'} · Planner: {plan.get('planner_recommendation')} · "
                       f"Notify: {', '.join(plan.get('customers_to_notify') or []) or 'none'}")

        with st.popover("Agent trace (tool calls)"):
            st.json(imp_rep.get("tool_trace") or [])

        if detail["status"] == "pending_approval":
            st.markdown("**5 · Your decision**")
            f1, f2, f3 = st.columns([2, 2, 3])
            name = f1.text_input("Your name", key=f"name-{inc['id']}")
            if f2.button("✅ Approve & create ticket", key=f"ok-{inc['id']}", type="primary", disabled=not name):
                api("POST", f"/incidents/{inc['id']}/decision", json={"approved": True, "by": name})
                st.rerun()
            reason = f3.text_input("Reason (to reject)", key=f"reason-{inc['id']}")
            if f3.button("Reject", key=f"no-{inc['id']}", disabled=not reason):
                api("POST", f"/incidents/{inc['id']}/decision", json={"approved": False, "reason": reason})
                st.rerun()
        else:
            st.info(f"Outcome: {detail.get('outcome')}"
                    + (f" · ticket #{detail['ticket_id']}" if detail.get("ticket_id") else ""))
