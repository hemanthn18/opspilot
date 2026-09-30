"""
erp.py - tools the Impact agent can call against the ERP database.

Safety design:
  - Every connection is opened READ-ONLY at the database level (mode=ro)
  - Free-form SQL is limited to a single SELECT statement (allow-list, not block-list)
  - Business numbers (lost cases, shortfalls) are computed in code, never by the LLM
"""

import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "erp.db"
MAX_ROWS = 50
LOOKAHEAD_HOURS = 24  # orders due within 24h after the line restarts are checked


def connect_readonly() -> sqlite3.Connection:
    """Read-only connection: even a malicious query cannot write."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def run_readonly_sql(query: str) -> dict:
    """Run one SELECT query. Anything else is rejected before it reaches the database."""
    q = query.strip().rstrip(";").strip()
    if not re.match(r"(?is)^(select|with)\b", q) or ";" in q:
        return {"error": "Only a single SELECT statement is allowed."}
    try:
        with connect_readonly() as conn:
            rows = conn.execute(q).fetchmany(MAX_ROWS)
            return {"rows": [dict(r) for r in rows], "truncated_at": MAX_ROWS}
    except sqlite3.Error as err:
        return {"error": str(err)}


def _shortfalls(conn, sku, schedule, start, down_start, down_end, horizon) -> dict:
    """Available-to-promise: allocate inventory + production made before each due date
    to open orders in due-date order. Returns {order_id: short_cases} for uncovered orders."""
    on_hand = conn.execute("SELECT on_hand_cases FROM inventory WHERE sku = ?", (sku,)).fetchone()[0]
    orders = conn.execute(
        "SELECT order_id, qty_cases, due_date FROM orders "
        "WHERE sku = ? AND status = 'open' AND due_date >= ? AND due_date <= ? ORDER BY due_date",
        (sku, start.isoformat(), horizon.isoformat())).fetchall()
    shorts, allocated = {}, 0
    for order in orders:
        due = datetime.fromisoformat(order["due_date"])
        produced = 0.0
        for run in schedule:
            if run["sku"] != sku:
                continue
            s, e = datetime.fromisoformat(run["start_time"]), datetime.fromisoformat(run["end_time"])
            rate = run["planned_cases"] / (e - s).total_seconds()
            # production from 'start' until the due date, skipping the downtime window
            for a, b in ((max(s, start), min(e, due, down_start)), (max(s, down_end, start), min(e, due))):
                if b > a:
                    produced += rate * (b - a).total_seconds()
        allocated += order["qty_cases"]
        gap = allocated - (on_hand + int(produced))
        if gap > 0:
            shorts[order["order_id"]] = min(order["qty_cases"], gap)
    return shorts


def estimate_impact(line_id: str, start_time: str, downtime_hours: float) -> dict:
    """
    Deterministic impact of stopping a line for `downtime_hours`.
      1) Lost production per SKU (planned cases in the window, prorated)
      2) Orders at risk BECAUSE of the downtime: compare an available-to-promise check
         with downtime against a no-downtime baseline, and keep only the extra shortfall.
    """
    start = datetime.fromisoformat(start_time)
    down_end = start + timedelta(hours=downtime_hours)
    horizon = down_end + timedelta(hours=LOOKAHEAD_HOURS)

    with connect_readonly() as conn:
        schedule = [dict(r) for r in conn.execute(
            "SELECT * FROM production_schedule WHERE line_id = ? AND end_time > ? AND start_time < ?",
            (line_id, start_time, horizon.isoformat()))]

        lost = {}
        for run in schedule:
            s, e = datetime.fromisoformat(run["start_time"]), datetime.fromisoformat(run["end_time"])
            overlap = (min(e, down_end) - max(s, start)).total_seconds()
            if overlap > 0:
                lost[run["sku"]] = lost.get(run["sku"], 0) + round(run["planned_cases"] * overlap / (e - s).total_seconds())

        orders_at_risk = []
        for sku in lost:
            with_downtime = _shortfalls(conn, sku, schedule, start, start, down_end, horizon)
            baseline = _shortfalls(conn, sku, schedule, start, start, start, horizon)
            for order_id, short in with_downtime.items():
                extra = short - baseline.get(order_id, 0)
                if extra > 0:
                    o = conn.execute("SELECT order_id, customer, sku, qty_cases, due_date FROM orders WHERE order_id = ?",
                                     (order_id,)).fetchone()
                    orders_at_risk.append({**dict(o), "short_cases": extra})

    return {
        "line_id": line_id,
        "downtime_window": [start.isoformat(), down_end.isoformat()],
        "lost_cases_by_sku": lost,
        "orders_at_risk": sorted(orders_at_risk, key=lambda o: o["due_date"]),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(estimate_impact("L1", "2026-09-22T04:54:00", 2.5), indent=2))
    print(run_readonly_sql("DELETE FROM orders"))
    print(run_readonly_sql("SELECT 1; DROP TABLE orders"))
    print(run_readonly_sql("SELECT sku, on_hand_cases FROM inventory LIMIT 2"))
