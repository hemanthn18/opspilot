"""
tickets.py - the ONLY code path that writes to the ERP.
Called by the graph only AFTER a human approves (least privilege + human-in-the-loop).
"""

import sqlite3
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "erp.db"


def create_ticket(line_id: str, fault_summary: str, root_cause: str,
                  proposed_action: str, approved_by: str) -> int:
    """Insert one maintenance ticket and return its id. Parameterized query = no SQL injection."""
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "INSERT INTO tickets (created_at, line_id, fault_summary, root_cause, proposed_action, status, approved_by) "
            "VALUES (?, ?, ?, ?, ?, 'open', ?)",
            (datetime.now().isoformat(timespec="seconds"), line_id, fault_summary,
             root_cause, proposed_action, approved_by))
        return cur.lastrowid
