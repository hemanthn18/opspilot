"""
eval_watcher.py - scores the Watcher against the injected faults (ground truth).

Metrics:
  - Recall:    how many real faults did we catch?
  - Precision: how many of our alerts were real faults (not false alarms)?
  - Lead time: how many minutes before the machine's own alarm did we detect it?
"""

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.watcher import watch  # noqa: E402

faults = json.loads((ROOT / "evals" / "faults.json").read_text())
anomalies = list(watch())


def minutes(a: str, b: str) -> float:
    return (datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds() / 60


matched_faults = set()
true_positives = 0
rows = []
for a in anomalies:
    match = next((f for f in faults
                  if f["line_id"] == a.line_id and f["sensor"] == a.sensor
                  and f["start"] <= a.detected_at <= f["end"]), None)
    if match:
        true_positives += 1
        matched_faults.add(match["fault_id"])
        alarm_min = match["duration_min"] * 0.5          # machine alarm fires at 50% of the fault
        detect_min = minutes(match["start"], a.detected_at)
        rows.append((match["fault_id"], match["fault_type"], detect_min, alarm_min - detect_min))
    else:
        rows.append(("FALSE ALARM", f"{a.line_id} {a.sensor}", None, None))

recall = len(matched_faults) / len(faults)
precision = true_positives / len(anomalies) if anomalies else 0.0

print(f"{'Fault':<12} {'Type':<18} {'Detected after':<16} {'Lead vs machine alarm'}")
for fid, ftype, det, lead in rows:
    if det is None:
        print(f"{fid:<12} {ftype:<18}")
    else:
        print(f"{fid:<12} {ftype:<18} {det:>5.0f} min        {lead:>5.0f} min earlier")

print(f"\nRecall:    {recall:.0%}  ({len(matched_faults)}/{len(faults)} faults caught)")
print(f"Precision: {precision:.0%}  ({true_positives}/{len(anomalies)} alerts were real)")
leads = [r[3] for r in rows if r[3] is not None]
if leads:
    print(f"Average lead time: {sum(leads) / len(leads):.0f} min before the machine's own alarm")
