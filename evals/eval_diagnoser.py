"""
eval_diagnoser.py - scores the Diagnoser against the injected faults (ground truth).

For each fault the Watcher caught, check:
  - Error code correct?  (did it identify the right failure mode)
  - Grounded?            (did it cite the right manual)
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPECTED_MANUAL = {
    "chiller_failure": "chiller_system.md",
    "bearing_wear": "filler_bearings.md",
    "co2_leak": "carbonation_co2.md",
    "conveyor_jam": "conveyor_system.md",
}

faults = json.loads((ROOT / "evals" / "faults.json").read_text())
results = json.loads((ROOT / "data" / "diagnoses.json").read_text())

code_ok = grounded_ok = 0
print(f"{'Fault':<6} {'Type':<16} {'Expected':<9} {'Got':<9} {'Code':<5} {'Cited right manual'}")
for r in results:
    a, d = r["anomaly"], r["diagnosis"]
    fault = next(f for f in faults if f["line_id"] == a["line_id"] and f["sensor"] == a["sensor"]
                 and f["start"] <= a["detected_at"] <= f["end"])
    code_match = d["likely_error_code"] == fault["error_code"]
    grounded = any(s.startswith(EXPECTED_MANUAL[fault["fault_type"]]) for s in d["sources"])
    code_ok += code_match
    grounded_ok += grounded
    print(f"{fault['fault_id']:<6} {fault['fault_type']:<16} {fault['error_code']:<9} {d['likely_error_code']:<9} "
          f"{'yes' if code_match else 'NO':<5} {'yes' if grounded else 'NO'}")

n = len(results)
print(f"\nDiagnosis accuracy: {code_ok / n:.0%} ({code_ok}/{n})")
print(f"Grounding rate:     {grounded_ok / n:.0%} ({grounded_ok}/{n})")
