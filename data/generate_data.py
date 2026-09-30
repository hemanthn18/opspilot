"""
generate_data.py
Creates synthetic data for OpsPilot:
  1. data/logs.jsonl      - time-series sensor logs for 3 bottling lines, with injected faults
  2. data/erp.db          - SQLite ERP database (lines, products, orders, inventory, schedule, tickets)
  3. data/manuals/*.md    - maintenance manuals and past incident notes (the RAG knowledge base)
  4. evals/faults.json    - ground truth: which faults were injected, where and when
"""

import json
import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)  # fixed seed = same data every run (reproducible)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MANUALS_DIR = DATA_DIR / "manuals"
EVALS_DIR = ROOT / "evals"
MANUALS_DIR.mkdir(parents=True, exist_ok=True)
EVALS_DIR.mkdir(parents=True, exist_ok=True)

START = datetime(2026, 9, 21, 0, 0)  # simulation start
DAYS = 3
INTERVAL_MIN = 1  # one reading per minute per line

LINES = ["L1", "L2", "L3"]

# Normal operating range for each sensor: (mean, noise)
NORMAL = {
    "temperature_c": (4.0, 0.3),       # product temperature at the filler
    "filler_speed_bpm": (600, 8),      # bottles per minute
    "pressure_bar": (3.5, 0.05),       # CO2 pressure for carbonation
    "vibration_mm_s": (2.0, 0.2),      # motor/bearing vibration
}

# ---------------------------------------------------------------------------
# 1. Fault injection plan (this is the ground truth / answer key for evals)
# ---------------------------------------------------------------------------
FAULT_TYPES = {
    "chiller_failure": {"sensor": "temperature_c", "error_code": "E-214"},
    "bearing_wear": {"sensor": "vibration_mm_s", "error_code": "E-331"},
    "co2_leak": {"sensor": "pressure_bar", "error_code": "E-118"},
    "conveyor_jam": {"sensor": "filler_speed_bpm", "error_code": "E-402"},
}

total_minutes = DAYS * 24 * 60
faults = []
fault_id = 1
for line in LINES:
    # 2 faults per line, spread out so they don't overlap
    chosen = random.sample(list(FAULT_TYPES), 2)
    for i, ftype in enumerate(chosen):
        start_min = random.randint(200, total_minutes // 2 - 200) + i * (total_minutes // 2)
        duration = random.randint(45, 120)
        faults.append({
            "fault_id": f"F{fault_id:02d}",
            "line_id": line,
            "fault_type": ftype,
            "sensor": FAULT_TYPES[ftype]["sensor"],
            "error_code": FAULT_TYPES[ftype]["error_code"],
            "start": (START + timedelta(minutes=start_min)).isoformat(),
            "end": (START + timedelta(minutes=start_min + duration)).isoformat(),
            "start_min": start_min,
            "duration_min": duration,
        })
        fault_id += 1


def active_fault(line, minute):
    for f in faults:
        if f["line_id"] == line and f["start_min"] <= minute < f["start_min"] + f["duration_min"]:
            return f
    return None


# ---------------------------------------------------------------------------
# 2. Sensor logs (JSON Lines: one JSON object per line)
# ---------------------------------------------------------------------------
log_count = 0
with open(DATA_DIR / "logs.jsonl", "w") as out:
    for minute in range(0, total_minutes, INTERVAL_MIN):
        ts = (START + timedelta(minutes=minute)).isoformat()
        for line in LINES:
            reading = {s: round(random.gauss(m, sd), 2) for s, (m, sd) in NORMAL.items()}
            reading["error_code"] = None

            f = active_fault(line, minute)
            if f:
                progress = (minute - f["start_min"]) / f["duration_min"]  # 0 -> 1, fault gets worse
                if f["fault_type"] == "chiller_failure":
                    reading["temperature_c"] = round(4.0 + 6.0 * progress + random.gauss(0, 0.3), 2)
                elif f["fault_type"] == "bearing_wear":
                    reading["vibration_mm_s"] = round(2.0 + 7.0 * progress + random.gauss(0, 0.3), 2)
                elif f["fault_type"] == "co2_leak":
                    reading["pressure_bar"] = round(3.5 - 1.2 * progress + random.gauss(0, 0.05), 2)
                elif f["fault_type"] == "conveyor_jam":
                    reading["filler_speed_bpm"] = round(max(0, 600 * (1 - progress) + random.gauss(0, 15)), 1)
                # machine raises its error code once the fault is past halfway
                if progress > 0.5:
                    reading["error_code"] = f["error_code"]

            out.write(json.dumps({"ts": ts, "line_id": line, **reading}) + "\n")
            log_count += 1

# ---------------------------------------------------------------------------
# 3. ERP database (SQLite)
# ---------------------------------------------------------------------------
db_path = DATA_DIR / "erp.db"
if db_path.exists():
    db_path.unlink()
conn = sqlite3.connect(db_path)
cur = conn.cursor()

cur.executescript("""
CREATE TABLE lines (
    line_id TEXT PRIMARY KEY,
    name TEXT,
    plant TEXT,
    capacity_bpm INTEGER
);
CREATE TABLE products (
    sku TEXT PRIMARY KEY,
    name TEXT,
    pack_size INTEGER
);
CREATE TABLE inventory (
    sku TEXT PRIMARY KEY REFERENCES products(sku),
    on_hand_cases INTEGER,
    safety_stock_cases INTEGER
);
CREATE TABLE production_schedule (
    schedule_id INTEGER PRIMARY KEY,
    line_id TEXT REFERENCES lines(line_id),
    sku TEXT REFERENCES products(sku),
    start_time TEXT,
    end_time TEXT,
    planned_cases INTEGER
);
CREATE TABLE orders (
    order_id TEXT PRIMARY KEY,
    customer TEXT,
    sku TEXT REFERENCES products(sku),
    qty_cases INTEGER,
    due_date TEXT,
    status TEXT
);
CREATE TABLE tickets (
    ticket_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT,
    line_id TEXT REFERENCES lines(line_id),
    fault_summary TEXT,
    root_cause TEXT,
    proposed_action TEXT,
    status TEXT,
    approved_by TEXT
);
""")

cur.executemany("INSERT INTO lines VALUES (?,?,?,?)", [
    ("L1", "Filler Line 1 - Cans", "Plant A", 600),
    ("L2", "Filler Line 2 - PET Bottles", "Plant A", 600),
    ("L3", "Filler Line 3 - Glass Bottles", "Plant A", 600),
])

products = [
    ("SKU-COLA-12C", "Cola 12oz Can 24-pack", 24),
    ("SKU-DIET-12C", "Diet Cola 12oz Can 24-pack", 24),
    ("SKU-LIME-20P", "Lime Soda 20oz PET 24-pack", 24),
    ("SKU-COLA-2LP", "Cola 2L PET 8-pack", 8),
    ("SKU-COLA-8G", "Cola 8oz Glass 24-pack", 24),
    ("SKU-GING-12G", "Ginger Ale 12oz Glass 24-pack", 24),
]
cur.executemany("INSERT INTO products VALUES (?,?,?)", products)

cur.executemany("INSERT INTO inventory VALUES (?,?,?)", [
    (sku, random.randint(150, 700), 400) for sku, _, _ in products
])

# Which SKUs each line can run
line_skus = {
    "L1": ["SKU-COLA-12C", "SKU-DIET-12C"],
    "L2": ["SKU-LIME-20P", "SKU-COLA-2LP"],
    "L3": ["SKU-COLA-8G", "SKU-GING-12G"],
}

# Production schedule: 4-hour runs, alternating SKUs on each line
schedule = []
sid = 1
for line, skus in line_skus.items():
    t = START
    k = 0
    while t < START + timedelta(days=DAYS):
        sku = skus[k % len(skus)]
        pack = next(p[2] for p in products if p[0] == sku)
        planned_cases = int(600 * 60 * 4 * 0.85 / pack)  # 85% efficiency over 4 hours
        schedule.append((sid, line, sku, t.isoformat(), (t + timedelta(hours=4)).isoformat(), planned_cases))
        sid += 1
        k += 1
        t += timedelta(hours=4)
cur.executemany("INSERT INTO production_schedule VALUES (?,?,?,?,?,?)", schedule)

customers = ["NorthMart DC 14", "ValueHub DC 3", "FreshCo DC 7", "BulkWay DC 2", "SunGrocer DC 9", "QuickStop Hub 5"]
orders = []
for i in range(1, 61):
    sku = random.choice(products)[0]
    due = START + timedelta(hours=random.randint(8, DAYS * 24 + 24))
    orders.append((f"SO-{10000 + i}", random.choice(customers), sku,
                   random.choice([800, 1200, 1600, 2400, 3200]), due.isoformat(), "open"))
cur.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?)", orders)

conn.commit()
conn.close()

# ---------------------------------------------------------------------------
# 4. Maintenance manuals and past incidents (knowledge base for RAG)
# ---------------------------------------------------------------------------
MANUALS = {
    "chiller_system.md": """# Product Chiller System - Maintenance Manual

## Normal operation
Product must enter the filler at 3.5 to 4.5 C. Warm product loses carbonation and causes foaming and underfilled containers.

## Error code E-214: Product temperature high
Symptoms: product temperature rises steadily above 5 C, foaming at the filler valves, underfill rejects increase.
Likely causes, in order of frequency:
1. Chiller refrigerant low or compressor tripped.
2. Glycol pump failure, so no coolant reaches the plate heat exchanger.
3. Fouled plate heat exchanger reducing heat transfer.

## Recommended actions
- Check chiller compressor status and reset if tripped.
- Verify glycol pump is running and flow is above 40 L/min.
- If temperature exceeds 8 C, stop the filler to avoid producing out-of-spec product.
Typical repair time: 1 to 3 hours. Spare part: glycol pump seal kit (part GP-220).
""",
    "filler_bearings.md": """# Filler Main Drive and Bearings - Maintenance Manual

## Normal operation
Main drive vibration should stay below 3.0 mm/s RMS.

## Error code E-331: Vibration high
Symptoms: vibration climbs gradually over hours, audible grinding, bearing housing warm to touch.
Likely causes:
1. Main drive bearing wear or lack of lubrication (most common).
2. Misaligned drive coupling.
3. Loose mounting bolts.

## Recommended actions
- Above 4.5 mm/s: schedule bearing inspection within the next shift.
- Above 7.0 mm/s: stop the line, replace the bearing to avoid shaft damage.
- Lubricate per schedule every 500 operating hours.
Typical repair time: 3 to 5 hours. Spare part: main drive bearing (part BR-6310).
""",
    "carbonation_co2.md": """# Carbonation and CO2 Supply - Maintenance Manual

## Normal operation
CO2 supply pressure should be 3.4 to 3.6 bar.

## Error code E-118: CO2 pressure low
Symptoms: pressure falls below 3.2 bar, flat product, low carbonation readings in QA checks.
Likely causes:
1. Leak at a CO2 line fitting or regulator diaphragm (most common).
2. CO2 bulk tank running low.
3. Faulty pressure regulator.

## Recommended actions
- Check bulk tank level first.
- Leak-test fittings with soapy water; tighten or replace fittings.
- Replace regulator diaphragm if pressure drifts after tightening.
Typical repair time: 0.5 to 2 hours. Spare part: regulator diaphragm kit (part CR-45).
""",
    "conveyor_system.md": """# Conveyor and Container Handling - Maintenance Manual

## Normal operation
Filler runs at 600 bottles per minute when infeed conveyors are clear.

## Error code E-402: Infeed starvation / jam
Symptoms: filler speed drops sharply toward zero, containers tipped or stuck at the starwheel, photo-eye blocked alarms.
Likely causes:
1. Fallen or damaged containers jamming the starwheel.
2. Worn conveyor chain or guide rails out of adjustment.
3. Upstream depalletizer stopped.

## Recommended actions
- Clear jammed containers and inspect starwheel for damage.
- Check guide rail width against container size change parts.
- Check upstream equipment status.
Typical repair time: 0.25 to 1.5 hours. Spare part: starwheel change parts (part SW-12).
""",
    "past_incidents.md": """# Past Incident Log (summaries)

## INC-2025-031 - Line 1 - Warm product
E-214 raised after temperature reached 7 C. Root cause: glycol pump seal leak. Replaced seal kit GP-220. Downtime 2.5 hours. Lesson: temperature rose for 40 minutes before the alarm; earlier detection would have saved product.

## INC-2025-058 - Line 3 - Grinding noise
Vibration trended from 2 to 6 mm/s over one shift. Root cause: main drive bearing dry. Bearing BR-6310 replaced. Downtime 4 hours.

## INC-2026-004 - Line 2 - Flat product
QA found low carbonation. CO2 pressure 2.9 bar. Root cause: cracked fitting after regulator. Fitting replaced. Downtime 1 hour.

## INC-2026-017 - Line 2 - Speed drop
Filler speed fell to 50 bpm. Root cause: damaged PET bottles jammed the starwheel after a size changeover with wrong guide rails. Downtime 45 minutes.
""",
}
for name, text in MANUALS.items():
    (MANUALS_DIR / name).write_text(text)

# ---------------------------------------------------------------------------
# 5. Ground truth for evaluation
# ---------------------------------------------------------------------------
with open(EVALS_DIR / "faults.json", "w") as fh:
    json.dump(faults, fh, indent=2)

print(f"Wrote {log_count} log readings to data/logs.jsonl")
print(f"Wrote ERP database to data/erp.db ({len(orders)} orders, {len(schedule)} schedule rows)")
print(f"Wrote {len(MANUALS)} documents to data/manuals/")
print(f"Wrote {len(faults)} injected faults to evals/faults.json")
for f in faults:
    print(f"  {f['fault_id']} {f['line_id']} {f['fault_type']:<16} {f['start']}  ({f['duration_min']} min)")
