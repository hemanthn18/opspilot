"""
watcher.py - Agent 1: the Watcher

Reads sensor telemetry and flags anomalies using a rolling z-score.
No LLM is used here: statistics are fast, cheap and deterministic.
The LLM-based agents only run after the Watcher raises an anomaly.
"""

import json
import statistics
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS_PATH = ROOT / "data" / "logs.jsonl"

WINDOW = 60          # rolling baseline = last 60 normal readings (1 hour)
Z_THRESHOLD = 4.0    # how many standard deviations counts as abnormal
CONSECUTIVE = 3      # must be abnormal 3 readings in a row (filters noise spikes)
RECOVERY = 10        # 10 normal readings in a row = incident is over
MIN_STD = 0.05       # floor so a very flat signal doesn't produce huge z-scores
BASELINE_GATE = 2.5  # only readings within 2.5 std of normal may update the baseline

# Which direction is "bad" for each sensor
DIRECTION = {
    "temperature_c": "up",
    "vibration_mm_s": "up",
    "pressure_bar": "down",
    "filler_speed_bpm": "down",
}


@dataclass
class Anomaly:
    line_id: str
    sensor: str
    detected_at: str        # timestamp of the reading that confirmed the anomaly
    value: float            # sensor value at detection
    baseline_mean: float    # what "normal" looked like
    z_score: float
    error_code_seen: str | None  # did the machine itself raise an alarm yet?


class SensorMonitor:
    """Tracks one sensor on one line."""

    def __init__(self, line_id: str, sensor: str):
        self.line_id = line_id
        self.sensor = sensor
        self.baseline = deque(maxlen=WINDOW)
        self.abnormal_streak = 0
        self.normal_streak = 0
        self.in_incident = False

    def update(self, ts: str, value: float, error_code: str | None) -> Anomaly | None:
        # Need a full window of history before we can judge anything
        if len(self.baseline) < WINDOW:
            self.baseline.append(value)
            return None

        mean = statistics.fmean(self.baseline)
        std = max(statistics.stdev(self.baseline), MIN_STD)
        z = (value - mean) / std
        abnormal = z > Z_THRESHOLD if DIRECTION[self.sensor] == "up" else z < -Z_THRESHOLD

        if abnormal:
            self.abnormal_streak += 1
            self.normal_streak = 0
            # Do NOT add abnormal values to the baseline, or "normal" would drift
            if self.abnormal_streak == CONSECUTIVE and not self.in_incident:
                self.in_incident = True
                return Anomaly(self.line_id, self.sensor, ts, value,
                               round(mean, 2), round(z, 1), error_code)
        else:
            self.abnormal_streak = 0
            # Baseline gating: a slow drift (e.g. a chiller slowly failing) would
            # otherwise creep into the baseline and hide itself. Only clearly
            # normal readings are allowed to update what "normal" means.
            if abs(z) < BASELINE_GATE:
                self.baseline.append(value)
            if self.in_incident:
                self.normal_streak += 1
                if self.normal_streak >= RECOVERY:
                    self.in_incident = False
                    self.normal_streak = 0
        return None


def watch(logs_path: Path = LOGS_PATH):
    """Stream through the logs and yield each new anomaly as it is detected."""
    monitors: dict[tuple[str, str], SensorMonitor] = {}
    with open(logs_path) as fh:
        for raw in fh:
            reading = json.loads(raw)
            for sensor in DIRECTION:
                key = (reading["line_id"], sensor)
                if key not in monitors:
                    monitors[key] = SensorMonitor(*key)
                anomaly = monitors[key].update(reading["ts"], reading[sensor], reading["error_code"])
                if anomaly:
                    yield anomaly


if __name__ == "__main__":
    anomalies = list(watch())
    print(f"Watcher found {len(anomalies)} anomalies:\n")
    for a in anomalies:
        alarm = a.error_code_seen or "no machine alarm yet"
        print(f"  {a.detected_at}  {a.line_id}  {a.sensor:<17} value={a.value:<7} "
              f"baseline={a.baseline_mean:<7} z={a.z_score:<6} ({alarm})")

    # Save for the next agents
    out = ROOT / "data" / "anomalies.json"
    out.write_text(json.dumps([asdict(a) for a in anomalies], indent=2))
    print(f"\nSaved to {out.relative_to(ROOT)}")
