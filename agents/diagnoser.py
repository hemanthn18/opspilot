"""
diagnoser.py - Agent 2: the Diagnoser

Input:  an anomaly from the Watcher
Steps:  1) build a search query  2) retrieve manual sections (RAG)
        3) ask the LLM for a root cause, grounded ONLY in those sections
        4) validate the JSON output against a schema
Output: a Diagnosis with root cause, actions and citations
"""

import json
import sys
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.llm import chat  # noqa: E402
from tools.rag import search  # noqa: E402

SENSOR_WORDS = {
    "temperature_c": "product temperature rising",
    "vibration_mm_s": "vibration increasing on the filler drive",
    "pressure_bar": "CO2 pressure dropping",
    "filler_speed_bpm": "filler speed dropping, bottles per minute falling",
}


class Diagnosis(BaseModel):
    """The contract the LLM must follow (structured output)."""
    likely_error_code: str = Field(description="e.g. E-214, or UNKNOWN")
    root_cause: str
    confidence: float = Field(ge=0, le=1)
    recommended_actions: list[str]
    spare_part: str | None = None
    estimated_repair_hours: float | None = None
    sources: list[str] = Field(description="ids of the manual sections used")


SYSTEM_PROMPT = """You are a maintenance diagnostics agent for a beverage bottling plant.
Use ONLY the manual excerpts provided. Do not use outside knowledge.
If the excerpts do not explain the anomaly, set likely_error_code to "UNKNOWN",
confidence below 0.3, and recommend human inspection.
Always cite the excerpt ids you used in "sources".
Reply with a JSON object with exactly these keys:
likely_error_code, root_cause, confidence, recommended_actions, spare_part,
estimated_repair_hours, sources."""


def diagnose(anomaly: dict) -> Diagnosis:
    # 1) Turn the anomaly into a natural-language search query
    query = f"{SENSOR_WORDS[anomaly['sensor']]}, error code, likely causes and actions"

    # 2) Retrieval: top 3 manual sections
    hits = search(query, k=3)
    excerpts = "\n\n".join(f"[{h['id']}]\n{h['text']}" for h in hits)

    # 3) Generation, grounded in the retrieved text
    user_prompt = f"""Anomaly detected by the Watcher:
- Line: {anomaly['line_id']}
- Sensor: {anomaly['sensor']}
- Current value: {anomaly['value']} (normal baseline about {anomaly['baseline_mean']})
- z-score: {anomaly['z_score']}
- Machine error code raised yet: {anomaly['error_code_seen'] or 'none'}

Manual excerpts:
{excerpts}"""

    response = chat(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": user_prompt}],
        response_format={"type": "json_object"},  # JSON mode
    )
    raw = response.choices[0].message.content

    # 4) Validate: never trust LLM output blindly
    try:
        diagnosis = Diagnosis.model_validate_json(raw)
    except ValidationError as err:
        return Diagnosis(likely_error_code="UNKNOWN", root_cause=f"Invalid model output: {err.errors()[0]['msg']}",
                         confidence=0.0, recommended_actions=["Escalate to a maintenance engineer"], sources=[])

    # Guardrail: citations must be sections we actually retrieved
    retrieved_ids = {h["id"] for h in hits}
    diagnosis.sources = [s for s in diagnosis.sources if s in retrieved_ids]
    if not diagnosis.sources:
        diagnosis.confidence = min(diagnosis.confidence, 0.3)
    return diagnosis


if __name__ == "__main__":
    anomalies = json.loads((ROOT / "data" / "anomalies.json").read_text())
    results = []
    for a in anomalies:
        d = diagnose(a)
        results.append({"anomaly": a, "diagnosis": d.model_dump()})
        print(f"\n{a['line_id']} {a['sensor']} at {a['detected_at']}")
        print(f"  -> {d.likely_error_code}: {d.root_cause}")
        print(f"     confidence {d.confidence}, part {d.spare_part}, ~{d.estimated_repair_hours}h")
        print(f"     sources: {', '.join(d.sources)}")
    (ROOT / "data" / "diagnoses.json").write_text(json.dumps(results, indent=2))
    print("\nSaved to data/diagnoses.json")
