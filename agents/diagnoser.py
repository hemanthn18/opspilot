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

from pydantic import BaseModel, Field, ValidationError, field_validator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.llm import chat_structured  # noqa: E402
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
    confidence: float = Field(description="0.0 to 1.0")
    recommended_actions: list[str]
    spare_part: str | None
    estimated_repair_hours: float | None
    sources: list[str] = Field(description="ids of the manual sections used, copied exactly")

    # Defensive parsing: accept a single string where a list is expected
    @field_validator("recommended_actions", "sources", mode="before")
    @classmethod
    def string_to_list(cls, value):
        return [value] if isinstance(value, str) else value

    # Keep confidence inside 0..1 even if the model drifts
    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, value):
        return max(0.0, min(1.0, value))


SYSTEM_PROMPT = """You are a maintenance diagnostics agent for a beverage bottling plant.
Use ONLY the manual excerpts provided. Do not use outside knowledge.
If the excerpts do not explain the anomaly, set likely_error_code to "UNKNOWN",
confidence below 0.3, and recommend human inspection.
Always cite the excerpt ids you used in "sources", copied exactly as shown in square brackets."""


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

    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}]

    # 4) Structured Outputs: the API enforces the Diagnosis schema,
    #    then Pydantic validates again (never trust LLM output blindly)
    try:
        diagnosis, raw = chat_structured(messages, Diagnosis)
        if diagnosis is None:
            raise ValueError(f"model refused or returned nothing: {raw}")
    except (ValidationError, ValueError) as err:
        print(f"  [warn] invalid model output: {err}")
        return Diagnosis(likely_error_code="UNKNOWN", root_cause="Invalid model output",
                         confidence=0.0, recommended_actions=["Escalate to a maintenance engineer"],
                         spare_part=None, estimated_repair_hours=None, sources=[])

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
