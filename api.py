"""
api.py - REST API for OpsPilot (FastAPI)

Endpoints
  GET  /health                      liveness check (used by Azure later)
  POST /incidents/run               run the Watcher and start the graph for each anomaly (background task)
  GET  /incidents                   list incidents with status
  GET  /incidents/{id}              full state of one incident
  GET  /incidents/{id}/trend        sensor readings around detection (for the chart)
  POST /incidents/{id}/decision     approve / reject -> resumes the paused graph

Run:  uvicorn api:app --reload
Docs: http://localhost:8000/docs   (auto-generated Swagger UI)
"""

import json
import threading
from dataclasses import asdict
from datetime import datetime, timedelta

from fastapi import BackgroundTasks, FastAPI, HTTPException
from langgraph.types import Command
from pydantic import BaseModel

from agents.watcher import LOGS_PATH, watch
from graph import build_graph

app = FastAPI(title="OpsPilot API", version="1.0")
graph = build_graph()          # one compiled graph + checkpointer for the whole app
incidents: dict[str, dict] = {}  # incident_id -> {"config": ..., "anomaly": ...}
run_lock = threading.Lock()


class Decision(BaseModel):
    approved: bool
    by: str = "unknown"
    reason: str | None = None


def incident_id(a: dict) -> str:
    return f"{a['line_id']}-{a['sensor']}-{a['detected_at']}".replace(":", "")


def status_of(incident: dict) -> dict:
    snapshot = graph.get_state(incident["config"])
    values = snapshot.values or {}
    if snapshot.interrupts:
        status = "pending_approval"
    elif values.get("outcome"):
        status = values["outcome"].split(" - ")[0]      # approved / rejected / escalated
    else:
        status = "processing"
    return {"status": status, **values}


def run_incident(iid: str):
    """Runs the graph until it finishes or pauses at the approval gate."""
    graph.invoke({"anomaly": incidents[iid]["anomaly"]}, incidents[iid]["config"])


def run_all():
    with run_lock:                                  # avoid two overlapping runs
        for a in (asdict(x) for x in watch()):
            iid = incident_id(a)
            if iid in incidents:                    # idempotent: skip incidents already seen
                continue
            incidents[iid] = {"anomaly": a, "config": {"configurable": {"thread_id": iid}}}
            try:
                run_incident(iid)
            except Exception as err:                # one failure must not stop the others
                incidents[iid]["error"] = str(err)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/incidents/run", status_code=202)
def start_run(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_all)
    return {"message": "Watcher started; incidents will appear as they are processed"}


@app.get("/incidents")
def list_incidents():
    out = []
    for iid, inc in incidents.items():
        s = status_of(inc)
        d = s.get("diagnosis") or {}
        p = s.get("action_plan") or {}
        out.append({"id": iid, "status": s["status"] if "error" not in inc else "error",
                    "line_id": inc["anomaly"]["line_id"], "sensor": inc["anomaly"]["sensor"],
                    "detected_at": inc["anomaly"]["detected_at"],
                    "error_code": d.get("likely_error_code"), "priority": p.get("priority"),
                    "ticket_id": s.get("ticket_id")})
    return sorted(out, key=lambda x: x["detected_at"])


@app.get("/incidents/{iid}")
def get_incident(iid: str):
    if iid not in incidents:
        raise HTTPException(404, "incident not found")
    return {"id": iid, **status_of(incidents[iid]), "error": incidents[iid].get("error")}


@app.get("/incidents/{iid}/trend")
def get_trend(iid: str, minutes_before: int = 90, minutes_after: int = 30):
    if iid not in incidents:
        raise HTTPException(404, "incident not found")
    a = incidents[iid]["anomaly"]
    t = datetime.fromisoformat(a["detected_at"])
    lo, hi = (t - timedelta(minutes=minutes_before)).isoformat(), (t + timedelta(minutes=minutes_after)).isoformat()
    points = []
    with open(LOGS_PATH) as fh:
        for raw in fh:
            r = json.loads(raw)
            if r["line_id"] == a["line_id"] and lo <= r["ts"] <= hi:
                points.append({"ts": r["ts"], "value": r[a["sensor"]]})
    return {"sensor": a["sensor"], "detected_at": a["detected_at"], "points": points}


@app.post("/incidents/{iid}/decision")
def decide(iid: str, decision: Decision):
    if iid not in incidents:
        raise HTTPException(404, "incident not found")
    if status_of(incidents[iid])["status"] != "pending_approval":
        raise HTTPException(409, "incident is not waiting for approval")   # 409 = conflict
    graph.invoke(Command(resume=decision.model_dump()), incidents[iid]["config"])
    return get_incident(iid)
