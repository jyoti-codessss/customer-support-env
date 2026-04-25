"""
app.py — FastAPI server exposing CustomerSupportEnv as an HTTP API.
Uses file-based session persistence for HuggingFace Spaces.
FIXED: /reset accepts empty body (task_id optional, defaults to 'billing_dispute_easy')

Includes:
  - /metrics    — Real-time agent performance dashboard
  - /metrics/report — Accepts metrics from inference.py
"""

import uuid
import os
import pickle
import json
import time
from typing import Dict, Optional, Any, List
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.environment import CustomerSupportEnv
from env.models import Action
from tasks.task_definitions import TASKS

app = FastAPI(
    title="CustomerSupportEnv",
    description="OpenEnv-compliant Customer Support Agent simulation environment with Advanced Multi-Agent Metrics",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSION_DIR = "/tmp/csenv_sessions"
os.makedirs(SESSION_DIR, exist_ok=True)

_memory_cache: Dict[str, CustomerSupportEnv] = {}

# ── Metrics Store ────────────────────────────────────────────────────────────

_metrics_store: Dict[str, Any] = {
    "total_tasks_attempted": 0,
    "tasks_completed": 0,
    "average_score": 0.0,
    "per_task_scores": {},
    "supervisor_interventions": 0,
    "supervisor_intervention_rate": 0.0,
    "self_improvement_deltas": {},
    "agent_confidence_avg": 0.0,
    "uptime_seconds": 0.0,
    "last_updated": None,
    "architecture": "hierarchical_multi_agent_v2",
    "layers": ["CustomerAgent (L1)", "SupervisorAgent (L2)", "OrchestratorAgent (L3)"],
}

_server_start_time = time.time()


# ── Session Management ───────────────────────────────────────────────────────

def _session_path(session_id: str) -> str:
    safe = session_id.replace("-", "")[:64]
    return os.path.join(SESSION_DIR, f"{safe}.pkl")


def save_session(session_id: str, env: CustomerSupportEnv):
    _memory_cache[session_id] = env
    try:
        with open(_session_path(session_id), "wb") as f:
            pickle.dump(env, f)
    except Exception:
        pass


def load_session(session_id: str) -> Optional[CustomerSupportEnv]:
    if session_id in _memory_cache:
        return _memory_cache[session_id]
    try:
        path = _session_path(session_id)
        if os.path.exists(path):
            with open(path, "rb") as f:
                env = pickle.load(f)
            _memory_cache[session_id] = env
            return env
    except Exception:
        pass
    return None


def delete_session(session_id: str):
    _memory_cache.pop(session_id, None)
    try:
        path = _session_path(session_id)
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


class StepRequest(BaseModel):
    session_id: str
    action: Action


# ── Core Endpoints (UNCHANGED) ──────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "CustomerSupportEnv",
        "version": "2.0.0",
        "architecture": "Hierarchical Multi-Agent (3-Layer)",
        "tasks": list(TASKS.keys()),
        "endpoints": ["/reset", "/step", "/state", "/grade", "/tasks", "/health", "/metrics"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks():
    return {
        task_id: {
            "difficulty": t["difficulty"],
            "initial_message": t["initial_user_message"][:100] + "...",
            "max_turns": t["max_turns"],
        }
        for task_id, t in TASKS.items()
    }


@app.post("/reset")
async def reset(request: Request):
    """
    Reset the environment.
    - Empty body → uses default task 'billing_dispute_easy'
    - {"task_id": "..."} → uses specified task
    """
    task_id = "billing_dispute_easy"

    try:
        body = await request.body()
        if body and body.strip():
            data = json.loads(body)
            if isinstance(data, dict):
                task_id = data.get("task_id", "billing_dispute_easy") or "billing_dispute_easy"
    except Exception:
        pass  # empty or invalid body → use default

    if task_id not in TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task: '{task_id}'. Available: {list(TASKS.keys())}"
        )

    session_id = str(uuid.uuid4())
    env = CustomerSupportEnv(task_id)
    obs = env.reset()
    save_session(session_id, env)

    return {
        "session_id": session_id,
        "observation": obs.dict(),
    }


@app.post("/step")
def step(request: StepRequest):
    env = load_session(request.session_id)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{request.session_id}' not found. Call /reset first."
        )

    obs, reward, done, info = env.step(request.action)

    if not done:
        save_session(request.session_id, env)
    else:
        delete_session(request.session_id)

    response = {
        "observation": obs.dict(),
        "reward": reward,
        "done": done,
        "info": info,
    }

    if done:
        score, breakdown = env.grade()
        response["final_score"] = score
        response["grade_breakdown"] = breakdown

    return response


@app.get("/state/{session_id}")
def get_state(session_id: str):
    env = load_session(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return env.state().dict()


@app.post("/grade/{session_id}")
def grade(session_id: str):
    env = load_session(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    score, breakdown = env.grade()
    return {"score": score, "breakdown": breakdown}


# ── Metrics Endpoints (NEW) ─────────────────────────────────────────────────

@app.get("/metrics")
def get_metrics():
    """
    Real-time agent performance dashboard.
    Returns metrics from the multi-agent inference system.
    """
    _metrics_store["uptime_seconds"] = round(time.time() - _server_start_time, 1)
    return {
        "status": "ok",
        "architecture": _metrics_store["architecture"],
        "layers": _metrics_store["layers"],
        "metrics": {
            "total_tasks_attempted": _metrics_store["total_tasks_attempted"],
            "tasks_completed": _metrics_store["tasks_completed"],
            "average_score": _metrics_store["average_score"],
            "per_task_scores": _metrics_store["per_task_scores"],
            "supervisor_interventions": _metrics_store["supervisor_interventions"],
            "supervisor_intervention_rate": _metrics_store["supervisor_intervention_rate"],
            "self_improvement_deltas": _metrics_store["self_improvement_deltas"],
            "agent_confidence_avg": _metrics_store["agent_confidence_avg"],
        },
        "server": {
            "uptime_seconds": _metrics_store["uptime_seconds"],
            "tasks_available": list(TASKS.keys()),
            "total_tasks_available": len(TASKS),
            "last_updated": _metrics_store["last_updated"],
        }
    }


@app.post("/metrics/report")
async def report_metrics(request: Request):
    """
    Accept metrics from the inference.py multi-agent system.
    Called after each task completes.
    """
    try:
        body = await request.body()
        if body:
            data = json.loads(body)
            if isinstance(data, dict):
                for key in _metrics_store:
                    if key in data:
                        _metrics_store[key] = data[key]
                _metrics_store["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                _metrics_store["tasks_completed"] = _metrics_store.get("total_tasks_attempted", 0)
        return {"status": "ok", "message": "Metrics updated"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ── Mount Gradio Demo ────────────────────────────────────────────────────────

try:
    import gradio as gr
    from demo import demo as gradio_demo
    app = gr.mount_gradio_app(app, gradio_demo, path="/demo")
except ImportError:
    pass  # gradio not installed — skip demo mount