"""
app.py — FastAPI server exposing CustomerSupportEnv as an HTTP API.
Uses file-based session persistence for HuggingFace Spaces.
FIXED: /reset accepts empty body (task_id optional, defaults to 'billing_dispute_easy')
"""

import uuid
import os
import pickle
import json
from typing import Dict, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.environment import CustomerSupportEnv
from env.models import Action
from tasks.task_definitions import TASKS

app = FastAPI(
    title="CustomerSupportEnv",
    description="OpenEnv-compliant Customer Support Agent simulation environment",
    version="1.0.0",
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


@app.get("/")
def root():
    return {
        "name": "CustomerSupportEnv",
        "version": "1.0.0",
        "tasks": list(TASKS.keys()),
        "endpoints": ["/reset", "/step", "/state", "/grade", "/tasks", "/health"],
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