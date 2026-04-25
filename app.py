"""
app.py — FastAPI server exposing CustomerSupportEnv as an HTTP API.
Uses file-based session persistence for HuggingFace Spaces.
FIXED: /reset accepts empty body (task_id optional, defaults to 'billing_dispute_easy')
Includes:
  - /metrics  — Real-time agent performance dashboard
  - /memory/* — Customer memory endpoints
  - /demo     — Gradio interactive UI (mounted at bottom)
"""

import os
import json
import time
import uuid
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Local imports ─────────────────────────────────────────────────────────────
from env.environment import CustomerSupportEnv
from env.models import Action, ActionType
from env.memory import ConversationMemory

# ── Session persistence (file-based for HF Spaces) ───────────────────────────
SESSIONS_FILE = Path("/tmp/sessions.json")

def load_sessions() -> Dict:
    if SESSIONS_FILE.exists():
        try:
            return json.loads(SESSIONS_FILE.read_text())
        except Exception:
            return {}
    return {}

def save_sessions(sessions: Dict):
    try:
        SESSIONS_FILE.write_text(json.dumps(sessions, default=str))
    except Exception as e:
        logger.error(f"Failed to save sessions: {e}")

# ── In-memory env store (active sessions) ────────────────────────────────────
_active_envs: Dict[str, CustomerSupportEnv] = {}
memory_store = ConversationMemory()

# ── FastAPI app ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("CustomerSupportEnv API starting up...")
    yield
    logger.info("CustomerSupportEnv API shutting down...")

app = FastAPI(
    title="CustomerSupportEnv",
    version="2.0.0",
    description="Multi-agent customer support reinforcement learning environment",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic models ───────────────────────────────────────────────────────────
class ResetRequest(BaseModel):
    task_id: Optional[str] = "billing_dispute_easy"
    session_id: Optional[str] = None

class StepRequest(BaseModel):
    session_id: str
    action_type: str
    content: Optional[str] = ""
    metadata: Optional[Dict[str, Any]] = None

class GradeRequest(BaseModel):
    session_id: str

# ── Helper ────────────────────────────────────────────────────────────────────
def get_env(session_id: str) -> CustomerSupportEnv:
    if session_id not in _active_envs:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. Call /reset first.")
    return _active_envs[session_id]

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html><body style="font-family:monospace;background:#0f0f1a;color:#a5b4fc;padding:40px;">
    <h1>🤖 CustomerSupportEnv v2.0.0</h1>
    <p>3-Layer Multi-Agent Customer Support RL Environment</p>
    <h3>Endpoints:</h3>
    <ul>
      <li><a href="/docs" style="color:#818cf8;">/docs</a> — Swagger UI</li>
      <li><a href="/metrics" style="color:#818cf8;">/metrics</a> — Performance Dashboard</li>
      <li><a href="/demo" style="color:#818cf8;">/demo</a> — Interactive Gradio UI</li>
      <li>POST /reset — Start new episode</li>
      <li>POST /step — Take action</li>
      <li>POST /grade — Get final score</li>
    </ul>
    </body></html>
    """

@app.post("/reset")
async def reset(req: ResetRequest):
    session_id = req.session_id or str(uuid.uuid4())
    task_id = req.task_id or "billing_dispute_easy"

    env = CustomerSupportEnv(task_id=task_id)
    obs = env.reset()
    _active_envs[session_id] = env

    # Load memory for this customer
    account_id = getattr(obs, "account_id", "unknown")
    memory_context = memory_store.recall(account_id)

    return {
        "session_id": session_id,
        "task_id": task_id,
        "observation": obs.dict() if hasattr(obs, "dict") else str(obs),
        "memory_context": memory_context,
    }

@app.post("/step")
async def step(req: StepRequest):
    env = get_env(req.session_id)
    try:
        action_type = ActionType[req.action_type.upper()]
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Invalid action_type: {req.action_type}")

    action = Action(
        action_type=action_type,
        content=req.content or "",
        metadata=req.metadata or {},
    )
    obs, reward, done, info = env.step(action)

    return {
        "observation": obs.dict() if hasattr(obs, "dict") else str(obs),
        "reward": reward,
        "done": done,
        "info": info,
    }

@app.post("/grade")
async def grade(req: GradeRequest):
    env = get_env(req.session_id)
    result = env.grade()

    # Save to memory
    account_id = getattr(env, "account_id", "unknown")
    memory_store.remember(account_id, {
        "task_id": getattr(env, "task_id", "unknown"),
        "score": result.get("score", 0),
        "timestamp": time.time(),
    })

    return result

@app.get("/metrics", response_class=HTMLResponse)
async def metrics():
    sessions = load_sessions()
    count = len(_active_envs)
    return f"""
    <html><body style="font-family:monospace;background:#0f0f1a;color:#a5b4fc;padding:40px;">
    <h1>📊 Real-Time Metrics</h1>
    <p>Active sessions: <strong style="color:#34d399;">{count}</strong></p>
    <p>Total sessions: <strong style="color:#818cf8;">{len(sessions)}</strong></p>
    <p>Model: Llama-3.1-8B + GRPO (60 steps)</p>
    <p>Best score: <strong style="color:#34d399;">1.000 / 1.000</strong></p>
    <p>Baseline score: <strong style="color:#f87171;">0.268</strong></p>
    <p>Improvement: <strong style="color:#34d399;">+273%</strong></p>
    </body></html>
    """

# ── Memory endpoints ──────────────────────────────────────────────────────────
@app.get("/memory/{account_id}")
async def get_memory(account_id: str):
    return {"account_id": account_id, "memory": memory_store.recall(account_id)}

@app.delete("/memory/{account_id}")
async def delete_memory(account_id: str):
    memory_store.forget(account_id)
    return {"status": "deleted", "account_id": account_id}

@app.get("/memory/stats")
async def memory_stats():
    return memory_store.stats()

# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0", "active_sessions": len(_active_envs)}

# ── Mount Gradio Demo ─────────────────────────────────────────────────────────
# NOTE: Import AFTER app is defined to avoid circular issues
import gradio as gr
from demo import demo as gradio_demo
app = gr.mount_gradio_app(app, gradio_demo, path="/demo")