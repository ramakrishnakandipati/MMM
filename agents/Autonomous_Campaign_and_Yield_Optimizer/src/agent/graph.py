"""Compiled LangGraph state machine with conditional halt routing and a HITL interrupt,
plus a small AgentService used by run_all.py, Flask and Streamlit (no logic duplication)."""
from __future__ import annotations

import threading
from typing import Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from ..utils.logger import reset_logs
from ..validation.step_validators import reset_report
from .nodes import make_nodes
from .state import AgentState, RunContext, initial_state

def to_py(o):
    """Recursively convert numpy/pandas scalars & containers to plain Python (checkpoint-safe)."""
    import numpy as np
    if isinstance(o, dict):
        return {(k if isinstance(k, (str, int)) else str(k)): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def _safe(fn):
    def wrapped(state):
        return to_py(fn(state))
    wrapped.__name__ = getattr(fn, "__name__", "node")
    return wrapped


STAGES = ["ingest", "features", "models", "optimize", "evaluate", "review", "execute"]


def _route(next_node: str):
    def f(state: AgentState) -> str:
        return "halt" if state.get("halted") else next_node
    return f


def build_graph(ctx: RunContext, checkpointer=None):
    nodes = make_nodes(ctx)
    g = StateGraph(AgentState)
    for name, fn in nodes.items():
        g.add_node(name, _safe(fn))
    g.add_edge(START, "ingest")
    for a, b in zip(STAGES, STAGES[1:]):
        g.add_conditional_edges(a, _route(b), {b: b, "halt": "halt"})
    g.add_conditional_edges("execute", lambda s: "halt" if s.get("halted") else END, {END: END, "halt": "halt"})
    g.add_edge("halt", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())


class AgentService:
    """Holds live runs (graph + RunContext) so UIs can start, inspect, and resume HITL runs."""
    _runs: dict[str, tuple] = {}
    _lock = threading.Lock()
    _errors: dict[str, str] = {}

    @classmethod
    def create(cls, auto_approve: bool = True, device: Optional[str] = None, reset: bool = True) -> str:
        if reset:
            reset_report()
            reset_logs("agent_decision_audit.jsonl", "pipeline_execution.jsonl")
            from ..utils.config import out_dir
            t = out_dir("metrics", "step_timings.csv")
            if t.exists():
                t.unlink()
        ctx = RunContext(auto_approve=auto_approve)
        if device:
            ctx.cfg["device"] = device
            from ..utils.device import get_device_manager
            ctx.dm = get_device_manager(device)
        graph = build_graph(ctx)
        cfg = {"configurable": {"thread_id": ctx.run_id}}
        with cls._lock:
            cls._runs[ctx.run_id] = (graph, ctx, cfg)
        return ctx.run_id

    @classmethod
    def execute(cls, run_id: str) -> dict:
        graph, ctx, cfg = cls._runs[run_id]
        graph.invoke(initial_state(ctx), cfg)
        return cls.status(run_id)

    @classmethod
    def start(cls, auto_approve: bool = True, device: Optional[str] = None, reset: bool = True) -> dict:
        return cls.execute(cls.create(auto_approve, device, reset))

    @classmethod
    def start_async(cls, auto_approve: bool = True, device: Optional[str] = None) -> str:
        rid = cls.create(auto_approve, device)
        threading.Thread(target=cls._safe_exec, args=(rid,), daemon=True).start()
        return rid

    @classmethod
    def _safe_exec(cls, rid: str):
        try:
            cls.execute(rid)
        except Exception as e:  # surfaced via status()
            cls._errors[rid] = f"{type(e).__name__}: {e}"

    @classmethod
    def status(cls, run_id: str) -> dict:
        graph, ctx, cfg = cls._runs[run_id]
        snap = graph.get_state(cfg)
        pending = [i.value for t in snap.tasks for i in t.interrupts]
        v = dict(snap.values)
        v["run_id"] = run_id
        v["error"] = cls._errors.get(run_id)
        v["waiting_for_human"] = bool(pending)
        v["pending_interrupt"] = pending[0] if pending else None
        v["next"] = list(snap.next)
        return v

    @classmethod
    def resume(cls, run_id: str, decision: str, reviewer: str = "human", note: str = "") -> dict:
        graph, ctx, cfg = cls._runs[run_id]
        graph.invoke(Command(resume={"decision": decision, "reviewer": reviewer, "note": note}), cfg)
        return cls.status(run_id)

    @classmethod
    def context(cls, run_id: str) -> RunContext:
        return cls._runs[run_id][1]

    @classmethod
    def latest(cls) -> Optional[str]:
        return list(cls._runs)[-1] if cls._runs else None
