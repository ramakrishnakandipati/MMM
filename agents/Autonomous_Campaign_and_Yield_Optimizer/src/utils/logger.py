"""Structured JSONL logging, stage timers and audit trail."""
from __future__ import annotations

import csv
import json
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone

from .config import out_dir

_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def write_jsonl(filename: str, record: dict) -> None:
    path = out_dir("logs", filename)
    rec = {"ts": _now(), **record}
    with _LOCK, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def log_event(stage: str, message: str, **kw) -> None:
    write_jsonl("pipeline_execution.jsonl", {"stage": stage, "message": message, **kw})


def audit(actor: str, decision: str, **kw) -> None:
    """Agent decision audit trail (also records LLM/mock source)."""
    write_jsonl("agent_decision_audit.jsonl", {"actor": actor, "decision": decision, **kw})


def read_jsonl(filename: str) -> list[dict]:
    path = out_dir("logs", filename)
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def reset_logs(*names: str) -> None:
    for n in names:
        p = out_dir("logs", n)
        if p.exists():
            p.unlink()


@contextmanager
def stage_timer(stage: str, device_manager=None):
    """Times a stage, appends to step_timings.csv and the pipeline log."""
    if device_manager is not None:
        device_manager.sync()
    t0 = time.perf_counter()
    rec = {"stage": stage, "seconds": None}
    try:
        yield rec
    finally:
        if device_manager is not None:
            device_manager.sync()
        rec["seconds"] = round(time.perf_counter() - t0, 4)
        path = out_dir("metrics", "step_timings.csv")
        new = not path.exists()
        with _LOCK, open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["timestamp", "stage", "seconds"])
            w.writerow([_now(), stage, rec["seconds"]])
        log_event(stage, "stage_complete", seconds=rec["seconds"])
