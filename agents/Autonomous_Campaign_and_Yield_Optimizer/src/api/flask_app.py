"""Flask REST + HTML control portal for the Autonomous Campaign & Yield Optimizer."""
from __future__ import annotations

import json
import time

from flask import Flask, jsonify, render_template, request, send_from_directory

from ..agent.graph import AgentService
from ..utils.config import out_dir
from ..utils.logger import read_jsonl, write_jsonl
from ..utils.visualizer import FLOW


def _json_file(*parts):
    p = out_dir(*parts)
    return json.loads(p.read_text()) if p.exists() else {}


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    @app.before_request
    def _t0():
        request._t0 = time.perf_counter()

    @app.after_request
    def _log(resp):
        if not request.path.startswith("/static"):
            write_jsonl("flask_access.jsonl", {"method": request.method, "path": request.path, "status": resp.status_code,
                                               "ms": round((time.perf_counter() - getattr(request, "_t0", time.perf_counter())) * 1000, 2)})
        return resp

    def _rid():
        return request.args.get("run_id") or (request.get_json(silent=True) or {}).get("run_id") or AgentService.latest()

    @app.get("/")
    def index():
        return render_template("index.html", steps=[s.replace("\n", " ") for s in FLOW])

    @app.get("/health")
    def health():
        return jsonify(ok=True)

    @app.post("/api/run")
    def run():
        body = request.get_json(silent=True) or {}
        rid = AgentService.start_async(auto_approve=bool(body.get("auto_approve", True)), device=body.get("device"))
        return jsonify(run_id=rid), 202

    @app.get("/api/status")
    def status():
        rid = _rid()
        if not rid:
            return jsonify(run_id=None, gates={}, step=0, running=False)
        st = AgentService.status(rid)
        st["running"] = bool(st["next"]) and not st["waiting_for_human"] and not st.get("halted") and not st.get("error")
        return jsonify(st)

    @app.get("/api/step/<int:n>")
    def step(n: int):
        rep = _json_file("validation", "step_validation_report.json")
        g = rep.get("gates", {}).get(str(n))
        if g is None:
            return jsonify(error=f"gate {n} not run"), 404
        return jsonify(g)

    @app.get("/api/validation")
    def validation():
        return jsonify(_json_file("validation", "step_validation_report.json"))

    @app.get("/api/metrics")
    def metrics():
        return jsonify(_json_file("metrics", "system_metrics.json"))

    @app.get("/api/telemetry")
    def telemetry():
        return jsonify(_json_file("metrics", "gpu_telemetry.json"))

    @app.get("/api/audit")
    def audit():
        return jsonify(read_jsonl("agent_decision_audit.jsonl")[-50:])

    @app.get("/api/hitl/pending")
    def hitl_pending():
        rid = _rid()
        if not rid:
            return jsonify(pending=None)
        st = AgentService.status(rid)
        return jsonify(run_id=rid, pending=st["pending_interrupt"], waiting=st["waiting_for_human"])

    def _decide(decision: str):
        body = request.get_json(silent=True) or {}
        rid = _rid()
        if not rid or not AgentService.status(rid)["waiting_for_human"]:
            return jsonify(error="no pending approval"), 409
        st = AgentService.resume(rid, decision, body.get("reviewer", "portal-user"), body.get("note", ""))
        return jsonify(hitl=st.get("hitl"), halted=st.get("halted"), step=st.get("step"))

    @app.post("/api/hitl/approve")
    def approve():
        return _decide("approve")

    @app.post("/api/hitl/reject")
    def reject():
        return _decide("reject")

    @app.get("/api/charts")
    def charts():
        return jsonify(sorted(p.name for p in out_dir("charts").glob("*.png")))

    @app.get("/charts/<path:name>")
    def chart(name):
        return send_from_directory(out_dir("charts"), name)

    return app


if __name__ == "__main__":
    import os
    create_app().run(host="127.0.0.1", port=int(os.environ.get("FLASK_PORT", 5000)), debug=False, threaded=True)
