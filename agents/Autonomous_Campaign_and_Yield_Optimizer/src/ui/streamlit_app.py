"""Streamlit Analytics & HITL Studio.  Run:  streamlit run src/ui/streamlit_app.py"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.agent.graph import AgentService
from src.utils.config import out_dir
from src.utils.logger import read_jsonl
from src.utils.visualizer import flow_diagram

st.set_page_config(page_title="Campaign & Yield Optimizer Studio", layout="wide", page_icon="📈")


def jload(*parts):
    p = out_dir(*parts)
    return json.loads(p.read_text()) if p.exists() else {}


st.title("Autonomous Campaign & Yield Optimizer — Studio")
if "rid" not in st.session_state:
    st.session_state.rid = AgentService.latest()

with st.sidebar:
    st.header("Run agent cycle")
    auto = st.checkbox("Auto-approve HITL (batch)", value=False,
                       help="Off = pause for human approval before execution")
    if st.button("▶ Run live cycle", type="primary"):
        st.session_state.rid = AgentService.start_async(auto_approve=auto)
        st.rerun()
    if st.button("⟳ Refresh"):
        st.rerun()
    rid = st.session_state.rid
    status = AgentService.status(rid) if rid else {}
    if rid:
        state = ("error" if status.get("error") else "awaiting approval" if status.get("waiting_for_human")
                 else "halted" if status.get("halted") else "complete" if status.get("step") == 6 else "running")
        st.write(f"**Run** `{rid}` — **{state}** (step {status.get('step', 0)}/6)")
        if state == "running":
            time.sleep(1.5)
            st.rerun()
    else:
        st.info("No run yet — start one.")

report = jload("validation", "step_validation_report.json")
metrics = jload("metrics", "system_metrics.json")
tab_flow, tab_data, tab_gpu, tab_val, tab_hitl = st.tabs(["Flow Visualizer", "Data Explorer", "GPU Metrics", "Validation", "HITL Studio"])

with tab_flow:
    gates = status.get("gates", {}) if rid else {}
    flow_diagram({int(k): ("pass" if v["passed"] else "fail") for k, v in gates.items()})
    st.image(str(out_dir("charts", "pipeline_flow.png")), width="stretch")
    if status.get("evaluation"):
        e = status["evaluation"]
        c = st.columns(4)
        c[0].metric("Lift vs static", f"{e['lift_vs_static_pct']:.1f}%")
        c[1].metric("Lift vs heuristic", f"{e['lift_vs_heuristic_pct']:.1f}%")
        c[2].metric("DR OPE lift", f"{e['dr_lift']:.1%}", help=f"95% CI {e['ci'][0]:.1%} .. {e['ci'][1]:.1%}")
        c[3].metric("Guardrail violations", e["guardrail_violations"])
    if status.get("plan"):
        p = status["plan"]
        st.subheader("Proposed plan")
        st.dataframe(pd.DataFrame({"previous": p["prev_budgets"], "proposed": p["budgets"]}).assign(
            change_pct=lambda d: (d.proposed / d.previous - 1) * 100).round(1))
        st.caption(f"Discount tier {p['discount']:.0%} · creative {p['creative']} · unit margin {p['unit_margin']:.2f} · "
                   f"{p['solver']} ({p['status']}) · clamps: {', '.join(p['clamps']) or 'none'}")
    for name in ("allocation.png", "response_curves.png", "benchmark_regimes.png", "ope_lift.png"):
        pth = out_dir("charts", name)
        if pth.exists():
            st.image(str(pth), caption=name, width="stretch")
    for k, label in (("diagnosis", "LLM diagnosis"), ("visual", "LLM visual QA")):
        if status.get(k):
            with st.expander(f"{label} [{status[k]['meta']['source']}]"):
                st.json(status[k])

with tab_data:
    p = out_dir("data", "features_snapshot.parquet")
    if p.exists():
        df = pd.read_parquet(p)
        st.write(f"{len(df):,} daily rows × {df.shape[1]} columns")
        sc = [c for c in df.columns if c.startswith("spend_")]
        st.subheader("Daily spend by channel"); st.line_chart(df.set_index("day_idx")[sc])
        st.subheader("Observed conversions & utilisation")
        st.line_chart(df.set_index("day_idx")[["conversions"]]); st.line_chart(df.set_index("day_idx")[["inventory_util"]])
        st.dataframe(df.tail(30))
        if (out_dir("charts", "drift_psi.png")).exists():
            st.image(str(out_dir("charts", "drift_psi.png")))
    else:
        st.info("Run a cycle to generate data.")

with tab_gpu:
    tele = jload("metrics", "gpu_telemetry.json")
    if tele:
        c = st.columns(4)
        c[0].metric("Device", tele.get("device_name", "-"))
        c[1].metric("VRAM (MB)", tele.get("vram_mb", 0))
        c[2].metric("CUDA", tele.get("cuda_version") or "n/a")
        c[3].metric("OPE bootstrap (ms)", tele.get("ope_bootstrap_ms", "-"))
        st.json(tele)
    tp = out_dir("metrics", "step_timings.csv")
    if tp.exists():
        t = pd.read_csv(tp)
        st.subheader("Step timings (s)"); st.bar_chart(t.groupby("stage")["seconds"].last())
    if metrics:
        st.subheader("System metrics"); st.json(metrics)

with tab_val:
    if report:
        s = report["summary"]
        st.metric("Gates passed", f"{s['gates_passed']}/6", delta=f"pass rate {s['pass_rate']:.0%}")
        for g in sorted(report["gates"].values(), key=lambda x: x["gate"]):
            with st.expander(f"Gate {g['gate']} — {g['name']} — {'✅ PASS' if g['passed'] else '❌ FAIL'}", expanded=not g["passed"]):
                st.dataframe(pd.DataFrame(g["checks"])[["name", "passed", "value", "threshold", "critical", "detail"]].astype(str))
    else:
        st.info("No validation report yet.")
    st.subheader("Agent decision audit")
    st.dataframe(pd.DataFrame(read_jsonl("agent_decision_audit.jsonl")).astype(str).tail(40))

with tab_hitl:
    if rid and status.get("waiting_for_human"):
        st.warning("Human approval required before the plan is executed.")
        st.json(status["pending_interrupt"])
        note = st.text_input("Reviewer note")
        c1, c2 = st.columns(2)
        if c1.button("✅ Approve", type="primary"):
            AgentService.resume(rid, "approve", "streamlit-user", note); st.rerun()
        if c2.button("⛔ Reject"):
            AgentService.resume(rid, "reject", "streamlit-user", note); st.rerun()
    elif rid:
        st.write("HITL status:", status.get("hitl"))
        st.caption("Untick 'Auto-approve HITL' in the sidebar and run a cycle to review a plan here.")
    else:
        st.info("Start a run first.")
