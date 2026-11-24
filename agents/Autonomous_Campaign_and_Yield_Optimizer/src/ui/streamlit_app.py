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
from src.agent.scenarios import SCENARIOS, run_all_scenarios, run_scenario
from src.utils.config import out_dir
from src.utils.logger import read_jsonl
from src.utils.visualizer import flow_diagram, scenario_chart

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
tab_sc, tab_flow, tab_data, tab_gpu, tab_val, tab_hitl = st.tabs(
    ["Scenario Studio", "Flow Visualizer", "Data Explorer", "GPU Metrics", "Validation", "HITL Studio"])

BADGE = {"PASS": ":green-background[PASS]", "PASS-HITL": ":orange-background[PASS + HITL]", "WARN": ":yellow-background[WARN]",
         "FAIL": ":red-background[FAIL]", "SKIPPED": ":gray-background[SKIPPED]"}


def base_context():
    """A run whose models are trained (step >= 4). Starts an auto-approved run if none exists."""
    for rid_ in reversed(list(AgentService._runs)):
        s_ = AgentService.status(rid_)
        if s_.get("step", 0) >= 4 and not s_.get("halted"):
            return AgentService.context(rid_)
    with st.spinner("Training GPU models on the synthetic world (about 30 s)..."):
        s_ = AgentService.start(auto_approve=True)
    st.session_state.rid = s_["run_id"]
    return AgentService.context(s_["run_id"])


def load_saved():
    p = out_dir("scenarios", "scenario_results.json")
    return json.loads(p.read_text()) if p.exists() else {}


with tab_sc:
    st.subheader("End-to-end scenario studio")
    st.caption("Each scenario pushes a market situation through all six steps (data, features + drift, GPU models, "
               "optimize + guardrails, offline replay + OPE, execute / HITL) using the models of a trained run. "
               "Results are written to outputs/scenarios/ and outputs/charts/scenario_*.png.")
    if "scenarios" not in st.session_state:
        st.session_state.scenarios = load_saved()
    colA, colB = st.columns([1, 2])
    with colA:
        labels = [s_["label"] for s_ in SCENARIOS]
        choice = st.radio("Pick an example", labels, key="sc_choice")
        sc_def = SCENARIOS[labels.index(choice)]
        if st.button("Run selected scenario"):
            ctx_ = base_context()
            res_ = run_scenario(ctx_, sc_def)
            scenario_chart(res_)
            st.session_state.scenarios = {**st.session_state.scenarios, sc_def["id"]: res_}
        if st.button("Run ALL scenarios", type="primary"):
            ctx_ = base_context()
            allres = run_all_scenarios(ctx_)
            for r_ in allres.values():
                scenario_chart(r_)
            st.session_state.scenarios = allres
            st.rerun()
    with colB:
        st.markdown(f"**Story.** {sc_def['story']}")
        st.markdown(f"**What to expect.** {sc_def['expect']}")
    res_all = st.session_state.scenarios
    if res_all:
        st.markdown("### All scenarios compared")
        rows = []
        for r_ in res_all.values():
            rp, pl = r_.get("replay"), r_.get("plan")
            rows.append({"scenario": r_["label"], "outcome": "HALTED" if r_["halted"] else "completed",
                         "discount": f"{pl['discount']:.0%}" if pl else "-",
                         "lift vs static %": round(rp["lift_vs_static_pct"], 1) if rp else None,
                         "lift vs heuristic %": round(rp["lift_vs_heuristic_pct"], 1) if rp else None,
                         "guardrail violations": rp["guardrail_violations"] if rp else "-",
                         "HITL": r_["hitl"]["status"]})
        df_sc = pd.DataFrame(rows)
        st.dataframe(df_sc, hide_index=True, width="stretch")
        st.bar_chart(df_sc.set_index("scenario")[["lift vs static %", "lift vs heuristic %"]])
    r_ = res_all.get(sc_def["id"])
    if r_:
        st.markdown(f"### Flow: {r_['label']}")
        cols = st.columns(6)
        for c_, stp in zip(cols, r_["steps"]):
            c_.markdown("**" + stp["name"] + "**\n\n" + BADGE.get(stp["status"], stp["status"]))
        for stp in r_["steps"]:
            st.markdown(f"- **{stp['name']}** - {BADGE.get(stp['status'], stp['status'])} {stp['detail']}")
        if r_["halted"]:
            st.error(f"HALT: {r_['halt_reason']}")
        else:
            pl = r_["plan"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Discount tier", f"{pl['discount']:.0%}")
            c2.metric("Lift vs static", f"{r_['replay']['lift_vs_static_pct']:+.1f}%")
            c3.metric("Lift vs heuristic", f"{r_['replay']['lift_vs_heuristic_pct']:+.1f}%")
            c4.metric("HITL", r_["hitl"]["status"].replace("_", " "))
            st.dataframe(pd.DataFrame({"previous": pl["prev_budgets"], "proposed": pl["budgets"]}).assign(
                change_pct=lambda d: (d.proposed / d.previous - 1) * 100).round(1), width="stretch")
            if r_["hitl"]["reasons"]:
                st.warning("Needs human approval: " + ", ".join(r_["hitl"]["reasons"]))
            with st.expander("Net margin by discount tier (planner's candidates)"):
                st.dataframe(pd.DataFrame(r_["net_by_tier"]).T.round(3), width="stretch")
            with st.expander("Gate 4 checks"):
                st.dataframe(pd.DataFrame(r_["gate4"])[["name", "passed", "value", "threshold"]].astype(str), width="stretch")
        png = out_dir("charts", f"scenario_{r_['id']}.png")
        if png.exists():
            st.image(str(png), width="stretch")
    else:
        st.info("Pick an example and press **Run selected scenario**, or **Run ALL scenarios**.")


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
