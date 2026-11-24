"""Six pipeline-stage LangGraph nodes (+ HITL review and halt), each with pre/post validation hooks."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from langgraph.types import interrupt

from ..data.state_builder import build_features, spend_cols, split_windows
from ..data.synthetic_generator import CHANNELS, CTX_COLS, generate_all
from ..evaluation.benchmark_runner import run_benchmark
from ..evaluation.metrics_aggregator import aggregate
from ..evaluation.ope_evaluator import evaluate_policy_ope
from ..guardrails.policy_engine import GuardrailEngine
from ..llm.client import LocalLLM
from ..models.contextual_bandit import BatchedLinUCB
from ..models.response_curves import ResponseModel
from ..models.uplift_elasticity import UpliftModel
from ..utils import visualizer as viz
from ..utils.config import out_dir
from ..utils.logger import audit, log_event, read_jsonl, stage_timer
from ..validation import step_validators as sv
from ..validation.contracts import BanditArtifact, ResponseArtifact, UpliftArtifact
from .state import AgentState, RunContext
from .tools import MockAdPlatformAPI, MockPricingAPI, Planner

SEVERE = "critical"


# ------------------------------------------------------------------------ helpers
def _gate_update(state: AgentState, result, step: int) -> dict:
    """Post-validation hook: persist gate, embed status into state, request halt on critical failure."""
    sv.record(result)
    gates = dict(state.get("gates", {}))
    gates[str(result.gate)] = {"passed": result.passed, "severity": result.severity,
                               "failed": [c.name for c in result.checks if not c.passed]}
    upd = {"gates": gates, "step": step}
    if not result.passed:
        failed = [c.name for c in result.checks if (not c.passed) and c.critical]
        upd.update(halted=True, halt_reason=f"Gate {result.gate} failed: {failed}")
    return upd


def _pre(state: AgentState, required_step: int, name: str) -> str | None:
    """Pre-validation hook: previous stage must have completed and passed."""
    if state.get("halted"):
        return state.get("halt_reason", "halted")
    if state.get("step", 0) < required_step:
        return f"{name}: prerequisite step {required_step} not completed"
    g = state.get("gates", {}).get(str(required_step))
    if required_step > 0 and (not g or not g["passed"]):
        return f"{name}: prerequisite gate {required_step} not passed"
    return None


def _blocked(reason: str) -> dict:
    return {"halted": True, "halt_reason": reason}


# ------------------------------------------------------------------------ Step 1
def make_nodes(ctx: RunContext) -> dict:
    cfg, dm = ctx.cfg, ctx.dm

    def ingest(state: AgentState) -> dict:
        with stage_timer("1_ingest", dm):
            data = generate_all(cfg)
            ctx.data.update(data)
            daily = data["daily"]
            daily.to_parquet(out_dir("data", "raw_daily.parquet"), index=False)
            res = sv.gate1_data_integrity(daily, cfg)
            audit("ingest", "synthetic data generated", rows=len(daily), run_id=ctx.run_id)
            return _gate_update(state, res, 1)

    # -------------------------------------------------------------------- Step 2
    def features(state: AgentState) -> dict:
        if (why := _pre(state, 1, "features")):
            return _blocked(why)
        with stage_timer("2_features", dm):
            d = cfg["data"]
            feat, spec = build_features(ctx.data["daily"])
            win = split_windows(feat, d["train_end"], d["live_end"])
            ctx.data.update(features=feat, spec=spec, windows=win)
            feat.to_parquet(out_dir("data", "features_snapshot.parquet"), index=False)
            res, rep = sv.gate2_features_drift(feat, spec, win["train"], win["live"])
            ctx.cache["drift"] = rep
            viz.data_distributions(ctx.data["daily"], spec["channels"])
            viz.drift_chart(rep)
            audit("features", "features built; drift checked", max_psi=rep["_max_psi"])
            return _gate_update(state, res, 2)

    # -------------------------------------------------------------------- Step 3
    def models(state: AgentState) -> dict:
        if (why := _pre(state, 2, "models")):
            return _blocked(why)
        with stage_timer("3_models", dm):
            d, W = cfg["data"], ctx.data["windows"]
            daily, world = ctx.data["daily"], ctx.data["world"]
            sc = spend_cols(world.n_ch)
            rm = ResponseModel(dm, cfg["response"])
            tr = np.arange(d["train_end"])
            ho = np.arange(d["train_end"], d["live_end"])
            rep = rm.fit(daily[sc].to_numpy(), daily["dow"].to_numpy(), daily["conversions"].to_numpy(), tr, ho,
                         seed=cfg["seed"])
            um = UpliftModel(dm, cfg["uplift"])
            urep = um.fit(ctx.data["uplift"], seed=cfg["seed"])
            bl = ctx.data["bandit_log"]
            bandit = BatchedLinUCB(dm, cfg["bandit"]["n_creatives"], cfg["bandit"]["discount_tiers"],
                                   cfg["bandit"]["alpha"], cfg["bandit"]["ridge_lambda"])
            bandit.update_batch(bl[CTX_COLS].to_numpy(), bl["arm"].to_numpy(), bl["reward"].to_numpy())
            brep = bandit.covariance_stats()
            guard = GuardrailEngine(ctx.gcfg)
            planner = Planner(rm, um, bandit, guard, cfg)
            ctx.models.update(resp=rm, uplift=um, bandit=bandit, guard=guard, planner=planner)
            curves = rm.curves(ctx.gcfg["max_total_budget"])
            ctx.cache["curves"] = curves
            true_curves = {c: np.array([world.hill(world.steady_adstock(np.eye(world.n_ch)[c] * g))[c] for g in curves[c][0]])
                           for c in curves}
            viz.response_curves(curves, CHANNELS, true_curves)
            dm.write_telemetry({"stage_3_models": {"resp_device": rep["device"], "uplift_device": urep["device"]}})
            res = sv.gate3_models(rep, curves, urep, brep, cfg["response"]["r2_min"])
            upd = _gate_update(state, res, 3)
            ResponseArtifact(**{k: rep[k] for k in ResponseArtifact.model_fields})
            UpliftArtifact(**{k: urep[k] for k in UpliftArtifact.model_fields})
            BanditArtifact(**{k: brep[k] for k in BanditArtifact.model_fields})
            # ---- post-hook: advisory LLM diagnosis (never gates the pipeline)
            live_ctx = W["live"].tail(7)[CTX_COLS].mean()
            tele = {"r2_train": rep["r2_train"], "r2_holdout": rep["r2_holdout"], "max_psi": ctx.cache["drift"]["_max_psi"],
                    "cate_mean": urep["cate_mean"], "cate_var": urep["cate_var"], "bandit": brep,
                    "inventory_util": float(live_ctx["inventory_util"]), "gate3_passed": res.passed,
                    "channel_decay": rep["lam"], "channel_beta": rep["beta"]}
            diag, meta = LocalLLM(cfg).diagnose(tele)
            audit("llm_diagnose", diag.summary, **meta, risk=diag.risk, hypotheses=[h.model_dump() for h in diag.hypotheses])
            upd["diagnosis"] = {**diag.model_dump(), "meta": meta}
            esc = list(state.get("escalations", []))
            if diag.recommend_escalation or diag.risk == "high":
                esc.append(f"llm_diagnosis_risk_{diag.risk}")
            upd["escalations"] = esc
            return upd

    # -------------------------------------------------------------------- Step 4
    def optimize(state: AgentState) -> dict:
        if (why := _pre(state, 3, "optimize")):
            return _blocked(why)
        with stage_timer("4_optimize", dm):
            W = ctx.data["windows"]
            planner: Planner = ctx.models["planner"]
            tail = W["live"].tail(7)
            prev = tail[spend_cols(planner.C)].mean().to_numpy()
            cm = tail[CTX_COLS].mean().to_numpy()
            gm = float(tail["gross_margin"].mean())
            ctx.cache.update(prev=prev, cm=cm, gm=gm)
            plan = planner.plan(prev, cm, gm)
            audit("optimize", "convex allocation + tier selection", status=plan["status"], discount=plan["discount"],
                  creative=plan["creative"], expected_net_per_day=plan["expected_net"], solver=plan["solver"])
            audit("guardrails", "plan checked/clamped", clamps=plan["clamps"], guardrail_ok=plan["guardrail_ok"])
            res = sv.gate4_optimization({k: v for k, v in plan.items() if k in
                                         ("status", "budgets", "prev_budgets", "discount", "creative", "unit_margin",
                                          "inventory_util", "expected_net", "clamps", "guardrail_ok", "solver")},
                                        ctx.gcfg, plan["status"])
            viz.allocation_chart(plan["prev_budgets"], plan["budgets"], plan["discount"])
            upd = _gate_update(state, res, 4)
            upd["plan"] = plan
            return upd

    # -------------------------------------------------------------------- Step 5
    def evaluate(state: AgentState) -> dict:
        if (why := _pre(state, 4, "evaluate")):
            return _blocked(why)
        with stage_timer("5_evaluate", dm):
            ope = evaluate_policy_ope(ctx.data["bandit_log"], dm, cfg, seed=cfg["seed"])
            ev = run_benchmark(ctx.data["world"], ctx.models["planner"], ctx.data["daily"], cfg)
            ev["ope"] = ope
            out_dir("evaluation", "ope_replay_results.json").write_text(json.dumps(ope, indent=2))
            res = sv.gate5_evaluation(ev, cfg["evaluation"]["target_lift_pct"])
            viz.ope_plot(ope)
            viz.benchmark_chart(ev)
            ctx.cache["evaluation"] = ev
            dm.write_telemetry({"ope_bootstrap_ms": ope["bootstrap_ms"], "ope_n_boot": ope["n_boot"]})
            audit("evaluate", "offline replay + OPE", lift_vs_static_pct=ev["lift_vs_static_pct"],
                  lift_vs_heuristic_pct=ev["lift_vs_heuristic_pct"], dr_lift=ope["dr_lift"], ci=[ope["lift_ci_low"], ope["lift_ci_high"]])
            upd = _gate_update(state, res, 5)
            upd["evaluation"] = {"lift_vs_static_pct": ev["lift_vs_static_pct"], "lift_vs_heuristic_pct": ev["lift_vs_heuristic_pct"],
                                 "dr_lift": ope["dr_lift"], "ci": [ope["lift_ci_low"], ope["lift_ci_high"]],
                                 "guardrail_violations": ev["guardrail_violations"]}
            return upd

    # -------------------------------------------------------------------- Step 6 (pre): visual QA + HITL
    def review(state: AgentState) -> dict:
        if (why := _pre(state, 5, "review")):
            return _blocked(why)
        plan, h = state["plan"], cfg["hitl"]
        with stage_timer("6a_visual_qa_hitl", dm):
            imgs = [out_dir("charts", "response_curves.png"), out_dir("charts", "allocation.png")]
            verdict, meta = LocalLLM(cfg).verify_visual(imgs, {"discount": plan["discount"], "creative": plan["creative"],
                                                                "budgets": plan["budgets"], "prev": plan["prev_budgets"]})
            audit("llm_visual_verify", "; ".join(verdict.findings)[:400], verdict=verdict.verdict, **meta)
            reasons = list(state.get("escalations", []))
            if verdict.verdict != "ok":
                reasons.append(f"visual_verdict_{verdict.verdict}")
            if plan["discount"] >= h["discount_tier_trigger"]:
                reasons.append(f"discount_tier_{plan['discount']:.0%}")
            tot_new, tot_prev = sum(plan["budgets"].values()), sum(plan["prev_budgets"].values())
            if abs(tot_new - tot_prev) / tot_prev > h["total_budget_change_pct"]:
                reasons.append(f"total_budget_change_{(tot_new - tot_prev) / tot_prev:+.1%}")
            if plan["unit_margin"] - ctx.gcfg["min_unit_margin"] < h["margin_buffer"]:
                reasons.append("unit_margin_near_floor")
            upd = {"visual": {**verdict.model_dump(), "meta": meta}, "escalations": reasons}
            if not h["enabled"] or not reasons:
                upd["hitl"] = {"required": False, "status": "not_required"}
                return upd
            if ctx.auto_approve:
                upd["hitl"] = {"required": True, "status": "auto_approved", "reviewer": "batch-policy", "reasons": reasons}
                audit("hitl", "auto-approved in batch mode", reasons=reasons)
                return upd
            decision = interrupt({"reasons": reasons, "plan": plan, "visual": upd["visual"], "diagnosis": state.get("diagnosis")})
            ok = str(decision.get("decision", "reject")).lower() == "approve"
            upd["hitl"] = {"required": True, "status": "approved" if ok else "rejected", "reviewer": decision.get("reviewer", "human"),
                           "note": decision.get("note", ""), "reasons": reasons}
            audit("hitl", f"human decision: {'approve' if ok else 'reject'}", reviewer=decision.get("reviewer"), note=decision.get("note"))
            if not ok:
                upd.update(halted=True, halt_reason="Plan rejected by human reviewer")
            return upd

    # -------------------------------------------------------------------- Step 6
    def execute(state: AgentState) -> dict:
        if (why := _pre(state, 5, "execute")):
            return _blocked(why)
        with stage_timer("6_execute", dm):
            plan = state["plan"]
            ads, pricing = MockAdPlatformAPI(), MockPricingAPI()
            r1 = ads.set_budgets(plan["budgets"], dry_run=True)
            r2 = pricing.set_promo(plan["discount"], plan["creative"], dry_run=True)
            approved = state.get("hitl", {}).get("status") in ("not_required", "auto_approved", "approved")
            payload = {"approved": approved, "guardrail_ok": plan["guardrail_ok"], "calls": [r1, r2], "dry_run": True,
                       "hitl_status": state.get("hitl", {}).get("status")}
            audit("execute", "dry-run execution of approved plan", **{k: payload[k] for k in ("approved", "dry_run")})
            ev = ctx.cache["evaluation"]
            ctx.cache["payload"] = payload
            # metrics artifacts must exist before Gate 6 inspects them
            dm.write_telemetry()
            res = sv.gate6_execution(payload, read_jsonl("agent_decision_audit.jsonl"))
            upd = _gate_update(state, res, 6)
            upd["executed"] = payload
            aggregate(ev, ev["ope"], dm.info(), {"run_id": ctx.run_id, "plan": plan})
            return upd

    def halt(state: AgentState) -> dict:
        audit("halt", state.get("halt_reason", "halted"), run_id=ctx.run_id)
        log_event("halt", state.get("halt_reason", ""))
        return {"halted": True}

    return {"ingest": ingest, "features": features, "models": models, "optimize": optimize,
            "evaluate": evaluate, "review": review, "execute": execute, "halt": halt}
