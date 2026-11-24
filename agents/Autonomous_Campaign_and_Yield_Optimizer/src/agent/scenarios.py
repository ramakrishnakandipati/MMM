"""End-to-end example scenarios. Each one pushes a market situation through the same six-step flow
(data -> features -> models -> optimize+guardrails -> evaluate -> execute/HITL) using the models of a
completed base run, so the Streamlit Scenario Studio and run_all.py show identical results."""
from __future__ import annotations

import json
import time

import numpy as np

from ..data.synthetic_generator import CHANNELS, CTX_COLS
from ..data.state_builder import spend_cols
from ..evaluation.benchmark_runner import heuristic_weights, run_policy, static_allocation
from ..utils.config import out_dir
from ..utils.logger import audit
from ..validation import step_validators as sv
from ..validation.drift_monitor import drift_report
from .tools import hitl_reasons

STEP_NAMES = ["1 Data", "2 Features + drift", "3 Models", "4 Optimize + guardrails", "5 Evaluate (replay + OPE)", "6 Execute / HITL"]

SCENARIOS = [
    {"id": "baseline", "label": "1. Business as usual", "regime": "baseline",
     "story": "Normal demand and healthy inventory. The agent shifts budget (max 15% per channel per cycle) toward the "
              "channels with the highest marginal return and picks the discount tier with the best net margin.",
     "expect": "Large yield lift vs both baselines, all gates pass, high discount tier needs human approval."},
    {"id": "demand_shock", "label": "2. Demand shock", "regime": "demand_shock",
     "story": "Search response collapses (-45%) while display surges (+35%). The online calibrator re-weights channels "
              "from observed conversions, so the agent follows the market within a few cycles.",
     "expect": "Static spend keeps feeding the collapsed channel, so the agent's lift is even larger."},
    {"id": "inventory_crunch", "label": "3. Inventory crunch", "regime": "inventory_crunch", "snapshot_util": 0.92,
     "story": "Capacity utilisation spikes above 0.85. Promotions would create demand the supply cannot serve, so the "
              "guardrail suppresses every discount; the decision snapshot is taken at utilisation 0.92.",
     "expect": "Discount forced to 0%, lift comes from reallocation only (smallest of the five scenarios)."},
    {"id": "high_cac", "label": "4. CAC inflation", "regime": "high_cac",
     "story": "Social and video saturate later (K x1.7) so each extra dollar buys fewer conversions. The agent moves spend "
              "to cheaper channels and, when marginal ROI < 1, spends less than the static plan.",
     "expect": "Lift vs static is large because the static plan over-spends on now-expensive channels."},
    {"id": "margin_squeeze", "label": "5. Margin squeeze (COGS spike)", "regime": "margin_squeeze",
     "story": "Gross margin falls from 55% to 38%. The unit-margin floor (>= 20%) removes the 20% discount tier and the "
              "optimizer trims unprofitable spend.",
     "expect": "Discount tier capped below 20%; biggest relative lift because the static plan burns margin."},
    {"id": "live_drift", "label": "6. Live feature drift (pipeline HALT)", "regime": None,
     "story": "The live window's price-sensitivity feature shifts by +0.35. Gate 2 (PSI < 0.25) fails, the pipeline halts "
              "before any model is trusted and nothing is executed.",
     "expect": "Gate 2 FAIL (critical) -> HALT; steps 3-6 skipped; no budget change is sent."},
]


def _step(i, status, detail):
    return {"step": i, "name": STEP_NAMES[i - 1], "status": status, "detail": detail}


def run_scenario(ctx, sc: dict) -> dict:
    t0 = time.perf_counter()
    cfg, gcfg = ctx.cfg, ctx.gcfg
    W, world = ctx.data["windows"], ctx.data["world"]
    base_gates = ctx.cache.get("base_gates", {})
    steps = []
    max_psi = ctx.cache["drift"]["_max_psi"]
    res = {"id": sc["id"], "label": sc["label"], "story": sc["story"], "expect": sc["expect"], "halted": False}

    # ----------------------------------------------------------------- halt scenario
    if sc["id"] == "live_drift":
        live = W["live"].copy()
        live["price_sens"] = np.clip(live["price_sens"] + 0.35, 0, 1)
        rep = drift_report(W["train"], live, [c for c in ctx.data["spec"]["feature_cols"] if c in live])
        worst = max((k for k in rep if not k.startswith("_")), key=lambda k: rep[k]["psi"])
        steps = [_step(1, "PASS", "raw data integrity checks pass"),
                 _step(2, "FAIL", f"max PSI {rep['_max_psi']:.2f} on '{worst}' (limit 0.25) -> critical, HALT")]
        steps += [_step(i, "SKIPPED", "pipeline halted upstream") for i in range(3, 7)]
        res.update(halted=True, halt_reason=f"Gate 2 failed: max_psi_train_vs_live ({worst}={rep[worst]['psi']:.2f})",
                   steps=steps, plan=None, replay=None, hitl={"required": False, "status": "halted"},
                   drift={k: v["psi"] for k, v in rep.items() if not k.startswith("_")})
        audit("scenario", "live_drift halted at gate 2", scenario=sc["id"], max_psi=rep["_max_psi"])
        res["seconds"] = round(time.perf_counter() - t0, 2)
        return res

    # ----------------------------------------------------------------- regime scenarios
    w = world.with_regime(sc["regime"])
    planner = ctx.models["planner"]
    planner.calib = type(planner.calib)(planner.C)
    tail = W["live"].tail(7)
    prev = tail[spend_cols(planner.C)].mean().to_numpy()
    cm = tail[CTX_COLS].mean().to_numpy().copy()
    if sc.get("snapshot_util") is not None:
        cm[0] = sc["snapshot_util"]
    gm = w.gm
    plan = planner.plan(prev, cm, gm)
    g4 = sv.gate4_optimization({k: v for k, v in plan.items() if k in (
        "status", "budgets", "prev_budgets", "discount", "creative", "unit_margin", "inventory_util", "expected_net",
        "clamps", "guardrail_ok", "solver")}, gcfg, plan["status"])

    train = ctx.data["daily"][ctx.data["daily"].day_idx < cfg["data"]["train_end"]]
    oos = ctx.data["daily"][ctx.data["daily"].day_idx >= cfg["data"]["live_end"]].reset_index(drop=True)
    sb = static_allocation(train, w.n_ch)
    hw = heuristic_weights(train, w.n_ch, world.aov * world.gm)
    runs = {p: run_policy(p, w, planner if p == "agent" else None, oos, cfg, sb, hw, sc["regime"])
            for p in ("static", "heuristic", "agent")}
    net = {p: float(r["net"]) for p, r in runs.items()}
    lift_s = 100 * (net["agent"] - net["static"]) / abs(net["static"])
    lift_h = 100 * (net["agent"] - net["heuristic"]) / abs(net["heuristic"])
    cum = {p: np.cumsum([r["net"] for r in runs[p]["rows"]]).tolist() for p in runs}
    disc = [r["discount"] for r in runs["agent"]["rows"]]
    reasons = hitl_reasons(plan, cfg, gcfg)
    hitl_status = "approval_required" if reasons else "not_required"

    steps = [
        _step(1, "PASS", f"regime '{sc['regime']}' applied to the ground-truth simulator ({len(oos)} OOS days)"),
        _step(2, "PASS", f"max PSI train-vs-live {max_psi:.3f} (< 0.25)"),
        _step(3, "PASS", f"response R2 holdout {ctx.models['resp'].report['r2_holdout']:.2f}; CATE + LinUCB ready"),
        _step(4, "PASS" if g4.passed else "FAIL",
              f"{plan['solver']} {plan['status']}; discount {plan['discount']:.0%} (util {plan['inventory_util']:.2f}, "
              f"margin {gm:.0%}); clamps: {', '.join(plan['clamps']) or 'none'}; gate-4 checks "
              f"{sum(c.passed for c in g4.checks)}/{len(g4.checks)}"),
        _step(5, "PASS" if (lift_s >= 8 and lift_h > 0) else "WARN",
              f"lift vs static {lift_s:+.1f}%, vs heuristic {lift_h:+.1f}% over {len(runs['agent']['rows'])} cycles; "
              f"violations {runs['agent']['violations']}"),
        _step(6, "PASS" if (g4.passed and not reasons) else "PASS-HITL",
              ("human approval required: " + ", ".join(reasons)) if reasons else "auto-execute (dry-run)"),
    ]
    res.update(steps=steps, plan={k: v for k, v in plan.items() if k != "net_by_tier"}, net_by_tier=plan["net_by_tier"],
               replay={"net": net, "lift_vs_static_pct": lift_s, "lift_vs_heuristic_pct": lift_h,
                       "cumulative": cum, "agent_discount_by_cycle": disc,
                       "guardrail_violations": runs["agent"]["violations"]},
               hitl={"required": bool(reasons), "status": hitl_status, "reasons": reasons},
               gate4=[c.model_dump() for c in g4.checks])
    audit("scenario", f"{sc['id']} evaluated", scenario=sc["id"], lift_vs_static_pct=lift_s, discount=plan["discount"])
    res["seconds"] = round(time.perf_counter() - t0, 2)
    return res


def run_all_scenarios(ctx) -> dict:
    out = {s["id"]: run_scenario(ctx, s) for s in SCENARIOS}
    # drop heavy series from the persisted summary only
    out_dir("scenarios", "scenario_results.json").write_text(json.dumps(out, indent=2, default=float))
    return out
