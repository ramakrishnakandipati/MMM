"""Replays Agent vs Static vs Heuristic policies on out-of-sample days across market regimes,
scored by the ground-truth simulator (known counterfactuals)."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from ..agent.tools import Planner
from ..data.synthetic_generator import CHANNELS, CTX_COLS, World
from ..data.state_builder import spend_cols
from ..utils.config import out_dir

POLICIES = ["agent", "static", "heuristic"]


def static_allocation(train: pd.DataFrame, n_ch: int) -> np.ndarray:
    total = train[spend_cols(n_ch)].sum(axis=1).mean()
    return np.full(n_ch, total / n_ch)


def heuristic_weights(train: pd.DataFrame, n_ch: int, margin_per_conv: float) -> np.ndarray:
    """ROAS-proportional: per-channel OLS conversions-per-$ x margin, floored at a small positive weight."""
    X = train[spend_cols(n_ch)].to_numpy()
    X1 = np.column_stack([np.ones(len(X)), X])
    coef = np.linalg.lstsq(X1, train["conversions"].to_numpy(), rcond=None)[0][1:]
    roas = np.maximum(coef * margin_per_conv, 0.05)
    return roas / roas.sum()


def run_policy(policy: str, world: World, planner: Planner | None, oos: pd.DataFrame, cfg: dict, static_b: np.ndarray,
               heur_w: np.ndarray, regime: str, seed: int = 0) -> dict:
    cyc = cfg["cadence"]["cycle_days"]
    n_ch = world.n_ch
    rng = np.random.default_rng(seed)
    ctx_all = oos[CTX_COLS].to_numpy().copy()
    if regime == "inventory_crunch":
        ctx_all[:, 0] = np.clip(ctx_all[:, 0] + 0.25, 0, 0.98)
    dow_all = oos["dow"].to_numpy()
    gm = world.gm
    base_world = world
    ad_state = static_b / (1 - world.lam)                      # same warm start for every policy
    prev = static_b.copy()
    rows, net_total, violations, plan_ms = [], 0.0, 0, []
    if planner is not None:
        planner.calib = type(planner.calib)(n_ch)              # fresh calibration per run
    total_budget = static_b.sum()
    for ci in range(len(oos) // cyc):
        sl = slice(ci * cyc, (ci + 1) * cyc)
        ctx, dows = ctx_all[sl], dow_all[sl]
        cm = ctx.mean(0)
        plan = None
        if policy == "static":
            b, d, k = static_b, 0.0, 0
        elif policy == "heuristic":
            b = total_budget * heur_w
            d = 0.10 if cm[0] < 0.70 and gm - 0.10 >= 0.2 else 0.0
            k = 0
        else:
            t0 = time.perf_counter()
            plan = planner.plan(prev, cm, gm)
            plan_ms.append((time.perf_counter() - t0) * 1000)
            ctx_check = {"inventory_util": float(cm[0]), "gm": gm}
            if (not plan["guardrail_ok"]) or planner.guard.evaluate(plan, ctx_check) or plan["status"] not in ("optimal", "optimal_inaccurate"):
                violations += 1
            b = np.array([plan["budgets"][c] for c in CHANNELS[:n_ch]])
            d, k = plan["discount"], plan["creative"]
        res = world.simulate_cycle(b, ad_state, ctx, dows, d, k, gm=gm, rng=rng)
        if policy == "agent":
            planner.observe(b, ad_state, res["daily_conv"], dows, plan)
        ad_state, prev = res["ad_state"], b
        net_total += res["net"]
        rows.append({"regime": regime, "policy": policy, "cycle": ci, "net": res["net"], "incr_conv": res["incr_conv"],
                     "spend": res["spend"], "promo_cost": res["promo_cost"], "discount": d, "creative": k,
                     "inventory_util": float(cm[0]), **{f"b_{c}": float(b[i]) for i, c in enumerate(CHANNELS[:n_ch])}})
    return {"net": net_total, "rows": rows, "violations": violations, "plan_ms": plan_ms}


def run_benchmark(world: World, planner: Planner, daily: pd.DataFrame, cfg: dict) -> dict:
    d = cfg["data"]
    train = daily[daily.day_idx < d["train_end"]]
    oos = daily[daily.day_idx >= d["live_end"]].reset_index(drop=True)
    n_ch = world.n_ch
    static_b = static_allocation(train, n_ch)
    heur_w = heuristic_weights(train, n_ch, world.aov * world.gm)
    all_rows, regimes, viol, plan_ms = [], {}, 0, []
    for regime in cfg["evaluation"]["regimes"]:
        w = world.with_regime(regime)
        res = {p: run_policy(p, w, planner if p == "agent" else None, oos, cfg, static_b, heur_w, regime) for p in POLICIES}
        for p in POLICIES:
            all_rows += res[p]["rows"]
        viol += res["agent"]["violations"]
        plan_ms += res["agent"]["plan_ms"]
        net = {p: res[p]["net"] for p in POLICIES}
        regimes[regime] = {
            "net": net,
            "lift_vs_static_pct": 100 * (net["agent"] - net["static"]) / abs(net["static"]),
            "lift_vs_heuristic_pct": 100 * (net["agent"] - net["heuristic"]) / abs(net["heuristic"]),
            "guardrail_violations": res["agent"]["violations"],
        }
    tot = {p: sum(r["net"][p] for r in regimes.values()) for p in POLICIES}
    replay = pd.DataFrame(all_rows)
    comp = replay.groupby(["regime", "policy"]).agg(net=("net", "sum"), incr_conv=("incr_conv", "sum"), spend=("spend", "sum"),
                                                    promo_cost=("promo_cost", "sum")).reset_index()
    comp.to_csv(out_dir("evaluation", "benchmark_comparison.csv"), index=False)
    replay.to_parquet(out_dir("data", "counterfactual_replay.parquet"), index=False)
    out_dir("evaluation", "regime_breakdown.json").write_text(json.dumps(regimes, indent=2))
    return {
        "regimes": regimes, "total_net": tot,
        "lift_vs_static_pct": 100 * (tot["agent"] - tot["static"]) / abs(tot["static"]),
        "lift_vs_heuristic_pct": 100 * (tot["agent"] - tot["heuristic"]) / abs(tot["heuristic"]),
        "guardrail_violations": viol, "plan_latency_ms_mean": float(np.mean(plan_ms)) if plan_ms else None,
        "plan_latency_ms_p95": float(np.percentile(plan_ms, 95)) if plan_ms else None,
        "n_cycles_per_regime": len(oos) // cfg["cadence"]["cycle_days"],
    }
