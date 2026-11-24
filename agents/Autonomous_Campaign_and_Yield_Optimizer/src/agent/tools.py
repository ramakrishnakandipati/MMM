"""Deterministic tool wrappers: GPU models + optimizer + guardrails + mock ad/pricing APIs.

`Planner.plan()` is the single decision function used by the LangGraph agent AND the
offline benchmark replay, so evaluation measures exactly what the agent would do.
"""
from __future__ import annotations

import numpy as np

from ..data.synthetic_generator import CHANNELS, CTX_COLS
from ..guardrails.policy_engine import GuardrailEngine
from ..models.contextual_bandit import BatchedLinUCB
from ..models.convex_optimizer import BudgetOptimizer
from ..models.response_curves import ResponseModel
from ..models.uplift_elasticity import UpliftModel


class OnlineCalibrator:
    """Ridge-RLS per-channel multipliers m_c on the learned response (prior m=1).

    obs_adj(t) ~= base_pred + sum_c m_c * pred_c(t). Lets the agent adapt to regime shifts
    using only the total conversions it actually observes after each cycle.
    """

    def __init__(self, n_ch: int, prior_sd: float = 0.35):
        self.n = n_ch
        self.lam = 1.0 / prior_sd ** 2
        self.A = np.eye(n_ch) * self.lam
        self.b = np.ones(n_ch) * self.lam
        self.m = np.ones(n_ch)

    def update(self, pred_c: np.ndarray, resid_target: np.ndarray, scale: float):
        """pred_c: (days, C) predicted channel contributions; resid_target: (days,) observed - base (adjusted)."""
        X, y = pred_c / scale, resid_target / scale
        self.A += X.T @ X
        self.b += X.T @ y
        self.m = np.clip(np.linalg.solve(self.A, self.b), 0.2, 3.0)


class Planner:
    def __init__(self, resp: ResponseModel, uplift: UpliftModel, bandit: BatchedLinUCB, guard: GuardrailEngine,
                 cfg: dict, n_tan: int = 41):
        self.resp, self.uplift, self.bandit, self.guard, self.cfg = resp, uplift, bandit, guard, cfg
        self.C = resp.n_ch
        self.opt = BudgetOptimizer(self.C, n_tan + 1)
        self.tiers = cfg["bandit"]["discount_tiers"]
        self.redemption = cfg["data"].get("redemption", 0.30)
        self.aov = cfg["data"]["aov"]
        self.bmax = guard.g["max_total_budget"]
        self.calib = OnlineCalibrator(self.C)
        self._tan = [resp.tangents(c, self.bmax, n_tan) for c in range(self.C)]

    # ------------------------------------------------------------------ one decision
    def plan(self, prev: np.ndarray, ctx: np.ndarray, gm: float) -> dict:
        util = float(ctx[0])
        g = self.guard
        lo, hi = g.bounds(prev)
        P = self.resp.params()
        org = P["base"]
        slopes = np.stack([t[0] for t in self._tan]) * self.calib.m[:, None]
        icpts = np.stack([t[1] for t in self._tan]) * self.calib.m[:, None]
        best, by_tier = None, {}
        for d in g.allowed_tiers(self.tiers, util, gm):
            L = self.uplift.rel_lift(ctx, d)
            k, _ = self.bandit.best_arm(ctx, allowed_tiers=[d])
            mk = self.bandit.creative_mult(ctx, d, k) if k > 0 else 1.0
            unit = self.aov * (gm - d * self.redemption)
            coef = mk * (1 + L) * unit
            sol = self.opt.solve(slopes, icpts, coef, lo, hi, self.bmax)
            const = org * (mk * (1 + L) * unit - self.aov * gm)
            net = sol["objective"] + const if np.isfinite(sol["objective"]) else -np.inf
            by_tier[d] = {"net": float(net), "creative": int(k), "L": float(L), "mk": float(mk), "status": sol["status"]}
            if best is None or net > best["net"]:
                best = {"net": net, "d": d, "k": k, "sol": sol, "L": L, "mk": mk}
        sol = best["sol"]
        plan = {
            "status": sol["status"],
            "budgets": {CHANNELS[c]: float(sol["b"][c]) for c in range(self.C)},
            "prev_budgets": {CHANNELS[c]: float(prev[c]) for c in range(self.C)},
            "discount": float(best["d"]), "creative": int(best["k"]),
            "unit_margin": float(gm - best["d"]), "inventory_util": util,
            "expected_net": float(best["net"]), "clamps": [], "solver": sol["solver"],
            "est_lift": float(best["L"]), "est_creative_mult": float(best["mk"]), "net_by_tier": by_tier,
        }
        plan = g.enforce(plan, {"inventory_util": util, "gm": gm})
        plan["guardrail_ok"] = not g.evaluate(plan, {"inventory_util": util, "gm": gm})
        return plan

    # ------------------------------------------------------------------ learning from outcome
    def observe(self, spend: np.ndarray, ad_state: np.ndarray, daily_conv: np.ndarray, dow_days: np.ndarray,
                plan: dict) -> None:
        """Update online calibration from the realised cycle (adjusting out estimated promo/creative effects)."""
        P = self.resp.params()
        lam, K, S, beta = map(np.array, (P["lam"], P["K"], P["S"], P["beta"]))
        st = ad_state.copy()
        preds = []
        for _ in dow_days:
            st = spend + lam * st
            xs = np.maximum(st, 0) ** S
            preds.append(beta * xs / (K ** S + xs + 1e-12))
        preds = np.array(preds)
        adj = np.array(daily_conv) / (self.resp.dow_mult()[np.asarray(dow_days, int)] * plan["est_creative_mult"] * (1 + plan["est_lift"]))
        # the dow factor multiplies the whole signal: remove it from both sides
        target = adj - P["base"]
        self.calib.update(preds, target, scale=max(float(np.mean(adj)), 1.0))


# ---------------------------------------------------------------------- mock ad / pricing APIs
class MockAdPlatformAPI:
    """Dry-run ad platform: records budget changes, never calls a real network."""

    def __init__(self):
        self.calls: list[dict] = []

    def set_budgets(self, budgets: dict[str, float], dry_run: bool = True) -> dict:
        rec = {"endpoint": "POST /campaigns/budgets", "payload": budgets, "dry_run": dry_run, "status": 200}
        self.calls.append(rec)
        return rec


class MockPricingAPI:
    def __init__(self):
        self.calls: list[dict] = []

    def set_promo(self, discount: float, creative: int, dry_run: bool = True) -> dict:
        rec = {"endpoint": "POST /pricing/promo", "payload": {"discount": discount, "creative": creative},
               "dry_run": dry_run, "status": 200}
        self.calls.append(rec)
        return rec
