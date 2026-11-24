import numpy as np
import pytest

from src.guardrails.policy_engine import GuardrailEngine
from src.models.convex_optimizer import BudgetOptimizer

G = {"max_total_budget": 100000.0, "max_channel_shift_pct": 0.15, "min_unit_margin": 0.20, "inventory_discount_cap": 0.85,
     "max_discount": 0.20, "min_channel_budget": 500.0}


def _tangents(beta, K, S, hi, n=41):
    pts = np.concatenate([[1.0], np.linspace(hi / (n - 1), hi, n - 1)])
    f = beta * pts ** S / (K ** S + pts ** S)
    df = beta * S * K ** S * pts ** (S - 1) / (K ** S + pts ** S) ** 2
    return df, f - df * pts


def _setup(C=4, seed=0):
    rng = np.random.default_rng(seed)
    sl, ic = [], []
    for _ in range(C):
        a, b = _tangents(rng.uniform(300, 1500), rng.uniform(5000, 40000), rng.uniform(0.6, 0.95), 100000.0)
        sl.append(a)
        ic.append(b)
    return np.array(sl), np.array(ic), sl[0].shape[0]


@pytest.mark.parametrize("seed", range(6))
def test_optimizer_respects_constraints_and_beats_start(seed):
    sl, ic, K = _setup(seed=seed)
    C = sl.shape[0]
    prev = np.random.default_rng(seed).uniform(8000, 22000, C)
    g = GuardrailEngine(G)
    lo, hi = g.bounds(prev)
    res = BudgetOptimizer(C, K).solve(sl, ic, 46.75, lo, hi, 100000.0)
    assert res["status"] == "optimal"
    b = res["b"]
    assert (b >= lo - 1e-3).all() and (b <= hi + 1e-3).all() and b.sum() <= 100000.0 + 1e-3
    assert (np.abs(b - prev) / prev <= 0.15 + 1e-6).all()
    f = lambda x: 46.75 * np.min(sl * x[:, None] + ic, axis=1).sum() - x.sum()
    assert f(b) >= f(prev) - 1e-3


def test_scipy_fallback_matches_cvxpy():
    sl, ic, K = _setup()
    prev = np.full(4, 15000.0)
    lo, hi = GuardrailEngine(G).bounds(prev)
    o = BudgetOptimizer(4, K)
    a = o.solve(sl, ic, 46.75, lo, hi, 100000.0)
    b = o._scipy(sl, ic, 46.75, lo, hi, 100000.0, 0.0)
    assert abs(a["objective"] - b["objective"]) < 1e-2 * max(1, abs(a["objective"]))


def test_allowed_tiers_rules():
    g = GuardrailEngine(G)
    tiers = [0.0, 0.05, 0.10, 0.20]
    assert g.allowed_tiers(tiers, 0.5, 0.55) == tiers
    assert g.allowed_tiers(tiers, 0.9, 0.55) == [0.0]              # inventory suppression
    assert g.allowed_tiers(tiers, 0.5, 0.38) == [0.0, 0.05, 0.10]  # margin floor removes the 20% tier


def test_enforce_clamps_violations():
    g = GuardrailEngine(G)
    plan = {"budgets": {"a": 30000.0, "b": 100.0}, "prev_budgets": {"a": 10000.0, "b": 10000.0}, "discount": 0.2}
    out = g.enforce(plan, {"inventory_util": 0.95, "gm": 0.55})
    assert out["discount"] == 0.0 and "discount_suppressed_inventory" in out["clamps"]
    assert out["budgets"]["a"] <= 11500.0 + 1e-6 and out["budgets"]["b"] >= 8500.0 - 1e-6
    assert out["guardrail_ok"] and sum(out["budgets"].values()) <= 100000.0


def test_enforce_scales_total_budget_cap():
    g = GuardrailEngine({**G, "max_total_budget": 20000.0})
    plan = {"budgets": {"a": 11500.0, "b": 11500.0}, "prev_budgets": {"a": 10000.0, "b": 10000.0}, "discount": 0.0}
    out = g.enforce(plan, {"inventory_util": 0.3, "gm": 0.55})
    assert sum(out["budgets"].values()) <= 20000.0 + 1e-6 and "budget_cap_scaled" in out["clamps"]


def test_margin_floor_removes_discount():
    g = GuardrailEngine(G)
    out = g.enforce({"budgets": {"a": 10000.0}, "prev_budgets": {"a": 10000.0}, "discount": 0.2},
                    {"inventory_util": 0.3, "gm": 0.35})
    assert out["discount"] == 0.0 and out["unit_margin"] >= 0.2
