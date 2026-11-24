import copy
import json

import numpy as np
import pytest

from src.data.state_builder import build_features, split_windows
from src.validation import step_validators as sv
from src.validation.drift_monitor import psi


def test_gate1_passes_on_clean_data(world_data, cfg):
    assert sv.gate1_data_integrity(world_data["daily"], cfg).passed


@pytest.mark.parametrize("col,idx,val,check", [
    ("spend_search", 5, -10.0, "spend_non_negative"),
    ("conversions", 3, np.nan, "no_nan_inf"),
    ("capacity_util", 9, 1.4, "capacity_util_in_[0,1]"),
    ("unit_cogs", 2, -1.0, "cogs_non_negative"),
])
def test_gate1_detects_corruption(world_data, cfg, col, idx, val, check):
    d = world_data["daily"].copy()
    d.loc[idx, col] = val
    r = sv.gate1_data_integrity(d, cfg)
    assert not r.passed and check in [c.name for c in r.checks if not c.passed]


def test_gate1_detects_gap_and_row_count(world_data, cfg):
    d = world_data["daily"].drop(index=[10]).reset_index(drop=True)
    names = [c.name for c in sv.gate1_data_integrity(d, cfg).checks if not c.passed]
    assert "timestamp_continuity" in names and "row_count" in names


def test_gate2_passes_then_fails_on_drift(world_data, cfg):
    d = cfg["data"]
    feat, spec = build_features(world_data["daily"])
    w = split_windows(feat, d["train_end"], d["live_end"])
    r, rep = sv.gate2_features_drift(feat, spec, w["train"], w["live"])
    assert r.passed, [c for c in r.checks if not c.passed]
    live = w["live"].copy()
    live["price_sens"] = np.clip(live["price_sens"] + 0.35, 0, 1)
    r2, rep2 = sv.gate2_features_drift(feat, spec, w["train"], live)
    assert not r2.passed and rep2["price_sens"]["psi"] > 0.25


def test_gate2_rejects_bad_decay(world_data, cfg):
    d = cfg["data"]
    feat, spec = build_features(world_data["daily"])
    w = split_windows(feat, d["train_end"], d["live_end"])
    spec = copy.deepcopy(spec)
    spec["decays"]["search"] = 1.2
    r, _ = sv.gate2_features_drift(feat, spec, w["train"], w["live"])
    assert not r.passed


def test_psi_basics():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert psi(a, rng.normal(0, 1, 5000)) < 0.05
    assert psi(a, rng.normal(1.5, 1, 5000)) > 0.25


def _plan(**kw):
    p = {"status": "optimal", "budgets": {"a": 11000.0, "b": 9000.0}, "prev_budgets": {"a": 10000.0, "b": 10000.0},
         "discount": 0.05, "creative": 1, "unit_margin": 0.5, "inventory_util": 0.5, "expected_net": 1000.0,
         "clamps": [], "guardrail_ok": True}
    p.update(kw)
    return p


G = {"max_total_budget": 25000.0, "max_channel_shift_pct": 0.15, "min_unit_margin": 0.20, "inventory_discount_cap": 0.85,
     "max_discount": 0.20}


def test_gate4_ok_and_each_violation():
    assert sv.gate4_optimization(_plan(), G, "optimal").passed
    bad = {
        "solver_status": sv.gate4_optimization(_plan(), G, "infeasible"),
        "budget": sv.gate4_optimization(_plan(budgets={"a": 12000.0, "b": 14000.0}, prev_budgets={"a": 12000.0, "b": 14000.0}), G, "optimal"),
        "shift": sv.gate4_optimization(_plan(budgets={"a": 13000.0, "b": 9000.0}), G, "optimal"),
        "inventory": sv.gate4_optimization(_plan(inventory_util=0.9, discount=0.05), G, "optimal"),
        "margin": sv.gate4_optimization(_plan(unit_margin=0.15), G, "optimal"),
    }
    for k, r in bad.items():
        assert not r.passed, k


def test_gate5_requires_significance():
    ev = {"ope": {"lift_ci_low": 0.1, "ips_lift": 0.2, "dr_lift": 0.2}, "lift_vs_static_pct": 20, "lift_vs_heuristic_pct": 10,
          "guardrail_violations": 0, "regimes": {str(i): {"lift_vs_static_pct": 5} for i in range(5)}}
    assert sv.gate5_evaluation(ev, 8).passed
    ev2 = {**ev, "ope": {**ev["ope"], "lift_ci_low": -0.01}}
    assert not sv.gate5_evaluation(ev2, 8).passed
    assert not sv.gate5_evaluation({**ev, "lift_vs_heuristic_pct": 3}, 8).passed


def test_enforce_halts_and_record_persists():
    sv.reset_report()
    bad = sv.gate4_optimization(_plan(unit_margin=0.1), G, "optimal")
    with pytest.raises(sv.HaltPipeline):
        sv.enforce(bad)
    rep = json.loads(sv.report_path().read_text())
    assert rep["gates"]["4"]["passed"] is False
