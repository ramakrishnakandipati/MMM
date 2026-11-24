import json

import pytest

from src.agent.graph import AgentService
from src.agent.scenarios import SCENARIOS, run_all_scenarios, run_scenario
from src.utils.config import out_dir


@pytest.fixture(scope="module")
def results(pipeline_run):
    ctx = AgentService.context(pipeline_run["run_id"])
    return run_all_scenarios(ctx)


def test_six_scenarios_defined_and_persisted(results):
    assert len(SCENARIOS) >= 5 and set(results) == {s["id"] for s in SCENARIOS}
    saved = json.loads(out_dir("scenarios", "scenario_results.json").read_text())
    assert set(saved) == set(results)


def test_every_scenario_has_six_steps(results):
    for r in results.values():
        assert [s["step"] for s in r["steps"]] == [1, 2, 3, 4, 5, 6]


def test_inventory_crunch_suppresses_discount(results):
    r = results["inventory_crunch"]
    assert r["plan"]["inventory_util"] > 0.85 and r["plan"]["discount"] == 0.0 and r["plan"]["guardrail_ok"]


def test_margin_squeeze_blocks_top_tier(results):
    r = results["margin_squeeze"]
    assert r["plan"]["discount"] < 0.20 and r["plan"]["unit_margin"] >= 0.20


def test_drift_halts_at_gate2_and_skips_rest(results):
    r = results["live_drift"]
    assert r["halted"] and r["steps"][1]["status"] == "FAIL"
    assert all(s["status"] == "SKIPPED" for s in r["steps"][2:]) and r["plan"] is None


def test_regime_scenarios_guardrail_clean_and_beat_static(results):
    for k, r in results.items():
        if r["halted"]:
            continue
        assert r["replay"]["guardrail_violations"] == 0 and r["replay"]["lift_vs_static_pct"] > 0, k
        assert r["steps"][3]["status"] == "PASS", k


def test_single_scenario_is_deterministic(pipeline_run, results):
    ctx = AgentService.context(pipeline_run["run_id"])
    again = run_scenario(ctx, next(s for s in SCENARIOS if s["id"] == "demand_shock"))
    assert again["replay"]["lift_vs_static_pct"] == pytest.approx(results["demand_shock"]["replay"]["lift_vs_static_pct"])
