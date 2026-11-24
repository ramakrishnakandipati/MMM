"""Six mandatory validation gates. Each returns a GateResult, is persisted to
outputs/validation/step_validation_report.json, and `enforce` halts on critical failure."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.config import out_dir
from ..utils.logger import log_event
from .contracts import AllocationPlan, DailyRow, FeatureRow, FeatureSpec, GateCheck, GateResult
from .drift_monitor import drift_report

_LOCK = threading.Lock()
GATE_NAMES = {
    1: "Raw & Synthetic Data Integrity",
    2: "Feature Engineering & Drift Check",
    3: "Model Calibration & Convergence",
    4: "Optimization & Guardrail Compliance",
    5: "Offline Evaluation & OPE Significance",
    6: "Execution & Output Artifact Integrity",
}


class HaltPipeline(RuntimeError):
    def __init__(self, result: GateResult):
        self.result = result
        failed = [c.name for c in result.checks if not c.passed and c.critical]
        super().__init__(f"Gate {result.gate} ({result.name}) failed critical checks: {failed}")


def _chk(name, passed, value=None, threshold=None, critical=True, detail="") -> GateCheck:
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        value = value.item()
    return GateCheck(name=name, passed=bool(passed), value=value, threshold=threshold, critical=critical, detail=detail)


def _finish(gate: int, checks: list[GateCheck], t0: float) -> GateResult:
    crit_fail = any((not c.passed) and c.critical for c in checks)
    warn = any((not c.passed) and not c.critical for c in checks)
    return GateResult(
        gate=gate, name=GATE_NAMES[gate], passed=not crit_fail, checks=checks,
        severity="critical" if crit_fail else ("warn" if warn else "ok"),
        seconds=round(time.perf_counter() - t0, 4), timestamp=datetime.now(timezone.utc).isoformat(),
    )


def report_path() -> Path:
    return out_dir("validation", "step_validation_report.json")


def reset_report() -> None:
    p = report_path()
    if p.exists():
        p.unlink()


def record(result: GateResult) -> GateResult:
    """Persist latest result per gate (+ history) and refresh the summary."""
    with _LOCK:
        p = report_path()
        rep = json.loads(p.read_text()) if p.exists() else {"gates": {}, "history": []}
        rep["gates"][str(result.gate)] = json.loads(result.model_dump_json())
        rep["history"].append({"gate": result.gate, "passed": result.passed, "timestamp": result.timestamp})
        gates = rep["gates"]
        rep["summary"] = {
            "gates_run": len(gates), "gates_passed": sum(g["passed"] for g in gates.values()),
            "pass_rate": round(sum(g["passed"] for g in gates.values()) / max(len(gates), 1), 4),
            "all_passed": all(g["passed"] for g in gates.values()) and len(gates) == 6,
            "updated": datetime.now(timezone.utc).isoformat(),
        }
        p.write_text(json.dumps(rep, indent=2))
    log_event(f"gate_{result.gate}", "gate_result", passed=result.passed, severity=result.severity)
    return result


def enforce(result: GateResult) -> GateResult:
    record(result)
    if not result.passed:
        raise HaltPipeline(result)
    return result


# --------------------------------------------------------------------------- Gate 1
def gate1_data_integrity(daily: pd.DataFrame, cfg: dict) -> GateResult:
    t0 = time.perf_counter()
    exp_rows = cfg["data"]["n_days"]
    num = daily.select_dtypes(include=[np.number])
    spend_cols = [c for c in daily.columns if c.startswith("spend_")]
    dates = pd.to_datetime(daily["date"])
    gaps = dates.diff().dropna().dt.days
    schema_bad = 0
    for rec in daily.head(200).to_dict("records"):
        try:
            DailyRow(**rec)
        except Exception:
            schema_bad += 1
    checks = [
        _chk("row_count", len(daily) == exp_rows, len(daily), f"== {exp_rows}"),
        _chk("no_nan_inf", bool(np.isfinite(num.to_numpy()).all()), int((~np.isfinite(num.to_numpy())).sum()), "== 0"),
        _chk("spend_non_negative", bool((daily[spend_cols] >= 0).all().all()), float(daily[spend_cols].min().min()), ">= 0"),
        _chk("cogs_non_negative", bool((daily["unit_cogs"] >= 0).all()), float(daily["unit_cogs"].min()), ">= 0"),
        _chk("capacity_util_in_[0,1]", bool(daily["capacity_util"].between(0, 1).all()),
             float(daily["capacity_util"].max()), "in [0,1]"),
        _chk("timestamp_continuity", bool((gaps == 1).all()) and dates.is_monotonic_increasing, int((gaps != 1).sum()), "0 gaps"),
        _chk("row_schema", schema_bad == 0, schema_bad, "== 0 invalid rows (sample 200)"),
    ]
    return _finish(1, checks, t0)


# --------------------------------------------------------------------------- Gate 2
def gate2_features_drift(feat: pd.DataFrame, spec: dict, train: pd.DataFrame, live: pd.DataFrame, psi_max: float = 0.25) -> tuple[GateResult, dict]:
    t0 = time.perf_counter()
    decays = list(spec["decays"].values())
    try:
        FeatureSpec(**spec)
        spec_ok = True
    except Exception:
        spec_ok = False
    bad = 0
    for rec in feat.sample(min(200, len(feat)), random_state=0).to_dict("records"):
        try:
            FeatureRow(**rec)
        except Exception:
            bad += 1
    cols = [c for c in spec["feature_cols"] if c in train.columns]
    rep = drift_report(train, live, cols)
    out_dir("validation", "drift_report.json").write_text(json.dumps(rep, indent=2))
    out_dir("validation", "schema_audit.json").write_text(json.dumps(
        {"feature_spec_ok": spec_ok, "invalid_rows_in_sample": bad, "feature_cols": spec["feature_cols"]}, indent=2))
    checks = [
        _chk("adstock_decay_in_[0,1)", all(0 <= d < 1 for d in decays), float(max(decays)), "0 <= decay < 1"),
        _chk("feature_spec_schema", spec_ok, spec_ok),
        _chk("feature_row_schema", bad == 0, bad, "== 0 invalid rows (sample 200)"),
        _chk("features_finite", bool(np.isfinite(feat[cols].to_numpy()).all()), None),
        _chk("max_psi_train_vs_live", rep["_max_psi"] < psi_max, rep["_max_psi"], f"< {psi_max}",
             detail=str({k: v["psi"] for k, v in rep.items() if not k.startswith('_') and v['psi'] >= 0.1})),
    ]
    return _finish(2, checks, t0), rep


# --------------------------------------------------------------------------- Gate 3
def gate3_models(resp: dict, curves: dict, uplift: dict, bandit: dict, r2_min: float = 0.80) -> GateResult:
    """curves: {channel: (grid_spend, steady_response)}"""
    t0 = time.perf_counter()
    mono_ok, conc_ok, worst = True, True, 0.0
    for ch, (g, r) in curves.items():
        d1 = np.diff(r) / np.diff(g)
        d2 = np.diff(d1)
        mono_ok &= bool((d1 > -1e-9).all())
        conc_ok &= bool((d2 < 1e-6 * max(abs(d1).max(), 1e-9)).all())
        worst = max(worst, float(d2.max() / max(abs(d1).max(), 1e-12)))
    checks = [
        _chk("response_r2_train", resp["r2_train"] >= r2_min, resp["r2_train"], f">= {r2_min}"),
        _chk("response_r2_holdout", resp["r2_holdout"] >= r2_min, resp["r2_holdout"], f">= {r2_min}"),
        _chk("response_params_finite", bool(np.isfinite(resp["beta"] + resp["K"] + resp["S"] + resp["lam"]).all()), None),
        _chk("adstock_decay_in_[0,1)", all(0 <= x < 1 for x in resp["lam"]), float(max(resp["lam"])), "< 1"),
        _chk("monotonic_increasing_response", mono_ok, mono_ok),
        _chk("diminishing_marginal_returns", conc_ok, worst, "second derivative <= 0"),
        _chk("cate_variance_bounds", 1e-10 < uplift["cate_var"] < 0.05 and np.isfinite(uplift["cate_var"]), uplift["cate_var"], "(1e-10, 0.05)"),
        _chk("cate_mean_finite", bool(np.isfinite(uplift["cate_mean"])), uplift["cate_mean"]),
        _chk("cate_corr_with_truth", (uplift.get("cate_corr_true") or 0) > 0.3, uplift.get("cate_corr_true"), "> 0.3", critical=False),
        _chk("bandit_covariance_finite", bool(bandit["finite_cov"]), bandit["finite_cov"]),
        _chk("bandit_covariance_pd", bandit["min_eig"] > 0, bandit["min_eig"], "> 0"),
    ]
    return _finish(3, checks, t0)


# --------------------------------------------------------------------------- Gate 4
def gate4_optimization(plan: dict, gcfg: dict, solver_status: str, hard_total: float | None = None) -> GateResult:
    t0 = time.perf_counter()
    p = AllocationPlan(**plan)
    bmax = hard_total or gcfg["max_total_budget"]
    shifts = []
    for c, b in p.budgets.items():
        prev = p.prev_budgets.get(c, 0.0)
        shifts.append(abs(b - prev) / prev if prev > 0 else 0.0)
    checks = [
        _chk("solver_status_optimal", solver_status in ("optimal", "optimal_inaccurate"), solver_status, "== optimal"),
        _chk("budgets_non_negative", all(b >= -1e-9 for b in p.budgets.values()), min(p.budgets.values()), ">= 0"),
        _chk("budget_conservation", sum(p.budgets.values()) <= bmax * (1 + 1e-6), sum(p.budgets.values()), f"<= {bmax}"),
        _chk("max_channel_shift", max(shifts) <= gcfg["max_channel_shift_pct"] + 1e-6, max(shifts), f"<= {gcfg['max_channel_shift_pct']}"),
        _chk("discount_suppressed_when_inventory_high",
             not (p.inventory_util > gcfg["inventory_discount_cap"] and p.discount > 0), p.discount,
             f"== 0 if util > {gcfg['inventory_discount_cap']}", detail=f"util={p.inventory_util:.3f}"),
        _chk("unit_margin_floor", p.unit_margin >= gcfg["min_unit_margin"] - 1e-9, p.unit_margin, f">= {gcfg['min_unit_margin']}"),
        _chk("max_discount", p.discount <= gcfg["max_discount"] + 1e-9, p.discount, f"<= {gcfg['max_discount']}"),
        _chk("expected_net_finite", bool(np.isfinite(p.expected_net)), p.expected_net),
    ]
    return _finish(4, checks, t0)


# --------------------------------------------------------------------------- Gate 5
def gate5_evaluation(ev: dict, target_lift_pct: float) -> GateResult:
    t0 = time.perf_counter()
    ope = ev["ope"]
    checks = [
        _chk("ope_dr_lift_ci_lower_gt_0", ope["lift_ci_low"] > 0, ope["lift_ci_low"], "> 0 (95% CI)"),
        _chk("ope_ips_dr_agree_sign", np.sign(ope["ips_lift"]) == np.sign(ope["dr_lift"]), f"ips={ope['ips_lift']:.4f} dr={ope['dr_lift']:.4f}", critical=False),
        _chk("lift_vs_static_ge_target", ev["lift_vs_static_pct"] >= target_lift_pct, ev["lift_vs_static_pct"], f">= {target_lift_pct}%"),
        _chk("lift_vs_heuristic_ge_target", ev["lift_vs_heuristic_pct"] >= target_lift_pct, ev["lift_vs_heuristic_pct"], f">= {target_lift_pct}%"),
        _chk("replay_guardrail_violations", ev["guardrail_violations"] == 0, ev["guardrail_violations"], "== 0"),
        _chk("all_regimes_evaluated", len(ev["regimes"]) >= 4, len(ev["regimes"]), ">= 4"),
        _chk("agent_beats_static_in_every_regime", all(r["lift_vs_static_pct"] > 0 for r in ev["regimes"].values()),
             min(r["lift_vs_static_pct"] for r in ev["regimes"].values()), "> 0", critical=False),
    ]
    return _finish(5, checks, t0)


# --------------------------------------------------------------------------- Gate 6
def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


REQUIRED_ARTIFACTS = [
    ("validation", "drift_report.json"), ("validation", "schema_audit.json"),
    ("evaluation", "benchmark_comparison.csv"), ("evaluation", "ope_replay_results.json"),
    ("evaluation", "regime_breakdown.json"), ("metrics", "gpu_telemetry.json"),
    ("metrics", "step_timings.csv"), ("data", "features_snapshot.parquet"),
    ("data", "counterfactual_replay.parquet"),
]


def gate6_execution(exec_payload: dict, audit_rows: list[dict], gate1_5_report: dict | None = None) -> GateResult:
    t0 = time.perf_counter()
    missing, empty, hashes = [], [], {}
    for parts in REQUIRED_ARTIFACTS:
        p = out_dir(*parts)
        if not p.exists():
            missing.append("/".join(parts))
        elif p.stat().st_size == 0:
            empty.append("/".join(parts))
        else:
            hashes["/".join(parts)] = file_sha(p)
    unparsable = []
    for parts in REQUIRED_ARTIFACTS:
        p = out_dir(*parts)
        if p.suffix == ".json" and p.exists():
            try:
                json.loads(p.read_text())
            except Exception:
                unparsable.append(p.name)
    actors = {r.get("actor") for r in audit_rows}
    llm_rows = [r for r in audit_rows if r.get("actor", "").startswith("llm")]
    checks = [
        _chk("required_artifacts_exist", not missing, len(missing), "== 0", detail=str(missing)),
        _chk("artifacts_non_empty", not empty, len(empty), "== 0", detail=str(empty)),
        _chk("json_artifacts_parse", not unparsable, len(unparsable), "== 0", detail=str(unparsable)),
        _chk("execution_is_dry_run_or_approved", exec_payload.get("approved", False), exec_payload.get("approved"),
             "plan approved (auto or human)"),
        _chk("execution_payload_guardrail_clean", exec_payload.get("guardrail_ok", False), exec_payload.get("guardrail_ok")),
        _chk("agent_audit_has_decisions", len(audit_rows) >= 6, len(audit_rows), ">= 6"),
        _chk("audit_has_llm_or_mock_entries", len(llm_rows) >= 1, len(llm_rows), ">= 1 (live or mock)"),
        _chk("audit_covers_agent_stages", {"optimize", "guardrails", "execute"} <= actors, len(actors), ">= 3 stages", critical=False, detail=str(sorted(map(str, actors)))),
    ]
    out_dir("validation", "artifact_hashes.json").write_text(json.dumps(hashes, indent=2))
    return _finish(6, checks, t0)
