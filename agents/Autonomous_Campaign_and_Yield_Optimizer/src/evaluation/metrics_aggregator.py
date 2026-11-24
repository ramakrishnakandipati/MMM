"""Compiles latency, GPU utilisation, yield lift and guardrail KPIs into system_metrics.json."""
from __future__ import annotations

import json

import pandas as pd

from ..utils.config import out_dir


def _py(o):
    import numpy as np
    if isinstance(o, dict):
        return {str(k): _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def aggregate(evaluation: dict, ope: dict, device_info: dict, extra: dict | None = None) -> dict:
    tp = out_dir("metrics", "step_timings.csv")
    timings = {}
    if tp.exists():
        df = pd.read_csv(tp)
        timings = df.groupby("stage")["seconds"].last().round(4).to_dict()
    rp = out_dir("validation", "step_validation_report.json")
    gates = json.loads(rp.read_text()).get("summary", {}) if rp.exists() else {}
    m = {
        "yield": {
            "lift_vs_static_pct": evaluation["lift_vs_static_pct"],
            "lift_vs_heuristic_pct": evaluation["lift_vs_heuristic_pct"],
            "total_net": evaluation["total_net"],
            "target_met": bool(evaluation["lift_vs_static_pct"] >= 8 and evaluation["lift_vs_heuristic_pct"] >= 8),
        },
        "ope": {k: ope[k] for k in ("ips_lift", "dr_lift", "lift_ci_low", "lift_ci_high", "bootstrap_ms", "n_boot")},
        "guardrails": {"violations": evaluation["guardrail_violations"]},
        "latency": {"plan_ms_mean": evaluation["plan_latency_ms_mean"], "plan_ms_p95": evaluation["plan_latency_ms_p95"],
                    "stage_seconds": timings},
        "gpu": device_info,
        "validation": gates,
        **(extra or {}),
    }
    out_dir("metrics", "system_metrics.json").write_text(json.dumps(_py(m), indent=2, default=str))
    return m
