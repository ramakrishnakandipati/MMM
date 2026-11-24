"""Injects Mermaid diagrams, live validation tables and metrics between marker comments in the docs."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from .config import ROOT, out_dir

MERMAID_FLOW = """```mermaid
flowchart LR
    subgraph S1[Step 1 - Data]
      A[Synthetic world<br/>ads + inventory + COGS<br/>counterfactual truth] --> G1{{Gate 1<br/>integrity}}
    end
    subgraph S2[Step 2 - Features]
      G1 --> B[Adstock + supply merge<br/>train / live / OOS split] --> G2{{Gate 2<br/>schema + PSI<0.25}}
    end
    subgraph S3[Step 3 - GPU models]
      G2 --> C1[Hill+adstock net<br/>AdamW + cosine]
      G2 --> C2[T-Learner CATE]
      G2 --> C3[Batched LinUCB]
      C1 & C2 & C3 --> G3{{Gate 3<br/>R2, concavity,<br/>CATE var, cov}}
      G3 -.telemetry.-> L1[(Qwen3-VL<br/>diagnose - advisory)]
    end
    subgraph S4[Step 4 - Decide]
      G3 --> D[CVXPY LP per discount tier<br/>+ guardrail clamp] --> G4{{Gate 4<br/>optimal, shift<=15%,<br/>inventory, margin}}
    end
    subgraph S5[Step 5 - Evaluate]
      G4 --> E[Replay vs static/heuristic<br/>5 regimes + GPU OPE IPS/DR] --> G5{{Gate 5<br/>CI>0, lift>=8%}}
    end
    subgraph S6[Step 6 - Execute]
      G5 --> V[Visual QA<br/>Qwen3-VL on charts] --> H{HITL?}
      H -- approve / auto --> X[Dry-run ad + pricing APIs] --> G6{{Gate 6<br/>artifact integrity}}
      H -- reject --> HALT((HALT))
    end
    G1 & G2 & G3 & G4 & G5 & G6 -. critical fail .-> HALT
```"""

MERMAID_LINEAGE = """```mermaid
flowchart TD
    W[World params<br/>beta,K,S,lambda,organic] --> RAW[(raw_daily.parquet)]
    RAW --> FEAT[(features_snapshot.parquet)]
    FEAT --> RM[ResponseModel] --> CURVES[steady-state curves] --> OPT[BudgetOptimizer]
    UPL[(uplift cohorts)] --> UM[UpliftModel] --> OPT
    BLOG[(bandit log + propensities)] --> LU[BatchedLinUCB] --> OPT
    OPT --> PLAN[AllocationPlan] --> GR[GuardrailEngine] --> PLAN2[guardrailed plan]
    PLAN2 --> SIM[Counterfactual simulator] --> REPLAY[(counterfactual_replay.parquet)]
    BLOG --> OPE[OPE IPS/DR + GPU bootstrap] --> OPEJ[(ope_replay_results.json)]
    REPLAY --> BENCH[(benchmark_comparison.csv)]
```"""


def _replace_block(text: str, name: str, body: str) -> str:
    pat = re.compile(rf"(<!-- BEGIN:{name} -->)(.*?)(<!-- END:{name} -->)", re.S)
    if pat.search(text):
        return pat.sub(lambda m: f"{m.group(1)}\n{body}\n{m.group(3)}", text)
    return text + f"\n<!-- BEGIN:{name} -->\n{body}\n<!-- END:{name} -->\n"


def _gate_table(report: dict) -> str:
    rows = ["| Gate | Stage | Result | Checks (pass/total) | Failed |", "|---|---|---|---|---|"]
    for g in sorted(report.get("gates", {}).values(), key=lambda x: x["gate"]):
        ok = sum(c["passed"] for c in g["checks"])
        failed = ", ".join(c["name"] for c in g["checks"] if not c["passed"]) or "-"
        rows.append(f"| {g['gate']} | {g['name']} | {'PASS' if g['passed'] else '**FAIL**'} | {ok}/{len(g['checks'])} | {failed} |")
    s = report.get("summary", {})
    rows.append(f"\n**Pass rate: {s.get('gates_passed', 0)}/6 gates ({s.get('pass_rate', 0):.0%})**")
    return "\n".join(rows)


def _metrics_block(m: dict) -> str:
    if not m:
        return "_Run `python run_all.py` to populate._"
    y, o, g = m["yield"], m["ope"], m.get("gpu", {})
    reg = json.loads(out_dir("evaluation", "regime_breakdown.json").read_text()) if out_dir("evaluation", "regime_breakdown.json").exists() else {}
    lines = [
        f"_Last run: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} on `{g.get('device_name', '?')}` "
        f"({g.get('vram_mb', 0)} MB VRAM, CUDA {g.get('cuda_version')})_", "",
        "| KPI | Value |", "|---|---|",
        f"| Net yield lift vs static | **{y['lift_vs_static_pct']:+.1f}%** |",
        f"| Net yield lift vs heuristic | **{y['lift_vs_heuristic_pct']:+.1f}%** |",
        f"| Target (>= +8%) met | {'yes' if y['target_met'] else 'NO'} |",
        f"| OPE DR lift (95% CI) | {o['dr_lift']:+.1%} [{o['lift_ci_low']:+.1%}, {o['lift_ci_high']:+.1%}] |",
        f"| OPE IPS lift | {o['ips_lift']:+.1%} |",
        f"| GPU bootstrap ({o['n_boot']:,} resamples) | {o['bootstrap_ms']:.0f} ms |",
        f"| Guardrail violations in replay | {m['guardrails']['violations']} |",
        f"| Plan latency mean / p95 | {m['latency']['plan_ms_mean']:.1f} / {m['latency']['plan_ms_p95']:.1f} ms |",
        "", "| Regime | Agent lift vs static | Agent lift vs heuristic | Violations |", "|---|---|---|---|",
    ]
    for k, v in reg.items():
        lines.append(f"| {k} | {v['lift_vs_static_pct']:+.1f}% | {v['lift_vs_heuristic_pct']:+.1f}% | {v['guardrail_violations']} |")
    lines += ["", "![benchmark](outputs/charts/benchmark_regimes.png)", "![ope](outputs/charts/ope_lift.png)",
              "![response](outputs/charts/response_curves.png)"]
    return "\n".join(lines)


def _scenarios_block() -> str:
    p = out_dir("scenarios", "scenario_results.json")
    if not p.exists():
        return "_Run `python run_all.py` to populate._"
    R = json.loads(p.read_text())
    rows = ["| # | Scenario | Outcome | Discount | Lift vs static | Lift vs heuristic | Step statuses (1-6) | HITL |",
            "|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(R.values(), 1):
        stat = " ".join({"PASS": "P", "PASS-HITL": "P+H", "WARN": "W", "FAIL": "F", "SKIPPED": "-"}[s["status"]] for s in r["steps"])
        if r["halted"]:
            rows.append(f"| {i} | {r['label']} | **HALT** | - | - | - | {stat} | halted |")
        else:
            rp = r["replay"]
            rows.append(f"| {i} | {r['label']} | completed | {r['plan']['discount']:.0%} | {rp['lift_vs_static_pct']:+.1f}% | "
                        f"{rp['lift_vs_heuristic_pct']:+.1f}% | {stat} | {r['hitl']['status']} |")
    out = ["\n".join(rows), "", "_P = pass, P+H = pass but human approval required, W = warn (below +8% target), F = fail, - = skipped_", ""]
    for i, r in enumerate(R.values(), 1):
        out += [f"#### {r['label']}", f"{r['story']}", "", f"_Expected:_ {r['expect']}", "",
                f"![{r['id']}](outputs/charts/scenario_{r['id']}.png)", ""]
    return "\n".join(out)


def _screenshot_block() -> str:
    shots = sorted(out_dir("screenshots").glob("*.png"))
    groups = [("Flask control portal", "flask_"), ("Streamlit - live run, HITL approval, validation, GPU", "streamlit_"),
              ("Streamlit - Scenario Studio (end-to-end examples)", "scenario_")]
    out = []
    for title, pre in groups:
        g = [s for s in shots if s.name.startswith(pre)]
        if g:
            out += [f"**{title}**", ""] + [f"![{s.stem}](outputs/screenshots/{s.name})" for s in g] + [""]
    return "\n".join(out)


def update_docs() -> list[str]:
    rp = out_dir("validation", "step_validation_report.json")
    mp = out_dir("metrics", "system_metrics.json")
    report = json.loads(rp.read_text()) if rp.exists() else {}
    metrics = json.loads(mp.read_text()) if mp.exists() else {}
    touched = []
    for name in ("README.md", "ARCHITECTURE_AND_DATA_FLOW.md"):
        p = ROOT / name
        if not p.exists():
            continue
        t = p.read_text(encoding="utf-8")
        t = _replace_block(t, "FLOW", MERMAID_FLOW)
        t = _replace_block(t, "LINEAGE", MERMAID_LINEAGE)
        t = _replace_block(t, "GATES", _gate_table(report))
        t = _replace_block(t, "METRICS", _metrics_block(metrics))
        t = _replace_block(t, "SCENARIOS", _scenarios_block())
        if list(out_dir("screenshots").glob("*.png")):
            t = _replace_block(t, "SCREENSHOTS", _screenshot_block())
        p.write_text(t, encoding="utf-8")
        touched.append(name)
    return touched
