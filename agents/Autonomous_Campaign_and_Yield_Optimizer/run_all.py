"""Master orchestrator: runs the full pipeline with validation gates, renders visuals,
launches both UIs, captures screenshots, and refreshes the docs.

    python run_all.py [--device auto|cuda|cpu] [--skip-ui-shots] [--skip-docs]
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.agent.graph import AgentService  # noqa: E402
from src.agent.scenarios import SCENARIOS, run_all_scenarios  # noqa: E402
from src.utils import visualizer as viz  # noqa: E402
from src.utils.config import out_dir  # noqa: E402
from src.utils.logger import log_event, reset_logs  # noqa: E402
from src.utils.readme_updater import update_docs  # noqa: E402


def banner(msg: str):
    print(f"\n=== {msg} " + "=" * max(3, 70 - len(msg)))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_http(url: str, timeout: float = 90) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def launch_browser(p):
    last = None
    for kw in ({"channel": "msedge"}, {"channel": "chrome"}, {}):
        try:
            return p.chromium.launch(**kw)
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"no usable Chromium browser for Playwright: {last}")


def capture_uis() -> dict:
    """Start Flask + Streamlit, drive both with Playwright, save PNGs. Best-effort; never fails the pipeline."""
    shots, procs = {}, []
    shot_dir = out_dir("screenshots")
    fport, sport = free_port(), free_port()
    env = {**os.environ, "FLASK_PORT": str(fport), "PYTHONPATH": str(ROOT), "PYTHONWARNINGS": "ignore"}
    try:
        from playwright.sync_api import sync_playwright
        procs.append(subprocess.Popen([sys.executable, "-m", "src.api.flask_app"], cwd=ROOT, env=env,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        procs.append(subprocess.Popen([sys.executable, "-m", "streamlit", "run", "src/ui/streamlit_app.py", "--server.port", str(sport),
                                       "--server.headless", "true", "--browser.gatherUsageStats", "false"], cwd=ROOT, env=env,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        flask_ok = wait_http(f"http://127.0.0.1:{fport}/health")
        st_ok = wait_http(f"http://127.0.0.1:{sport}/_stcore/health")
        print(f"  flask up: {flask_ok}  streamlit up: {st_ok}")
        with sync_playwright() as p:
            br = launch_browser(p)
            if flask_ok:
                try:
                    pg = br.new_page(viewport={"width": 1280, "height": 1500})
                    pg.goto(f"http://127.0.0.1:{fport}/")
                    pg.screenshot(path=str(shot_dir / "flask_01_idle.png"))
                    pg.click("#run")
                    pg.wait_for_function("document.querySelector('#runstate').textContent==='complete'", timeout=240000)
                    time.sleep(1.5)
                    pg.screenshot(path=str(shot_dir / "flask_02_complete.png"), full_page=True)
                    shots["flask"] = True
                    pg.close()
                except Exception as e:  # noqa: BLE001
                    print(f"  flask capture failed: {type(e).__name__}: {e}")
            if st_ok:
                try:
                    pg = br.new_page(viewport={"width": 1400, "height": 2600})
                    pg.goto(f"http://127.0.0.1:{sport}/")
                    pg.get_by_role("button", name="Run live cycle").wait_for(timeout=60000)
                    pg.get_by_role("button", name="Run live cycle").click()
                    pg.get_by_role("tab", name="HITL Studio").click()
                    pg.get_by_text("Human approval required").wait_for(timeout=240000)
                    time.sleep(1)
                    pg.screenshot(path=str(shot_dir / "streamlit_01_hitl_pending.png"), full_page=True)
                    pg.locator("button", has_text="Approve").click()
                    pg.wait_for_function("document.querySelector(\"section[data-testid='stSidebar']\").innerText.includes('complete')",
                                         timeout=180000)
                    pg.get_by_role("tab", name="Flow Visualizer").click()
                    time.sleep(3)
                    pg.screenshot(path=str(shot_dir / "streamlit_02_flow.png"), full_page=True)
                    pg.get_by_role("tab", name="Validation").click()
                    time.sleep(2)
                    pg.screenshot(path=str(shot_dir / "streamlit_03_validation.png"), full_page=True)
                    pg.get_by_role("tab", name="GPU Metrics").click()
                    time.sleep(2)
                    pg.screenshot(path=str(shot_dir / "streamlit_04_gpu.png"), full_page=True)
                    # ---- Scenario Studio: run all examples, capture overview + one end-to-end flow per scenario
                    pg.get_by_role("tab", name="Scenario Studio").click()
                    pg.locator("button", has_text="Run ALL scenarios").click()
                    pg.get_by_text("All scenarios compared").wait_for(timeout=180000)
                    time.sleep(3)
                    pg.screenshot(path=str(shot_dir / "scenario_00_overview.png"), full_page=True)
                    for i, sc in enumerate(SCENARIOS, start=1):
                        pg.locator("label", has_text=sc["label"]).first.click()
                        pg.get_by_role("heading", name="Flow: " + sc["label"]).wait_for(timeout=60000)
                        time.sleep(2.5)
                        pg.screenshot(path=str(shot_dir / f"scenario_{i:02d}_{sc['id']}.png"), full_page=True)
                    shots["scenarios"] = len(SCENARIOS)
                    shots["streamlit"] = True
                    pg.close()
                except Exception as e:  # noqa: BLE001
                    print(f"  streamlit capture failed: {type(e).__name__}: {e}")
            br.close()
    except Exception as e:  # noqa: BLE001
        print(f"  UI capture skipped: {type(e).__name__}: {e}")
    finally:
        for pr in procs:
            pr.terminate()
    return shots


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="auto|cuda|mps|cpu")
    ap.add_argument("--skip-ui-shots", action="store_true")
    ap.add_argument("--skip-docs", action="store_true")
    args = ap.parse_args()
    t_all = time.perf_counter()

    reset_logs("flask_access.jsonl")
    if not args.skip_ui_shots:
        banner("UI screenshots (Flask + Streamlit via Playwright; UI-driven runs, then canonical run below)")
        shots = capture_uis()
        print(f"  captured: {shots}  files: {sorted(p.name for p in out_dir('screenshots').glob('*.png'))}")
    banner("Pipeline (LangGraph, 6 steps with validation gates)")
    st = AgentService.start(auto_approve=True, device=args.device)
    ctx = AgentService.context(st["run_id"])
    print(f"  run_id={st['run_id']}  device={ctx.dm.info()['device_name']} ({ctx.dm.type})  halted={st.get('halted')}")

    banner("Validation gates")
    report = json.loads(out_dir("validation", "step_validation_report.json").read_text())
    for g in sorted(report["gates"].values(), key=lambda x: x["gate"]):
        ok = sum(c["passed"] for c in g["checks"])
        print(f"  Gate {g['gate']}  {'PASS' if g['passed'] else 'FAIL'}  {ok}/{len(g['checks'])} checks  {g['name']}")
        for c in g["checks"]:
            if not c["passed"]:
                print(f"      - FAILED {c['name']}: value={c['value']} threshold={c['threshold']} {c['detail']}")
    summ = report["summary"]
    print(f"  pass rate: {summ['gates_passed']}/6")

    ev = st.get("evaluation")
    if ev:
        banner("Evaluation")
        print(f"  net-yield lift vs static    : {ev['lift_vs_static_pct']:+.1f}%")
        print(f"  net-yield lift vs heuristic : {ev['lift_vs_heuristic_pct']:+.1f}%")
        print(f"  DR OPE lift                 : {ev['dr_lift']:+.1%}  95% CI [{ev['ci'][0]:+.1%}, {ev['ci'][1]:+.1%}]")
        print(f"  guardrail violations        : {ev['guardrail_violations']}")
    if st.get("diagnosis"):
        print(f"  LLM diagnose source={st['diagnosis']['meta']['source']}  visual source={st['visual']['meta']['source']} "
              f"verdict={st['visual']['verdict']}  HITL={st['hitl']['status']}")

    banner("End-to-end scenarios (6 examples through the same 6-step flow)")
    from src.utils import visualizer as _v
    sc_res = run_all_scenarios(ctx)
    for r in sc_res.values():
        _v.scenario_chart(r)
        if r["halted"]:
            print(f"  {r['label']:<40} HALTED  {r['halt_reason']}")
        else:
            rp = r["replay"]
            print(f"  {r['label']:<40} discount {r['plan']['discount']:>4.0%}  lift vs static {rp['lift_vs_static_pct']:+6.1f}%  "
                  f"vs heuristic {rp['lift_vs_heuristic_pct']:+6.1f}%  HITL={r['hitl']['status']}")

    banner("Visualisations")
    statuses = {int(k): ("pass" if v["passed"] else "fail") for k, v in st["gates"].items()}
    viz.flow_diagram(statuses)
    print(f"  charts: {sorted(p.name for p in out_dir('charts').glob('*'))}")

    if not args.skip_docs:
        banner("Docs")
        print(f"  updated: {update_docs()}")

    total = time.perf_counter() - t_all
    log_event("run_all", "complete", seconds=round(total, 2), all_gates_passed=summ["all_passed"])
    print(f"\nTotal wall time: {total:.1f}s")
    target_ok = ev is not None and ev["lift_vs_static_pct"] >= 8 and ev["lift_vs_heuristic_pct"] >= 8
    return 0 if (summ["all_passed"] and target_ok and not st.get("halted")) else 1


if __name__ == "__main__":
    sys.exit(main())
