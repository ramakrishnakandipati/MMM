import os

import pytest

from src.agent import nodes as nodes_mod
from src.agent.graph import AgentService
from src.llm.client import LocalLLM


def test_happy_path_all_gates_pass(pipeline_run):
    st = pipeline_run
    assert not st["halted"], st.get("halt_reason")
    assert st["step"] == 6
    assert set(st["gates"]) == {"1", "2", "3", "4", "5", "6"} and all(g["passed"] for g in st["gates"].values())
    assert st["plan"]["guardrail_ok"] and st["executed"]["dry_run"]
    assert st["evaluation"]["lift_vs_static_pct"] >= 8 and st["evaluation"]["lift_vs_heuristic_pct"] >= 8
    assert st["diagnosis"]["meta"]["source"] == "mock" and st["visual"]["meta"]["source"] == "mock"


def test_audit_log_has_all_actors(pipeline_run):
    from src.utils.logger import read_jsonl
    actors = {r["actor"] for r in read_jsonl("agent_decision_audit.jsonl")}
    assert {"ingest", "features", "optimize", "guardrails", "evaluate", "execute", "llm_diagnose", "llm_visual_verify"} <= actors


def test_halt_on_corrupted_data(monkeypatch):
    real = nodes_mod.generate_all

    def corrupt(cfg):
        d = real(cfg)
        d["daily"].loc[7, "spend_search"] = -5.0
        return d

    monkeypatch.setattr(nodes_mod, "generate_all", corrupt)
    st = AgentService.start(auto_approve=True)
    assert st["halted"] and "Gate 1" in st["halt_reason"]
    assert st["step"] == 1
    assert st.get("plan") is None            # nothing downstream ran


def test_hitl_interrupt_approve_and_reject():
    rid = AgentService.create(auto_approve=False)
    AgentService.execute(rid)
    st = AgentService.status(rid)
    assert st["waiting_for_human"] and st["step"] == 5 and st["pending_interrupt"]["reasons"]
    assert not st.get("executed")
    st = AgentService.resume(rid, "approve", "tester")
    assert not st["waiting_for_human"] and st["step"] == 6 and st["hitl"]["status"] == "approved"

    rid2 = AgentService.create(auto_approve=False)
    AgentService.execute(rid2)
    st2 = AgentService.resume(rid2, "reject", "tester", "no")
    assert st2["halted"] and "rejected" in st2["halt_reason"].lower() and not st2.get("executed")


def test_llm_mock_fallback_when_unreachable(cfg):
    os.environ.pop("DISABLE_LLM", None)
    try:
        llm = LocalLLM({**cfg, "llm": {**cfg["llm"], "enabled": True, "base_url": "http://127.0.0.1:9", "probe_ports": [9]}})
        d, meta = llm.diagnose({"r2_holdout": 0.7, "max_psi": 0.3})
    finally:
        os.environ["DISABLE_LLM"] = "1"
    assert meta["source"] == "mock" and d.risk in ("medium", "high")


def test_live_llm_if_reachable(cfg):
    os.environ.pop("DISABLE_LLM", None)
    try:
        llm = LocalLLM(cfg)
        if not llm.available:
            pytest.skip("local LLM server not reachable")
        d, meta = llm.diagnose({"r2_holdout": 0.9, "max_psi": 0.1})
        assert meta["source"] == "llm" and d.summary
    finally:
        os.environ["DISABLE_LLM"] = "1"
