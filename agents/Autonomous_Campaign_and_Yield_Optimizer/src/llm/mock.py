"""Deterministic offline stand-in used in CI / when the local server is paused."""
from __future__ import annotations

import numpy as np


def diagnose(t: dict) -> dict:
    hyps, risk = [], "low"
    r2 = t.get("r2_holdout", 1.0)
    if r2 < 0.85:
        hyps.append({"cause": "response model has limited explanatory power (noisy conversions)",
                     "evidence": f"holdout R2={r2:.3f}", "confidence": 0.5})
        risk = "medium"
    if t.get("max_psi", 0) >= 0.2:
        hyps.append({"cause": "feature drift between training and live windows",
                     "evidence": f"max PSI={t['max_psi']:.3f}", "confidence": 0.7})
        risk = "high" if t["max_psi"] >= 0.25 else "medium"
    if t.get("inventory_util", 0) > 0.85:
        hyps.append({"cause": "supply constraint: discounts suppressed by inventory guardrail",
                     "evidence": f"util={t['inventory_util']:.2f}", "confidence": 0.9})
    if not hyps:
        hyps.append({"cause": "no anomaly detected", "evidence": "all monitored telemetry within bounds", "confidence": 0.8})
    return {"summary": "Deterministic mock diagnosis.", "hypotheses": hyps, "risk": risk,
            "recommend_escalation": risk == "high"}


def visual(paths, context: dict) -> dict:
    findings, verdict = [], "ok"
    try:
        from PIL import Image
        for p in paths:
            a = np.asarray(Image.open(p).convert("L"), dtype=float)
            if a.std() < 5:
                findings.append(f"{getattr(p, 'name', p)}: image nearly blank/uniform")
                verdict = "anomaly"
    except Exception as e:
        findings.append(f"image unreadable: {e}")
        verdict = "anomaly"
    if not findings:
        findings.append("charts non-blank; deterministic mock cannot assess curve shape (numeric Gate 3 already verified it)")
    return {"verdict": verdict, "findings": findings, "confidence": 0.4}
