import json

DIAG_SYSTEM = (
    "You are a marketing-science diagnostics assistant for an autonomous campaign budget optimizer. "
    "Given pipeline telemetry, identify likely root causes of any anomaly or risk. "
    "Reply with ONLY a JSON object: {\"summary\": str, \"hypotheses\": [{\"cause\": str, \"evidence\": str, \"confidence\": 0..1}], "
    "\"risk\": \"low\"|\"medium\"|\"high\", \"recommend_escalation\": bool}. No prose outside JSON."
)

VISUAL_SYSTEM = (
    "You are a visual QA reviewer for marketing response-curve and budget-allocation charts. "
    "Check: response curves are monotonic increasing with diminishing returns; no flat/blank/garbled plots; "
    "allocation bars sum plausibly and no channel shows an abrupt implausible jump. "
    "Reply with ONLY JSON: {\"verdict\": \"ok\"|\"anomaly\"|\"escalate\", \"findings\": [str], \"confidence\": 0..1}."
)


def diagnose_messages(telemetry: dict) -> list[dict]:
    return [{"role": "system", "content": DIAG_SYSTEM},
            {"role": "user", "content": "Telemetry:\n" + json.dumps(telemetry, default=str)[:6000]}]


def visual_prompt(context: dict) -> str:
    return "Review the attached chart image(s). Context: " + json.dumps(context, default=str)[:1500]
