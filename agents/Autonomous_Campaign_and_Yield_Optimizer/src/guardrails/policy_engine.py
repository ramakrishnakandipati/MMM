"""Deterministic guardrail engine: declarative OPA-style rules + clamp/repair logic."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Rule:
    id: str
    description: str
    predicate: Callable[[dict, dict], bool]   # (plan, ctx) -> True if compliant


class GuardrailEngine:
    def __init__(self, g: dict):
        self.g = g
        self.rules = [
            Rule("budget_cap", "Total daily budget must not exceed B_max",
                 lambda p, c: sum(p["budgets"].values()) <= g["max_total_budget"] * (1 + 1e-9)),
            Rule("channel_shift", "Single-cycle channel change <= max_channel_shift_pct",
                 lambda p, c: all(abs(b - p["prev_budgets"][k]) <= g["max_channel_shift_pct"] * p["prev_budgets"][k] + 1e-6
                                  for k, b in p["budgets"].items() if p["prev_budgets"][k] > 0)),
            Rule("inventory_discount_suppression", "No discounts when inventory utilization above cap",
                 lambda p, c: not (c["inventory_util"] > g["inventory_discount_cap"] and p["discount"] > 0)),
            Rule("unit_margin_floor", "Discounted unit margin >= floor",
                 lambda p, c: c["gm"] - p["discount"] >= g["min_unit_margin"] - 1e-9),
            Rule("max_discount", "Discount tier <= max_discount", lambda p, c: p["discount"] <= g["max_discount"] + 1e-9),
            Rule("non_negative", "Budgets non-negative", lambda p, c: all(b >= -1e-9 for b in p["budgets"].values())),
        ]

    # ---- constraints used by the planner -------------------------------------------------
    def allowed_tiers(self, tiers: list[float], util: float, gm: float) -> list[float]:
        out = []
        for d in tiers:
            if d > self.g["max_discount"] + 1e-12:
                continue
            if gm - d < self.g["min_unit_margin"] - 1e-12:
                continue
            if d > 0 and util > self.g["inventory_discount_cap"]:
                continue
            out.append(d)
        return out or [0.0]

    def bounds(self, prev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = self.g["max_channel_shift_pct"]
        lo = np.maximum(prev * (1 - s), self.g["min_channel_budget"] * 0 + 0.0)
        floor = self.g["min_channel_budget"]
        lo = np.where(prev * (1 - s) < floor, np.minimum(floor, prev * (1 + s)), prev * (1 - s))
        hi = np.minimum(prev * (1 + s), self.g["max_total_budget"])
        return lo, hi

    # ---- verification & repair ----------------------------------------------------------
    def evaluate(self, plan: dict, ctx: dict) -> list[str]:
        return [r.id for r in self.rules if not r.predicate(plan, ctx)]

    def enforce(self, plan: dict, ctx: dict) -> dict:
        """Clamp / repair a plan deterministically; records every clamp."""
        g, clamps = self.g, []
        chans = list(plan["budgets"])
        prev = np.array([plan["prev_budgets"][c] for c in chans], float)
        b = np.array([plan["budgets"][c] for c in chans], float)
        lo, hi = self.bounds(prev)
        b2 = np.clip(b, lo, hi)
        if np.abs(b2 - b).max() > 1e-6:
            clamps.append("channel_shift_clamped")
        b = b2
        if b.sum() > g["max_total_budget"]:
            excess = b.sum() - g["max_total_budget"]
            room = b - lo
            b = b - room / max(room.sum(), 1e-12) * min(excess, room.sum())
            clamps.append("budget_cap_scaled")
        d = plan["discount"]
        if d > 0 and ctx["inventory_util"] > g["inventory_discount_cap"]:
            d = 0.0
            clamps.append("discount_suppressed_inventory")
        if ctx["gm"] - d < g["min_unit_margin"]:
            d = 0.0
            clamps.append("discount_removed_margin_floor")
        if d > g["max_discount"]:
            d = g["max_discount"]
            clamps.append("discount_capped")
        out = dict(plan)
        out["budgets"] = {c: float(v) for c, v in zip(chans, b)}
        out["discount"] = float(d)
        out["unit_margin"] = float(ctx["gm"] - d)
        out["clamps"] = plan.get("clamps", []) + clamps
        out["guardrail_ok"] = not self.evaluate(out, ctx)
        return out
