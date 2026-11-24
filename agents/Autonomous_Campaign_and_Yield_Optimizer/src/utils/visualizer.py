"""Data-side, flow-side and model-evaluation charts (PNG + SVG for the flow diagram)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .config import out_dir

PALETTE = ["#2a6fdb", "#e8833a", "#2fa36b", "#9b59b6", "#c0392b", "#16a2b8"]
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "figure.dpi": 110,
                     "axes.prop_cycle": matplotlib.cycler(color=PALETTE)})


def _save(fig, name: str) -> str:
    p = out_dir("charts", name)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(p)


def data_distributions(daily, channels) -> str:
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for i, c in enumerate(channels):
        ax[0].hist(daily[f"spend_{c}"], bins=40, alpha=0.55, label=c, color=PALETTE[i])
    ax[0].set(title="Daily spend distribution", xlabel="$ / day")
    ax[0].legend(fontsize=7)
    ax[1].plot(daily["day_idx"], daily["conversions"], lw=0.8)
    ax[1].set(title="Observed conversions", xlabel="day")
    ax[2].plot(daily["day_idx"], daily["inventory_util"], lw=0.9, color=PALETTE[4])
    ax[2].axhline(0.85, ls="--", color="grey", lw=0.8)
    ax[2].text(2, 0.86, "discount cap 0.85", fontsize=7)
    ax[2].set(title="Inventory / capacity utilisation", xlabel="day")
    return _save(fig, "data_distributions.png")


def response_curves(curves: dict, channels, true_curves: dict | None = None) -> str:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for c, (g, r) in curves.items():
        ax[0].plot(g, r, color=PALETTE[c], label=channels[c])
        if true_curves is not None:
            ax[0].plot(g, true_curves[c], color=PALETTE[c], ls=":", lw=1)
        ax[1].plot(g[1:], np.diff(r) / np.diff(g), color=PALETTE[c], label=channels[c])
    ax[0].set(title="Steady-state response (solid=fitted, dotted=truth)", xlabel="daily spend $", ylabel="conversions/day")
    ax[1].set(title="Marginal conversions per $ (diminishing returns)", xlabel="daily spend $")
    ax[0].legend(fontsize=7)
    return _save(fig, "response_curves.png")


def allocation_chart(prev: dict, new: dict, discount: float) -> str:
    ch = list(new)
    x = np.arange(len(ch))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(x - 0.2, [prev[c] for c in ch], 0.4, label="previous", color="#b7c4d6")
    ax.bar(x + 0.2, [new[c] for c in ch], 0.4, label="proposed", color=PALETTE[0])
    ax.set_xticks(x, ch)
    ax.set(title=f"Budget allocation (discount tier {discount:.0%})", ylabel="$ / day")
    ax.legend()
    return _save(fig, "allocation.png")


def ope_plot(ope: dict, lifts: np.ndarray | None = None) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    lo, hi, m = ope["lift_ci_low"], ope["lift_ci_high"], ope["dr_lift"]
    if lifts is not None:
        ax.hist(lifts, bins=60, color="#b7c4d6")
    ax.axvline(m, color=PALETTE[0], label=f"DR lift {m:.1%}")
    ax.axvspan(lo, hi, color=PALETTE[0], alpha=0.15, label=f"95% CI [{lo:.1%}, {hi:.1%}]")
    ax.axvline(ope["ips_lift"], color=PALETTE[1], ls="--", label=f"IPS {ope['ips_lift']:.1%}")
    ax.axvline(0, color="grey", lw=0.8)
    ax.set(title=f"OPE bootstrap (N={ope['n_boot']:,})", xlabel="relative lift vs static arm")
    ax.legend(fontsize=7)
    return _save(fig, "ope_lift.png")


def benchmark_chart(ev: dict) -> str:
    regs = list(ev["regimes"])
    x = np.arange(len(regs))
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, p in enumerate(["static", "heuristic", "agent"]):
        ax.bar(x + (i - 1) * 0.27, [ev["regimes"][r]["net"][p] / 1e6 for r in regs], 0.27, label=p,
               color=["#b7c4d6", "#e8b27a", PALETTE[0]][i])
    ax.set_xticks(x, regs, rotation=15)
    ax.set(title="Net contribution margin per regime (out-of-sample replay)", ylabel="$ millions")
    ax.legend()
    return _save(fig, "benchmark_regimes.png")


def drift_chart(rep: dict) -> str:
    keys = [k for k in rep if not k.startswith("_")]
    v = [rep[k]["psi"] for k in keys]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.barh(keys, v, color=[PALETTE[4] if x >= 0.25 else PALETTE[0] for x in v])
    ax.axvline(0.25, ls="--", color="grey")
    ax.set(title="PSI train vs live (threshold 0.25)", xlabel="PSI")
    return _save(fig, "drift_psi.png")


FLOW = ["1 Data\nGate 1", "2 Features\n+ Drift\nGate 2", "3 Models\n+ Diagnose\nGate 3", "4 Optimize\n+ Guardrails\nGate 4",
        "5 Evaluate\nOPE\nGate 5", "6 Execute\n+ Visual QA\nGate 6"]


def flow_diagram(statuses: dict | None = None) -> list[str]:
    statuses = statuses or {}
    fig, ax = plt.subplots(figsize=(14, 3.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    for i, label in enumerate(FLOW):
        x = 0.3 + i * 2.28
        st = statuses.get(i + 1)
        fc = {"pass": "#d6f0e0", "fail": "#f8d7da"}.get(st, "#e6edf7")
        ax.add_patch(FancyBboxPatch((x, 1.0), 1.9, 1.5, boxstyle="round,pad=0.05", fc=fc, ec="#2a6fdb", lw=1.4))
        ax.text(x + 0.95, 1.75, label, ha="center", va="center", fontsize=8.5)
        if i < 5:
            ax.add_patch(FancyArrowPatch((x + 1.95, 1.75), (x + 2.27, 1.75), arrowstyle="-|>", mutation_scale=12, color="#555"))
    ax.text(7, 0.55, "FAIL(critical) -> HALT   |   high-impact / LLM-escalation -> HITL approval   |   LLM is advisory only",
            ha="center", fontsize=8, color="#555")
    ax.set_title("Autonomous Campaign & Yield Optimizer - control flow", fontsize=11)
    p1 = out_dir("charts", "pipeline_flow.png")
    p2 = out_dir("charts", "pipeline_flow.svg")
    fig.savefig(p1, bbox_inches="tight", facecolor="white")
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    return [str(p1), str(p2)]


_STATUS_COLOR = {"PASS": "#d6f0e0", "PASS-HITL": "#fde8d2", "WARN": "#fff3c4", "FAIL": "#f8d7da", "SKIPPED": "#ececec"}


def scenario_chart(res: dict) -> str:
    """Flow strip + allocation + cumulative-net panels for one scenario."""
    fig = plt.figure(figsize=(13, 6.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.8, 2.2])
    ax0 = fig.add_subplot(gs[0, :])
    ax0.set_xlim(0, 12.6)
    ax0.set_ylim(0, 2)
    ax0.axis("off")
    for i, st in enumerate(res["steps"]):
        x = 0.1 + i * 2.1
        ax0.add_patch(FancyBboxPatch((x, 0.3), 1.9, 1.4, boxstyle="round,pad=0.04", fc=_STATUS_COLOR.get(st["status"], "#fff"),
                                     ec="#2a6fdb", lw=1.2))
        ax0.text(x + 0.95, 1.2, st["name"].replace(" + ", "\n+ ").replace(" (", "\n("), ha="center", va="center", fontsize=7.5)
        ax0.text(x + 0.95, 0.55, st["status"], ha="center", va="center", fontsize=8, fontweight="bold")
        if i < 5:
            ax0.add_patch(FancyArrowPatch((x + 1.93, 1.0), (x + 2.07, 1.0), arrowstyle="-|>", mutation_scale=9, color="#555"))
    fig.suptitle(res["label"], fontsize=12, fontweight="bold")
    ax1, ax2 = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    if res.get("plan"):
        p = res["plan"]
        ch = list(p["budgets"])
        x = np.arange(len(ch))
        ax1.bar(x - 0.2, [p["prev_budgets"][c] for c in ch], 0.4, label="previous", color="#b7c4d6")
        ax1.bar(x + 0.2, [p["budgets"][c] for c in ch], 0.4, label="proposed", color=PALETTE[0])
        ax1.set_xticks(x, ch)
        ax1.set(title=f"Next-cycle budgets | discount {p['discount']:.0%} | util {p['inventory_util']:.2f}", ylabel="$ / day")
        ax1.legend(fontsize=7)
        cum = res["replay"]["cumulative"]
        for name, col in (("static", "#b7c4d6"), ("heuristic", "#e8b27a"), ("agent", PALETTE[0])):
            ax2.plot(np.array(cum[name]) / 1e3, label=name, color=col, lw=2 if name == "agent" else 1.4)
        ax2.set(title=f"Cumulative net margin (lift vs static {res['replay']['lift_vs_static_pct']:+.0f}%, "
                      f"vs heuristic {res['replay']['lift_vs_heuristic_pct']:+.0f}%)", xlabel="weekly cycle", ylabel="$ thousands")
        ax2.legend(fontsize=7)
    else:
        d = res["drift"]
        keys = sorted(d, key=d.get)
        ax1.barh(keys, [d[k] for k in keys], color=[PALETTE[4] if d[k] >= 0.25 else PALETTE[0] for k in keys])
        ax1.axvline(0.25, ls="--", color="grey")
        ax1.set(title="Feature PSI, train vs shifted live window (limit 0.25)")
        ax2.axis("off")
        ax2.text(0.5, 0.6, "PIPELINE HALTED\n" + res["halt_reason"] + "\n\nno models trusted - no budget or price change sent",
                 ha="center", va="center", fontsize=11, color="#8a1f17", wrap=True)
    return _save(fig, f"scenario_{res['id']}.png")
