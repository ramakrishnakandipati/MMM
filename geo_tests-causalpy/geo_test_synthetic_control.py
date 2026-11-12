"""Hospitality geo-test with CausalPy (Bayesian Synthetic Control, PyMC).

Design
  * 15 synthetic markets x 52 weeks: 5 treated, 10 control. Campaign starts week 41.
  * A true +12% revenue lift is injected into treated geos from week 41 onward.
  * The 5 treated geos are averaged into one treated series; the 10 controls are donors.
  * Model: cp.SyntheticControl + cp.pymc_models.WeightedSumFitter (Dirichlet weights on
    donors, no intercept). Fit on the pre-period only; the counterfactual is predicted
    for the post-period.

NOTE: the experiment class is `cp.SyntheticControl`; the PyMC model is
`cp.pymc_models.WeightedSumFitter` (there is no `pymc_models.SyntheticControl`).

Run (WSL2, env `causalpy`, CausalPy 0.9):
    python geo_test_synthetic_control.py --quick     # fast smoke test
    python geo_test_synthetic_control.py             # full run
"""
from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")   # tiny model: CPU is faster, avoids GPU OOM

import causalpy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
OUT = Path(__file__).parent / "outputs"


# --------------------------------------------------------------------------- config
@dataclass(frozen=True)
class Config:
    seed: int = 42
    n_treated: int = 5
    n_control: int = 10
    n_weeks: int = 52
    intervention_week: int = 41          # weeks 1-40 pre, 41-52 intervention
    true_lift: float = 0.12              # +12% revenue in treated geos
    start_date: str = "2025-01-06"       # a Monday
    campaign_spend: float = 250_000.0    # total media spend in the test (all treated geos)
    gross_margin: float = 0.35           # flow-through for profit ROAS
    draws: int = 1000
    tune: int = 1000
    chains: int = 4
    target_accept: float = 0.95


# --------------------------------------------------------------------------- data
def make_long_panel(cfg: Config) -> pd.DataFrame:
    """Long panel: date, geo_id, revenue, group ('treatment_group'/'control_group')."""
    rng = np.random.default_rng(cfg.seed)
    weeks = np.arange(1, cfg.n_weeks + 1)
    dates = pd.date_range(cfg.start_date, periods=cfg.n_weeks, freq="W-MON")

    # Shared drivers: trend, yearly hospitality seasonality (summer peak), AR(1) demand factor
    trend = 1 + 0.002 * weeks
    season = 1 + 0.18 * np.sin(2 * np.pi * (weeks - 14) / 52) + 0.06 * np.sin(4 * np.pi * weeks / 52)
    factor = np.zeros(cfg.n_weeks)
    for t in range(1, cfg.n_weeks):
        factor[t] = 0.7 * factor[t - 1] + rng.normal(0, 0.03)

    n_geo = cfg.n_treated + cfg.n_control
    rows = []
    for g in range(n_geo):
        treated = g < cfg.n_treated
        scale = rng.uniform(80_000, 140_000)          # weekly revenue level (comparable across geos)
        season_amp = rng.uniform(0.97, 1.03)          # near-homogeneous seasonality (regional peers)
        loading = rng.uniform(0.97, 1.03)             # near-homogeneous exposure to the demand factor
        noise = rng.normal(0, 0.02, cfg.n_weeks)
        rev = scale * trend * (1 + season_amp * (season - 1)) * (1 + loading * factor) * (1 + noise)
        if treated:
            rev = np.where(weeks >= cfg.intervention_week, rev * (1 + cfg.true_lift), rev)
        for w in range(cfg.n_weeks):
            rows.append((dates[w], f"geo_{g + 1:02d}", rev[w],
                         "treatment_group" if treated else "control_group"))
    return pd.DataFrame(rows, columns=["date", "geo_id", "revenue", "group"])


def to_wide(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Wide frame (one column per geo) plus a `treated_avg` column = mean of treated geos."""
    wide = df.pivot(index="date", columns="geo_id", values="revenue")
    groups = df.drop_duplicates("geo_id").set_index("geo_id")["group"]
    treated = sorted(groups[groups == "treatment_group"].index)
    controls = sorted(groups[groups == "control_group"].index)
    wide["treated_avg"] = wide[treated].mean(axis=1)   # keeps scale inside the donors' convex hull
    return wide, treated, controls


# --------------------------------------------------------------------------- model
def fit_synthetic_control(wide: pd.DataFrame, controls: list[str], cfg: Config,
                          treatment_time: pd.Timestamp) -> cp.SyntheticControl:
    model = cp.pymc_models.WeightedSumFitter(sample_kwargs=dict(
        draws=cfg.draws, tune=cfg.tune, chains=cfg.chains, target_accept=cfg.target_accept,
        random_seed=cfg.seed, progressbar=False,
        nuts_sampler="numpyro"))   # numpyro: fast, needs no C compiler
    return cp.SyntheticControl(
        wide, treatment_time,
        control_units=controls,          # predictors: control geos (pre-period fit)
        treated_units=["treated_avg"],   # dependent variable
        model=model)


# --------------------------------------------------------------------------- inference
def impact_draws(result: cp.SyntheticControl, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Posterior draws of post-period (impact, counterfactual), shape (samples, weeks)."""
    imp = result.post_impact.isel(treated_units=0).stack(sample=("chain", "draw"))
    impact = imp.transpose("sample", "obs_ind").values * scale          # back to $
    actual = result.datapost["treated_avg"].values * scale
    return impact, actual[None, :] - impact


def hdi(x: np.ndarray, prob: float = 0.94) -> tuple[float, float]:
    s = np.sort(x)
    k = int(np.ceil(prob * len(s)))
    i = np.argmin(s[k - 1:] - s[: len(s) - k + 1])
    return float(s[i]), float(s[i + k - 1])


def report(result, cfg: Config, n_treated: int, scale: float) -> dict:
    impact, cf = impact_draws(result, scale)
    actual = result.datapost["treated_avg"].values * scale
    ate = impact.mean(axis=1)                         # avg weekly effect per treated geo
    cum = impact.sum(axis=1) * n_treated              # cumulative across all treated geos
    rel = impact.sum(axis=1) / cf.sum(axis=1)
    inc_rev = cum
    print("\n=== Causal impact (post-period) ===")
    print(f"ATT per treated geo-week : {ate.mean():>12,.0f}   94% HDI {tuple(round(v) for v in hdi(ate))}")
    print(f"Cumulative lift (all geos): {cum.mean():>12,.0f}   94% HDI {tuple(round(v) for v in hdi(cum))}")
    print(f"Relative lift             : {rel.mean():>12.1%}   94% HDI "
          f"({hdi(rel)[0]:.1%}, {hdi(rel)[1]:.1%})   [true: {cfg.true_lift:.0%}]")
    print(f"P(lift > 0)               : {(cum > 0).mean():.3f}")
    roas = inc_rev / cfg.campaign_spend
    print(f"\nROAS (incremental rev / spend): {roas.mean():.2f}x   94% HDI "
          f"({hdi(roas)[0]:.2f}, {hdi(roas)[1]:.2f}) | profit ROAS @ {cfg.gross_margin:.0%} margin: "
          f"{(roas * cfg.gross_margin).mean():.2f}x")
    print("\nCausalPy effect summary:\n", result.effect_summary().text)
    return dict(ate=ate, cum=cum, rel=rel, roas=roas, actual=actual)


# --------------------------------------------------------------------------- plots
def make_plots(result, wide, cfg: Config, scale: float) -> None:
    OUT.mkdir(exist_ok=True)
    # Standard CausalPy figure: (a) observed vs counterfactual, (b) pointwise impact,
    # (c) cumulative impact.
    fig, _ = result.plot(show=False)
    fig.savefig(OUT / "causalpy_standard_plot.png", dpi=130, bbox_inches="tight")

    # Explicit three-panel version computed from posterior draws.
    impact, cf = impact_draws(result, scale)
    dates = result.datapost.index
    fig, ax = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    lo, hi = np.percentile(cf, [3, 97], axis=0)
    ax[0].plot(wide.index, wide["treated_avg"] * scale, "k", label="Actual (treated avg)")
    ax[0].plot(dates, cf.mean(0), "C0--", label="Counterfactual")
    ax[0].fill_between(dates, lo, hi, color="C0", alpha=0.25)
    ax[0].set_title("a) Observed vs counterfactual baseline")
    ax[0].legend()
    lo, hi = np.percentile(impact, [3, 97], axis=0)
    ax[1].plot(dates, impact.mean(0), "C1")
    ax[1].fill_between(dates, lo, hi, color="C1", alpha=0.25)
    ax[1].axhline(0, color="grey", lw=0.8)
    ax[1].set_title("b) Pointwise causal impact (per treated geo-week)")
    c = impact.cumsum(axis=1)
    lo, hi = np.percentile(c, [3, 97], axis=0)
    ax[2].plot(dates, c.mean(0), "C2")
    ax[2].fill_between(dates, lo, hi, color="C2", alpha=0.25)
    ax[2].axhline(0, color="grey", lw=0.8)
    ax[2].set_title("c) Cumulative causal impact (per treated geo)")
    for a in ax:
        a.axvline(dates[0], color="red", ls=":", lw=1)
    fig.tight_layout()
    fig.savefig(OUT / "causal_impact_panels.png", dpi=130)
    plt.close("all")


# --------------------------------------------------------------------------- main
def main(quick: bool = False) -> None:
    cfg = Config(draws=300, tune=300, chains=2) if quick else Config()
    df = make_long_panel(cfg)
    print(df.groupby("group").geo_id.nunique().to_string(), "\n", df.head(3).to_string())
    wide, treated, controls = to_wide(df, cfg)
    treatment_time = wide.index[cfg.intervention_week - 1]
    print(f"\nTreatment starts {treatment_time.date()} (week {cfg.intervention_week}); "
          f"pre-period weeks: {cfg.intervention_week - 1}")

    # Rescale to ~1 (pre-period mean of treated avg) so the sampler is well-conditioned;
    # all reported $ figures are converted back.
    scale = float(wide["treated_avg"].iloc[: cfg.intervention_week - 1].mean())
    result = fit_synthetic_control(wide / scale, controls, cfg, treatment_time)
    result.print_coefficients()
    import arviz as az
    diag = az.summary(result.idata, var_names=["beta", "y_hat_sigma"])
    div = int(result.idata.sample_stats["diverging"].sum())
    print(f"Diagnostics: max R-hat {diag['r_hat'].max():.3f} | min ESS-bulk {diag['ess_bulk'].min():.0f} "
          f"| divergences {div}")
    report(result, cfg, len(treated), scale)
    make_plots(result, wide, cfg, scale)
    print(f"\nPlots saved to {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)

# =====================================================================================
# HOW A HOSPITALITY CMO SHOULD READ THE ACTUAL-vs-COUNTERFACTUAL GAP
#
# * Counterfactual = revenue (or bookings) the treated markets would have earned with NO
#   campaign, built from the control markets' pre-period relationship. Actual = what
#   really happened. The gap, summed over the post-period, is INCREMENTAL revenue: guests
#   who booked because of the campaign, excluding those who would have booked anyway.
# * ROAS = incremental revenue / media spend. Report the posterior interval, not a point
#   estimate; if the 94% HDI of the cumulative gap includes 0, the test is inconclusive.
# * Revenue is not profit. Multiply by flow-through margin (net of OTA commission,
#   cost-to-serve) for profit ROAS; ROAS > 1/margin is needed to break even on spend.
# * Bookings lag stays: include a tail after the campaign ends so stays booked during
#   the test but consumed later are counted, and watch for a post-campaign dip
#   (pull-forward demand rather than new demand).
# * Check spillover: nearby control markets receiving the ads understate the lift.
#   Validate with a placebo test (fake start date in the pre-period -> effect ~ 0).
# =====================================================================================
