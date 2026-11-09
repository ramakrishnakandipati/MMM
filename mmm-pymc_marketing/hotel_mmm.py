"""Hospitality Marketing Mix Model with pymc-marketing (1.3.x API).

Run inside WSL2 in the `mmm-pymc` env:
    python hotel_mmm.py --quick      # smoke test (tiny sampler settings)
    python hotel_mmm.py              # full run
    python hotel_mmm.py --data my.csv

Edit the CONFIG block for your own data. Without --data, a synthetic weekly
hotel dataset is generated so the full workflow can be exercised end to end.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import arviz as az
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymc_extras.prior import Prior

from pymc_marketing.mmm import MMM, DelayedAdstock, LogisticSaturation

warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------------------------------------------------------
# CONFIG - replace with your own schema
# ----------------------------------------------------------------------------
DATE_COL = "date_week"
TARGET_COL = "bookings"
CHANNELS = [
    "paid_search_brand",
    "paid_search_generic",
    "meta_ads",
    "display",
    "ota_promotions",
    "metasearch",
]
CONTROLS = [
    "average_daily_rate_adr",
    "local_events",
    "competitor_promo_flag",
    "macro_holiday_flag",
]
# Booking-window lag per channel (weeks): short = lower-funnel/brand/metasearch,
# long = upper-funnel awareness that converts weeks later.
SHORT_LEAD = ["paid_search_brand", "paid_search_generic", "metasearch", "ota_promotions"]
LONG_LEAD = ["meta_ads", "display"]
L_MAX = 8
OUT = Path("outputs")


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def make_synthetic(n_weeks: int = 156, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n_weeks, freq="W-MON")
    t = np.arange(n_weeks)
    season = 1 + 0.35 * np.sin(2 * np.pi * (t - 14) / 52) + 0.1 * np.cos(4 * np.pi * t / 52)
    df = pd.DataFrame({DATE_COL: dates})
    scale = dict(zip(CHANNELS, [8e3, 12e3, 15e3, 6e3, 10e3, 7e3]))
    for c in CHANNELS:
        spend = scale[c] * season * rng.gamma(4, 0.25, n_weeks)
        spend[rng.random(n_weeks) < 0.05] = 0.0  # a few zero-spend weeks
        df[c] = spend
    df["average_daily_rate_adr"] = 140 + 25 * (season - 1) + rng.normal(0, 4, n_weeks)
    df["local_events"] = (rng.random(n_weeks) < 0.1).astype(float)
    df["competitor_promo_flag"] = (rng.random(n_weeks) < 0.15).astype(float)
    df["macro_holiday_flag"] = (rng.random(n_weeks) < 0.08).astype(float)

    def adstock(x, a, l=L_MAX):
        w = a ** np.arange(l)
        return np.convolve(x, w / w.sum())[: len(x)]

    def sat(x, lam):
        return (1 - np.exp(-lam * x)) / (1 + np.exp(-lam * x))

    beta = dict(zip(CHANNELS, [900, 700, 800, 300, 600, 500]))
    alpha = dict(zip(CHANNELS, [0.2, 0.3, 0.7, 0.75, 0.25, 0.3]))
    y = 1500 * season - 4.0 * (df["average_daily_rate_adr"] - 140)
    y = y + 250 * df["local_events"] - 120 * df["competitor_promo_flag"] + 300 * df["macro_holiday_flag"]
    for c in CHANNELS:
        x = adstock(df[c].to_numpy() / df[c].max(), alpha[c])
        y = y + beta[c] * sat(x, 3.0)
    df[TARGET_COL] = np.maximum(y + rng.normal(0, 60, n_weeks), 0)
    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Missing dates, zero-spend weeks, NaNs, multicollinearity (VIF)."""
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    full = pd.date_range(df[DATE_COL].min(), df[DATE_COL].max(), freq="W-MON")
    missing = full.difference(pd.DatetimeIndex(df[DATE_COL]))
    print(f"[validate] rows={len(df)}  missing weeks={len(missing)}")
    if len(missing):
        print("  missing:", list(missing.date)[:10])
        df = (
            df.set_index(DATE_COL).reindex(full).rename_axis(DATE_COL).reset_index()
        )
        df[CHANNELS] = df[CHANNELS].fillna(0.0)
        df[[TARGET_COL, *CONTROLS]] = df[[TARGET_COL, *CONTROLS]].interpolate()
    nans = df[[TARGET_COL, *CHANNELS, *CONTROLS]].isna().sum()
    assert nans.sum() == 0, f"NaNs remain:\n{nans[nans > 0]}"
    zero = (df[CHANNELS] == 0).mean().round(3)
    print("[validate] share of zero-spend weeks:\n", zero.to_string())

    # VIF via inverse correlation matrix diagonal (no statsmodels needed)
    cols = [*CHANNELS, *[c for c in CONTROLS if df[c].nunique() > 1]]
    corr = np.corrcoef(df[cols].to_numpy().T)
    vif = pd.Series(np.diag(np.linalg.pinv(corr)), index=cols).round(2)
    print("[validate] VIF (>5 moderate, >10 severe):\n", vif.to_string())
    if (vif > 10).any():
        print("  WARNING: severe collinearity - consider tighter priors / lift-test calibration")
    return df


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def build_mmm(df: pd.DataFrame) -> MMM:
    """Spend-share informed priors; separate adstock for short vs long lead.

    Two MMMs can't mix adstock types per channel in one model, so we use
    DelayedAdstock (peak lag learned) for all channels - short-lead channels
    will learn theta ~ 0 (geometric-like), long-lead channels a positive delay.
    """
    spend_share = (df[CHANNELS].sum() / df[CHANNELS].sum().sum()).to_numpy()
    n = len(CHANNELS)

    adstock = DelayedAdstock(
        l_max=L_MAX,
        priors={
            "alpha": Prior("Beta", alpha=2, beta=3, dims="channel"),
        },
    )
    saturation = LogisticSaturation(
        priors={
            "lam": Prior("Gamma", mu=3, sigma=1.5, dims="channel"),
            "beta": Prior("HalfNormal", sigma=spend_share * n * 0.5, dims="channel"),
        }
    )
    return MMM(
        date_column=DATE_COL,
        target_column=TARGET_COL,
        channel_columns=CHANNELS,
        control_columns=CONTROLS,
        adstock=adstock,
        saturation=saturation,
        yearly_seasonality=3,          # Fourier terms for travel seasonality
        time_varying_intercept=True,   # HSGP baseline drift (demand trend)
    )


# ----------------------------------------------------------------------------
# Workflow
# ----------------------------------------------------------------------------
def _date_first(da, n_dates: int) -> np.ndarray:
    """(date, samples) array from a DataArray with any dim order."""
    da = da.squeeze(drop=True)
    other = [d for d in da.dims if d != "date"]
    return da.transpose("date", *other).values.reshape(n_dates, -1)


def main(args: argparse.Namespace) -> None:
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(args.data, parse_dates=[DATE_COL]) if args.data else make_synthetic()
    df = validate_data(df)
    X, y = df.drop(columns=[TARGET_COL]), df[TARGET_COL]

    mmm = build_mmm(df)

    # 1. Prior predictive check -------------------------------------------------
    prior = mmm.sample_prior_predictive(X, y, samples=200)
    fig, ax = plt.subplots(figsize=(10, 4))
    t_scale = float(mmm.get_scales_as_xarray()["target_scale"])
    lo, hi = np.percentile(_date_first(prior["y"], len(df)) * t_scale, [3, 97], axis=1)
    ax.fill_between(df[DATE_COL], lo, hi, alpha=0.3, label="prior 94% band")
    ax.plot(df[DATE_COL], y, "k", lw=1, label="observed")
    ax.set(title="Prior predictive check", ylabel=TARGET_COL)
    ax.legend()
    fig.savefig(OUT / "prior_predictive.png", dpi=120, bbox_inches="tight")
    print(f"[prior] observed range {y.min():.0f}-{y.max():.0f}; prior band {lo.min():.0f}-{hi.max():.0f}")

    # 2. Fit ----------------------------------------------------------------------
    sk = dict(draws=200, tune=200, chains=2) if args.quick else dict(
        draws=1500, tune=1500, chains=4, target_accept=0.95
    )
    if not args.quick:
        sk["nuts_sampler"] = "numpyro"  # JAX/GPU sampler
    mmm.fit(X, y, random_seed=42, **sk)
    idata = mmm.idata

    # 3. Diagnostics ----------------------------------------------------------------
    summ = az.summary(idata.posterior if hasattr(idata, "posterior") else idata,
                      var_names=["~intercept_latent_process"], filter_vars="like")
    summ.to_csv(OUT / "summary.csv")
    print("[diag] max r_hat:", float(summ["r_hat"].max()), " min ess_bulk:", float(summ["ess_bulk"].min()))
    div = int(idata.sample_stats["diverging"].sum())
    print(f"[diag] divergences: {div}")
    if summ["r_hat"].max() > 1.01 or div:
        print("  WARNING: convergence issues - raise target_accept/tune or tighten priors")

    mmm.sample_posterior_predictive(X, extend_idata=True, combined=True)
    az.plot_trace(idata.posterior, var_names=["saturation_beta", "adstock_alpha"])
    plt.savefig(OUT / "trace.png", dpi=100, bbox_inches="tight")

    post = mmm.idata.posterior_predictive["y"] if "y" in mmm.idata.posterior_predictive else None
    if post is not None:
        arr = _date_first(post, len(df)) * t_scale  # model output is max-scaled
        m = arr.mean(1)
        lo, hi = np.percentile(arr, [3, 97], axis=1)
        r2 = 1 - ((y - m) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        mape = float(np.mean(np.abs((y - m) / y)))
        print(f"[ppc] R2={r2:.3f}  MAPE={mape:.2%}  coverage94={np.mean((y >= lo) & (y <= hi)):.2%}")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.fill_between(df[DATE_COL], lo, hi, alpha=0.3)
        ax.plot(df[DATE_COL], m, label="posterior mean")
        ax.plot(df[DATE_COL], y, "k", lw=1, label="observed")
        ax.legend(); ax.set_title("Posterior predictive check")
        fig.savefig(OUT / "ppc.png", dpi=120, bbox_inches="tight")

    # 4. Decomposition, ROAS, HDIs -----------------------------------------------------
    contrib = mmm.compute_mean_contributions_over_time()
    contrib.to_csv(OUT / "contributions_over_time.csv")
    print("[decomp] columns:", list(contrib.columns))
    # Draw-level incremental contribution per channel -> ROAS with HDI.
    cf = mmm.compute_counterfactual_contributions_dataset()
    rows = []
    for ch in CHANNELS:
        if ch not in cf:
            continue
        tot = cf[ch].sum(dim="date") if "date" in cf[ch].dims else cf[ch]
        draws = tot.stack(s=("chain", "draw")).values.ravel()
        spend = df[ch].sum()
        roas = draws / spend
        hdi = az.hdi(roas, prob=0.94)
        rows.append(dict(channel=ch, spend=spend, contribution=draws.mean(),
                         roas_mean=roas.mean(), roas_hdi_3=hdi[0], roas_hdi_97=hdi[1]))
    roas_df = pd.DataFrame(rows).set_index("channel")
    roas_df.to_csv(OUT / "roas.csv")
    print("[roas] (bookings per currency unit of spend; multiply by ADR for revenue ROAS)\n",
          roas_df.round(4).to_string())

    # 5. Budget optimisation ----------------------------------------------------------------
    last = pd.Timestamp(df[DATE_COL].max())
    horizon = 8
    start = last + pd.Timedelta(weeks=1)
    end = last + pd.Timedelta(weeks=horizon)
    total_budget = float(df[CHANNELS].tail(horizon).sum().sum())
    opt = mmm.budget_optimizer(start_date=start, end_date=end)
    allocation = opt.allocate_budget(total_budget=total_budget)
    result = allocation[0] if isinstance(allocation, tuple) else allocation
    print(f"[opt] total budget {total_budget:,.0f} over {horizon} weeks")
    alloc = result.budgets.to_series().to_frame("optimal_budget")
    alloc["current_budget"] = df[CHANNELS].tail(horizon).sum()
    alloc.to_csv(OUT / "optimal_allocation.csv")
    print(alloc.round(0).to_string())
    print(f"Outputs in {OUT.resolve()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", help="CSV with the CONFIG columns")
    p.add_argument("--quick", action="store_true", help="tiny sampler for smoke tests")
    main(p.parse_args())
