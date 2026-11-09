"""Hospitality MMM with Google Meridian (v2.x API, national-level, weekly).

Run in WSL2, env `mmm-meridian`:
    python hotel_meridian.py --quick     # smoke test
    python hotel_meridian.py             # full run
    python hotel_meridian.py --data hotel_weekly.csv

Search for "PLUG IN" for places where real data/assumptions belong.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # avoid JAX grabbing all GPU memory
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow_probability.substrates import jax as tfp

from meridian.analysis import analyzer, optimizer
from meridian.data import data_frame_input_data_builder as dfb
from meridian.model import model, prior_distribution, spec

warnings.filterwarnings("ignore")
tfd = tfp.distributions
OUT = Path("outputs")

# ----------------------------------------------------------------------------
# CONFIG - PLUG IN your schema (weekly, one row per week, national level)
# ----------------------------------------------------------------------------
TIME = "time"                       # 'YYYY-MM-DD' week start
KPI = "bookings"                    # total bookings (room nights is also fine)
REV_PER_KPI = "revenue_per_booking"  # ADR x avg length of stay; gives revenue ROI
CHANNELS = {                        # channel -> (impressions col, spend col)
    "brand_search":   ("brand_search_impr", "brand_search_spend"),
    "nonbrand_search": ("nonbrand_search_impr", "nonbrand_search_spend"),
    "paid_social":    ("paid_social_impr", "paid_social_spend"),
    "ota":            ("ota_impr", "ota_spend"),
    "hotel_ads":      ("hotel_ads_impr", "hotel_ads_spend"),
    "display":        ("display_impr", "display_spend"),
}
CONTROLS = [
    "competitor_price_index", "consumer_confidence", "fuel_price",
    "avg_temp_c", "local_event_flag", "public_holiday_flag", "promo_flag",
]
# Short booking window (high intent, decays fast) vs long window (awareness).
# Geometric decay is Meridian's default; 'binomial' gives a delayed, flatter tail.
ADSTOCK = {
    "brand_search": "geometric", "nonbrand_search": "geometric",
    "hotel_ads": "geometric", "ota": "geometric",
    "paid_social": "binomial", "display": "binomial",
}
MAX_LAG = 8  # weeks of carryover; hotels with 4-8 week booking windows need this


# ----------------------------------------------------------------------------
# Dummy data (replace with: df = pd.read_csv(...))
# ----------------------------------------------------------------------------
def make_dummy(n_weeks: int = 156, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n_weeks)
    season = 1 + 0.35 * np.sin(2 * np.pi * (t - 14) / 52)
    df = pd.DataFrame({TIME: pd.date_range("2022-01-03", periods=n_weeks, freq="W-MON").strftime("%Y-%m-%d")})
    base_spend = dict(zip(CHANNELS, [8e3, 12e3, 15e3, 10e3, 7e3, 6e3]))
    cpm = dict(zip(CHANNELS, [40, 30, 8, 20, 25, 4]))  # cost per 1000 impressions
    for ch, (impr, spend) in CHANNELS.items():
        s = base_spend[ch] * season * rng.gamma(4, 0.25, n_weeks)
        df[spend] = s
        df[impr] = s / cpm[ch] * 1000 * rng.normal(1, 0.05, n_weeks)
    df["competitor_price_index"] = 100 + 8 * np.sin(2 * np.pi * t / 52 + 1) + rng.normal(0, 2, n_weeks)
    df["consumer_confidence"] = 100 + np.cumsum(rng.normal(0, 0.8, n_weeks))
    df["fuel_price"] = 3.5 + np.cumsum(rng.normal(0, 0.03, n_weeks))
    df["avg_temp_c"] = 15 + 10 * np.sin(2 * np.pi * (t - 12) / 52) + rng.normal(0, 2, n_weeks)
    for c in ["local_event_flag", "public_holiday_flag", "promo_flag"]:
        df[c] = (rng.random(n_weeks) < 0.1).astype(float)

    def ads(x, a):
        w = a ** np.arange(MAX_LAG)
        return np.convolve(x, w / w.sum())[:n_weeks]

    beta = dict(zip(CHANNELS, [500, 450, 400, 350, 600, 150]))
    y = 1500 * season + 200 * df["local_event_flag"] + 150 * df["public_holiday_flag"] \
        + 100 * df["promo_flag"] - 3 * (df["competitor_price_index"] - 100)
    for ch, (_, spend) in CHANNELS.items():
        x = ads(df[spend].to_numpy() / df[spend].max(), 0.3 if ADSTOCK[ch] == "geometric" else 0.6)
        y = y + beta[ch] * x / (x + 0.5)
    df[KPI] = np.maximum(y + rng.normal(0, 50, n_weeks), 1).round()
    df[REV_PER_KPI] = 160 * 2.1 * (1 + 0.1 * (season - 1)) + rng.normal(0, 5, n_weeks)
    return df


# ----------------------------------------------------------------------------
# Input data
# ----------------------------------------------------------------------------
def build_input(df: pd.DataFrame):
    media_cols = [v[0] for v in CHANNELS.values()]
    spend_cols = [v[1] for v in CHANNELS.values()]
    b = dfb.DataFrameInputDataBuilder(kpi_type="non_revenue", default_time_column=TIME,
                                      default_media_time_column=TIME, default_kpi_column=KPI,
                                      default_revenue_per_kpi_column=REV_PER_KPI)
    return (
        b.with_kpi(df)
        .with_revenue_per_kpi(df)          # -> ROI reported in revenue terms
        .with_controls(df, control_cols=CONTROLS)
        .with_media(df, media_cols=media_cols, media_spend_cols=spend_cols,
                    media_channels=list(CHANNELS))
        .build()
    )


# ----------------------------------------------------------------------------
# Priors
# ----------------------------------------------------------------------------
def build_spec() -> spec.ModelSpec:
    """ROI priors are LogNormal (always positive). Tighter + higher on high-intent
    channels (brand search, hotel ads), wider on awareness channels.
    PLUG IN: set medians from past experiments / finance targets."""
    names = list(CHANNELS)
    med = {"brand_search": 3.0, "hotel_ads": 3.0, "nonbrand_search": 2.0,
           "ota": 1.5, "paid_social": 1.0, "display": 0.8}
    sig = {"brand_search": 0.3, "hotel_ads": 0.3, "nonbrand_search": 0.5,
           "ota": 0.5, "paid_social": 0.7, "display": 0.8}
    roi_m = tfd.LogNormal(loc=np.log([med[c] for c in names]).astype("float64"),
                          scale=np.array([sig[c] for c in names], dtype="float64"))
    prior = prior_distribution.PriorDistribution(roi_m=roi_m)
    return spec.ModelSpec(
        prior=prior,
        paid_media_prior_type="roi",
        max_lag=MAX_LAG,
        adstock_decay_spec=ADSTOCK,
        knots=26,   # time-varying baseline ~ one knot per 2 weeks for 3y data; fewer = smoother
    )


# ----------------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(args.data) if args.data else make_dummy()
    data = build_input(df)
    mmm = model.Meridian(input_data=data, model_spec=build_spec())

    mmm.sample_prior(200 if args.quick else 500)
    cfg = dict(n_chains=2, n_adapt=100, n_burnin=100, n_keep=100) if args.quick else \
        dict(n_chains=4, n_adapt=500, n_burnin=500, n_keep=1000)
    mmm.sample_posterior(seed=1, **cfg)

    az = analyzer.Analyzer(mmm)
    print("[diag] R-hat summary:\n", az.rhat_summary())
    acc = az.predictive_accuracy()
    print("[diag] predictive accuracy:\n", acc)

    # ROI / mROI (median + 90% CI): revenue per currency unit of spend
    roi = az.roi()               # shape (chains, draws, channel) after aggregating geos
    mroi = az.marginal_roi()
    names = list(CHANNELS)

    def summ(a):
        a = np.asarray(a).reshape(-1, len(names))
        return pd.DataFrame({"mean": a.mean(0), "p05": np.percentile(a, 5, 0),
                             "p95": np.percentile(a, 95, 0)}, index=names)

    roi_df, mroi_df = summ(roi), summ(mroi)
    roi_df.to_csv(OUT / "roi.csv"); mroi_df.to_csv(OUT / "mroi.csv")
    print("[roi]\n", roi_df.round(2).to_string())
    print("[mroi] (next euro/dollar; <1 = past break-even)\n", mroi_df.round(2).to_string())

    # Budget optimisation: same total budget, reallocated (+/-50% per channel)
    opt = optimizer.BudgetOptimizer(mmm)
    res = opt.optimize(fixed_budget=True, spend_constraint_lower=0.5, spend_constraint_upper=0.5,
                       use_kpi=False)
    res.output_optimization_summary(filename="optimization_summary.html", filepath=str(OUT))
    print("[opt] summary written to", (OUT / "optimization_summary.html").resolve())
    try:
        opt_spend = np.asarray(res.optimized_data.spend if hasattr(res.optimized_data, "spend")
                               else res.optimized_data.media_spend)
        print(pd.DataFrame({"optimized_spend": np.ravel(opt_spend)[: len(names)]}, index=names).round(0))
    except Exception as e:  # summary HTML is the source of truth
        print("  (see HTML summary for allocation)", type(e).__name__)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data")
    p.add_argument("--quick", action="store_true")
    main(p.parse_args())
