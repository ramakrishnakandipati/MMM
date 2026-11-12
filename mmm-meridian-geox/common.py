"""Shared configuration + synthetic hospitality data generator for the GeoX study.

Everything stochastic is seeded here so the four stage scripts stay consistent:
the first 180 days of the "no-treatment world" are identical at design time
(stage 1/2) and at analysis time (stage 3/4).

Run with the `mmm-meridian` conda env (WSL):
    /home/rama/miniforge3/envs/mmm-meridian/bin/python <script>.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths / logging
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def setup_logging() -> None:
    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")


# --------------------------------------------------------------------------- #
# Study calendar (hospitality-aware)
# --------------------------------------------------------------------------- #
SEED = 20260712
PRETEST_START = pd.Timestamp("2026-01-12")      # Monday
PRETEST_DAYS = 180                              # ~25.7 weeks => 25+ full DoW cycles
PRETEST_END = PRETEST_START + pd.Timedelta(days=PRETEST_DAYS - 1)   # Fri 2026-07-10
TEST_DAYS = 28                                  # 4 whole weeks, Sat -> Fri
COOLDOWN_DAYS = 14                              # >= booking-conversion lag (7-14d)
TEST_START = PRETEST_END + pd.Timedelta(days=1)                      # Sat 2026-07-11
TEST_END = TEST_START + pd.Timedelta(days=TEST_DAYS - 1)
COOLDOWN_START = TEST_END + pd.Timedelta(days=1)
COOLDOWN_END = TEST_END + pd.Timedelta(days=COOLDOWN_DAYS)
TOTAL_DAYS = PRETEST_DAYS + TEST_DAYS + COOLDOWN_DAYS

# --------------------------------------------------------------------------- #
# Ground truth for the simulated experiment
# --------------------------------------------------------------------------- #
TRUE_IROAS = 2.0           # incremental booking revenue per incremental $ of spend
HEAVY_UP_PCT = 0.75        # planned +75% spend in treatment DMAs
DELIVERY_FACTOR = 0.93     # platforms rarely deliver 100% of the extra budget
SPEND_TO_REV = 0.03        # BAU paid search+social spend / booking revenue
SEARCH_SHARE = 0.65        # remainder is paid social

# Booking-conversion lag kernel (days from ad exposure to booking), truncated at 14d.
LAG_MAX = 14
_lag_days = np.arange(0, LAG_MAX + 1)
_lag_w = (_lag_days + 0.5) ** (2 - 1) * np.exp(-(_lag_days + 0.5) / 3.0)   # gamma(k=2, theta=3)
LAG_KERNEL = _lag_w / _lag_w.sum()

# Booking-date day-of-week profile (Mon..Sun). Leisure travel is researched/booked
# Sun-Tue evenings and mid-week; Fri/Sat are the weakest *booking* days.
DOW_PROFILE = np.array([1.12, 1.08, 1.03, 0.98, 0.92, 0.84, 1.03])

# --------------------------------------------------------------------------- #
# 60 US DMAs: (name, region, avg daily booking revenue $k, peak occupancy, renovation)
# --------------------------------------------------------------------------- #
_DMAS = [
    # Northeast
    ("New York", "NE", 520, .93, 0), ("Boston", "NE", 150, .88, 0), ("Philadelphia", "NE", 140, .86, 0),
    ("Washington DC", "NE", 160, .89, 0), ("Pittsburgh", "NE", 55, .80, 0), ("Hartford", "NE", 45, .78, 0),
    ("Albany", "NE", 35, .76, 0), ("Buffalo", "NE", 38, .77, 0), ("Portland ME", "NE", 22, .84, 0),
    ("Providence", "NE", 40, .79, 0), ("Baltimore", "NE", 60, .82, 0),
    # Southeast
    ("Atlanta", "SE", 130, .87, 0), ("Charlotte", "SE", 70, .84, 0), ("Raleigh", "SE", 65, .83, 0),
    ("Nashville", "SE", 75, .90, 0), ("Charleston SC", "SE", 30, .91, 0), ("Savannah", "SE", 24, .90, 1),
    ("Myrtle Beach", "SE", 28, .96, 0), ("Memphis", "SE", 35, .78, 0), ("Birmingham", "SE", 38, .79, 0),
    ("Asheville", "SE", 20, .88, 0), ("Richmond", "SE", 42, .81, 0),
    # Florida
    ("Orlando", "FL", 300, .97, 0), ("Miami", "FL", 240, .96, 0), ("Tampa", "FL", 110, .89, 0),
    ("Jacksonville", "FL", 55, .84, 0), ("Fort Myers", "FL", 40, .90, 0),
    ("West Palm Beach", "FL", 60, .88, 0), ("Pensacola", "FL", 22, .85, 0),
    # Midwest
    ("Chicago", "MW", 210, .86, 0), ("Detroit", "MW", 70, .78, 0), ("Minneapolis", "MW", 80, .81, 0),
    ("Cleveland", "MW", 50, .77, 0), ("Columbus", "MW", 46, .79, 0), ("Indianapolis", "MW", 44, .78, 0),
    ("St. Louis", "MW", 52, .78, 0), ("Milwaukee", "MW", 34, .77, 0), ("Kansas City", "MW", 40, .78, 0),
    # Texas / South Central
    ("Dallas", "TX", 150, .85, 0), ("Houston", "TX", 130, .83, 0), ("Austin", "TX", 85, .87, 0),
    ("San Antonio", "TX", 60, .84, 0), ("New Orleans", "TX", 70, .88, 0),
    # West
    ("Los Angeles", "WE", 330, .89, 0), ("San Diego", "WE", 120, .90, 0), ("San Francisco", "WE", 180, .88, 0),
    ("Seattle", "WE", 110, .86, 0), ("Portland OR", "WE", 60, .82, 0), ("Phoenix", "WE", 100, .84, 0),
    ("Sacramento", "WE", 40, .80, 0), ("Honolulu", "WE", 160, .96, 0), ("Las Vegas", "WE", 250, .95, 0),
    ("Palm Springs", "WE", 26, .88, 1), ("Santa Barbara", "WE", 18, .87, 0), ("Fresno", "WE", 24, .76, 0),
    # Mountain
    ("Denver", "MT", 95, .85, 0), ("Salt Lake City", "MT", 50, .82, 0), ("Boise", "MT", 22, .80, 0),
    ("Albuquerque", "MT", 28, .79, 0), ("Colorado Springs", "MT", 18, .81, 0),
]

# Regional annual seasonality: (peak day-of-year, amplitude)
_SEASON = {"NE": (200, .18), "SE": (120, .10), "FL": (65, .20), "MW": (190, .12),
           "TX": (130, .08), "WE": (210, .10), "MT": (30, .15)}

FLAGSHIP_TOP_N = 4          # largest-volume DMAs are treated as flagship / brand-sensitive
OCC_THRESHOLD = 0.95        # >=95% seasonal occupancy => inventory-constrained
BROKEN_TRACKING_GEO = "Fresno"   # fault injected in stage 1 (spend > 0, zero bookings)


def geo_metadata() -> pd.DataFrame:
    md = pd.DataFrame(_DMAS, columns=["location", "region", "avg_daily_rev_k", "peak_occupancy", "renovation"])
    md["flagship"] = md["avg_daily_rev_k"].rank(ascending=False, method="first") <= FLAGSHIP_TOP_N
    return md


def operational_exclusions(md: pd.DataFrame) -> pd.DataFrame:
    """Geos that must be kept out of the experiment for operational reasons."""
    reason = pd.Series("", index=md.index)
    reason[md["flagship"]] += "flagship;"
    reason[md["peak_occupancy"] >= OCC_THRESHOLD] += "occupancy>=95%;"
    reason[md["renovation"] == 1] += "renovation;"
    out = md.assign(exclusion_reason=reason.str.rstrip(";"))
    return out[out["exclusion_reason"] != ""]


# --------------------------------------------------------------------------- #
# Synthetic data generator
# --------------------------------------------------------------------------- #
def _ar1(rng, n, phi, sd, size=None):
    shape = (n,) if size is None else (n, size)
    e = rng.normal(0, sd, shape)
    x = np.zeros(shape)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def build_untreated_panel(seed: int = SEED) -> pd.DataFrame:
    """Full 222-day *no-treatment world* panel (pre-test + test + cooldown).

    Columns: date, location, conversions (booking revenue $), spend (search+social),
    spend_search, spend_social.
    """
    rng = np.random.default_rng(seed)
    md = geo_metadata()
    G, T = len(md), TOTAL_DAYS
    dates = pd.date_range(PRETEST_START, periods=T, freq="D")
    doy = dates.dayofyear.values
    dow = dates.dayofweek.values

    size = md["avg_daily_rev_k"].values * 1000 * rng.lognormal(0, 0.05, G)
    trend = rng.normal(0.06, 0.08, G)                            # annualised growth per DMA
    dow_tilt = rng.uniform(0.6, 1.4, G)                          # leisure vs business DoW shape
    t_years = np.arange(T)[:, None] / 365.0

    # regional seasonality
    season = np.ones((T, G))
    for r, (pk, amp) in _SEASON.items():
        idx = np.where(md["region"].values == r)[0]
        season[:, idx] = 1 + amp * np.cos(2 * np.pi * (doy[:, None] - pk) / 365.0)

    dow_eff = 1 + (DOW_PROFILE[dow][:, None] - 1) * dow_tilt[None, :]
    national = np.exp(_ar1(rng, T, 0.7, 0.025))[:, None]
    regional = np.ones((T, G))
    for r in _SEASON:
        idx = np.where(md["region"].values == r)[0]
        regional[:, idx] = np.exp(_ar1(rng, T, 0.8, 0.03))[:, None]
    idio = rng.lognormal(0, 0.08, (T, G))

    expected = size[None, :] * np.exp(trend[None, :] * t_years) * season * dow_eff * national * regional
    revenue = expected * idio

    # BAU paid spend follows *expected* demand (bid/budget pacing), not realised noise
    spend = SPEND_TO_REV * expected * rng.lognormal(0, 0.12, (T, G))
    df = pd.DataFrame({
        "date": np.repeat(dates.values, G),
        "location": np.tile(md["location"].values, T),
        "conversions": revenue.ravel(),
        "spend": spend.ravel(),
    })
    df["spend_search"] = df["spend"] * SEARCH_SHARE
    df["spend_social"] = df["spend"] * (1 - SEARCH_SHARE)
    return df


def split_windows(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "pretest": df[df["date"] <= PRETEST_END],
        "test": df[(df["date"] >= TEST_START) & (df["date"] <= TEST_END)],
        "cooldown": df[(df["date"] >= COOLDOWN_START) & (df["date"] <= COOLDOWN_END)],
    }


def save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def banner(title: str) -> None:
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)
