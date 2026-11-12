"""STAGE 3 - Simulate the live test + cooldown with a KNOWN ground-truth incremental lift.

Data-generating process for treatment DMAs, test window only:
    extra_spend[g,t]  = BAU_spend[g,t] * HEAVY_UP_PCT * DELIVERY_FACTOR * noise
    incr_revenue[g,t] = TRUE_IROAS * sum_k LAG_KERNEL[k] * extra_spend[g,t-k] * noise
i.e. every incremental $ buys TRUE_IROAS of booking revenue, but the bookings arrive over the
following 0-14 days (search-to-book lag), so ~18% of the effect lands AFTER the test ends.

    python 03_simulate_posttest.py
"""
import numpy as np
import pandas as pd

import meridian_geox as geox
from common import *   # noqa: F401,F403

setup_logging()
pd.set_option("display.width", 200)

design = geox.Design.load_from_json((OUT / "design.json").read_text())
treat = sorted(design.designs["cell_1"].treatment_geos)

# --------------------------------------------------------------------------- #
# Baseline world: SAME seed as stage 1 => identical history; only test/cooldown are new
# --------------------------------------------------------------------------- #
world = build_untreated_panel()
world.loc[world.location == BROKEN_TRACKING_GEO, "conversions"] = 0.0     # tracking still broken
pre_clean = pd.read_csv(OUT / "pretest_clean.csv", parse_dates=["date"])
pre_world = split_windows(world)["pretest"]
m = pre_clean.merge(pre_world, on=["date", "location"], suffixes=("", "_w"))
ratio = (m.conversions / m.conversions_w).replace([np.inf], np.nan).dropna()
print(f"sanity: cleaned pre-test vs regenerated baseline  median ratio={ratio.median():.4f} "
      f"(1.0 expected; deviations = injected faults / repairs)")

# --------------------------------------------------------------------------- #
# Apply the heavy-up treatment + lagged incremental bookings
# --------------------------------------------------------------------------- #
rng = np.random.default_rng(SEED + 7)
dates = pd.date_range(PRETEST_START, periods=TOTAL_DAYS)
w_rev = world.pivot(index="date", columns="location", values="conversions")
w_sp = world.pivot(index="date", columns="location", values="spend")
T = len(dates)
in_test = ((dates >= TEST_START) & (dates <= TEST_END))

extra_spend = pd.DataFrame(0.0, index=dates, columns=w_sp.columns)
sd = 0.08
delivery = DELIVERY_FACTOR * rng.lognormal(-sd**2 / 2, sd, (in_test.sum(), len(treat)))   # pacing noise
extra_spend.loc[in_test, treat] = w_sp.loc[in_test, treat].values * HEAVY_UP_PCT * delivery

# lagged response: convolve each geo's extra spend with the booking-lag kernel
incr_rev = pd.DataFrame(0.0, index=dates, columns=w_sp.columns)
for g in treat:
    resp = np.convolve(extra_spend[g].values, LAG_KERNEL)[:T]
    s = 0.15
    incr_rev[g] = TRUE_IROAS * resp * rng.lognormal(-s**2 / 2, s, T)

obs_rev = w_rev + incr_rev
obs_sp = w_sp + extra_spend
panel = (pd.concat({"conversions": obs_rev, "spend": obs_sp}, axis=1).stack(level=1, future_stack=True)
         .rename_axis(["date", "location"]).reset_index())

# stitch: keep the *cleaned* historical pre-test exactly as the analyst would have it
post = panel[panel.date > PRETEST_END]
analysis_data = pd.concat([pre_clean, post[["date", "location", "conversions", "spend"]]], ignore_index=True)
analysis_data = analysis_data.sort_values(["date", "location"]).reset_index(drop=True)
analysis_data.to_csv(OUT / "analysis_data.csv", index=False)

# --------------------------------------------------------------------------- #
# Ground truth bookkeeping
# --------------------------------------------------------------------------- #
tot_spend = float(extra_spend.sum().sum())
mask_test = in_test
rev_test = float(incr_rev.loc[mask_test, treat].sum().sum())
rev_cool = float(incr_rev.loc[dates > TEST_END, treat].sum().sum())
rev_cool7 = float(incr_rev.loc[(dates > TEST_END) & (dates <= TEST_END + pd.Timedelta(days=7)), treat].sum().sum())
truth = dict(
    true_iroas_parameter=TRUE_IROAS,
    realised_iroas_lifetime=(rev_test + rev_cool) / tot_spend,
    realised_iroas_test_window_only=rev_test / tot_spend,
    realised_iroas_test_plus_7d=(rev_test + rev_cool7) / tot_spend,
    incremental_revenue_cooldown_first7d=rev_cool7,
    incremental_spend=tot_spend,
    incremental_revenue_test_window=rev_test,
    incremental_revenue_cooldown=rev_cool,
    share_of_effect_after_test_end=rev_cool / (rev_test + rev_cool),
    treatment_geos=treat,
    test_start=str(TEST_START.date()), test_end=str(TEST_END.date()),
    cooldown_end=str(COOLDOWN_END.date()),
    planned_heavy_up_pct=HEAVY_UP_PCT, delivery_factor=DELIVERY_FACTOR,
)
save_json(truth, OUT / "ground_truth.json")

banner("STAGE 3 - SIMULATED POST-TEST DATA")
print(f"window rows: {len(analysis_data):,}  | {analysis_data.date.min().date()} -> {analysis_data.date.max().date()}")
print(f"test     : {TEST_START.date()} -> {TEST_END.date()} ({TEST_DAYS}d, Sat-Fri, no US holidays)")
print(f"cooldown : {COOLDOWN_START.date()} -> {COOLDOWN_END.date()} ({COOLDOWN_DAYS}d, no extra spend, bookings still landing)")
print(f"treatment DMAs ({len(treat)}): {', '.join(treat)}")
print(pd.Series({k: v for k, v in truth.items() if k not in ("treatment_geos",)}).to_string())

# daily path of the true incremental effect
daily = incr_rev[treat].sum(axis=1)
by_week = daily.groupby(((dates - TEST_START).days // 7)).sum()
print("\ntrue incremental revenue per week since test start ($):")
print(by_week[by_week.index >= 0].round(0).rename_axis("week_since_test_start").to_string())

# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
checks = {
    "lifetime iROAS == parameter (±3%)": abs(truth["realised_iroas_lifetime"] / TRUE_IROAS - 1) < 0.03,
    "all lagged bookings land inside cooldown (nothing after day 14)": float(daily[dates > COOLDOWN_END].sum()) < 1e-6,
    "test-window read under-states effect (lag)": truth["realised_iroas_test_window_only"] < truth["realised_iroas_lifetime"],
    "share after test end in 10-30%": 0.10 < truth["share_of_effect_after_test_end"] < 0.30,
    "control & excluded geos untouched": float(incr_rev.drop(columns=treat).abs().sum().sum()) == 0.0,
    "no extra spend outside the test window": float(extra_spend.loc[~in_test].abs().sum().sum()) == 0.0,
    "geo set matches design": set(analysis_data.location) == set(pre_clean.location),
}
banner("VERIFICATION")
for k, v in checks.items():
    print(f"[{'PASS' if v else 'FAIL'}] {k}")
assert all(checks.values()), "stage-3 verification failed"
