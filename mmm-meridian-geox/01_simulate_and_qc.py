"""STAGE 1 - Synthetic 180-day DMA panel + pre-test data-quality checks.

1. Build a realistic hospitality panel (60 DMAs x 180 days).
2. Inject known faults: missing dates, duplicate rows, a tracking-broken DMA
   (spend > 0, zero bookings), a booking-engine outage, local event spikes.
3. Run the built-in `meridian_geox` quality diagnostics on the DIRTY data and show
   what they catch. They cover: missing days, duplicates, zero-response, spend>0 with
   no response, aggregate pre-test date outliers.
4. Add the checks geox does not do (per-DMA event spikes) and write a CLEAN file.

    python 01_simulate_and_qc.py
"""
import datetime as dt
import json

import numpy as np
import pandas as pd

import meridian_geox as geox
from common import *   # noqa: F401,F403

setup_logging()
pd.set_option("display.width", 200, "display.max_colwidth", 90, "display.max_columns", 30)

# --------------------------------------------------------------------------- #
# 1. Clean synthetic pre-test panel
# --------------------------------------------------------------------------- #
full = build_untreated_panel()
md = geo_metadata()
md.to_csv(OUT / "geo_metadata.csv", index=False)
pre = split_windows(full)["pretest"][["date", "location", "conversions", "spend"]].copy()
banner("1. SYNTHETIC PRE-TEST PANEL (clean, before fault injection)")
print(f"rows={len(pre):,}  DMAs={pre.location.nunique()}  days={pre.date.nunique()}  "
      f"{pre.date.min().date()} -> {pre.date.max().date()}")
wk = pre.groupby(pre.date.dt.dayofweek).conversions.mean()
print("avg total booking revenue by weekday (Mon..Sun, index 100 = mean):")
print((wk / wk.mean() * 100).round(1).to_string())
print("\nrevenue by region ($k/day):")
print(pre.merge(md[["location", "region"]]).groupby("region").conversions.sum().div(PRETEST_DAYS * 1e3).round(0).to_string())

# --------------------------------------------------------------------------- #
# 2. Fault injection (known ground truth for the QC demo)
# --------------------------------------------------------------------------- #
rng = np.random.default_rng(SEED + 1)
dates = pd.date_range(PRETEST_START, periods=PRETEST_DAYS)
dirty = pre.copy()
faults = {}

# (a) booking-engine outage: ~75% of bookings lost on one day, partial recovery next day
outage_days = [dates[95], dates[96]]
dirty.loc[dirty.date == outage_days[0], "conversions"] *= 0.25
dirty.loc[dirty.date == outage_days[1], "conversions"] *= 0.60
faults["outage_dates"] = [str(d.date()) for d in outage_days]

# (b) local events: demand spikes in single DMAs (e.g. SXSW-like, music festival)
events = {"Austin": (dates[62], 4, 1.9), "Nashville": (dates[120], 3, 1.7)}
for geo, (d0, n, mult) in events.items():
    m = (dirty.location == geo) & dirty.date.between(d0, d0 + pd.Timedelta(days=n - 1))
    dirty.loc[m, "conversions"] *= mult
faults["event_spikes"] = {g: f"{d0.date()} +{n}d x{mult}" for g, (d0, n, mult) in events.items()}

# (c) tracking broken in one DMA: spend flows, zero bookings recorded
dirty.loc[dirty.location == BROKEN_TRACKING_GEO, "conversions"] = 0.0
faults["zero_response_geo"] = BROKEN_TRACKING_GEO

# (d) two entire dates missing + sporadic missing geo-days
missing_dates = [dates[40], dates[41]]
dirty = dirty[~dirty.date.isin(missing_dates)]
dirty = dirty.drop(dirty.sample(frac=0.002, random_state=3).index)
faults["missing_dates"] = [str(d.date()) for d in missing_dates]

# (e) duplicate geo-date records
dirty = pd.concat([dirty, dirty.sample(25, random_state=5)], ignore_index=True)
faults["duplicate_rows"] = 25
dirty = dirty.sort_values(["date", "location"]).reset_index(drop=True)
dirty.to_csv(OUT / "pretest_raw_dirty.csv", index=False)
save_json(faults, OUT / "injected_faults.json")
banner("2. INJECTED FAULTS (ground truth for the QC demo)")
print(json.dumps(faults, indent=2))

# --------------------------------------------------------------------------- #
# 3. Built-in meridian_geox diagnostics on the DIRTY data
# --------------------------------------------------------------------------- #
# A DesignConfig is only needed because the checker reads experiment_types from it.
cfg = geox.DesignConfig(experiment_duration=dt.timedelta(days=TEST_DAYS),
                        experiment_types=geox.ExperimentType.HEAVY_UP,
                        methodology=geox.Methodology.TBR)
qc_cfg = geox.QualityCheckConfig(exclude_geos_no_response=True, exclude_outlier_dates=True)
banner("3. meridian_geox.check_design_data_quality  (DIRTY data)")
qc_dirty = geox.check_design_data_quality(dirty, cfg, qc_cfg)
print(qc_dirty.quality_metrics[["metric", "value", "message"]].to_string(index=False))
print("\noutlier_geos :", sorted(qc_dirty.outlier_geos))
print("outlier_dates:", sorted(str(d.date()) for d in qc_dirty.outlier_dates))
# Library note: duplicates are only *flagged* ("will be aggregated"); its pivot SUMS them,
# so a duplicated row silently double counts bookings -> we de-duplicate ourselves.

# --------------------------------------------------------------------------- #
# 4. Cleaning + the checks geox does not cover
# --------------------------------------------------------------------------- #
log = []
clean = dirty.drop_duplicates(["date", "location"], keep="first")
log.append(("duplicate rows removed", len(dirty) - len(clean)))

# 4a. complete the date x DMA grid; fill gaps with the same-weekday neighbour average
grid = pd.MultiIndex.from_product([dates, md.location], names=["date", "location"])
clean = clean.set_index(["date", "location"]).reindex(grid)
n_missing = int(clean.conversions.isna().sum())
for col in ["conversions", "spend"]:
    w = clean[col].unstack("location")
    same_dow = (w.shift(7) + w.shift(-7)) / 2
    w = w.fillna(same_dow).interpolate(limit_direction="both")
    clean[col] = w.stack().reindex(clean.index)
log.append(("missing geo-days imputed (same-weekday avg)", n_missing))
clean = clean.reset_index()

# 4b. per-DMA local event / data glitch detector (robust z on DoW-adjusted residuals)
w = clean.pivot(index="date", columns="location", values="conversions")
wl = np.log(w.where(w > 0))
dow_med = wl.groupby(wl.index.dayofweek).transform("median")
resid = wl - dow_med
resid = resid - resid.rolling(21, center=True, min_periods=7).median()
mad = (resid - resid.median()).abs().median() * 1.4826
z = resid / mad
core = (z.abs() > 4.0) & (resid.abs() > 0.35)
# multi-day events: extend a flagged run to adjacent days that are still clearly elevated
near = z.abs() > 2.5
flag = core | (near & (core.shift(1, fill_value=False) | core.shift(-1, fill_value=False)))
flag = flag | (near & flag.shift(1, fill_value=False)) | (near & flag.shift(-1, fill_value=False))
flag.loc[outage_days] = False       # the outage hits all DMAs; handled at date level by geox
hits = flag.stack()[lambda s: s].reset_index()
hits.columns = ["date", "location", "flag"]
banner("4. EXTRA CHECK - per-DMA event spikes (not covered by geox's aggregate-level check)")
print(hits.groupby("location").agg(first=("date", "min"), last=("date", "max"), days=("date", "count")).to_string())
# winsorise flagged geo-days to the expected level (keeps geo in the design)
exp_level = np.exp(dow_med + resid.rolling(21, center=True, min_periods=7).median().fillna(0)
                   + resid.median())
for _, r in hits.iterrows():
    clean.loc[(clean.date == r.date) & (clean.location == r.location), "conversions"] = exp_level.loc[r.date, r.location]
log.append(("local-event geo-days winsorised", len(hits)))

clean[["date", "location", "conversions", "spend"]].to_csv(OUT / "pretest_clean.csv", index=False)

# --------------------------------------------------------------------------- #
# 5. Re-run the library check on the CLEAN file
# --------------------------------------------------------------------------- #
banner("5. meridian_geox.check_design_data_quality  (CLEANED data)")
qc_clean = geox.check_design_data_quality(clean[["date", "location", "conversions", "spend"]], cfg, qc_cfg)
print(qc_clean.quality_metrics[["metric", "value", "message"]].to_string(index=False)
      if len(qc_clean.quality_metrics) else "no warnings")
print("outlier_geos :", sorted(qc_clean.outlier_geos), " -> auto-excluded from design & analysis by geox")
print("outlier_dates:", sorted(str(d.date()) for d in qc_clean.outlier_dates), " -> auto-excluded by geox")

banner("CLEANING LOG")
print(pd.DataFrame(log, columns=["action", "count"]).to_string(index=False))

# --------------------------------------------------------------------------- #
# Verification: did every injected fault get caught?
# --------------------------------------------------------------------------- #
checks = {
    "duplicates flagged by geox": any("Duplicate" in m for m in qc_dirty.quality_metrics.metric),
    "broken-tracking geo flagged": BROKEN_TRACKING_GEO in qc_dirty.outlier_geos,
    "outage date(s) flagged": any(pd.Timestamp(d) in qc_dirty.outlier_dates for d in faults["outage_dates"]),
    "event spikes flagged (extra check)": set(hits.location) >= set(events),
    "no residual duplicates": not clean.duplicated(["date", "location"]).any(),
    "grid complete (no NaN)": bool(clean.conversions.notna().all()),
}
banner("VERIFICATION")
for k, v in checks.items():
    print(f"[{'PASS' if v else 'FAIL'}] {k}")
assert all(checks.values()), "stage-1 verification failed"
