"""STAGE 2 - Stratified heavy-up design, operational exclusions, MDE / power / duration.

Uses the CLEAN pre-test panel written by stage 1.

    python 02_design_power.py
"""
import datetime as dt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", message="FigureCanvasAgg")
warnings.filterwarnings("ignore", message="The balance properties of Sobol")

import meridian_geox as geox
from common import *   # noqa: F401,F403

setup_logging()
pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 100)

pre = pd.read_csv(OUT / "pretest_clean.csv", parse_dates=["date"])
md = pd.read_csv(OUT / "geo_metadata.csv")

# --------------------------------------------------------------------------- #
# 1. Operational geo-exclusions (inventory + brand constraints)
# --------------------------------------------------------------------------- #
excl = operational_exclusions(md)
banner("1. OPERATIONAL EXCLUSIONS (cannot be treatment OR control)")
print(excl[["location", "region", "avg_daily_rev_k", "peak_occupancy", "renovation", "exclusion_reason"]].to_string(index=False))
# Why exclude: at >=95% occupancy extra demand can't be fulfilled (lift is capped at 0),
# and renovation DMAs have shrinking / non-stationary inventory -> both break the
# "control predicts treatment" assumption behind TBR.
BUDGET_PCT = HEAVY_UP_PCT
constraints = geox.Constraints(
    excluded_geos=set(excl.location),
    budget_constraint=geox.Budget(budget_pct=BUDGET_PCT),     # +50% spend in treatment DMAs
    max_conversions_percent=0.30,                              # treatment <= 30% of revenue
)
qc = geox.QualityCheckConfig(exclude_geos_no_response=True, exclude_outlier_dates=True)


def make_config(days: int, **kw) -> geox.DesignConfig:
    return geox.DesignConfig(
        experiment_duration=dt.timedelta(days=days),
        experiment_types=geox.ExperimentType.HEAVY_UP,
        methodology=geox.Methodology.TBR,                       # only methodology geox 1.0.1 implements
        geo_assignment_rule=geox.GeoAssignmentRule.STRATIFIED_SAMPLING,
        num_strata=4,                                           # k-means strata on volume/trend/seasonality
        alpha=0.10, power=0.80, test_type=geox.TestType.TWO_SIDED,
        n_candidates=50_000, n_ranked_candidates=100, design_output_count=10,
        seed=42, **kw)


# --------------------------------------------------------------------------- #
# 2. Duration sweep: MDE vs. test length (accounting for the booking lag)
# --------------------------------------------------------------------------- #
banner("2. DURATION SWEEP  (best design per duration; alpha=0.10 two-sided, power=0.80)")
rows, sets = [], {}
for days in (14, 21, 28, 35, 42):
    ds = geox.run_design(pre, make_config(days), constraints, qc)
    sets[days] = ds
    best = ds.design_metrics.iloc[0]
    rows.append(dict(test_days=days, r2=best.r2, mde_pct=100 * best.mde, mde_abs_rev=best.mde_abs,
                     incr_spend=best.budget, min_detectable_iROAS=best.mde_abs / best.budget,
                     treat_geos=int(best.treatment_geo_count), treat_rev_share_pct=best.treatment_conversions_pct,
                     aa_pvalue=best["p_value (AA)"]))
sweep = pd.DataFrame(rows)
print(sweep.round(3).to_string(index=False))
sweep.to_csv(OUT / "design_duration_sweep.csv", index=False)
print("\nmin_detectable_iROAS = MDE revenue / incremental spend  (what iROAS the test can resolve at 80% power)")

# --------------------------------------------------------------------------- #
# 3. Cooldown sizing from the booking-lag kernel
# --------------------------------------------------------------------------- #
banner("3. COOLDOWN SIZING - share of lifetime incremental revenue captured in the read window")
print("lag kernel (days since ad -> booking), mean lag = %.1f d, P(lag<=7)=%.0f%%, P(lag<=14)=%.0f%%"
      % ((LAG_KERNEL * np.arange(LAG_MAX + 1)).sum(), 100 * LAG_KERNEL[:8].sum(), 100 * LAG_KERNEL.sum()))
cap = []
for D in (14, 21, 28, 35, 42):
    row = {"test_days": D}
    spend_profile = np.ones(D)                                        # flat extra spend
    resp = np.convolve(spend_profile, LAG_KERNEL)                      # bookings per day, len D+LAG_MAX
    for c in (0, 7, 14, 21):
        row[f"cooldown_{c}d"] = 100 * resp[: D + c].sum() / resp.sum()
    cap.append(row)
cap = pd.DataFrame(cap)
print(cap.round(1).to_string(index=False))
print("-> with a 14d cooldown ~100% of the lifetime effect is observed; reading only the test window "
      "under-states iROAS by the gap shown in the cooldown_0d column.")
cap.to_csv(OUT / "cooldown_capture.csv", index=False)

# --------------------------------------------------------------------------- #
# 4. Final design (28 days = 4 whole weeks, Sat->Fri) and power curve
# --------------------------------------------------------------------------- #
banner("4. FINAL DESIGN - top-10 candidates for the 28-day test")
DUR = TEST_DAYS
final_set = sets[DUR]
print(final_set.design_metrics[["design_id", "cell", "design_methodology", "r2", "mde", "mde_abs", "p_value (AA)",
                                "budget", "treatment_conversions_pct", "treatment_geo_count"]]
      .assign(design_id=lambda d: d.design_id.astype(str).str[:8]).round(4).to_string(index=False))
best_id = final_set.design_metrics.iloc[0].design_id
design = final_set.designs[best_id]
cell = design.designs["cell_1"]

# expected effect under planning assumption iROAS (lag-adjusted to the 28-day read window)
cap28 = cap.loc[cap.test_days == DUR, "cooldown_0d"].iloc[0] / 100
z_a, z_p = stats.norm.ppf(1 - 0.10 / 2), stats.norm.ppf(0.80)
se = cell.minimum_detectable_effect / (z_a + z_p)            # implied std. error of % lift
treat_rev = final_set.design_metrics.iloc[0].mde_abs / cell.minimum_detectable_effect   # treatment-geo revenue in window
banner("POWER CURVE for the chosen design (test-window read vs. test+cooldown read)")
pw = []
for iroas in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
    for label, share in (("test window only", cap28), ("test + 14d cooldown", 1.0)):
        lift = iroas * cell.budget * share / treat_rev        # fraction of treatment-geo revenue
        power = stats.norm.cdf(lift / se - z_a) + stats.norm.cdf(-lift / se - z_a)
        pw.append(dict(true_iROAS=iroas, read=label, expected_lift_pct=100 * lift, power=power))
pw = pd.DataFrame(pw)
print(pw.pivot(index="true_iROAS", columns="read", values="power").round(2).to_string())
print(f"\nMDE (80% power) = {100*cell.minimum_detectable_effect:.2f}% of treatment-geo revenue "
      f"= ${final_set.design_metrics.iloc[0].mde_abs:,.0f};  extra spend required (+{BUDGET_PCT:.0%}) = ${cell.budget:,.0f}")
pw.to_csv(OUT / "power_curve.csv", index=False)

# --------------------------------------------------------------------------- #
# 5. Control vs Treatment assignment table
# --------------------------------------------------------------------------- #
geos_in_design = sorted(set(pre.location) - set(design.excluded_geos))
strata = dict(zip(geos_in_design, np.asarray(design.geo_stratum_labels).astype(int)))
avg = pre.groupby("location").agg(avg_daily_rev=("conversions", "mean"), avg_daily_spend=("spend", "mean"))
rows = []
for g in md.location:
    if g in cell.treatment_geos:
        grp, why = "TREATMENT", "heavy-up +%d%% paid search+social" % (BUDGET_PCT * 100)
    elif g in design.control_geos:
        grp, why = "CONTROL", "business-as-usual spend"
    else:
        grp = "EXCLUDED"
        why = excl.set_index("location").exclusion_reason.get(g, "data-quality (spend>0, zero bookings)")
    rows.append((g, grp, strata.get(g, -1), why))
asg = (pd.DataFrame(rows, columns=["location", "group", "stratum", "reason"])
       .merge(md[["location", "region", "peak_occupancy"]]).merge(avg, on="location").sort_values(["group", "stratum", "location"]))
asg.to_csv(OUT / "assignment_table.csv", index=False)
banner("5. FINAL ASSIGNMENT TABLE")
print(asg.round({"avg_daily_rev": 0, "avg_daily_spend": 0}).to_string(index=False))
print("\ngroup sizes:", asg.group.value_counts().to_dict())
summary = asg.groupby("group").agg(n=("location", "size"), avg_daily_rev=("avg_daily_rev", "sum"), avg_daily_spend=("avg_daily_spend", "sum"))
print(summary.round(0).to_string())
print("\ntreatment-by-stratum balance:")
print(pd.crosstab(asg[asg.group != "EXCLUDED"].stratum, asg[asg.group != "EXCLUDED"].group).to_string())

# persist for stage 3/4
(OUT / "design.json").write_text(design.export_to_json())
reloaded = geox.Design.load_from_json((OUT / "design.json").read_text())
assert reloaded.designs["cell_1"].treatment_geos == cell.treatment_geos

geox.plot_design(design)
for i, n in enumerate(plt.get_fignums(), 1):
    plt.figure(n).savefig(OUT / f"design_pretest_fit_{i}.png", dpi=110, bbox_inches="tight")

# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
tg, cg = set(cell.treatment_geos), set(design.control_geos)
checks = {
    "no overlap treatment/control": not (tg & cg),
    "operational exclusions respected": not ((tg | cg) & set(excl.location)),
    "broken-tracking geo excluded": BROKEN_TRACKING_GEO not in (tg | cg),
    "A/A p-value >= alpha": cell.p_value >= 0.10,
    "treatment share of revenue <= 30%": final_set.design_metrics.iloc[0].treatment_conversions_pct <= 30.0,
    "design JSON round-trip": True,
    "MDE plausible (0.5%-5% of treatment revenue)": bool(sweep.mde_pct.between(0.5, 5).all()),
    "min detectable iROAS < planning iROAS (2.0)": float(sweep.loc[sweep.test_days == DUR, "min_detectable_iROAS"].iloc[0]) < TRUE_IROAS,
}
# NOTE: pct-MDE is NOT strictly monotone in test length: errors are autocorrelated
# (regional shocks), each duration re-draws its own best candidate, and longer tests
# shorten the training window. Pick duration on the booking cycle, not on this column alone.
banner("VERIFICATION")
for k, v in checks.items():
    print(f"[{'PASS' if v else 'FAIL'}] {k}")
assert all(checks.values()), "stage-2 verification failed"
