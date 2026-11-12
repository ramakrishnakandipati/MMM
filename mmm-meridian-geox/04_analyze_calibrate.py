"""STAGE 4 - Counterfactual analysis, estimator comparison, Meridian MMM prior.

A. meridian_geox.analyze()  (TBR - the only methodology implemented in geox 1.0.1; its
   confidence intervals come from design-aware placebo inference: placebo geo splits are
   drawn with the same stratified-sampling rule + R^2 filter as the real design).
   Run twice: (i) test window only, (ii) test + 14d cooldown, to show the booking-lag bias.
B. Reference Synthetic Control and Synthetic DiD (custom, ~60 lines each; NOT in geox)
   with the same design-aware placebo idea: pseudo-treatment groups are drawn from the
   control pool with the SAME stratum composition as the real treatment group.
C. Turn the iROAS posterior-like distribution into a Meridian `roi_m` LogNormal prior.

    python 04_analyze_calibrate.py
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats

import meridian_geox as geox
from common import *   # noqa: F401,F403

warnings.filterwarnings("ignore", message="FigureCanvasAgg")
warnings.filterwarnings("ignore", message="The balance properties of Sobol")
setup_logging()
pd.set_option("display.width", 220, "display.max_columns", 40)

data = pd.read_csv(OUT / "analysis_data.csv", parse_dates=["date"])
design = geox.Design.load_from_json((OUT / "design.json").read_text())
truth = json.loads((OUT / "ground_truth.json").read_text())
ALPHA = 0.10
qc = geox.QualityCheckConfig(exclude_geos_no_response=True, exclude_outlier_dates=True)

# --------------------------------------------------------------------------- #
# A. meridian_geox.analyze()
# --------------------------------------------------------------------------- #
def run_geox(end_date: pd.Timestamp) -> geox.AnalysisResult:
    cfg = geox.AnalysisConfig(design=design, analysis_start_date=TEST_START, analysis_end_date=end_date,
                              alpha=ALPHA, n_placebo_candidates=20_000, n_top_placebos=500, min_placebo_r2=0.6)
    return geox.analyze(data, cfg, qc)


banner("A0. geox analysis-phase data-quality check")
chk_cfg = geox.AnalysisConfig(design=design, analysis_start_date=TEST_START, analysis_end_date=TEST_END)
qcr = geox.check_analysis_data_quality(data, chk_cfg, qc)
print(qcr.quality_metrics[["metric", "value"]].to_string(index=False) if len(qcr.quality_metrics) else "clean")
print("auto-excluded dates:", sorted(str(d.date()) for d in qcr.outlier_dates))

res_test = run_geox(TEST_END)
res_7d = run_geox(TEST_END + pd.Timedelta(days=7))
res_full = run_geox(COOLDOWN_END)


def est_row(name, window, lift, inc_spend, truth_rev, truth_iroas, extra=None):
    """lift: (point, lo, hi, p); inc_spend: scalar; returns a dict with iROAS."""
    p, lo, hi, pv = lift
    r = dict(method=name, window=window, lift_usd=p, lift_lo=lo, lift_hi=hi, p_value=pv,
             incr_spend=inc_spend, iROAS=p / inc_spend, iROAS_lo=lo / inc_spend, iROAS_hi=hi / inc_spend,
             true_iROAS=truth_iroas, true_lift=truth_rev)
    r["covers_truth"] = r["iROAS_lo"] <= truth_iroas <= r["iROAS_hi"]
    return r


def geox_row(res, window, truth_rev, truth_iroas):
    m = res.results["cell_1"]
    inc_spend = m.lift.point_estimate / m.icpd.point_estimate
    r = est_row("geox TBR", window, (m.lift.point_estimate, m.lift.lower_bound, m.lift.upper_bound, m.lift.p_value),
                inc_spend, truth_rev, truth_iroas)
    # use geox's own iCPD interval (identical in construction) rather than re-deriving it
    r.update(iROAS=m.icpd.point_estimate, iROAS_lo=m.icpd.lower_bound, iROAS_hi=m.icpd.upper_bound)
    r["covers_truth"] = r["iROAS_lo"] <= truth_iroas <= r["iROAS_hi"]
    r["pct_lift"] = 100 * m.percent_lift.point_estimate
    return r


T_REV = {"test": truth["incremental_revenue_test_window"],
         "test+7d": truth["incremental_revenue_test_window"] + truth["incremental_revenue_cooldown_first7d"],
         "test+14d": truth["incremental_revenue_test_window"] + truth["incremental_revenue_cooldown"]}
T_ROAS = {"test": truth["realised_iroas_test_window_only"], "test+7d": truth["realised_iroas_test_plus_7d"],
          "test+14d": truth["realised_iroas_lifetime"]}
rows = [geox_row(res_test, "test", T_REV["test"], T_ROAS["test"]),
        geox_row(res_7d, "test+7d", T_REV["test+7d"], T_ROAS["test+7d"]),
        geox_row(res_full, "test+14d", T_REV["test+14d"], T_ROAS["test+14d"])]

banner("A1. geox.analyze() - TBR, design-aware placebo inference (alpha=0.10 => 90% intervals)")
for w, r_ in (("test", res_test), ("test+7d", res_7d), ("test+14d", res_full)):
    m = r_.results["cell_1"]
    print(f"[{w}] lift ${m.lift.point_estimate:,.0f}  90% CI [{m.lift.lower_bound:,.0f}, {m.lift.upper_bound:,.0f}]  "
          f"p={m.lift.p_value:.4f} | %lift {100*m.percent_lift.point_estimate:.2f}% "
          f"[{100*m.percent_lift.lower_bound:.2f}%, {100*m.percent_lift.upper_bound:.2f}%] | "
          f"iROAS {m.icpd.point_estimate:.3f} [{m.icpd.lower_bound:.3f}, {m.icpd.upper_bound:.3f}] sd={m.icpd.standard_deviation:.3f}")
print("geox auto-excluded geos :", sorted(res_full.excluded_geos))
print("geox auto-excluded dates:", sorted(str(d.date()) for d in res_full.excluded_dates))

geox.plot_analysis(res_full)
for i, n in enumerate(plt.get_fignums(), 1):
    plt.figure(n).savefig(OUT / f"analysis_tbr_{i}.png", dpi=110, bbox_inches="tight")
plt.close("all")

# --------------------------------------------------------------------------- #
# B. Reference SC / SDID with design-aware placebo inference
# --------------------------------------------------------------------------- #
excl_dates = {pd.Timestamp(d) for d in res_full.excluded_dates}
excl_geos = set(res_full.excluded_geos)
use = data[~data.location.isin(excl_geos) & ~data.date.isin(excl_dates)]
Y = use.pivot(index="date", columns="location", values="conversions")
S = use.pivot(index="date", columns="location", values="spend")
treated = sorted(design.designs["cell_1"].treatment_geos)
controls = sorted(set(Y.columns) - set(treated))
geos_design = sorted(set(data.location) - set(design.excluded_geos))
strat = dict(zip(geos_design, np.asarray(design.geo_stratum_labels).astype(int)))
PRE_LEN = 126                                                     # 18 whole weeks of local history
pre_idx = Y.index[Y.index < TEST_START][-PRE_LEN:]
win = {"test": Y.index[(Y.index >= TEST_START) & (Y.index <= TEST_END)],
       "test+14d": Y.index[(Y.index >= TEST_START) & (Y.index <= COOLDOWN_END)]}


def simplex_ls(A, b, ridge=0.0):
    """min ||A w - b||^2 + ridge ||w||^2  s.t. w >= 0, sum w = 1."""
    n = A.shape[1]
    AtA, Atb = A.T @ A + ridge * np.eye(n), A.T @ b
    f = lambda w: w @ AtA @ w - 2 * Atb @ w
    g = lambda w: 2 * (AtA @ w - Atb)
    r = optimize.minimize(f, np.full(n, 1 / n), jac=g, bounds=[(0, 1)] * n, method="SLSQP",
                          constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(n)}],
                          options={"maxiter": 200, "ftol": 1e-12})
    return r.x


def sc_weights(Ypre_c, ypre_t):
    return simplex_ls(Ypre_c.T, ypre_t)


def sdid_unit_weights(Ypre_c, ypre_t, n_tr, t_post):
    sigma = np.diff(Ypre_c, axis=1).std()
    zeta2 = (n_tr * t_post) ** 0.5 * sigma ** 2                         # zeta^2 = (N_tr T_post)^(1/2) sigma^2
    A = (Ypre_c - Ypre_c.mean(1, keepdims=True)).T
    return simplex_ls(A, ypre_t - ypre_t.mean(), ridge=zeta2 * A.shape[0])     # zeta^2 * T_pre * ||w||^2


def sdid_time_weights(Ypre_c, Ypost_c):
    M = Ypre_c - Ypre_c.mean(0, keepdims=True)                          # remove unit-mean (lambda_0)
    b = Ypost_c.mean(1) - Ypost_c.mean(1).mean()
    return simplex_ls(M, b, ridge=1e-9 * (M ** 2).sum())


def effects(df, tr, co, wname, which="both"):
    """Per-day average effect on the treated group under SC and SDID, for window `wname`."""
    pre = df.loc[pre_idx]
    post = df.loc[win[wname]]
    Ypre_c, Ypost_c = pre[co].values.T, post[co].values.T              # (N_co, T)
    ypre_t, ypost_t = pre[tr].values.mean(1), post[tr].values.mean(1)
    scale = Ypre_c.mean()
    Ypre_c, Ypost_c, ypre_t, ypost_t = (a / scale for a in (Ypre_c, Ypost_c, ypre_t, ypost_t))
    w_sc = sc_weights(Ypre_c, ypre_t)
    e_sc = (ypost_t - w_sc @ Ypost_c) * scale
    om = sdid_unit_weights(Ypre_c, ypre_t, len(tr), Ypost_c.shape[1])
    lam = sdid_time_weights(Ypre_c, Ypost_c)
    e_sd = ((ypost_t - lam @ ypre_t) - (om @ Ypost_c - lam @ (om @ Ypre_c))) * scale
    return e_sc, e_sd


def run_estimators(df, tr, co):
    """Return {(method, window): total effect over window} for a treated/control split."""
    out = {}
    for wname in win:
        e_sc, e_sd = effects(df, tr, co, wname)
        out[("SC (custom)", wname)] = len(tr) * e_sc.sum()
        out[("SDID (custom)", wname)] = len(tr) * e_sd.sum()
    return out


def placebo_split(rng):
    """Design-aware placebo: stratified sampling over the CONTROL pool with the design's stratum labels.

    Same total group size as the real treatment group, allocated proportionally to each stratum's share of
    the control pool (largest-remainder rounding) - the same rule geox uses to draw its own placebo splits.
    Copying the real treated stratum counts would be wrong: they were drawn from treated+control geos together,
    so applying them to controls alone over-samples small strata and biases the placebo distribution.
    """
    pools = {s: [g for g in controls if strat[g] == s] for s in sorted({strat[g] for g in controls})}
    share = {s: len(p) * len(treated) / len(controls) for s, p in pools.items()}
    alloc = {s: int(np.floor(v)) for s, v in share.items()}
    for s in sorted(share, key=lambda s: share[s] - alloc[s], reverse=True)[: len(treated) - sum(alloc.values())]:
        alloc[s] += 1
    pick = []
    for s, k in alloc.items():
        pick += list(rng.choice(pools[s], size=k, replace=False))
    return sorted(pick)


import time
t0 = time.time()
real_rev = run_estimators(Y, treated, controls)
real_sp = run_estimators(S, treated, controls)
print(f"\n[timing] one SC+SDID pass over both windows: {time.time()-t0:.1f}s")

def holdout_r2(tr, co):
    """Out-of-sample R^2 of the SC fit: weights from the older pre-test days, scored on the last TEST_DAYS pre days
    (mirrors geox's `min_placebo_r2` filter, which keeps only splits whose control predicts treatment well)."""
    fit_idx, ho_idx = pre_idx[:-TEST_DAYS], pre_idx[-TEST_DAYS:]
    A, b = Y.loc[fit_idx, co].values, Y.loc[fit_idx, tr].values.mean(1)
    s = A.mean()
    w = sc_weights((A / s).T, b / s)
    y, yhat = Y.loc[ho_idx, tr].values.mean(1), Y.loc[ho_idx, co].values @ w
    return 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()


# Design-aware placebos: (1) stratified draws with the real stratum mix, (2) keep the best-fitting quartile+,
# because the real treatment group was ALSO chosen as a well-matched split out of tens of thousands of candidates.
N_DRAW, K = 500, 150
rng = np.random.default_rng(2026)
draws = [placebo_split(rng) for _ in range(N_DRAW)]
r2s = np.array([holdout_r2(pt, [g for g in controls if g not in pt]) for pt in draws])
r2_real = holdout_r2(treated, controls)
keep = np.argsort(-r2s)[:K]
print(f"\nholdout R^2 - real treatment group: {r2_real:.3f} | all {N_DRAW} placebo draws: median {np.median(r2s):.3f} "
      f"| kept top {K}: min {r2s[keep].min():.3f}, median {np.median(r2s[keep]):.3f}")
plc = {k: [] for k in real_rev}
for i in keep:
    pt = draws[i]
    pc = [g for g in controls if g not in pt]
    for k, v in run_estimators(Y, pt, pc).items():
        plc[k].append(v)
print(f"[timing] total incl. {N_DRAW} draws / {K} kept placebos: {time.time()-t0:.1f}s")

banner("B0. PLACEBO NULL DISTRIBUTIONS (true effect = 0 by construction) vs. observed effect, $ revenue over window")
diag = []
for (meth, w), lift in real_rev.items():
    e = np.array(plc[(meth, w)])
    diag.append(dict(method=meth, window=w, placebo_mean=e.mean(), placebo_median=np.median(e), placebo_sd=e.std(),
                     p05=np.quantile(e, .05), p95=np.quantile(e, .95), observed=lift, observed_z=(lift - e.mean()) / e.std()))
print(pd.DataFrame(diag).round(0).to_string(index=False))

for (meth, w), lift in real_rev.items():
    e = np.array(plc[(meth, w)])
    lo, hi = lift - np.quantile(e, 1 - ALPHA / 2), lift - np.quantile(e, ALPHA / 2)     # placebo-error inversion
    pv = (1 + np.sum(np.abs(e) >= abs(lift))) / (1 + K)
    rows.append(est_row(meth, w, (lift, lo, hi, pv), real_sp[(meth, w)], T_REV[w], T_ROAS[w]))

res = pd.DataFrame(rows)
banner("B. ESTIMATOR COMPARISON (90% intervals; truth = what the simulator actually delivered in that window)")
show = res[["method", "window", "lift_usd", "lift_lo", "lift_hi", "p_value", "incr_spend", "iROAS", "iROAS_lo", "iROAS_hi",
            "true_iROAS", "covers_truth"]].copy()
for c in ("lift_usd", "lift_lo", "lift_hi", "incr_spend"):
    show[c] = show[c].round(0).astype(int)
print(show.round({"iROAS": 3, "iROAS_lo": 3, "iROAS_hi": 3, "true_iROAS": 3, "p_value": 4}).to_string(index=False))
res.to_csv(OUT / "estimator_comparison.csv", index=False)
res["bias_pct"] = 100 * (res.iROAS / res.true_iROAS - 1)
print("\nbias vs truth (%):")
print(res.pivot(index="method", columns="window", values="bias_pct").round(1).to_string())

# --------------------------------------------------------------------------- #
# C. iROAS -> Meridian ROI prior
# --------------------------------------------------------------------------- #
banner("C. MERIDIAN MMM CALIBRATION - iROAS to roi_m LogNormal prior")
import tensorflow_probability as tfp
from meridian.model import prior_distribution as pdist
from meridian.model import spec

tfd = tfp.distributions
prim = res[(res.method == "geox TBR") & (res.window == "test+14d")].iloc[0]
m = res_full.results["cell_1"].icpd
print(f"PRIMARY experiment read (pre-specified: test + full 14d cooldown, unbiased for the lag):\n"
      f"  iROAS {m.point_estimate:.3f}  90% CI [{m.lower_bound:.3f}, {m.upper_bound:.3f}]  sd={m.standard_deviation:.3f}")


def lognormal_prior(est: geox.Estimate, inflate: float = 1.0):
    """iROAS Estimate -> LogNormal(mu, sigma) for Meridian's roi_m.

    * CI strictly positive  -> match the 90% interval to the prior's 5th/95th percentiles
      (Meridian helper `lognormal_dist_from_range`; median = geometric mid-point of the CI).
    * CI reaches <= 0       -> log of a negative bound is undefined, so match mean and sd
      (`lognormal_dist_from_mean_std`) using the point estimate and geox's standard deviation.
    `inflate` widens sigma to hedge the gap between a 4-week snapshot and the whole MMM window.
    """
    if est.lower_bound > 0:
        d, how = pdist.lognormal_dist_from_range(est.lower_bound, est.upper_bound, mass_percent=1 - ALPHA), "range-matched"
    else:
        d, how = pdist.lognormal_dist_from_mean_std(max(est.point_estimate, 1e-3), est.standard_deviation), "moment-matched"
    return tfd.LogNormal(loc=d.loc, scale=d.scale * inflate), how


INFLATE = 1.25          # a 4-week snapshot vs. a multi-year MMM: seasonality, saturation, marginal-vs-average ROI
m7 = res_7d.results["cell_1"].icpd
prior_q, how_q = lognormal_prior(m)                    # primary read, as measured
prior_w, how_w = lognormal_prior(m, INFLATE)           # RECOMMENDED: primary read, widened
prior_7, how_7 = lognormal_prior(m7)                   # sensitivity: test + 7d cooldown read
cands = [("primary test+14d, as measured", prior_q, how_q), (f"primary test+14d x{INFLATE} sigma (RECOMMENDED)", prior_w, how_w),
         ("sensitivity: test+7d read", prior_7, how_7)]
q = [0.05, 0.50, 0.95]
tbl = pd.DataFrame({
    "prior": [c[0] for c in cands], "method": [c[2] for c in cands],
    "mu": [float(c[1].loc) for c in cands], "sigma": [float(c[1].scale) for c in cands],
    "mean": [float(c[1].mean()) for c in cands],
    "p05": [float(c[1].quantile(q[0])) for c in cands], "p50": [float(c[1].quantile(q[1])) for c in cands],
    "p95": [float(c[1].quantile(q[2])) for c in cands],
    "P(prior>truth)": [float(1 - c[1].cdf(truth['realised_iroas_lifetime'])) for c in cands],
})
print(tbl.round(4).to_string(index=False))

roi_prior = pdist.PriorDistribution(roi_m=prior_w)                       # broadcast to every paid channel
model_spec = spec.ModelSpec(prior=roi_prior)                              # media_prior_type defaults to 'roi'
print("\nPriorDistribution(roi_m=LogNormal(mu=%.4f, sigma=%.4f)) built; ModelSpec ok: %s"
      % (float(prior_w.loc), float(prior_w.scale), type(model_spec).__name__))
print("\n--- paste into your Meridian pipeline ---")
print(f"""import tensorflow_probability as tfp
from meridian.model import prior_distribution, spec
prior = prior_distribution.PriorDistribution(
    roi_m=tfp.distributions.LogNormal({float(prior_w.loc):.4f}, {float(prior_w.scale):.4f}))   # paid search + paid social
model_spec = spec.ModelSpec(prior=prior)   # media_prior_type='roi' (default)
# optional: restrict the prior to the test weeks via ModelSpec(roi_calibration=...) (CalibrationSpec)""")

# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
lab = [f"{r.method}\n{r.window}" for r in res.itertuples()]
y = np.arange(len(res))[::-1]
for yi, r in zip(y, res.itertuples()):
    ax[0].plot([r.iROAS_lo, r.iROAS_hi], [yi, yi], lw=3, color="#2a7ab0" if r.covers_truth else "#c0392b")
    ax[0].plot(r.iROAS, yi, "o", color="k")
    ax[0].plot(r.true_iROAS, yi, "|", color="green", ms=18, mew=3)
ax[0].set_yticks(y); ax[0].set_yticklabels(lab, fontsize=8); ax[0].set_xlabel(r"iROAS (\$ booking revenue per incremental \$ of spend)")
ax[0].set_title("Estimates with 90% CI   (green tick = ground truth for that window)")
xs = np.linspace(0.01, max(6.0, float(prior_w.quantile(0.995))), 400)
for d, nm, c in ((prior_q, "as measured", "#2a7ab0"), (prior_w, "recommended (x1.25 sigma)", "#e67e22"),
                 (prior_7, "test+7d read", "#7f8c8d")):
    ax[1].plot(xs, d.prob(xs), label=nm, color=c)
ax[1].axvline(truth["realised_iroas_lifetime"], color="green", ls="--", label="ground truth")
ax[1].set_xlabel("roi_m"); ax[1].set_title("Meridian roi_m prior"); ax[1].legend()
fig.tight_layout(); fig.savefig(OUT / "estimators_and_prior.png", dpi=120)

# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
g = res[res.method == "geox TBR"].set_index("window")
prior_ok = (abs(float(prior_q.quantile(0.05)) / m.lower_bound - 1) < 0.01 and abs(float(prior_q.quantile(0.95)) / m.upper_bound - 1) < 0.01
            if how_q == "range-matched" else
            abs(float(prior_q.mean()) / m.point_estimate - 1) < 0.01 and abs(float(prior_q.stddev()) / m.standard_deviation - 1) < 0.01)
checks = {
    "geox TBR detects the effect in the powered window (test, p<0.10)": g.loc["test", "p_value"] < ALPHA,
    "geox TBR 90% CI covers ground truth: test window": bool(g.loc["test", "covers_truth"]),
    "geox TBR 90% CI covers ground truth: test+14d": bool(g.loc["test+14d", "covers_truth"]),
    "geox TBR lag effect visible: test-only truth < lifetime truth": T_ROAS["test"] < T_ROAS["test+14d"],
    "geox TBR test+14d iROAS within 35% of lifetime truth": abs(prim.iROAS / truth["realised_iroas_lifetime"] - 1) < 0.35,
    "geox TBR CI ordered & finite": bool(np.isfinite(g[["iROAS_lo", "iROAS", "iROAS_hi"]]).all().all()
                                         and (g.iROAS_lo <= g.iROAS).all() and (g.iROAS <= g.iROAS_hi).all()),
    "custom SC/SDID: placebo nulls centred (|mean| < 0.5 sd)": bool(
        all(abs(np.mean(v)) < 0.5 * np.std(v) for v in plc.values())),
    "custom SC/SDID: every CI contains its own point estimate": bool(
        ((res.iROAS_lo <= res.iROAS) & (res.iROAS <= res.iROAS_hi)).all()),
    f"prior reproduces experiment estimate ({how_q})": bool(prior_ok),
    "PriorDistribution / ModelSpec constructed": isinstance(model_spec, spec.ModelSpec),
}
banner("VERIFICATION")
for k, v in checks.items():
    print(f"[{'PASS' if v else 'FAIL'}] {k}")
print(f"[INFO] 90% intervals covering the window-specific ground truth: {int(res.covers_truth.sum())}/{len(res)} "
      f"(a single simulated test cannot validate calibration - expect ~1 miss in 10 over many tests)")
print(f"[INFO] p-values (test window): " + ", ".join(f"{r.method}={r.p_value:.3f}" for r in res[res.window == 'test'].itertuples()))
print(f"[INFO] p-values (test+14d)   : " + ", ".join(f"{r.method}={r.p_value:.3f}" for r in res[res.window == 'test+14d'].itertuples()))
assert all(checks.values()), "stage-4 verification failed"
