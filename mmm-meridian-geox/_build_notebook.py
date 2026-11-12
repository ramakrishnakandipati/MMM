"""Builds + executes geox_incrementality_demo.ipynb from the four verified stage scripts.

Stage code is reused verbatim (split at its section headers) so the notebook and scripts cannot drift;
visualisation cells are added after each stage. nbformat/nbclient come from a throw-away dir (/tmp/nbtools)
appended LAST on sys.path, so the conda env itself is untouched.
"""
import re
import sys
from pathlib import Path

sys.path.append("/tmp/nbtools")
import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
RULE = re.compile(r"^# -{60,} #\s*$")

md = lambda s: nbf.v4.new_markdown_cell(s.strip())
code = lambda s: nbf.v4.new_code_cell(s.strip("\n"))


def sections(path: Path):
    """-> (preamble_code, [(title, code), ...]) splitting at '# ---- #\\n# title\\n# ---- #' headers."""
    lines = path.read_text().splitlines()
    heads = [i for i in range(len(lines) - 2)
             if RULE.match(lines[i]) and lines[i + 1].startswith("# ") and RULE.match(lines[i + 2])]
    pre_end = heads[0] if heads else len(lines)
    pre = "\n".join(lines[:pre_end])
    pre = re.sub(r'^""".*?"""\n', "", pre, count=1, flags=re.S)
    out = []
    for k, h in enumerate(heads):
        end = heads[k + 1] if k + 1 < len(heads) else len(lines)
        out.append((lines[h + 1][2:].strip(), "\n".join(lines[h + 3:end]).strip("\n")))
    return pre, out


def clean(src: str) -> str:
    src = src.replace('matplotlib.use("Agg")', "")
    src = src.replace('plt.close("all")', "plt.show()")
    return src


def stage_cells(fname, intro, extra):
    pre, secs = sections(ROOT / fname)
    cells = [md(intro), code(clean(pre))]
    for title, body in secs:
        cells += [md(f"#### {title}"), code(clean(body))]
        for key, cell_list in extra.items():
            if title.startswith(key):
                cells += cell_list
    return cells


# --------------------------------------------------------------------------- #
# visualisation cells (use variables left in the kernel by the stage code)
# --------------------------------------------------------------------------- #
VIZ_STYLE = code('''
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": .25, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.autolayout": True})
C = dict(treat="#d35400", ctrl="#2a7ab0", excl="#95a5a6", good="#27ae60", bad="#c0392b")
''')

viz1 = [md("##### Visual: what the panel looks like, and what the QC found"), code('''
fig, ax = plt.subplots(2, 2, figsize=(14, 8))
tot_d = dirty.groupby("date").conversions.sum() / 1e6
tot_c = clean.groupby("date").conversions.sum() / 1e6
ax[0, 0].plot(tot_d.index, tot_d.values, color=C["bad"], lw=1, label="raw (dirty)")
ax[0, 0].plot(tot_c.index, tot_c.values, color=C["ctrl"], lw=1.2, label="cleaned")
for d in faults["missing_dates"]: ax[0, 0].axvline(pd.Timestamp(d), color="k", ls=":", lw=1)
ax[0, 0].set_title("Total daily booking revenue ($M): outage dip and 2 missing dates"); ax[0, 0].legend()

for geo, c in (("Austin", C["bad"]), ("Nashville", C["treat"])):
    s_d = dirty[dirty.location == geo].set_index("date").conversions / 1e3
    s_c = clean[clean.location == geo].set_index("date").conversions / 1e3
    ax[0, 1].plot(s_d.index, s_d.values, color=c, lw=.8, alpha=.6)
    ax[0, 1].plot(s_c.index, s_c.values, color=c, lw=1.6, label=f"{geo} cleaned")
ax[0, 1].set_title("Local event spikes (thin = raw, thick = winsorised)"); ax[0, 1].legend()

wk_ = clean.groupby(clean.date.dt.dayofweek).conversions.mean(); wk_ = wk_ / wk_.mean() * 100
ax[1, 0].bar(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], wk_.values, color=C["ctrl"])
ax[1, 0].axhline(100, color="k", lw=1); ax[1, 0].set_ylim(70, 120)
ax[1, 0].set_title("Booking-date day-of-week index (100 = mean)")

reg = clean.merge(md[["location", "region"]]).groupby(["date", "region"]).conversions.sum().unstack() / 1e3
reg.rolling(7).mean().plot(ax=ax[1, 1], lw=1.4); ax[1, 1].set_title("Regional seasonality, 7-day mean ($k/day)")
plt.show()

cm = clean.pivot(index="location", columns="date", values="conversions")
cm = cm.div(cm.mean(1), axis=0).loc[md.sort_values("avg_daily_rev_k", ascending=False).location]
fig, a = plt.subplots(figsize=(14, 7))
im = a.imshow(cm.values, aspect="auto", cmap="viridis", vmin=.5, vmax=1.6); a.grid(False)
a.set_yticks(range(len(cm))[::3]); a.set_yticklabels(cm.index[::3], fontsize=7)
a.set_title("Revenue indexed to each DMA's own mean (rows sorted by size, columns = 180 days)")
fig.colorbar(im, ax=a, shrink=.7); plt.show()
''')]

viz2 = [md("##### Visual: duration trade-off, power, and the Control vs Treatment split"), code('''
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
ax[0].plot(sweep.test_days, sweep.mde_pct, "o-", color=C["ctrl"]); ax[0].set_xlabel("test days"); ax[0].set_ylabel("MDE, % of treatment revenue")
ax2 = ax[0].twinx(); ax2.plot(sweep.test_days, sweep.min_detectable_iROAS, "s--", color=C["treat"]); ax2.set_ylabel("min detectable iROAS", color=C["treat"]); ax2.grid(False)
ax[0].set_title("Duration sweep (best design per length)")
for lab, g_ in pw.groupby("read"): ax[1].plot(g_.true_iROAS, g_.power, "o-", label=lab)
ax[1].axhline(.8, color="k", ls=":"); ax[1].axvline(TRUE_IROAS, color=C["good"], ls="--", label="planning iROAS 2.0")
ax[1].set_xlabel("true iROAS"); ax[1].set_ylabel("power"); ax[1].set_title("Power curve, chosen 28-day design"); ax[1].legend(fontsize=8)
cap.set_index("test_days")[["cooldown_0d","cooldown_7d","cooldown_14d","cooldown_21d"]].plot(kind="bar", ax=ax[2], rot=0)
ax[2].set_ylim(60, 101); ax[2].set_ylabel("% of lifetime effect observed"); ax[2].set_title("Booking-lag capture by cooldown")
plt.show()

colors = {"TREATMENT": C["treat"], "CONTROL": C["ctrl"], "EXCLUDED": C["excl"]}
fig, ax = plt.subplots(1, 2, figsize=(15, 5.2))
for gname, g_ in asg.groupby("group"):
    ax[0].scatter(g_.avg_daily_rev / 1e3, g_.avg_daily_spend / 1e3, s=60, c=colors[gname], label=f"{gname} ({len(g_)})", alpha=.85, edgecolor="w")
for r_ in asg[asg.group == "EXCLUDED"].itertuples(): ax[0].annotate(r_.location, (r_.avg_daily_rev / 1e3, r_.avg_daily_spend / 1e3), fontsize=6.5)
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("avg daily revenue ($k)"); ax[0].set_ylabel("avg daily spend ($k)")
ax[0].set_title("Assignment by DMA size (log-log)"); ax[0].legend()
ct = pd.crosstab(asg[asg.group != "EXCLUDED"].stratum, asg[asg.group != "EXCLUDED"].group)
ct.plot(kind="bar", ax=ax[1], color=[colors[c] for c in ct.columns], rot=0)
ax[1].set_title("Stratified sampling: DMAs per stratum"); ax[1].set_xlabel("stratum")
plt.show()
asg[["location","group","stratum","region","reason"]].reset_index(drop=True)
''')]

viz3 = [md("##### Visual: the injected ground truth (what a perfect analyst would recover)"), code('''
tr_rev = analysis_data[analysis_data.location.isin(treat)].groupby("date").conversions.sum()
co_rev = analysis_data[analysis_data.location.isin(list(design.control_geos))].groupby("date").conversions.sum()
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
idx = (tr_rev / tr_rev.loc[:PRETEST_END].mean())
ax[0].plot(idx.index, idx.values, color=C["treat"], label="treatment (indexed)")
idx2 = (co_rev / co_rev.loc[:PRETEST_END].mean()); ax[0].plot(idx2.index, idx2.values, color=C["ctrl"], label="control (indexed)")
ax[0].axvspan(TEST_START, TEST_END, color="orange", alpha=.15, label="test"); ax[0].axvspan(COOLDOWN_START, COOLDOWN_END, color="grey", alpha=.15, label="cooldown")
ax[0].set_xlim(PRETEST_END - pd.Timedelta(days=60), COOLDOWN_END); ax[0].legend(fontsize=8); ax[0].set_title("Treatment vs control revenue (eyeballing is not inference)")
dly = incr_rev[treat].sum(axis=1) / 1e3; sp = extra_spend[treat].sum(axis=1) / 1e3
ax[1].bar(sp.index, sp.values, color=C["treat"], alpha=.5, label="incremental spend ($k)")
ax[1].plot(dly.index, dly.values, color=C["good"], lw=2, label="true incremental revenue ($k)")
ax[1].axvline(TEST_END, color="k", ls="--"); ax[1].set_xlim(TEST_START - pd.Timedelta(days=3), COOLDOWN_END); ax[1].legend(fontsize=8)
ax[1].set_title("Spend stops on day 28, bookings keep landing")
cs = dly.cumsum() / dly.sum() * 100
ax[2].plot(cs.index, cs.values, color=C["good"], lw=2); ax[2].axvline(TEST_END, color="k", ls="--")
ax[2].axhline(100 - 100 * truth["share_of_effect_after_test_end"], color=C["bad"], ls=":")
ax[2].set_xlim(TEST_START, COOLDOWN_END); ax[2].set_ylabel("% of lifetime effect"); ax[2].set_title("Cumulative effect: ~82% by test end")
plt.show()
''')]

viz4a = [md("##### Visual: geox TBR output (counterfactual, cumulative lift)"), code('''
geox.plot_analysis(res_full)
plt.show()
''')]

viz4b = [md("##### Visual: placebo null distributions vs. the observed effect"), code('''
fig, ax = plt.subplots(2, 3, figsize=(16, 7))
for a, ((meth, w), lift) in zip(ax.ravel(), real_rev.items()):
    e = np.array(plc[(meth, w)]) / 1e3
    a.hist(e, bins=25, color=C["excl"], alpha=.85)
    a.axvline(lift / 1e3, color=C["treat"], lw=2.5, label="observed")
    a.axvline(0, color="k", lw=1)
    a.set_title(f"{meth} | {w}", fontsize=10); a.set_xlabel("placebo effect, $k over window")
ax[0, 0].legend()
fig.suptitle("Placebo (no-treatment) effects from design-aware pseudo-splits; observed effect should sit in the tail", y=1.02)
plt.show()
''')]

viz4c = [md("##### Visual: estimators with 90% CI vs. window-specific truth, and the Meridian prior"), code('''
fig, ax = plt.subplots(1, 2, figsize=(15, 5.2))
y_ = np.arange(len(res))[::-1]
for yi, r in zip(y_, res.itertuples()):
    ax[0].plot([r.iROAS_lo, r.iROAS_hi], [yi, yi], lw=4, color=C["ctrl"] if r.covers_truth else C["bad"], solid_capstyle="round")
    ax[0].plot(r.iROAS, yi, "o", color="k"); ax[0].plot(r.true_iROAS, yi, "|", color=C["good"], ms=20, mew=3)
ax[0].axvline(0, color="k", lw=.8)
ax[0].set_yticks(y_); ax[0].set_yticklabels([f"{r.method} | {r.window}" for r in res.itertuples()], fontsize=8)
ax[0].set_xlabel(r"iROAS (\\$ booking revenue per incremental \\$ spend)"); ax[0].set_title("Estimate (dot), 90% CI (bar), ground truth (green tick)")
xs = np.linspace(0.01, max(6.0, float(prior_w.quantile(0.995))), 500)
for d, nm, c in ((prior_q, "as measured", C["ctrl"]), (prior_w, "recommended (x1.25 sigma)", C["treat"]), (prior_7, "test+7d read", C["excl"])):
    ax[1].plot(xs, d.prob(xs), label=nm, color=c, lw=2)
ax[1].axvline(truth["realised_iroas_lifetime"], color=C["good"], ls="--", label="ground truth")
ax[1].set_xlabel("roi_m"); ax[1].set_title("Meridian roi_m LogNormal prior"); ax[1].legend()
plt.show()
res[["method","window","iROAS","iROAS_lo","iROAS_hi","true_iROAS","p_value","covers_truth"]].round(3)
''')]

# --------------------------------------------------------------------------- #
nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "mmm-meridian", "display_name": "mmm-meridian", "language": "python"}
cells = [md("""
# Geo-incrementality test for a hospitality brand with `meridian-geox` → Meridian ROI prior

Synthetic, end-to-end and reproducible: **60 US DMAs**, daily booking revenue and paid search + social spend, a **+75 % heavy-up** on a stratified
treatment group, a 28-day test and 14-day cooldown with a **known true iROAS of 2.0**.

| Stage | What happens |
|---|---|
| 1 | Simulate 180-day panel, inject faults, run `check_design_data_quality`, clean |
| 2 | `DesignConfig` (HEAVY_UP, stratified, TBR), operational exclusions, MDE / power / duration / cooldown, assignment table |
| 3 | Simulate test + cooldown with booking lag and a known lift (the *Google / Meta execution checklist* is in `README.md`) |
| 4 | `analyze()` (TBR) + custom Synthetic Control / SDID with design-aware placebo inference; iROAS → Meridian `roi_m` prior |

**Library note:** `meridian-geox` 1.0.1 implements only TBR. SC and SDID below are custom reference implementations, labelled `(custom)`.
Every stage ends with a `VERIFICATION` block of PASS/FAIL asserts. Run with the `mmm-meridian` kernel; nothing extra to install.
"""), VIZ_STYLE]

cells += stage_cells("01_simulate_and_qc.py", "## Stage 1 — Synthetic data and data-quality diagnostics",
                     {"5. meridian_geox": viz1})
cells += stage_cells("02_design_power.py", "## Stage 2 — Design: stratified sampling, exclusions, MDE / power / duration",
                     {"5. Control vs Treatment": viz2})
cells += stage_cells("03_simulate_posttest.py", "## Stage 3 — Simulated test + cooldown with a known ground truth",
                     {"Ground truth": viz3})
cells += stage_cells("04_analyze_calibrate.py", "## Stage 4 — Analysis, estimator comparison, Meridian prior",
                     {"A. meridian_geox.analyze": viz4a, "B. Reference SC": viz4b, "C. iROAS": viz4c})
cells.append(md("""
## Reading the results
* The 28-day test window alone under-states iROAS (≈ 18 % of bookings land after the test ends); reading test + 14 d recovers it but widens the TBR interval.
* A single simulated test with one seed cannot validate interval calibration; repeat over seeds before trusting coverage claims.
* A heavy-up measures **marginal** iROAS of the extra spend, usually below the **average** ROI that Meridian's `roi_m` represents. The prior is therefore widened (σ × 1.25).
* Platform execution checklist (Google Ads / Meta Ads) and the hospitality hyperparameter rationale: see `README.md`.
"""))
nb.cells = cells

NB = ROOT / "geox_incrementality_demo.ipynb"
nbf.write(nb, NB)
print(f"built {len(cells)} cells -> executing ...", flush=True)
client = NotebookClient(nb, timeout=1500, kernel_name="mmm-meridian", resources={"metadata": {"path": str(ROOT)}},
                        allow_errors=False)
try:
    client.execute()
finally:
    nbf.write(nb, NB)
n_img = sum(1 for c in nb.cells if c.cell_type == "code" for o in c.outputs if "image/png" in o.get("data", {}))
print(f"executed OK; figures embedded: {n_img}")
