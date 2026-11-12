# Hospitality geo-incrementality test with `meridian-geox` (heavy-up) → Meridian MMM prior

Synthetic, end-to-end, reproducible. 60 US DMAs, daily `booking_revenue` (mapped to geox's `conversions`)
and combined Paid Search + Paid Social `spend`.

## Run (WSL, existing `mmm-meridian` env — nothing to install)

```bash
cd /mnt/d/Ramakrishna_D/Python/GitProjects/MMM/mmm-meridian-geox
P=/home/rama/miniforge3/envs/mmm-meridian/bin/python
$P 01_simulate_and_qc.py && $P 02_design_power.py && $P 03_simulate_posttest.py && $P 04_analyze_calibrate.py
```

Every script ends with a `VERIFICATION` block of `PASS/FAIL` checks and exits non-zero on failure.
Logs from the last full run are in `outputs/logs/`. Versions: `meridian-geox 1.0.1`, `google-meridian 2.2.0`.

| Stage | Script | What it does |
|---|---|---|
| 1 | `01_simulate_and_qc.py` | 60 DMA × 180-day panel, injects faults, runs `check_design_data_quality`, cleans |
| 2 | `02_design_power.py` | `DesignConfig` (HEAVY_UP, stratified, TBR), exclusions, MDE/power sweep, assignment table |
| 3 | (below) | Platform checklist — non-code |
| 3b | `03_simulate_posttest.py` | Test + cooldown data with known ground-truth iROAS and booking lag |
| 4 | `04_analyze_calibrate.py` | `geox.analyze()`, SC/SDID comparison, Meridian `roi_m` prior |

## Where `meridian-geox` 1.0.1 differs from the brief (important)

* **Only TBR is implemented.** `Methodology.SDID` exists but `run_design`/`analyze` raise for anything but TBR, and
  there is no Synthetic Control. Stage 4 therefore runs **geox TBR** and adds **custom reference Synthetic Control and
  SDID** (≈60 lines, clearly labelled `(custom)`) with the same design-aware placebo idea.
* **Schema:** `date, location, conversions, spend`. Revenue → `conversions`; Search+Social are summed into `spend`
  (the library has one spend column per *cell*, not per channel).
* **Quality checks cover:** duplicates (flag only; the library's pivot *sums* them → de-duplicate yourself), spend>0 with
  zero response, aggregate pre-test date outliers, >30 % missing days, >50 % zero response. **Not covered:** single-DMA
  event spikes → stage 1 adds a robust per-DMA detector.
* iROAS = geox `icpd` (incremental conversions per incremental dollar; with revenue as the KPI it *is* iROAS).

## Results of the verified run

| | |
|---|---|
| Ground truth | iROAS parameter 2.0; realised lifetime **1.99**; only **1.63** lands inside the 28-day window (18 % of bookings arrive after the test ends) |
| Design | 13 treatment / 37 control / 10 excluded DMAs; MDE 2.48 % of treatment revenue; min. detectable iROAS ≈ 1.08; A/A p = 0.65 |
| geox TBR, test window | iROAS **1.94** [0.58, 3.16], p = 0.002 |
| geox TBR, test + 14 d | iROAS **2.13** [−0.27, 3.97], p = 0.23 (wider: cumulative placebo error grows with horizon) |
| SC / SDID (custom), test + 14 d | 2.32 [0.77, 3.39] / 2.10 [1.07, 3.70] |
| Meridian prior | `LogNormal(0.5873, 0.7261)` (median 1.80, 90 % range 0.55–5.94) |

Honest reading: a single simulated experiment; all 7 intervals cover the window-specific truth, which proves
nothing about calibration on its own. The real takeaways are the **lag bias** (test-window-only read is a lower bound
by construction), and the **precision cost of extending the read window** with TBR.

## Hyperparameters through a hospitality lens

| Choice | Value | Why |
|---|---|---|
| Pre-test length | 180 d (~25 weeks) | Needs ≥ 3× test length (library rule) and many full weekly cycles; hospitality seasonality is slow (regional peaks), so a long window stabilises TBR while still being local enough to avoid last-year drift |
| Test length | 28 d, Sat→Fri | **Whole weeks only**: booking-date demand is weekday-skewed (Mon +12 %, Sat −16 % in this data). Must avoid holidays/school breaks, which shift booking *and* lead-time behaviour |
| Cooldown | 14 d | Search→book lag mean ≈ 5 d, 100 % within 14 d. Capture table: 28-d test alone sees 82 % of the effect; +7 d → 98 %; +14 d → 100 % |
| Read window | Pre-specify it | Longer window = unbiased but wider (TBR); powered MDE applies to the *test* window. We report 0/7/14-d cooldown reads and use the full 14 d as primary for the prior |
| Heavy-up | +75 % | +50 % left min-detectable iROAS (1.6) uncomfortably close to the assumed 2.0 |
| Exclusions | flagship, occupancy ≥ 95 %, renovation | At ≥ 95 % occupancy extra demand can't be fulfilled (lift capped); renovation makes inventory non-stationary |
| `alpha`, `power` | 0.10 two-sided, 0.80 | 90 % intervals, as requested |
| Stay- vs booking-date | KPI = booking-date revenue | If you can only measure **stay-date** revenue, lags are 3–8 weeks and the cooldown must be much longer |

## 3. In-platform execution checklist (Google Ads / Meta Ads)

**Before launch (T-14 → T-1)**
- [ ] Freeze the geo list from `outputs/assignment_table.csv`; no re-assignment mid-test.
- [ ] Verify booking-engine conversion tracking is live in *all* DMAs (a zero-booking DMA is auto-excluded by geox).
- [ ] Lock BAU settings in control DMAs: budgets, bid strategies, target ROAS/CPA, audiences, creative rotation.
- [ ] Confirm no regional promotions, rate-parity changes, loyalty/price events, OTA/metasearch campaigns, TV/OOH or PR pushes planned for any DMA during test **and** cooldown.
- [ ] Check property calendar: no renovations/closures/overbooking in treatment DMAs; occupancy headroom ≥ 5 pts for the +75 % uplift.
- [ ] Record each platform's actual BAU daily spend per DMA for the last 4 weeks (needed to verify delivery).

**Google Ads (Search)**
- [ ] Do **not** edit existing campaigns' geo targeting in-flight (resets learning). Duplicate campaigns scoped to treatment DMAs only, or use a campaign-level geo target + *budget multiplier* (×1.75) for treatment DMAs.
- [ ] Target with **"Presence"** (people *in* the location), not "Presence or interest", so treatment ads don't leak to other DMAs; exclude the treatment DMAs from the national/legacy campaign to avoid double serving.
- [ ] Use shared budgets or explicit daily budgets per DMA group; remove tCPA/tROAS caps that would throttle the added spend, or raise them proportionally.
- [ ] Brand-keyword campaigns: decide explicitly whether they are in or out of the test (they inflate organic cannibalisation).

**Meta Ads (Paid Social)**
- [ ] Duplicate ad sets per cell with location = treatment DMAs (DMA-level custom locations) and exclude them from the BAU ad set; same creative, same optimisation event.
- [ ] Budgets ×1.75 on treatment ad sets; avoid Advantage+ budget reallocation across cells; keep learning-phase edits to zero.
- [ ] Use "People living in this location"; check audience overlap between cells.

**During the test (daily)**
- [ ] Pacing: treatment spend ≈ BAU × 1.75 (±10 %); we delivered 93 % in simulation — platforms under-deliver. Escalate if < 85 % for 3 days.
- [ ] Control spend within ±10 % of BAU. No campaign edits, no new promotions, no site outages (log any).
- [ ] Keep a change log (date, DMA, what changed) — outlier dates can be excluded via `excluded_dates`.

**After**
- [ ] Keep BAU spend in **all** DMAs through the 14-day cooldown (bookings still arriving); do not launch new promos.
- [ ] Pull final spend and bookings by DMA/day, then run stage 4.

## 4. Feeding the result to Meridian

The prior is built from the geox iROAS `Estimate` with Meridian's own helpers:
* CI strictly positive → `lognormal_dist_from_range(lo, hi, mass_percent=0.90)` (5th/95th percentiles = CI).
* CI reaches ≤ 0 (as in the primary read) → `lognormal_dist_from_mean_std(point, sd)`.
* Then widen σ ×1.25 (a 4-week snapshot ≠ the full MMM horizon) → `PriorDistribution(roi_m=...)` → `ModelSpec(prior=...)`.

Caveats when using it: (1) a heavy-up measures **marginal** iROAS of the *extra* spend, typically below the *average* ROI
Meridian's `roi_m` represents; (2) the test can't separate Search from Social, so one prior is shared (or aggregate them
into a single channel); (3) restrict it to the test weeks with `ModelSpec(roi_calibration=...)` if the MMM spans periods
with very different seasonality.

## Known limitations of this synthetic study
Linear response (no saturation) in the DGP; no cross-DMA spillover (in production keep buffer DMAs or check
border-market leakage); the +14-day lag kernel is assumed known in simulation (estimate it from historical
click-to-book data in practice); one random seed — run multiple seeds to assess interval calibration.
