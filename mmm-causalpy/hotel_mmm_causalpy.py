"""Hospitality Marketing Mix Model on CausalPy (custom PyMCModel subclass).

CausalPy 0.9 ships no MMM class and no adstock/saturation, so this subclasses
`causalpy.pymc_models.PyMCModel` and builds the full Bayesian MMM inside
`build_model`, reusing CausalPy's fit / sample_prior / posterior-predictive API.

Run in WSL2, env `causalpy`:
    python hotel_mmm_causalpy.py --quick     # smoke test
    python hotel_mmm_causalpy.py             # full run

Model (all on standardised scales; y is divided by its mean):
    S_c(t)  = logistic_sat( geometric_adstock( spend_c / max_c ) )
    y_t     ~ N( b0 + b_trend*t + Fourier_K(t)@g
                 + sum_c beta_c S_c(t)                 # direct media effects
                 + gamma_adr*ADR_z + gamma_ev*event + gamma_macro*macro_z
                 + delta*organic_z , sigma )           # organic = halo mediator
    organic_z ~ N( m0 + h * S_social(t) + Fourier_K(t)@gm , sigma_m )   # halo path
Priors: beta, delta, h, gamma_ev > 0 (HalfNormal); gamma_adr < 0 (-HalfNormal).
Social total effect = direct (beta_social) + halo (delta * h).
Organic traffic is a mediator, so it is NOT treated as a plain control: the
direct+indirect decomposition keeps the social estimate unbiased.
"""
from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from causalpy.pymc_models import PyMCModel
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
OUT = Path("outputs")
CHANNELS = ["ota_ads", "metasearch", "paid_social", "brand_search"]
SPEND = [f"spend_{c}" for c in CHANNELS]
SOCIAL = CHANNELS.index("paid_social")
L_MAX = 8            # weeks of carryover considered
K_FOURIER = 3        # yearly harmonics
REV_PER_BOOKING_NOTE = "revenue ROAS = bookings ROAS x mean ADR (room_bookings treated as room-nights)"

# ----------------------------------------------------------------------------
# 1. Transforms (NumPy, used for data synthesis, curves and optimisation)
# ----------------------------------------------------------------------------
def geometric_adstock(x: np.ndarray, alpha: np.ndarray, l_max: int = L_MAX) -> np.ndarray:
    """Normalised geometric adstock along axis -2.

    x: (..., T, C); alpha: broadcastable to (..., 1, C). Weights alpha**l, l<l_max,
    normalised to sum to 1 so total effect is not inflated by carryover.
    """
    alpha = np.asarray(alpha)[..., None, :]
    w = alpha ** np.arange(l_max).reshape((l_max,) + (1,) * alpha.ndim)  # (L, ..., 1, C)
    w = w / w.sum(0)
    out = np.zeros(np.broadcast_shapes(x.shape, w.shape[1:]))
    T = x.shape[-2]
    for l in range(l_max):
        shifted = np.zeros_like(x)
        shifted[..., l:, :] = x[..., : T - l, :]
        out = out + w[l] * shifted
    return out


def logistic_saturation(x: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """(1 - exp(-lam x)) / (1 + exp(-lam x)); 0 at x=0, saturates at 1."""
    lam = np.asarray(lam)[..., None, :]
    return (1 - np.exp(-lam * x)) / (1 + np.exp(-lam * x))


# ----------------------------------------------------------------------------
# 2. Synthetic data (156 weeks) with known ground truth
# ----------------------------------------------------------------------------
TRUTH = dict(
    alpha=np.array([0.25, 0.30, 0.65, 0.20]),
    lam=np.array([2.5, 3.0, 2.0, 4.0]),
    beta=np.array([0.20, 0.16, 0.12, 0.14]),   # in units of mean bookings
    gamma_adr=-0.10, gamma_ev=0.10, gamma_macro=0.04, delta=0.06, h=0.9,
)


def fourier(t: np.ndarray, k: int = K_FOURIER) -> np.ndarray:
    cols = []
    for j in range(1, k + 1):
        cols += [np.sin(2 * np.pi * j * t / 52.18), np.cos(2 * np.pi * j * t / 52.18)]
    return np.column_stack(cols)


def make_data(n: int = 156, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    date = pd.date_range("2022-01-03", periods=n, freq="W-MON")
    F = fourier(t)
    season = 0.6 * F[:, 0] - 0.3 * F[:, 1] + 0.15 * F[:, 2]       # summer peak, winter dip
    df = pd.DataFrame({"date": date})
    base = {"ota_ads": 9e3, "metasearch": 7e3, "paid_social": 6e3, "brand_search": 4e3}
    for c in CHANNELS:
        s = base[c] * (1 + 0.5 * season) * rng.gamma(6, 1 / 6, n)   # spend chases the season
        s[rng.random(n) < 0.04] = 0.0                               # occasional pauses
        df[f"spend_{c}"] = s
    df["regional_event_flag"] = (rng.random(n) < 0.09).astype(float)
    df["avg_daily_rate_adr"] = 165 + 28 * season + 6 * df["regional_event_flag"] + rng.normal(0, 4, n)
    df["macro_economic_index"] = 100 + np.cumsum(rng.normal(0, 0.7, n))

    x = df[SPEND].to_numpy() / df[SPEND].max().to_numpy()
    S = logistic_saturation(geometric_adstock(x, TRUTH["alpha"]), TRUTH["lam"])
    adr_z = (df["avg_daily_rate_adr"] - df["avg_daily_rate_adr"].mean()) / df["avg_daily_rate_adr"].std()
    mac_z = (df["macro_economic_index"] - df["macro_economic_index"].mean()) / df["macro_economic_index"].std()
    org_z = 0.5 * season + TRUTH["h"] * S[:, SOCIAL] + rng.normal(0, 0.35, n)   # halo: social -> organic
    df["organic_search_traffic"] = 6000 + 1500 * org_z

    org_std = (df["organic_search_traffic"] - df["organic_search_traffic"].mean()) / df["organic_search_traffic"].std()
    y_rel = (1.0 + 0.35 * season + 0.0004 * (t - n / 2) + S @ TRUTH["beta"]
             + TRUTH["gamma_adr"] * adr_z + TRUTH["gamma_ev"] * df["regional_event_flag"]
             + TRUTH["gamma_macro"] * mac_z + TRUTH["delta"] * org_std + rng.normal(0, 0.03, n))
    df["room_bookings"] = (1400 * y_rel).round()
    return df


# ----------------------------------------------------------------------------
# 3. Design matrix
# ----------------------------------------------------------------------------
class Scales:
    """Everything needed to map between raw and model units."""

    def __init__(self, df: pd.DataFrame):
        self.spend_max = df[SPEND].max().to_numpy()
        self.y_mean = float(df["room_bookings"].mean())
        self.z = {c: (df[c].mean(), df[c].std())
                  for c in ["avg_daily_rate_adr", "macro_economic_index", "organic_search_traffic"]}

    def zscore(self, df: pd.DataFrame, c: str) -> np.ndarray:
        m, s = self.z[c]
        return ((df[c] - m) / s).to_numpy()


def build_design(df: pd.DataFrame, sc: Scales) -> tuple[xr.DataArray, xr.DataArray, dict[str, Any]]:
    n = len(df)
    t = np.arange(n)
    F = fourier(t)
    cols: dict[str, np.ndarray] = {}
    for i, c in enumerate(CHANNELS):
        cols[f"x_{c}"] = df[SPEND[i]].to_numpy() / sc.spend_max[i]
    cols["adr_z"] = sc.zscore(df, "avg_daily_rate_adr")
    cols["event"] = df["regional_event_flag"].to_numpy()
    cols["macro_z"] = sc.zscore(df, "macro_economic_index")
    cols["organic_z"] = sc.zscore(df, "organic_search_traffic")
    cols["trend"] = (t - t.mean()) / t.std()
    for j in range(F.shape[1]):
        cols[f"fourier_{j}"] = F[:, j]
    names = list(cols)
    X = xr.DataArray(np.column_stack(list(cols.values())), dims=["obs_ind", "coeffs"],
                     coords={"obs_ind": df["date"].to_numpy(), "coeffs": names})
    y = xr.DataArray((df["room_bookings"].to_numpy() / sc.y_mean)[:, None],
                     dims=["obs_ind", "treated_units"],
                     coords={"obs_ind": df["date"].to_numpy(), "treated_units": ["room_bookings"]})
    coords = {"coeffs": names, "obs_ind": df["date"].to_numpy(), "treated_units": ["room_bookings"]}
    return X, y, coords


# ----------------------------------------------------------------------------
# 4. CausalPy model
# ----------------------------------------------------------------------------
class HospitalityMMM(PyMCModel):
    """Bayesian MMM with learned adstock/saturation and a social->organic halo."""

    default_priors: dict[str, Any] = {}   # priors are defined inline below

    # media-specific prior beliefs: short booking windows for search/OTA, long for social
    ALPHA_AB = np.array([[2, 5], [2, 4], [4, 2], [2, 5]], dtype=float)

    def build_model(self, X: xr.DataArray, y: xr.DataArray, coords: dict[str, Any] | None) -> None:
        names = list(coords["coeffs"])
        idx = {n: i for i, n in enumerate(names)}
        media_i = [idx[f"x_{c}"] for c in CHANNELS]
        four_i = [i for n, i in idx.items() if n.startswith("fourier_")]
        with self:
            self.add_coords({**coords, "channel": CHANNELS, "fourier": [names[i] for i in four_i]})
            Xd = pm.Data("X", X.values, dims=["obs_ind", "coeffs"])
            yd = pm.Data("y", y.values, dims=["obs_ind", "treated_units"])
            n_obs = int(X.shape[0])  # static: adstock lag slices need a fixed length

            # --- media: adstock (decay alpha) then logistic saturation (lam)
            alpha = pm.Beta("alpha", alpha=self.ALPHA_AB[:, 0], beta=self.ALPHA_AB[:, 1], dims="channel")
            lam = pm.Gamma("lam", alpha=3.0, beta=1.0, dims="channel")
            beta = pm.HalfNormal("beta", sigma=0.3, dims="channel")
            xm = Xd[:, media_i]
            w = alpha[None, :] ** pt.arange(L_MAX)[:, None].astype("float64")      # (L, C)
            w = w / w.sum(0, keepdims=True)
            # lag operators as constant matrices (avoids symbolic-length slicing)
            shift = np.stack([np.eye(n_obs, k=-l) for l in range(L_MAX)])      # (L, T, T)
            ad = sum(w[l][None, :] * pt.dot(pt.constant(shift[l]), xm) for l in range(L_MAX))
            S = (1 - pt.exp(-lam[None, :] * ad)) / (1 + pt.exp(-lam[None, :] * ad))
            S = pm.Deterministic("S", S, dims=["obs_ind", "channel"])

            # --- seasonality, trend, controls
            g_four = pm.Normal("gamma_fourier", 0, 0.5, dims="fourier")
            g_trend = pm.Normal("gamma_trend", 0, 0.2)
            g_adr = -pm.HalfNormal("neg_gamma_adr", sigma=0.15)       # strictly negative
            pm.Deterministic("gamma_adr", g_adr)
            g_ev = pm.HalfNormal("gamma_event", sigma=0.15)           # events lift demand
            g_mac = pm.Normal("gamma_macro", 0, 0.1)
            delta = pm.HalfNormal("delta", sigma=0.2)                 # organic -> bookings
            b0 = pm.Normal("intercept", 1.0, 0.3)

            # --- halo: social drives organic search traffic
            h = pm.HalfNormal("halo", sigma=1.0)
            m0 = pm.Normal("organic_intercept", 0, 1)
            gm = pm.Normal("organic_fourier", 0, 0.7, dims="fourier")
            sig_m = pm.HalfNormal("sigma_organic", 0.7)
            org = Xd[:, idx["organic_z"]]
            mu_org = m0 + h * S[:, SOCIAL] + pt.dot(Xd[:, four_i], gm)
            pm.Normal("organic_obs", mu_org, sig_m, observed=org)

            mu = (b0 + g_trend * Xd[:, idx["trend"]] + pt.dot(Xd[:, four_i], g_four)
                  + pt.dot(S, beta) + g_adr * Xd[:, idx["adr_z"]] + g_ev * Xd[:, idx["event"]]
                  + g_mac * Xd[:, idx["macro_z"]] + delta * org)
            mu = pm.Deterministic("mu", mu[:, None], dims=["obs_ind", "treated_units"])
            sigma = pm.HalfNormal("sigma", 0.1)
            pm.Normal("y_hat", mu, sigma, observed=yd, dims=["obs_ind", "treated_units"])


# ----------------------------------------------------------------------------
# 5. Posterior-based analytics
# ----------------------------------------------------------------------------
def posterior_arrays(idata, n_draws: int | None = None, seed: int = 0) -> dict[str, np.ndarray]:
    p = idata.posterior.stack(s=("chain", "draw"))
    keys = ["alpha", "lam", "beta", "delta", "halo"]
    out = {k: p[k].transpose("s", ...).values for k in keys}
    if n_draws and n_draws < out["alpha"].shape[0]:
        sel = np.random.default_rng(seed).choice(out["alpha"].shape[0], n_draws, replace=False)
        out = {k: v[sel] for k, v in out.items()}
    return out


def channel_response(post: dict[str, np.ndarray], x_scaled: np.ndarray) -> np.ndarray:
    """Saturated adstocked response S, shape (draws, T, C), from scaled spend (T, C)."""
    ad = geometric_adstock(np.broadcast_to(x_scaled, (post["alpha"].shape[0],) + x_scaled.shape),
                           post["alpha"])
    return logistic_saturation(ad, post["lam"])


def incremental_bookings(post: dict[str, np.ndarray], S: np.ndarray, y_mean: float) -> np.ndarray:
    """(draws, T, C) incremental bookings. Social also carries the halo path."""
    inc = S * post["beta"][:, None, :]
    inc[..., SOCIAL] += (post["delta"] * post["halo"])[:, None] * S[..., SOCIAL]
    return inc * y_mean


def roas_table(idata, df: pd.DataFrame, sc: Scales, n_draws: int = 500) -> pd.DataFrame:
    post = posterior_arrays(idata, n_draws)
    x = df[SPEND].to_numpy() / sc.spend_max
    inc = incremental_bookings(post, channel_response(post, x), sc.y_mean).sum(1)      # (draws, C)
    spend = df[SPEND].sum().to_numpy()
    adr = df["avg_daily_rate_adr"].mean()
    r = inc / spend
    rows = {"spend": spend, "bookings_per_1k": r.mean(0) * 1e3,
            "bookings_p3": np.percentile(r, 3, 0) * 1e3, "bookings_p97": np.percentile(r, 97, 0) * 1e3,
            "roas_revenue": r.mean(0) * adr, "roas_p3": np.percentile(r, 3, 0) * adr,
            "roas_p97": np.percentile(r, 97, 0) * adr}
    return pd.DataFrame(rows, index=CHANNELS)


def truth_roas(df: pd.DataFrame, sc: Scales) -> np.ndarray:
    x = df[SPEND].to_numpy() / sc.spend_max
    S = logistic_saturation(geometric_adstock(x, TRUTH["alpha"]), TRUTH["lam"])
    inc = S * TRUTH["beta"]
    h_std = TRUTH["h"] * 1500 / df["organic_search_traffic"].std()   # truth halo in standardised organic units
    inc[:, SOCIAL] += TRUTH["delta"] * h_std * S[:, SOCIAL]
    return inc.sum(0) * sc.y_mean / df[SPEND].sum().to_numpy() * df["avg_daily_rate_adr"].mean()


# ----------------------------------------------------------------------------
# 6. Budget optimisation for next quarter
# ----------------------------------------------------------------------------
def optimise_budget(idata, df: pd.DataFrame, sc: Scales, horizon: int = 13, total_budget: float | None = None,
                    lower: float = 0.5, upper: float = 2.0, n_draws: int = 200) -> pd.DataFrame:
    """Reallocate a fixed quarterly budget across channels (constant weekly spend per channel).

    Carryover from the last L_MAX-1 observed weeks is included. Objective = posterior-mean
    incremental bookings over the horizon (halo included for social). SLSQP with bounds
    [lower, upper] x current allocation and a budget-sum equality.
    """
    post = posterior_arrays(idata, n_draws)
    hist = df[SPEND].to_numpy()[-(L_MAX - 1):] / sc.spend_max
    current = df[SPEND].to_numpy()[-horizon:].sum(0)
    budget = float(current.sum() if total_budget is None else total_budget)

    def inc_bookings(alloc: np.ndarray) -> np.ndarray:           # (draws,) total incremental bookings
        weekly = (alloc / horizon / sc.spend_max)[None, :]
        path = np.vstack([hist, np.repeat(weekly, horizon, 0)])
        S = channel_response(post, path)[:, -horizon:, :]
        return incremental_bookings(post, S, sc.y_mean).sum(1)

    scale = budget
    obj = lambda a: -inc_bookings(a * scale).sum(1).mean() / 1e3
    x0 = current / current.sum()
    cons = [{"type": "eq", "fun": lambda a: a.sum() - 1.0}]
    bnds = [(lower * s, upper * s) for s in x0]
    res = minimize(obj, x0, method="SLSQP", bounds=bnds, constraints=cons, options={"maxiter": 200})
    best = res.x * scale
    cur_draws, best_draws = inc_bookings(current), inc_bookings(best)
    uplift = best_draws - cur_draws
    out = pd.DataFrame({"current_spend": current, "optimal_spend": best,
                        "change_pct": (best / current - 1) * 100}, index=CHANNELS)
    out.attrs.update(bookings_current=cur_draws.mean(), bookings_optimal=best_draws.mean(),
                     uplift_mean=uplift.mean(), uplift_p3=np.percentile(uplift, 3),
                     uplift_p97=np.percentile(uplift, 97), success=bool(res.success), budget=budget)
    return out


# ----------------------------------------------------------------------------
# 7. Plots
# ----------------------------------------------------------------------------
def _q(a: np.ndarray, axis: int = 0):
    return a.mean(axis), np.percentile(a, 3, axis), np.percentile(a, 97, axis)


def plot_parameters(idata, path: Path) -> None:
    p = idata.posterior.stack(s=("chain", "draw"))
    items: list[tuple[str, np.ndarray]] = []
    for var in ["beta", "alpha", "lam"]:
        for i, c in enumerate(CHANNELS):
            items.append((f"{var}[{c}]", p[var].transpose("s", ...).values[:, i]))
    for var in ["gamma_adr", "gamma_event", "gamma_macro", "gamma_trend", "delta", "halo", "sigma"]:
        items.append((var, p[var].values))
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(items) + 1))
    for k, (name, a) in enumerate(items[::-1]):
        m, lo, hi = _q(a)
        ax.plot([lo, hi], [k, k], color="C0", lw=2)
        ax.plot(m, k, "o", color="C0")
    ax.set_yticks(range(len(items)), [n for n, _ in items[::-1]], fontsize=8)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_title("Posterior mean and 94% interval")
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def plot_fit(idata, df: pd.DataFrame, sc: Scales, path: Path) -> dict[str, float]:
    mu = idata.posterior["mu"].stack(s=("chain", "draw")).squeeze(drop=True).transpose("obs_ind", "s").values * sc.y_mean
    m, lo, hi = _q(mu, 1)
    y = df["room_bookings"].to_numpy()
    r2 = 1 - ((y - m) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    mape = float(np.mean(np.abs((y - m) / y)))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(df["date"], lo, hi, alpha=0.3, label="94% interval (mean fit)")
    ax.plot(df["date"], m, label="fitted")
    ax.plot(df["date"], y, "k", lw=1, label="actual")
    ax.set(title=f"Fitted vs actual room bookings (R2={r2:.3f}, MAPE={mape:.1%})", ylabel="weekly bookings")
    ax.legend(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return dict(r2=r2, mape=mape)


def plot_roas(tab: pd.DataFrame, truth: np.ndarray | None, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(tab))
    ax.bar(x, tab["roas_revenue"], yerr=[tab["roas_revenue"] - tab["roas_p3"], tab["roas_p97"] - tab["roas_revenue"]],
           capsize=4, alpha=0.8, label="estimated (94% HDI)")
    if truth is not None:
        ax.plot(x, truth, "k_", ms=22, mew=2, label="synthetic truth")
    ax.axhline(1, color="r", ls="--", lw=0.8)
    ax.set_xticks(x, tab.index, rotation=15)
    ax.set(ylabel="revenue ROAS (revenue per $1)", title="Channel ROAS (paid social includes halo)")
    ax.legend(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def plot_response_curves(idata, df: pd.DataFrame, sc: Scales, path: Path) -> None:
    post = posterior_arrays(idata, 200)
    fig, axes = plt.subplots(1, len(CHANNELS), figsize=(14, 3.2), sharey=False)
    for i, (ax, c) in enumerate(zip(axes, CHANNELS)):
        grid = np.linspace(0, 2.5, 60)          # weekly spend as multiple of current mean
        mean_sp = df[SPEND[i]].mean()
        x = np.zeros((60, len(CHANNELS)))
        x[:, i] = grid * mean_sp / sc.spend_max[i]
        S = logistic_saturation(x[None] * 0 + x[None], post["lam"])   # steady state: adstock sums to 1
        resp = S[..., i] * post["beta"][:, i:i + 1]
        if i == SOCIAL:
            resp = resp + S[..., i] * (post["delta"] * post["halo"])[:, None]
        resp = resp * sc.y_mean
        m, lo, hi = _q(resp)
        ax.fill_between(grid * mean_sp, lo, hi, alpha=0.3); ax.plot(grid * mean_sp, m)
        ax.axvline(mean_sp, color="gray", ls=":")
        ax.set_title(c, fontsize=9); ax.set_xlabel("weekly spend")
    axes[0].set_ylabel("weekly incremental bookings")
    fig.suptitle("Steady-state response (saturation) curves; dotted = current mean spend")
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def plot_budget(opt: pd.DataFrame, path: Path) -> None:
    ax = opt[["current_spend", "optimal_spend"]].plot.bar(figsize=(7, 4), rot=15)
    ax.set(ylabel="quarterly spend", title="Next-quarter budget: current vs optimal")
    ax.figure.savefig(path, dpi=130, bbox_inches="tight"); plt.close(ax.figure)


# ----------------------------------------------------------------------------
# 8. Main
# ----------------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(args.data, parse_dates=["date"]) if args.data else make_data()
    sc = Scales(df)
    X, y, coords = build_design(df, sc)

    sk: dict[str, Any] = dict(draws=300, tune=300, chains=2) if args.quick else \
        dict(draws=1500, tune=1500, chains=4, target_accept=0.95)
    sk.update(random_seed=42, progressbar=False)
    if not args.no_numpyro:
        sk["nuts_sampler"] = "numpyro"        # no C compiler needed; JAX/GPU capable
    model = HospitalityMMM(sample_kwargs=sk)
    idata = model.fit(X, y, coords=coords)
    idata.to_netcdf(OUT / "idata.nc")

    # diagnostics
    import arviz as az
    summ = az.summary(idata.posterior, var_names=["alpha", "lam", "beta", "gamma_adr", "gamma_event",
                                                  "gamma_macro", "delta", "halo", "sigma"])
    summ.to_csv(OUT / "summary.csv")
    div = int(idata.sample_stats["diverging"].sum())
    print(summ[["mean", "sd", "r_hat", "ess_bulk"]].round(3).to_string())
    print(f"[diag] max r_hat={summ['r_hat'].max():.3f}  min ess_bulk={summ['ess_bulk'].min():.0f}  divergences={div}")

    plot_parameters(idata, OUT / "parameters.png")
    fit = plot_fit(idata, df, sc, OUT / "fit_vs_actual.png")
    print(f"[fit] R2={fit['r2']:.3f}  MAPE={fit['mape']:.2%}")

    tab = roas_table(idata, df, sc)
    tab.to_csv(OUT / "roas.csv")
    truth = truth_roas(df, sc) if not args.data else None
    plot_roas(tab, truth, OUT / "roas.png")
    plot_response_curves(idata, df, sc, OUT / "response_curves.png")
    show = tab.round(3)
    if truth is not None:
        show["truth_roas"] = truth.round(3)
    print("[roas]", REV_PER_BOOKING_NOTE, "\n", show.to_string())

    opt = optimise_budget(idata, df, sc)
    opt.to_csv(OUT / "optimal_allocation.csv")
    plot_budget(opt, OUT / "budget_allocation.png")
    a = opt.attrs
    print(f"[opt] budget {a['budget']:,.0f} over 13 wks, converged={a['success']}\n", opt.round(1).to_string())
    print(f"[opt] expected incremental bookings {a['bookings_current']:,.0f} -> {a['bookings_optimal']:,.0f} "
          f"(uplift {a['uplift_mean']:,.0f}, 94% [{a['uplift_p3']:,.0f}, {a['uplift_p97']:,.0f}])")
    print(f"Outputs in {OUT.resolve()}")


if __name__ == "__main__":
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    p = argparse.ArgumentParser()
    p.add_argument("--data", help="CSV with the same columns as the synthetic data")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--no-numpyro", action="store_true", help="use PyMC's own NUTS (slow without g++)")
    main(p.parse_args())
