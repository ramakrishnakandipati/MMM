"""Synthetic multi-channel ad + supply world with known counterfactual ground truth.

The `World` object is the single source of truth for the simulator: the generator uses it
to create logged data, and the benchmark uses it (with regime perturbations) to score
policies on out-of-sample days.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CHANNELS = ["search", "social", "video", "display", "affiliate"]
CTX_COLS = ["inventory_util", "price_sens", "loyalty", "recency", "mobile_share", "weekend"]
DISCOUNT_TIERS = [0.0, 0.05, 0.10, 0.20]
N_CREATIVES = 4


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class World:
    beta: np.ndarray            # saturation conversions/day per channel
    K: np.ndarray               # half-saturation (adstocked $)
    S: np.ndarray               # Hill shape (<=1 => concave)
    lam: np.ndarray             # geometric adstock decay
    organic: float = 350.0      # organic conversions/day with zero spend
    dow: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.02, 1.05, 1.1, 0.92, 0.88]))
    aov: float = 85.0
    gm: float = 0.55
    redemption: float = 0.30    # share of conversions that redeem the promotion
    creative_gamma: np.ndarray = field(default_factory=lambda: np.zeros((N_CREATIVES, len(CTX_COLS))))
    p0_w: np.ndarray = field(default_factory=lambda: np.zeros(len(CTX_COLS)))
    util_shift: float = 0.0     # regime: pushes inventory utilization up
    name: str = "baseline"

    @property
    def n_ch(self) -> int:
        return len(self.beta)

    # ---- demand side -------------------------------------------------
    def hill(self, x: np.ndarray) -> np.ndarray:
        """x: adstocked spend per channel (..., C) -> conversions contribution (..., C)."""
        x = np.maximum(x, 0.0)
        xs = x ** self.S
        return self.beta * xs / (self.K ** self.S + xs + 1e-12)

    def steady_adstock(self, b: np.ndarray) -> np.ndarray:
        return b / (1.0 - self.lam)

    # ---- promo / creative ------------------------------------------------
    @staticmethod
    def rel_sensitivity(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return 0.03 + 0.22 * _sigmoid(5.0 * (x[:, 1] - 0.5) - 2.0 * (x[:, 2] - 0.5))

    def supply_factor(self, util: np.ndarray) -> np.ndarray:
        return np.clip((1.0 - np.minimum(util + self.util_shift, 0.995)) / 0.3, 0.0, 1.0)

    def lift(self, x: np.ndarray, d: float) -> np.ndarray:
        x = np.atleast_2d(x)
        if d <= 0:
            return np.zeros(len(x))
        return self.rel_sensitivity(x) * (d / 0.10) ** 0.8 * self.supply_factor(x[:, 0])

    def creative_mult(self, x: np.ndarray, k: int) -> np.ndarray:
        x = np.atleast_2d(x)
        return 1.0 + (x - 0.5) @ self.creative_gamma[k]

    def p0(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return 0.03 + 0.09 * _sigmoid((x - 0.5) @ self.p0_w)

    # ---- one-cycle simulation -----------------------------------------------
    def simulate_cycle(self, spend: np.ndarray, ad_state: np.ndarray, ctx_days: np.ndarray, dow_days: np.ndarray,
                       d: float, creative: int, gm: float | None = None,
                       rng: np.random.Generator | None = None) -> dict:
        """Run len(ctx_days) days at constant daily `spend`; returns net yield and next adstock state.

        Net = incr_conv * AOV * GM  -  ad spend  -  promo discount cost
        incr_conv = conv(with spend, promo) - conv(no spend, no promo)
        promo cost = d * AOV * redemption * conv_total
        """
        gm = self.gm if gm is None else gm
        st = ad_state.copy()
        net = 0.0
        incr_total = conv_total_sum = promo_total = spend_total = 0.0
        daily_conv = []
        for ctx, dw_i in zip(ctx_days, dow_days):
            dw = float(self.dow[int(dw_i)])
            st = spend + self.lam * st
            ad_part = self.hill(st).sum()
            mk = float(self.creative_mult(ctx, creative)[0])
            lf = float(self.lift(ctx, d)[0])
            conv = (self.organic + ad_part) * dw * mk * (1.0 + lf)
            daily_conv.append(conv * (rng.lognormal(0, 0.03) if rng is not None else 1.0))
            conv0 = self.organic * dw
            incr = conv - conv0
            promo = d * self.aov * self.redemption * conv
            net += incr * self.aov * gm - spend.sum() - promo
            incr_total += incr
            conv_total_sum += conv
            promo_total += promo
            spend_total += spend.sum()
        return {"net": net, "incr_conv": incr_total, "conv": conv_total_sum, "promo_cost": promo_total,
                "spend": spend_total, "ad_state": st, "daily_conv": np.array(daily_conv)}

    def with_regime(self, regime: str) -> "World":
        w = copy.deepcopy(self)
        w.name = regime
        if regime == "demand_shock":      # one channel collapses, another surges
            w.beta = w.beta * np.array([0.55, 1.0, 1.0, 1.35, 1.0])
        elif regime == "inventory_crunch":
            w.util_shift = 0.25
        elif regime == "high_cac":         # saturation point moves out => weaker response
            w.K = w.K * np.array([1.0, 1.7, 1.7, 1.0, 1.0])
        elif regime == "margin_squeeze":   # COGS spike
            w.gm = 0.38
        elif regime != "baseline":
            raise ValueError(regime)
        return w


def default_world(n_channels: int = 5, aov: float = 85.0, gm: float = 0.55, seed: int = 7) -> World:
    rng = np.random.default_rng(seed + 1)
    # Heterogeneous channels: very different ROI curves so allocation matters.
    beta = 1.3 * np.array([800.0, 1500.0, 1100.0, 500.0, 600.0])[:n_channels]
    K = np.array([14000.0, 60000.0, 48000.0, 40000.0, 9000.0])[:n_channels]
    S = np.array([0.85, 0.75, 0.80, 0.90, 0.70])[:n_channels]
    lam = np.array([0.30, 0.50, 0.65, 0.20, 0.10])[:n_channels]
    gamma = np.zeros((N_CREATIVES, len(CTX_COLS)))
    gamma[1:] = rng.normal(0, 0.35, size=(N_CREATIVES - 1, len(CTX_COLS)))
    p0_w = rng.normal(0, 1.2, size=len(CTX_COLS))
    return World(beta=beta, K=K, S=S, lam=lam, aov=aov, gm=gm, creative_gamma=gamma, p0_w=p0_w)


# --------------------------------------------------------------------------------------
def _context_series(n: int, rng: np.random.Generator) -> pd.DataFrame:
    t = np.arange(n)
    ar = lambda s: np.convolve(rng.normal(0, s, n), np.ones(3) / 3, mode="same")
    df = pd.DataFrame({
        "inventory_util": np.clip(0.62 + 0.24 * np.sin(2 * np.pi * t / 90) + ar(0.12), 0.25, 0.98),
        "price_sens": np.clip(0.5 + 0.28 * np.sin(2 * np.pi * t / 60 + 1) + ar(0.12), 0.02, 0.98),
        "loyalty": np.clip(0.5 + 0.15 * np.sin(2 * np.pi * t / 180) + ar(0.08), 0.02, 0.98),
        "recency": np.clip(0.5 + ar(0.15), 0.02, 0.98),
        "mobile_share": np.clip(0.6 + 0.1 * np.sin(2 * np.pi * t / 180) + ar(0.08), 0.02, 0.98),
    })
    df["weekend"] = (t % 7 >= 5).astype(float)
    return df


def sample_contexts(n: int, rng: np.random.Generator) -> np.ndarray:
    x = np.column_stack([
        rng.uniform(0.25, 0.98, n), rng.beta(2, 2, n), rng.beta(2, 2, n),
        rng.beta(2, 2, n), rng.beta(3, 2, n), rng.binomial(1, 2 / 7, n).astype(float),
    ])
    return x


def generate_daily(world: World, cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    n, C = cfg["n_days"], world.n_ch
    bmax = cfg["b_max_daily"]
    ctx = _context_series(n, rng)
    # Spend log: persistent 3-day blocks + daily noise, random shares => identifiable curves.
    n_blocks = n // 3 + 1
    total = bmax * rng.uniform(0.45, 1.0, n_blocks)
    shares = rng.dirichlet(np.full(C, 6.0), n_blocks)
    spend = np.repeat(total[:, None] * shares, 3, axis=0)[:n] * rng.uniform(0.85, 1.15, (n, C))
    dow = np.arange(n) % 7
    st = np.zeros(C)
    contrib = np.zeros((n, C))
    for t in range(n):
        st = spend[t] + world.lam * st
        contrib[t] = world.hill(st)
    # day-of-week multiplies the whole signal (matches World.simulate_cycle)
    conv_true = (world.organic + contrib.sum(1)) * world.dow[dow]
    conv = conv_true * rng.lognormal(0, 0.03, n)
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "day_idx": np.arange(n),
        "dow": dow,
    })
    for c in range(C):
        df[f"spend_{CHANNELS[c]}"] = spend[:, c]
    for col in CTX_COLS:
        df[col] = ctx[col].values
    df["aov"] = world.aov * rng.normal(1.0, 0.01, n)
    df["unit_cogs"] = df["aov"] * (1 - world.gm) * rng.normal(1.0, 0.02, n)
    df["gross_margin"] = 1.0 - df["unit_cogs"] / df["aov"]
    df["capacity_util"] = df["inventory_util"]
    df["conversions"] = conv
    df["true_organic"] = world.organic * world.dow[dow]
    for c in range(C):
        df[f"true_contrib_{CHANNELS[c]}"] = contrib[:, c] * world.dow[dow]
    return df


def generate_uplift(world: World, n: int, rng: np.random.Generator, cohort: int = 2000) -> pd.DataFrame:
    x = sample_contexts(n, rng)
    t = rng.binomial(1, 0.5, n).astype(float)
    p = world.p0(x) * (1.0 + t * world.lift(x, 0.10))
    y = p + rng.normal(0, np.sqrt(p * (1 - p) / cohort))
    df = pd.DataFrame(x, columns=CTX_COLS)
    df["treated"], df["outcome"] = t, y
    df["true_cate"] = world.p0(x) * world.lift(x, 0.10)
    df["p0"] = world.p0(x)
    return df


def bandit_reward(world: World, x: np.ndarray, k: int, d: float, gm: float | None = None) -> np.ndarray:
    gm = world.gm if gm is None else gm
    p = world.p0(x) * world.creative_mult(x, k) * (1.0 + world.lift(x, d))
    return world.aov * p * (gm - d * world.redemption)


def generate_bandit_log(world: World, n: int, rng: np.random.Generator, tiers=DISCOUNT_TIERS) -> pd.DataFrame:
    x = sample_contexts(n, rng)
    n_arms = N_CREATIVES * len(tiers)
    W0 = rng.normal(0, 0.6, size=(len(CTX_COLS), n_arms))
    logits = (x - 0.5) @ W0
    prob = np.exp(logits - logits.max(1, keepdims=True))
    prob /= prob.sum(1, keepdims=True)
    cum = prob.cumsum(1)
    a = (rng.uniform(size=(n, 1)) > cum).sum(1).clip(0, n_arms - 1)
    k, j = a // len(tiers), a % len(tiers)
    r = np.zeros(n)
    for ki in range(N_CREATIVES):
        for ji, d in enumerate(tiers):
            m = (k == ki) & (j == ji)
            if m.any():
                r[m] = bandit_reward(world, x[m], ki, d)
    r = r + rng.normal(0, 0.15, n)
    df = pd.DataFrame(x, columns=CTX_COLS)
    df["arm"], df["creative"], df["tier_idx"] = a, k, j
    df["propensity"] = prob[np.arange(n), a]
    df["reward"] = r
    # logged full propensity matrix for off-policy evaluation of arbitrary target policies
    for i in range(n_arms):
        df[f"logp_{i}"] = prob[:, i]
    return df


def generate_all(cfg: dict, seed: int | None = None) -> dict:
    seed = cfg_seed(cfg) if seed is None else seed
    dcfg = cfg["data"]
    world = default_world(dcfg["n_channels"], dcfg["aov"], dcfg["gross_margin"], seed)
    rng = np.random.default_rng(seed)
    daily = generate_daily(world, dcfg, rng)
    uplift = generate_uplift(world, cfg["uplift"]["n_samples"], rng)
    blog = generate_bandit_log(world, cfg["bandit"]["n_log"], rng, cfg["bandit"]["discount_tiers"])
    return {"world": world, "daily": daily, "uplift": uplift, "bandit_log": blog}


def cfg_seed(cfg: dict) -> int:
    return int(cfg.get("seed", 7))
