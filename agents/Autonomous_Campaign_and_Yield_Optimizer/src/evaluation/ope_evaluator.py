"""Off-policy evaluation (IPS + Doubly Robust) with a GPU bootstrap (N=10,000 resamples)."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

from ..data.synthetic_generator import CTX_COLS
from ..models.contextual_bandit import BatchedLinUCB, featurize
from ..utils.device import DeviceManager


def evaluate_policy_ope(log: pd.DataFrame, dm: DeviceManager, cfg: dict, seed: int = 0) -> dict:
    """Target = greedy LinUCB trained on first half of the log; baseline = static arm (creative 0, no discount).
    Evaluation uses the held-out second half only."""
    n = len(log)
    half = n // 2
    tr, te = log.iloc[:half], log.iloc[half:].reset_index(drop=True)
    tiers = cfg["bandit"]["discount_tiers"]
    bandit = BatchedLinUCB(dm, cfg["bandit"]["n_creatives"], tiers, cfg["bandit"]["alpha"], cfg["bandit"]["ridge_lambda"])
    bandit.update_batch(tr[CTX_COLS].to_numpy(), tr["arm"].to_numpy(), tr["reward"].to_numpy())
    dev = dm.device
    t0 = time.perf_counter()
    X = te[CTX_COLS].to_numpy()
    q = bandit.scores(X, explore=False)                             # (n, A) reward-model predictions
    pi = q.argmax(1)                                                # greedy target policy
    a = torch.as_tensor(te["arm"].to_numpy(), device=dev)
    r = torch.as_tensor(te["reward"].to_numpy(), dtype=torch.float32, device=dev)
    p = torch.as_tensor(te["propensity"].to_numpy(), dtype=torch.float32, device=dev).clamp_min(1e-2)
    ar = torch.arange(n - half, device=dev)
    q_a = q[ar, a]

    def dr_ips(target_arm):
        match = (a == target_arm).float()
        ips = match / p * r
        dr = q[ar, target_arm] + match / p * (r - q_a)
        return ips, dr

    ips_t, dr_t = dr_ips(pi)
    ips_s, dr_s = dr_ips(torch.zeros_like(a))                       # static arm 0
    lift_ips = float(ips_t.mean() / ips_s.mean() - 1)
    lift_dr = float(dr_t.mean() / dr_s.mean() - 1)

    # --- GPU Monte Carlo bootstrap ---
    nb, m = cfg["ope"]["n_boot"], len(dr_t)
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    lifts, chunk = [], 1000
    for i in range(0, nb, chunk):
        idx = torch.randint(0, m, (min(chunk, nb - i), m), device=dev, generator=g)
        lifts.append(dr_t[idx].mean(1) / dr_s[idx].mean(1) - 1)
    lifts = torch.cat(lifts)
    dm.sync()
    alpha = (1 - cfg["ope"]["ci"]) / 2
    lo, hi = torch.quantile(lifts, alpha).item(), torch.quantile(lifts, 1 - alpha).item()
    return {
        "n_eval": int(m), "n_boot": int(nb), "device": str(dev),
        "value_target_dr": float(dr_t.mean()), "value_static_dr": float(dr_s.mean()),
        "value_logging": float(r.mean()),
        "ips_lift": lift_ips, "dr_lift": lift_dr,
        "lift_ci_low": lo, "lift_ci_high": hi,
        "boot_mean": float(lifts.mean()), "boot_std": float(lifts.std()),
        "bootstrap_ms": round((time.perf_counter() - t0) * 1000, 1),
        "target_arm_hist": np.bincount(pi.cpu().numpy(), minlength=bandit.n_arms).tolist(),
    }
