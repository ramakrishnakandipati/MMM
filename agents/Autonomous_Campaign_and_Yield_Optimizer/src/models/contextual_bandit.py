"""GPU-batched LinUCB / Thompson Sampling over (creative, discount_tier) arms."""
from __future__ import annotations

import numpy as np
import torch

from ..utils.device import DeviceManager


def featurize(x: np.ndarray | torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(x, dtype=torch.float32)
    x = x.unsqueeze(0) if x.ndim == 1 else x
    util_price = (x[:, 0] * x[:, 1]).unsqueeze(1)
    return torch.cat([torch.ones(len(x), 1), x, (x - 0.5) ** 2, util_price], dim=1)   # d = 1+6+6+1 = 14


class BatchedLinUCB:
    """Disjoint LinUCB: per-arm A (d x d) and b (d). All arms solved in one batched linalg call."""

    def __init__(self, dm: DeviceManager, n_creatives: int, tiers: list[float], alpha: float = 0.5,
                 ridge: float = 1.0, d: int = 14):
        self.dm, self.tiers, self.n_cr, self.alpha = dm, list(tiers), n_creatives, alpha
        self.n_arms = n_creatives * len(tiers)
        dev = dm.device
        self.A = ridge * torch.eye(d, device=dev).repeat(self.n_arms, 1, 1)
        self.b = torch.zeros(self.n_arms, d, device=dev)
        self.n_seen = torch.zeros(self.n_arms, device=dev)

    def arm(self, creative: int, tier_idx: int) -> int:
        return creative * len(self.tiers) + tier_idx

    def update_batch(self, x: np.ndarray, arms: np.ndarray, r: np.ndarray) -> None:
        """Batched ridge update (equivalent to sequential updates, order-independent)."""
        dev = self.dm.device
        f = featurize(x).to(dev)
        a = torch.as_tensor(arms, dtype=torch.long, device=dev)
        rr = torch.as_tensor(r, dtype=torch.float32, device=dev)
        outer = f.unsqueeze(2) * f.unsqueeze(1)
        self.A.index_add_(0, a, outer)
        self.b.index_add_(0, a, f * rr.unsqueeze(1))
        self.n_seen.index_add_(0, a, torch.ones_like(rr))

    def _inv(self):
        return torch.linalg.inv(self.A)

    @torch.no_grad()
    def scores(self, x: np.ndarray, explore: bool = True) -> torch.Tensor:
        """(n, n_arms) UCB scores."""
        f = featurize(x).to(self.dm.device)
        Ainv = self._inv()
        theta = torch.einsum("aij,aj->ai", Ainv, self.b)
        mean = f @ theta.T
        if not explore:
            return mean
        var = torch.einsum("ni,aij,nj->na", f, Ainv, f).clamp_min(0)
        return mean + self.alpha * var.sqrt()

    @torch.no_grad()
    def thompson_scores(self, x: np.ndarray, v: float = 0.5, seed: int | None = None) -> torch.Tensor:
        dev = self.dm.device
        g = torch.Generator(device=dev)
        if seed is not None:
            g.manual_seed(seed)
        Ainv = self._inv()
        theta = torch.einsum("aij,aj->ai", Ainv, self.b)
        Lc = torch.linalg.cholesky(0.5 * (Ainv + Ainv.transpose(1, 2)) + 1e-8 * torch.eye(Ainv.shape[1], device=dev))
        z = torch.randn(theta.shape, device=dev, generator=g)
        sample = theta + v * torch.einsum("aij,aj->ai", Lc, z)
        return featurize(x).to(dev) @ sample.T

    def best_arm(self, x: np.ndarray, allowed_tiers: list[float] | None = None, explore: bool = False) -> tuple[int, int]:
        s = self.scores(x, explore=explore)[0].clone()
        if allowed_tiers is not None:
            for a in range(self.n_arms):
                if self.tiers[a % len(self.tiers)] not in allowed_tiers:
                    s[a] = -float("inf")
        a = int(torch.argmax(s))
        return a // len(self.tiers), a % len(self.tiers)

    def creative_mult(self, x: np.ndarray, tier: float, creative: int) -> float:
        """Predicted reward ratio vs. creative 0 at the same tier (clipped) -- used by the planner."""
        j = self.tiers.index(tier)
        s = self.scores(x, explore=False)[0]
        a0, a1 = self.arm(0, j), self.arm(creative, j)
        return float(torch.clamp(s[a1] / s[a0].clamp_min(1e-6), 0.8, 1.3)) if s[a0] > 0 else 1.0

    def covariance_stats(self) -> dict:
        Ainv = self._inv()
        sym = 0.5 * (self.A + self.A.transpose(1, 2))
        eig = torch.linalg.eigvalsh(sym)
        finite = bool(torch.isfinite(Ainv).all() and torch.isfinite(self.A).all())
        return {"n_arms": self.n_arms, "finite_cov": finite, "min_eig": float(eig.min()),
                "max_cond": float((eig.max(1).values / eig.min(1).values).max()), "total_updates": float(self.n_seen.sum())}
