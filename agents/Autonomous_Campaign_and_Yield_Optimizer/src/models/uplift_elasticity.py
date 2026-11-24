"""Two-head T-Learner (shared trunk) estimating CATE of a 10% discount, trained on GPU."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..data.synthetic_generator import CTX_COLS
from ..utils.device import DeviceManager

Y_SCALE = 10.0


class TLearnerNet(nn.Module):
    def __init__(self, d_in: int, hidden: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.h0 = nn.Sequential(nn.Linear(hidden, 32), nn.SiLU(), nn.Linear(32, 1))   # control outcome
        self.h1 = nn.Sequential(nn.Linear(hidden, 32), nn.SiLU(), nn.Linear(32, 1))   # treated outcome

    def forward(self, x):
        z = self.trunk(x)
        return self.h0(z).squeeze(-1), self.h1(z).squeeze(-1)


class UpliftModel:
    def __init__(self, dm: DeviceManager, cfg: dict):
        self.dm, self.cfg = dm, cfg
        self.net: TLearnerNet | None = None
        self.report: dict = {}

    def fit(self, df: pd.DataFrame, seed: int = 0) -> dict:
        torch.manual_seed(seed)
        dev = self.dm.device
        x = torch.tensor(df[CTX_COLS].to_numpy(), dtype=torch.float32, device=dev)
        t = torch.tensor(df["treated"].to_numpy(), dtype=torch.float32, device=dev)
        y = torch.tensor(df["outcome"].to_numpy() * Y_SCALE, dtype=torch.float32, device=dev)
        n = len(x)
        perm = torch.randperm(n, device=dev)
        val, tr = perm[: n // 10], perm[n // 10:]
        self.net = TLearnerNet(x.shape[1]).to(dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.cfg["lr"], weight_decay=1e-4)
        ep, bs = self.cfg["epochs"], self.cfg["batch_size"]
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep)
        for _ in range(ep):
            self.net.train()
            idx = tr[torch.randperm(len(tr), device=dev)]
            for i in range(0, len(idx), bs):
                b = idx[i:i + bs]
                with self.dm.autocast():
                    m0, m1 = self.net(x[b])
                    pred = torch.where(t[b] > 0.5, m1, m0)
                loss = ((pred.float() - y[b]) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            sched.step()
        self.net.eval()
        with torch.no_grad():
            m0, m1 = self.net(x[val])
            pred = torch.where(t[val] > 0.5, m1, m0)
            val_mse = float(((pred - y[val]) ** 2).mean())
        cate = self.cate(df[CTX_COLS].to_numpy())
        corr = None
        if "true_cate" in df:
            corr = float(np.corrcoef(cate, df["true_cate"].to_numpy())[0, 1])
        q = np.quantile(cate, [0.25, 0.5, 0.75])
        seg = np.digitize(cate, q)
        segments = {f"seg_{k}": {"n": int((seg == k).sum()), "mean_cate": float(cate[seg == k].mean())} for k in range(4)}
        self.report = {"cate_mean": float(cate.mean()), "cate_var": float(cate.var()), "cate_corr_true": corr,
                       "n_cohorts": int(len(cate)), "val_mse": val_mse, "segments": segments, "device": str(dev)}
        return self.report

    @torch.no_grad()
    def _heads(self, x: np.ndarray):
        xt = torch.tensor(np.atleast_2d(x), dtype=torch.float32, device=self.dm.device)
        m0, m1 = self.net(xt)
        return (m0.cpu().numpy() / Y_SCALE), (m1.cpu().numpy() / Y_SCALE)

    def cate(self, x: np.ndarray) -> np.ndarray:
        m0, m1 = self._heads(x)
        return m1 - m0

    def rel_lift(self, x: np.ndarray, d: float) -> float:
        """Estimated relative conversion lift at discount d for mean context x (curvature 0.8 = modelling prior)."""
        if d <= 0:
            return 0.0
        m0, m1 = self._heads(x)
        rel = float(np.clip((m1 - m0) / np.maximum(m0, 1e-4), 0.0, 1.0).mean())
        return rel * (d / 0.10) ** 0.8
