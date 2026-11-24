"""GPU Hill saturation + geometric adstock response model, all channels fit jointly."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.device import DeviceManager

SPEND_SCALE = 20000.0


class HillAdstockNet(nn.Module):
    def __init__(self, n_ch: int, adstock_len: int, y0: float, init_seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(init_seed)
        self.n_ch, self.L, self.y0 = n_ch, adstock_len, y0
        self.raw_lam = nn.Parameter(torch.randn(n_ch, generator=g) * 0.8)
        self.log_K = nn.Parameter(torch.randn(n_ch, generator=g) * 0.4 + 0.3)
        self.raw_S = nn.Parameter(torch.randn(n_ch, generator=g) * 0.6)
        self.log_beta = nn.Parameter(torch.randn(n_ch, generator=g) * 0.3 - 0.3)
        self.log_base = nn.Parameter(torch.tensor(0.0))
        self.dow_emb = nn.Parameter(torch.zeros(7))

    # constrained parameterisation -> concave, bounded Hill
    def lam(self):  return 0.95 * torch.sigmoid(self.raw_lam)
    def S(self):    return 0.5 + 0.5 * torch.sigmoid(self.raw_S)          # (0.5, 1.0): concave
    def K(self):    return torch.exp(self.log_K)
    def beta(self): return torch.exp(self.log_beta)
    def base(self): return torch.exp(self.log_base)

    def adstock(self, s: torch.Tensor) -> torch.Tensor:
        """s: (T, C) normalised spend -> (T, C) via learnable geometric kernel."""
        lam = self.lam()
        k = torch.arange(self.L, device=s.device, dtype=s.dtype)
        kernel = lam[None, :] ** k[:, None]                       # (L, C)
        x = F.pad(s.T.unsqueeze(0), (self.L - 1, 0))              # (1, C, T+L-1)
        w = kernel.flip(0).T.unsqueeze(1)                         # (C, 1, L)
        return F.conv1d(x, w, groups=self.n_ch).squeeze(0).T

    def hill(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp_min(1e-9)
        xs = x ** self.S()
        return self.beta() * xs / (self.K() ** self.S() + xs)

    def forward(self, s: torch.Tensor, dow: torch.Tensor) -> torch.Tensor:
        mult = torch.exp(self.dow_emb - self.dow_emb.mean())[dow]
        return (self.base() + self.hill(self.adstock(s)).sum(1)) * mult


def r2_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


class ResponseModel:
    """Trainer + inference wrapper in original (dollar / conversion) units."""

    def __init__(self, dm: DeviceManager, cfg: dict):
        self.dm, self.cfg = dm, cfg
        self.net: HillAdstockNet | None = None
        self.n_ch = None
        self.report: dict = {}

    def fit(self, spend: np.ndarray, dow: np.ndarray, y: np.ndarray, train_idx: np.ndarray, hold_idx: np.ndarray,
            restarts: int = 3, seed: int = 0) -> dict:
        dev = self.dm.device
        self.n_ch = spend.shape[1]
        self.y0 = float(y[train_idx].mean())
        s = torch.tensor(spend / SPEND_SCALE, dtype=torch.float32, device=dev)
        d = torch.tensor(dow, dtype=torch.long, device=dev)
        yt = torch.tensor(y / self.y0, dtype=torch.float32, device=dev)
        tr = torch.tensor(train_idx, device=dev)
        ep, lr = self.cfg["epochs"], self.cfg["lr"]
        best = None
        for r in range(restarts):
            net = HillAdstockNet(self.n_ch, self.cfg["adstock_len"], self.y0, init_seed=seed + r).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep)
            for e in range(ep):
                opt.zero_grad(set_to_none=True)
                pred = net(s, d)
                loss = F.mse_loss(pred[tr], yt[tr])
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
                sched.step()
            with torch.no_grad():
                final = float(F.mse_loss(net(s, d)[tr], yt[tr]))
            if best is None or final < best[0]:
                best = (final, net)
        self.net = best[1].eval()
        with torch.no_grad():
            pred = (self.net(s, d) * self.y0).cpu().numpy()
        self.report = {
            "r2_train": r2_score(y[train_idx], pred[train_idx]),
            "r2_holdout": r2_score(y[hold_idx], pred[hold_idx]),
            "final_loss": best[0], "epochs": ep, "device": str(dev), "restarts": restarts,
            **self.params(),
        }
        return self.report

    # ------------------------------------------------------------------ inference
    def params(self) -> dict:
        n = self.net
        return {
            "beta": (n.beta().detach().cpu().numpy() * self.y0).tolist(),
            "K": (n.K().detach().cpu().numpy() * SPEND_SCALE).tolist(),   # adstocked $
            "S": n.S().detach().cpu().numpy().tolist(),
            "lam": n.lam().detach().cpu().numpy().tolist(),
            "base": float(n.base().detach().cpu()) * self.y0,
        }

    def dow_mult(self) -> np.ndarray:
        e = self.net.dow_emb.detach().cpu()
        return torch.exp(e - e.mean()).numpy()

    def steady_response(self, b: np.ndarray) -> np.ndarray:
        """Daily steady-state conversions contribution per channel for constant daily spend b (C,)."""
        p = self.params()
        lam, K, S, beta = map(np.array, (p["lam"], p["K"], p["S"], p["beta"]))
        x = np.maximum(b, 0) / (1 - lam)
        xs = x ** S
        return beta * xs / (K ** S + xs + 1e-12)

    def curves(self, b_max: float, n: int = 60) -> dict:
        grid = np.linspace(0, b_max, n)
        out = {}
        for c in range(self.n_ch):
            r = []
            for g in grid:
                b = np.zeros(self.n_ch)
                b[c] = g
                r.append(self.steady_response(b)[c])
            out[c] = (grid, np.array(r))
        return out

    def tangents(self, c: int, b_hi: float, n: int = 40):
        """Tangent lines (slope, intercept) of channel-c steady response on [0, b_hi] for PWL concave upper hull."""
        p = self.params()
        lam, K, S, beta = (np.array(p[k])[c] for k in ("lam", "K", "S", "beta"))
        pts = np.concatenate([[1.0], np.linspace(b_hi / n, b_hi, n)])
        x = pts / (1 - lam)
        xs = x ** S
        f = beta * xs / (K ** S + xs)
        # d f / d b = beta * S * K^S * x^(S-1) / (K^S + x^S)^2 / (1-lam)
        df = beta * S * K ** S * x ** (S - 1) / (K ** S + xs) ** 2 / (1 - lam)
        return df, f - df * pts
