"""Constrained cross-channel budget allocation.

Each channel's steady-state response is concave (Hill with S<=1). We replace it by the
minimum of tangent lines (an LP-representable concave upper envelope, exact as #tangents grows):

    max   coef * sum_c r_c - sum_c b_c
    s.t.  r_c <= a_ck * b_c + c_ck          (tangents)
          lo_c <= b_c <= hi_c               (per-cycle shift cap + floors)
          sum_c b_c <= B_max
"""
from __future__ import annotations

import time

import cvxpy as cp
import numpy as np
from scipy.optimize import linprog


class BudgetOptimizer:
    def __init__(self, n_ch: int, n_tan: int = 41):
        self.C, self.K = n_ch, n_tan
        self.b = cp.Variable(n_ch)
        self.r = cp.Variable(n_ch)
        self.slope = cp.Parameter((n_ch, n_tan))
        self.icpt = cp.Parameter((n_ch, n_tan))
        self.coef = cp.Parameter(nonneg=True)
        self.lo = cp.Parameter(n_ch, nonneg=True)
        self.hi = cp.Parameter(n_ch, nonneg=True)
        self.total = cp.Parameter(nonneg=True)
        cons = [self.r[:, None] <= cp.multiply(self.slope, self.b[:, None]) + self.icpt,
                self.b >= self.lo, self.b <= self.hi, cp.sum(self.b) <= self.total]
        self.prob = cp.Problem(cp.Maximize(self.coef * cp.sum(self.r) - cp.sum(self.b)), cons)

    def solve(self, slopes: np.ndarray, icpts: np.ndarray, coef: float, lo: np.ndarray, hi: np.ndarray, total: float) -> dict:
        t0 = time.perf_counter()
        self.slope.value, self.icpt.value = slopes, icpts
        self.coef.value, self.lo.value, self.hi.value, self.total.value = max(coef, 0.0), np.maximum(lo, 0), np.maximum(hi, 0), total
        try:
            self.prob.solve(solver=cp.CLARABEL)
            status = self.prob.status
            if status in ("optimal", "optimal_inaccurate") and self.b.value is not None:
                return {"status": status, "b": np.array(self.b.value), "objective": float(self.prob.value),
                        "solver": "cvxpy/CLARABEL", "seconds": time.perf_counter() - t0}
        except Exception:
            pass
        return self._scipy(slopes, icpts, coef, lo, hi, total, t0)

    def _scipy(self, slopes, icpts, coef, lo, hi, total, t0) -> dict:
        """Fallback LP with HiGHS. Variables [b_1..b_C, r_1..r_C]."""
        C, K = self.C, self.K
        c = np.concatenate([np.ones(C), -coef * np.ones(C)])
        A, ub = [], []
        for ch in range(C):
            for k in range(K):
                row = np.zeros(2 * C)
                row[C + ch], row[ch] = 1.0, -slopes[ch, k]
                A.append(row)
                ub.append(icpts[ch, k])
        row = np.zeros(2 * C)
        row[:C] = 1.0
        A.append(row)
        ub.append(total)
        bounds = [(lo[i], hi[i]) for i in range(C)] + [(None, None)] * C
        res = linprog(c, A_ub=np.array(A), b_ub=np.array(ub), bounds=bounds, method="highs")
        if res.status == 0:
            return {"status": "optimal", "b": res.x[:C], "objective": float(-res.fun), "solver": "scipy/HiGHS",
                    "seconds": time.perf_counter() - t0}
        return {"status": "infeasible", "b": np.clip(lo, 0, None), "objective": float("nan"), "solver": "scipy/HiGHS",
                "seconds": time.perf_counter() - t0}
