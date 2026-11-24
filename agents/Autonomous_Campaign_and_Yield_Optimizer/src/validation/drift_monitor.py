"""PSI / KS feature drift and telemetry anomaly checks."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 5, eps: float = 1e-4) -> float:
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(train: pd.DataFrame, live: pd.DataFrame, cols: list[str]) -> dict:
    rep = {}
    for c in cols:
        ks = stats.ks_2samp(train[c], live[c])
        rep[c] = {"psi": round(psi(train[c].to_numpy(), live[c].to_numpy()), 5),
                  "ks_stat": round(float(ks.statistic), 5), "ks_p": round(float(ks.pvalue), 5)}
    rep["_max_psi"] = max(v["psi"] for k, v in rep.items() if not k.startswith("_"))
    return rep


def telemetry_anomalies(train: pd.Series, live: pd.Series, z_thresh: float = 4.0) -> dict:
    mu, sd = float(train.mean()), float(train.std() + 1e-9)
    z = (live - mu) / sd
    return {"mean_z": float(z.mean()), "max_abs_z": float(z.abs().max()), "n_outliers": int((z.abs() > z_thresh).sum())}
