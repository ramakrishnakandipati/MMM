"""Merges demand-side ad signals (adstock) with supply-side yield/COGS into model-ready state."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .synthetic_generator import CHANNELS, CTX_COLS

PRIOR_DECAY = {"search": 0.3, "social": 0.5, "video": 0.6, "display": 0.2, "affiliate": 0.1}


def geometric_adstock(x: np.ndarray, decay: float) -> np.ndarray:
    out = np.zeros_like(x, dtype=float)
    carry = 0.0
    for i, v in enumerate(x):
        carry = v + decay * carry
        out[i] = carry
    return out


def spend_cols(n_ch: int | None = None) -> list[str]:
    cols = [f"spend_{c}" for c in CHANNELS]
    return cols[:n_ch] if n_ch else cols


def build_features(daily: pd.DataFrame, decays: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Returns (features frame, FeatureSpec dict)."""
    decays = decays or PRIOR_DECAY
    f = daily.copy()
    for c in CHANNELS:
        if f"spend_{c}" in f:
            f[f"adstock_{c}"] = geometric_adstock(f[f"spend_{c}"].to_numpy(), decays[c])
    sc = [c for c in f.columns if c.startswith("spend_")]
    f["total_spend"] = f[sc].sum(axis=1)
    f["unit_margin"] = f["gross_margin"]
    f["dow_sin"] = np.sin(2 * np.pi * f["dow"] / 7)
    f["dow_cos"] = np.cos(2 * np.pi * f["dow"] / 7)
    spec = {
        "channels": [c for c in CHANNELS if f"spend_{c}" in f],
        "decays": {c: decays[c] for c in CHANNELS if f"spend_{c}" in f},
        "context_cols": CTX_COLS,
        "feature_cols": [c for c in f.columns if c.startswith("adstock_")] + CTX_COLS + ["total_spend", "unit_margin"],
    }
    return f, spec


def split_windows(df: pd.DataFrame, train_end: int, live_end: int) -> dict:
    return {
        "train": df[df.day_idx < train_end].reset_index(drop=True),
        "live": df[(df.day_idx >= train_end) & (df.day_idx < live_end)].reset_index(drop=True),
        "oos": df[df.day_idx >= live_end].reset_index(drop=True),
    }
