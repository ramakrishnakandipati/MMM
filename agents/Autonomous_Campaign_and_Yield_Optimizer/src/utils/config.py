"""Config + path helpers."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def out_dir(*parts: str) -> Path:
    base = Path(os.environ.get("OUTPUT_DIR", ROOT / "outputs"))
    p = base.joinpath(*parts)
    (p if not p.suffix else p.parent).mkdir(parents=True, exist_ok=True)
    return p


@lru_cache(maxsize=None)
def _load(name: str) -> dict:
    with open(ROOT / "config" / name, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def agent_config() -> dict:
    return _load("agent_config.yaml")


def guardrail_config() -> dict:
    return _load("guardrails.yaml")
