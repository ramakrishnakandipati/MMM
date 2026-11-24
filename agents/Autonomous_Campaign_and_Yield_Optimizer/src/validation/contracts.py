"""Pydantic v2 contracts for stage inputs, state and output artifacts."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DailyRow(BaseModel):
    model_config = ConfigDict(extra="allow")
    day_idx: int = Field(ge=0)
    dow: int = Field(ge=0, le=6)
    inventory_util: float = Field(ge=0, le=1)
    capacity_util: float = Field(ge=0, le=1)
    unit_cogs: float = Field(ge=0)
    aov: float = Field(gt=0)
    gross_margin: float = Field(gt=0, lt=1)
    conversions: float = Field(ge=0)


class FeatureRow(BaseModel):
    model_config = ConfigDict(extra="allow")
    total_spend: float = Field(ge=0)
    unit_margin: float = Field(gt=0, lt=1)
    price_sens: float = Field(ge=0, le=1)
    loyalty: float = Field(ge=0, le=1)
    recency: float = Field(ge=0, le=1)
    mobile_share: float = Field(ge=0, le=1)
    weekend: float = Field(ge=0, le=1)

    @field_validator("*", mode="before")
    @classmethod
    def _finite(cls, v):
        if isinstance(v, float) and v != v:
            raise ValueError("NaN")
        return v


class FeatureSpec(BaseModel):
    channels: list[str]
    decays: dict[str, float]
    context_cols: list[str]
    feature_cols: list[str]


class ResponseArtifact(BaseModel):
    r2_train: float
    r2_holdout: float
    beta: list[float]
    K: list[float]
    S: list[float]
    lam: list[float]
    base: float
    final_loss: float
    epochs: int
    device: str


class UpliftArtifact(BaseModel):
    cate_mean: float
    cate_var: float
    cate_corr_true: Optional[float] = None
    n_cohorts: int


class BanditArtifact(BaseModel):
    n_arms: int
    finite_cov: bool
    min_eig: float
    max_cond: float


class AllocationPlan(BaseModel):
    """Proposed + guardrailed action for one cycle."""
    status: str
    budgets: dict[str, float]
    prev_budgets: dict[str, float]
    discount: float = Field(ge=0)
    creative: int = Field(ge=0)
    unit_margin: float
    inventory_util: float
    expected_net: float
    clamps: list[str] = []
    guardrail_ok: bool = True
    solver: str = "cvxpy"


class GateCheck(BaseModel):
    name: str
    passed: bool
    value: Optional[float | str | bool] = None
    threshold: Optional[str] = None
    critical: bool = True
    detail: str = ""


class GateResult(BaseModel):
    gate: int
    name: str
    passed: bool
    checks: list[GateCheck]
    severity: Literal["ok", "warn", "critical"] = "ok"
    seconds: float = 0.0
    timestamp: str = ""
