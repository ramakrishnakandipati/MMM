import numpy as np
import torch

from src.data.state_builder import spend_cols
from src.evaluation.ope_evaluator import evaluate_policy_ope
from src.models.contextual_bandit import BatchedLinUCB, featurize
from src.models.response_curves import ResponseModel
from src.models.uplift_elasticity import UpliftModel
from src.utils.device import DeviceManager


def test_device_cpu_fallback_and_telemetry():
    d = DeviceManager("cpu")
    assert d.type == "cpu" and not d.amp_enabled
    t = d.write_telemetry()
    assert {"device_name", "vram_mb", "cuda_version"} <= set(t)
    assert DeviceManager("cuda").type in ("cuda", "cpu")        # never raises without a GPU


def test_response_model_fits_and_is_concave(world_data, cfg, dm):
    daily, d = world_data["daily"], cfg["data"]
    rm = ResponseModel(dm, {**cfg["response"], "epochs": 500})
    rep = rm.fit(daily[spend_cols(5)].to_numpy(), daily["dow"].to_numpy(), daily["conversions"].to_numpy(),
                 np.arange(d["train_end"]), np.arange(d["train_end"], d["live_end"]), restarts=2)
    assert rep["r2_holdout"] > 0.7 and all(0 <= x < 1 for x in rep["lam"])
    for c, (g, r) in rm.curves(100000.0).items():
        d1 = np.diff(r) / np.diff(g)
        assert (d1 > -1e-9).all() and (np.diff(d1) < 1e-6).all()


def test_uplift_cate_correlates_with_truth(world_data, cfg, dm):
    um = UpliftModel(dm, {**cfg["uplift"], "epochs": 25})
    rep = um.fit(world_data["uplift"])
    assert rep["cate_corr_true"] > 0.5 and rep["cate_var"] > 0


def test_linucb_batched_matches_closed_form_ridge(dm):
    rng = np.random.default_rng(0)
    x = rng.uniform(size=(500, 6)).astype(np.float32)
    arms = rng.integers(0, 8, 500)
    r = rng.normal(size=500)
    b = BatchedLinUCB(dm, 2, [0.0, 0.05, 0.1, 0.2], ridge=1.0)
    b.update_batch(x, arms, r)
    f = featurize(x).numpy().astype(np.float64)
    for a in (0, 3, 7):
        m = arms == a
        theta = np.linalg.solve(f[m].T @ f[m] + np.eye(14), f[m].T @ r[m])
        got = torch.linalg.solve(b.A[a].double(), b.b[a].double()).cpu().numpy()
        np.testing.assert_allclose(got, theta, rtol=1e-3, atol=1e-3)
    st = b.covariance_stats()
    assert st["finite_cov"] and st["min_eig"] > 0
    s = b.scores(x[:5])
    assert s.shape == (5, 8) and torch.isfinite(s).all()
    assert torch.isfinite(b.thompson_scores(x[:5], seed=1)).all()


def test_best_arm_respects_allowed_tiers(dm):
    b = BatchedLinUCB(dm, 2, [0.0, 0.1], ridge=1.0)
    b.b[3] = 100.0
    _, j = b.best_arm(np.full(6, 0.5), allowed_tiers=[0.0])
    assert j == 0


def test_ope_gpu_bootstrap_ci(world_data, cfg, dm):
    c = {**cfg, "ope": {**cfg["ope"], "n_boot": 2000}}
    res = evaluate_policy_ope(world_data["bandit_log"], dm, c)
    assert res["lift_ci_low"] <= res["dr_lift"] <= res["lift_ci_high"]
    assert res["lift_ci_low"] > 0                       # learned policy beats the static arm on held-out data
    assert abs(res["ips_lift"] - res["dr_lift"]) < 0.2
