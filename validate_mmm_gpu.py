"""Validate GPU acceleration in an MMM environment. Exits 1 on any failed check.

Run inside mmm-pymc, causalpy or mmm-meridian; library checks follow what is installed.
"""
import importlib.util
import os
import subprocess
import sys
import threading
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
failures = []


def check(name, passed, detail=""):
    print(f"[{'PASS' if passed else 'FAIL'}] {name}  {detail}", flush=True)
    if not passed:
        failures.append(name)


def installed(module):
    return importlib.util.find_spec(module) is not None


def adstock_hill(spend, decay, half, lags=8):
    """Geometric adstock followed by Hill saturation; spend is geo x week x channel."""
    weights = decay[None, :] ** jnp.arange(lags)[:, None]
    weights = weights / weights.sum(0)
    stacked = jnp.stack([jnp.roll(spend, k, axis=1) for k in range(lags)])
    carried = (stacked * weights[:, None, None, :]).sum(0)
    return carried**2 / (carried**2 + half**2)


@jax.jit
def media_effects(key, geos=2000, weeks=260, channels=20):
    """Simulate media response and recover channel effects with OLS normal equations."""
    k_spend, k_noise = jax.random.split(key)
    true_beta = jnp.linspace(0.5, 2.5, channels)
    spend = jax.random.gamma(k_spend, 2.0, (geos, weeks, channels))
    decay = jnp.linspace(0.2, 0.7, channels)
    x = adstock_hill(spend, decay, jnp.full(channels, 2.0)).reshape(-1, channels)
    y = 3.0 + x @ true_beta + jax.random.normal(k_noise, (x.shape[0],))
    d = jnp.column_stack([jnp.ones(x.shape[0]), x])
    beta = jnp.linalg.solve(d.T @ d, d.T @ y)
    return jnp.abs(beta[1:] - true_beta).max()


def timed(device):
    key = jax.device_put(jax.random.PRNGKey(0), device)
    media_effects(key).block_until_ready()  # first call compiles
    start = time.perf_counter()
    error = media_effects(key).block_until_ready()
    return float(error), time.perf_counter() - start, error.devices()


print(f"Python {sys.version.split()[0]} | JAX {jax.__version__}")
print(subprocess.run(
    ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used", "--format=csv,noheader"],
    capture_output=True, text=True).stdout.strip())

gpus = [d for d in jax.devices() if d.platform == "gpu"]
check("JAX sees GPU", bool(gpus), str(jax.devices()))
if not gpus:
    sys.exit(1)

error_gpu, t_gpu, devices = timed(gpus[0])
error_cpu, t_cpu, _ = timed(jax.devices("cpu")[0])
peak = gpus[0].memory_stats()["peak_bytes_in_use"] / 2**30
check("Media matrix ran on GPU", all(d.platform == "gpu" for d in devices), f"peak {peak:.2f} GiB")
check("Channel effects recovered", error_gpu < 0.05, f"max error {error_gpu:.4f}, CPU {error_cpu:.4f}")
print(f"       Time: GPU {t_gpu:.2f}s, CPU {t_cpu:.2f}s, speedup {t_cpu / t_gpu:.1f}x", flush=True)



class GpuWatch:
    """Poll GPU bytes in use while a library fits, proving the fit touched the GPU."""

    def __init__(self, read):
        self.read = read

    def _poll(self):
        while not self.stop.wait(0.02):
            self.top = max(self.top, self.read())

    def __enter__(self):
        self.base = self.top = self.read()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start
        self.stop.set()
        self.thread.join()
        self.used = (self.top - self.base) / 2**20


def jax_in_use():
    return gpus[0].memory_stats()["bytes_in_use"]


def report_fit(name, watch, chains):
    check(f"{name} used GPU during fit", watch.used > 0, f"+{watch.used:.1f} MiB, {watch.seconds:.1f}s")
    check(f"{name} returned 4 posterior chains", chains == 4, f"chains={chains}")


NUMPYRO = {
    "chains": 4, "draws": 300, "tune": 300, "nuts_sampler": "numpyro",
    "nuts_sampler_kwargs": {"chain_method": "vectorized"},
}

if installed("pymc_marketing"):
    import numpy as np
    import pandas as pd
    from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

    rng = np.random.default_rng(1)
    n = 104
    spend = rng.gamma(2.0, 1.0, (n, 3))
    X = pd.DataFrame(spend, columns=["tv", "search", "social"])
    X.insert(0, "date", pd.date_range("2024-01-01", periods=n, freq="W-MON"))
    y = pd.Series(10 + spend @ np.array([1.5, 0.8, 0.3]) + rng.normal(0, 0.5, n), name="y")
    mmm = MMM(date_column="date", channel_columns=["tv", "search", "social"], target_column="y",
              adstock=GeometricAdstock(l_max=8), saturation=LogisticSaturation())
    with GpuWatch(jax_in_use) as watch:
        idata = mmm.fit(X, y, progressbar=False, random_seed=1, **NUMPYRO)
    report_fit("PyMC-Marketing MMM", watch, idata["posterior"].sizes["chain"])

if installed("causalpy"):
    import causalpy as cp
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(1)
    t = np.arange(100)
    df = pd.DataFrame({"t": t, "y": 2 + 0.1 * t + 5.0 * (t >= 70) + rng.normal(0, 0.5, 100)})
    model = cp.pymc_models.LinearRegression(sample_kwargs={**NUMPYRO, "progressbar": False, "random_seed": 1})
    with GpuWatch(jax_in_use) as watch:
        result = cp.InterruptedTimeSeries(df, treatment_time=70, formula="y ~ 1 + t", model=model)
    report_fit("CausalPy ITS", watch, result.idata.posterior.sizes["chain"])
    effect = float(result.post_impact.mean())
    check("CausalPy recovers intervention effect", abs(effect - 5.0) < 0.5, f"estimate {effect:.3f}, truth 5.0")

if installed("meridian"):
    import tensorflow as tf
    from meridian import backend as meridian_backend
    from meridian.data import test_utils
    from meridian.model.model import Meridian
    from meridian.model.spec import ModelSpec

    tf_gpus = tf.config.list_physical_devices("GPU")
    check("TensorFlow sees GPU", bool(tf_gpus), str(tf_gpus))
    backend_name = str(meridian_backend.config.get_backend()).lower()
    print(f"       Meridian backend: {backend_name}", flush=True)
    read = jax_in_use if "jax" in backend_name else (lambda: tf.config.experimental.get_memory_info("GPU:0")["current"])
    data = test_utils.sample_input_data_non_revenue_revenue_per_kpi(
        n_geos=10, n_times=50, n_media_times=53, n_controls=2, n_media_channels=3, seed=0)
    meridian = Meridian(input_data=data, model_spec=ModelSpec())
    with GpuWatch(read) as watch:
        meridian.sample_posterior(n_chains=4, n_adapt=100, n_burnin=100, n_keep=100, seed=1)
    report_fit("Meridian", watch, meridian.inference_data.posterior.sizes["chain"])

print("\nRESULT:", f"FAILED: {failures}" if failures else "All checks passed.")
sys.exit(1 if failures else 0)
