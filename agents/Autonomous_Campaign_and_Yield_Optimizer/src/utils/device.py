"""Hardware manager: CUDA -> MPS -> CPU with mixed-precision config and telemetry."""
from __future__ import annotations

import contextlib
import json
import platform
import time
from typing import Optional

import torch

from .config import out_dir


class DeviceManager:
    def __init__(self, preference: str = "auto"):
        self.preference = preference
        self.device = self._select(preference)
        self.type = self.device.type
        # AMP only helps on CUDA; small tabular nets stay in fp32 elsewhere.
        self.amp_enabled = self.type == "cuda"
        self.amp_dtype = torch.bfloat16 if (self.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    @staticmethod
    def _select(pref: str) -> torch.device:
        pref = (pref or "auto").lower()
        if pref in ("auto", "cuda") and torch.cuda.is_available():
            return torch.device("cuda")
        if pref in ("auto", "mps") and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def autocast(self):
        if self.amp_enabled:
            return torch.autocast(device_type="cuda", dtype=self.amp_dtype)
        return contextlib.nullcontext()

    def sync(self):
        if self.type == "cuda":
            torch.cuda.synchronize()

    def info(self) -> dict:
        d = {
            "device": str(self.device),
            "device_type": self.type,
            "device_name": platform.processor() or "cpu",
            "vram_mb": 0,
            "cuda_version": None,
            "torch_version": torch.__version__,
            "amp_enabled": self.amp_enabled,
            "amp_dtype": str(self.amp_dtype) if self.amp_enabled else None,
            "fallback_used": self.preference not in ("auto",) and self.type != self.preference,
        }
        if self.type == "cuda":
            p = torch.cuda.get_device_properties(0)
            d.update(device_name=p.name, vram_mb=int(p.total_memory / 2**20), cuda_version=torch.version.cuda,
                     compute_capability=f"{p.major}.{p.minor}")
        elif self.type == "mps":
            d["device_name"] = "Apple MPS"
        return d

    def utilization(self) -> dict:
        u = {"timestamp": time.time(), "device": self.type}
        if self.type == "cuda":
            u["mem_allocated_mb"] = round(torch.cuda.memory_allocated() / 2**20, 1)
            u["mem_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
            try:
                u["gpu_util_pct"] = torch.cuda.utilization()
            except Exception:
                u["gpu_util_pct"] = None
        return u

    def write_telemetry(self, extra: Optional[dict] = None) -> dict:
        data = {**self.info(), "utilization": self.utilization(), **(extra or {})}
        path = out_dir("metrics", "gpu_telemetry.json")
        existing = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except Exception:
                existing = {}
        existing.update(data)
        path.write_text(json.dumps(existing, indent=2, default=str))
        return existing


_DM: Optional[DeviceManager] = None


def get_device_manager(preference: Optional[str] = None) -> DeviceManager:
    global _DM
    if _DM is None or (preference is not None and preference != _DM.preference):
        if preference is None:
            from .config import agent_config
            preference = agent_config().get("device", "auto")
        _DM = DeviceManager(preference)
    return _DM
