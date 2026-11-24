"""Local OpenAI-compatible LLM client (qwen3-vl-30b-a3b-instruct) with port auto-probe and mock fallback.

The LLM is ADVISORY ONLY: its verdicts can add HITL escalation but can never relax a guardrail.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, Field, ValidationError

from . import mock, prompts


class Hypothesis(BaseModel):
    cause: str
    evidence: str = ""
    confidence: float = Field(ge=0, le=1, default=0.5)


class Diagnosis(BaseModel):
    summary: str
    hypotheses: list[Hypothesis] = []
    risk: Literal["low", "medium", "high"] = "low"
    recommend_escalation: bool = False


class VisualVerdict(BaseModel):
    verdict: Literal["ok", "anomaly", "escalate"] = "ok"
    findings: list[str] = []
    confidence: float = Field(ge=0, le=1, default=0.5)


def _extract_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.S)
    if m:
        text = m.group(1)
    else:
        i, j = text.find("{"), text.rfind("}")
        if i != -1 and j > i:
            text = text[i:j + 1]
    return json.loads(text)


def _image_data_uri(path: str | Path, max_side: int = 768) -> str:
    from PIL import Image
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class LocalLLM:
    def __init__(self, cfg: dict):
        self.cfg = cfg["llm"]
        self.model = self.cfg["model"]
        self.enabled = self.cfg.get("enabled", True) and not os.environ.get("DISABLE_LLM")
        self.timeout = self.cfg.get("timeout_s", 180)
        self.base_url: str | None = None
        self.last_probe: dict = {}

    # ---------------------------------------------------------------- discovery
    def candidates(self) -> list[str]:
        first = os.environ.get("LOCAL_LLM_BASE_URL") or self.cfg.get("base_url")
        urls = [first.strip().rstrip("/")] if first else []
        for port in self.cfg.get("probe_ports", [1234, 8000, 11434, 8080]):
            u = f"http://127.0.0.1:{port}"
            if u not in urls:
                urls.append(u)
        return urls

    def probe(self, force: bool = False) -> str | None:
        if self.base_url and not force:
            return self.base_url
        self.base_url, self.last_probe = None, {}
        if not self.enabled:
            return None
        for u in self.candidates():
            try:
                r = httpx.get(f"{u}/v1/models", timeout=2.0)
                ids = [m.get("id") for m in r.json().get("data", [])]
                self.last_probe[u] = ids
                if self.model in ids:
                    self.base_url = u
                    return u
            except Exception as e:
                self.last_probe[u] = f"unreachable: {type(e).__name__}"
        return None

    @property
    def available(self) -> bool:
        return self.probe() is not None

    # ---------------------------------------------------------------- raw chat
    def _chat(self, messages: list[dict]) -> str:
        r = httpx.post(f"{self.base_url}/v1/chat/completions", timeout=self.timeout, json={
            "model": self.model, "messages": messages, "temperature": self.cfg.get("temperature", 0.0),
            "max_tokens": self.cfg.get("max_tokens", 700)})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def _structured(self, messages: list[dict], schema_cls, retries: int = 1):
        last = None
        for _ in range(retries + 1):
            raw = self._chat(messages)
            try:
                return schema_cls(**_extract_json(raw)), raw
            except (json.JSONDecodeError, ValidationError, ValueError) as e:
                last = e
                messages = messages + [{"role": "assistant", "content": raw},
                                       {"role": "user", "content": "Invalid JSON for the schema. Reply with ONLY the corrected JSON object."}]
        raise ValueError(f"LLM output not parseable: {last}")

    # ---------------------------------------------------------------- tasks
    def diagnose(self, telemetry: dict) -> tuple[Diagnosis, dict]:
        t0 = time.perf_counter()
        if self.available:
            try:
                res, raw = self._structured(prompts.diagnose_messages(telemetry), Diagnosis)
                return res, {"source": "llm", "model": self.model, "base_url": self.base_url,
                             "latency_s": round(time.perf_counter() - t0, 2)}
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
        else:
            err = "server unavailable"
        if not self.cfg.get("use_mock_if_unavailable", True):
            raise RuntimeError(f"LLM unavailable: {err}")
        return Diagnosis(**mock.diagnose(telemetry)), {"source": "mock", "model": "deterministic-mock", "reason": err,
                                                       "latency_s": round(time.perf_counter() - t0, 3)}

    def verify_visual(self, image_paths: list[str | Path], context: dict) -> tuple[VisualVerdict, dict]:
        t0 = time.perf_counter()
        if self.available:
            try:
                content = [{"type": "text", "text": prompts.visual_prompt(context)}]
                for p in image_paths:
                    content.append({"type": "image_url", "image_url": {"url": _image_data_uri(p)}})
                res, raw = self._structured([{"role": "system", "content": prompts.VISUAL_SYSTEM},
                                             {"role": "user", "content": content}], VisualVerdict)
                return res, {"source": "llm", "model": self.model, "base_url": self.base_url,
                             "latency_s": round(time.perf_counter() - t0, 2)}
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
        else:
            err = "server unavailable"
        if not self.cfg.get("use_mock_if_unavailable", True):
            raise RuntimeError(f"LLM unavailable: {err}")
        return VisualVerdict(**mock.visual(image_paths, context)), {"source": "mock", "model": "deterministic-mock",
                                                                   "reason": err, "latency_s": round(time.perf_counter() - t0, 3)}
