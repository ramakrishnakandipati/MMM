# Autonomous Campaign & Yield Optimizer Agent

GPU-accelerated, LangGraph-orchestrated agent that maximises **Incremental Net Contribution Margin** across paid
channels and promotional levers:

```
Net Yield = (Incremental Conversions x AOV x Gross Margin %) - Ad Spend - Promo Discount Cost
```

It combines a PyTorch Hill+adstock response network, a two-head T-Learner (CATE of discounts), a batched GPU LinUCB
bandit (creative x discount-tier arms), and a CVXPY convex budget allocator, wrapped in six validation gates,
financial guardrails, human-in-the-loop approval, offline replay + GPU doubly-robust OPE, two UIs (Flask and
Streamlit) and an advisory local multimodal LLM (`qwen3-vl-30b-a3b-instruct`).

> **Important:** all results below come from a **synthetic world with known ground truth** (see
> `src/data/synthetic_generator.py`). They demonstrate that the system works end to end, not real-world uplift.

## Control flow

<!-- BEGIN:FLOW -->
```mermaid
flowchart LR
    subgraph S1[Step 1 - Data]
      A[Synthetic world<br/>ads + inventory + COGS<br/>counterfactual truth] --> G1{{Gate 1<br/>integrity}}
    end
    subgraph S2[Step 2 - Features]
      G1 --> B[Adstock + supply merge<br/>train / live / OOS split] --> G2{{Gate 2<br/>schema + PSI<0.25}}
    end
    subgraph S3[Step 3 - GPU models]
      G2 --> C1[Hill+adstock net<br/>AdamW + cosine]
      G2 --> C2[T-Learner CATE]
      G2 --> C3[Batched LinUCB]
      C1 & C2 & C3 --> G3{{Gate 3<br/>R2, concavity,<br/>CATE var, cov}}
      G3 -.telemetry.-> L1[(Qwen3-VL<br/>diagnose - advisory)]
    end
    subgraph S4[Step 4 - Decide]
      G3 --> D[CVXPY LP per discount tier<br/>+ guardrail clamp] --> G4{{Gate 4<br/>optimal, shift<=15%,<br/>inventory, margin}}
    end
    subgraph S5[Step 5 - Evaluate]
      G4 --> E[Replay vs static/heuristic<br/>5 regimes + GPU OPE IPS/DR] --> G5{{Gate 5<br/>CI>0, lift>=8%}}
    end
    subgraph S6[Step 6 - Execute]
      G5 --> V[Visual QA<br/>Qwen3-VL on charts] --> H{HITL?}
      H -- approve / auto --> X[Dry-run ad + pricing APIs] --> G6{{Gate 6<br/>artifact integrity}}
      H -- reject --> HALT((HALT))
    end
    G1 & G2 & G3 & G4 & G5 & G6 -. critical fail .-> HALT
```
<!-- END:FLOW -->

## Live validation status

<!-- BEGIN:GATES -->
| Gate | Stage | Result | Checks (pass/total) | Failed |
|---|---|---|---|---|
| 1 | Raw & Synthetic Data Integrity | PASS | 7/7 | - |
| 2 | Feature Engineering & Drift Check | PASS | 5/5 | - |
| 3 | Model Calibration & Convergence | PASS | 11/11 | - |
| 4 | Optimization & Guardrail Compliance | PASS | 8/8 | - |
| 5 | Offline Evaluation & OPE Significance | PASS | 7/7 | - |
| 6 | Execution & Output Artifact Integrity | PASS | 8/8 | - |

**Pass rate: 6/6 gates (100%)**
<!-- END:GATES -->

## Live metrics

<!-- BEGIN:METRICS -->
_Last run: 2026-11-24 21:38 UTC on `NVIDIA GeForce RTX 3090` (24575 MB VRAM, CUDA 12.8)_

| KPI | Value |
|---|---|
| Net yield lift vs static | **+36.4%** |
| Net yield lift vs heuristic | **+18.5%** |
| Target (>= +8%) met | yes |
| OPE DR lift (95% CI) | +27.9% [+27.1%, +28.6%] |
| OPE IPS lift | +19.4% |
| GPU bootstrap (10,000 resamples) | 35 ms |
| Guardrail violations in replay | 0 |
| Plan latency mean / p95 | 11.3 / 18.3 ms |

| Regime | Agent lift vs static | Agent lift vs heuristic | Violations |
|---|---|---|---|
| baseline | +38.7% | +18.0% | 0 |
| demand_shock | +41.4% | +22.8% | 0 |
| inventory_crunch | +6.5% | +2.7% | 0 |
| high_cac | +44.0% | +19.0% | 0 |
| margin_squeeze | +103.0% | +53.9% | 0 |

![benchmark](outputs/charts/benchmark_regimes.png)
![ope](outputs/charts/ope_lift.png)
![response](outputs/charts/response_curves.png)
<!-- END:METRICS -->

## Data lineage

<!-- BEGIN:LINEAGE -->
```mermaid
flowchart TD
    W[World params<br/>beta,K,S,lambda,organic] --> RAW[(raw_daily.parquet)]
    RAW --> FEAT[(features_snapshot.parquet)]
    FEAT --> RM[ResponseModel] --> CURVES[steady-state curves] --> OPT[BudgetOptimizer]
    UPL[(uplift cohorts)] --> UM[UpliftModel] --> OPT
    BLOG[(bandit log + propensities)] --> LU[BatchedLinUCB] --> OPT
    OPT --> PLAN[AllocationPlan] --> GR[GuardrailEngine] --> PLAN2[guardrailed plan]
    PLAN2 --> SIM[Counterfactual simulator] --> REPLAY[(counterfactual_replay.parquet)]
    BLOG --> OPE[OPE IPS/DR + GPU bootstrap] --> OPEJ[(ope_replay_results.json)]
    REPLAY --> BENCH[(benchmark_comparison.csv)]
```
<!-- END:LINEAGE -->

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate           # Python 3.10+ (developed on 3.12)
pip install torch --index-url https://download.pytorch.org/whl/cu128   # or the CPU wheel
pip install -r requirements.txt
playwright install chromium                               # optional; run_all falls back to Edge/Chrome
python run_all.py                                         # full pipeline + gates + charts + UI screenshots + docs
python run_all.py --device cpu --skip-ui-shots            # CPU-only, no browsers
pytest -q                                                 # tests (CPU always; CUDA tests skip without a GPU)
python -m src.api.flask_app                               # Flask portal   http://127.0.0.1:5000
streamlit run src/ui/streamlit_app.py                     # Streamlit studio
```

`run_all.py` exits non-zero if any gate fails or the +8% yield target is missed.

## Local LLM (advisory only)

`src/llm/client.py` talks to an OpenAI-compatible server. Base URL: `LOCAL_LLM_BASE_URL` (default
`http://127.0.0.1:1234`); if that does not list `qwen3-vl-30b-a3b-instruct` it probes ports 1234, 8000, 11434, 8080.
Two uses: (1) text root-cause diagnosis of telemetry after Gate 3, (2) visual QA of the response-curve and allocation
charts before execution. Output is Pydantic-validated JSON; **it can only add HITL escalations, never relax a
guardrail**, and every call is logged to `outputs/logs/agent_decision_audit.jsonl` with `source: llm|mock`. If the
server is down a deterministic mock is used so CI and gate pass-rate never depend on it.

## Screenshots

<!-- BEGIN:SCREENSHOTS -->
![flask_01_idle](outputs/screenshots/flask_01_idle.png)
![flask_02_complete](outputs/screenshots/flask_02_complete.png)
![streamlit_01_hitl_pending](outputs/screenshots/streamlit_01_hitl_pending.png)
![streamlit_02_flow](outputs/screenshots/streamlit_02_flow.png)
![streamlit_03_validation](outputs/screenshots/streamlit_03_validation.png)
![streamlit_04_gpu](outputs/screenshots/streamlit_04_gpu.png)
<!-- END:SCREENSHOTS -->

## Layout

See `ARCHITECTURE_AND_DATA_FLOW.md` for topology, data dictionary, math, and per-step contracts.
Config: `config/agent_config.yaml`, `config/guardrails.yaml`. Artifacts: `outputs/`.
