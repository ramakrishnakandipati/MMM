"""Agent state: serialisable AgentState (checkpointed by LangGraph) + heavy RunContext (kept in memory)."""
from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, TypedDict

from ..utils.config import agent_config, guardrail_config
from ..utils.device import DeviceManager, get_device_manager


class AgentState(TypedDict, total=False):
    run_id: str
    step: int                      # last completed pipeline step (1..6)
    gates: dict                    # {"1": {passed, severity, checks...}} embedded step validation status
    halted: bool
    halt_reason: str
    plan: dict                     # guardrailed allocation proposal
    diagnosis: dict                # LLM / mock diagnostics (advisory)
    visual: dict                   # LLM / mock visual verdict (advisory)
    escalations: list              # reasons requiring human approval
    hitl: dict                     # {required, status: not_required|auto_approved|pending|approved|rejected, reviewer, note}
    evaluation: dict               # headline evaluation summary
    executed: dict                 # dry-run execution payload
    artifacts: dict


@dataclass
class RunContext:
    """Heavy, non-serialisable objects shared by all nodes of one run."""
    cfg: dict = field(default_factory=lambda: copy.deepcopy(agent_config()))
    gcfg: dict = field(default_factory=lambda: copy.deepcopy(guardrail_config()))
    dm: Optional[DeviceManager] = None
    auto_approve: bool = True
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    data: dict = field(default_factory=dict)       # world, daily, uplift, bandit_log, features, windows
    models: dict = field(default_factory=dict)     # resp, uplift, bandit, planner
    cache: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.dm is None:
            self.dm = get_device_manager(self.cfg.get("device", "auto"))
        self.cfg["hitl"] = dict(self.cfg["hitl"], auto_approve_in_batch=self.auto_approve)


def initial_state(ctx: RunContext) -> AgentState:
    return AgentState(run_id=ctx.run_id, step=0, gates={}, halted=False, halt_reason="", escalations=[],
                      hitl={"required": False, "status": "not_required"}, artifacts={})
