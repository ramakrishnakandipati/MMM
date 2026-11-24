import copy
import os
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
os.environ["DISABLE_LLM"] = "1"          # tests use the deterministic mock; live-LLM test opts in explicitly


@pytest.fixture(scope="session", autouse=True)
def _isolated_outputs(tmp_path_factory):
    os.environ["OUTPUT_DIR"] = str(tmp_path_factory.mktemp("outputs"))
    yield


@pytest.fixture(scope="session")
def cfg():
    from src.utils.config import agent_config
    return copy.deepcopy(agent_config())


@pytest.fixture(scope="session")
def dm():
    from src.utils.device import DeviceManager
    return DeviceManager("auto")


@pytest.fixture(scope="session")
def world_data(cfg):
    from src.data.synthetic_generator import generate_all
    return generate_all(cfg)


@pytest.fixture(scope="session")
def pipeline_run():
    """One full auto-approved pipeline run shared by tests."""
    from src.agent.graph import AgentService
    return AgentService.start(auto_approve=True)
