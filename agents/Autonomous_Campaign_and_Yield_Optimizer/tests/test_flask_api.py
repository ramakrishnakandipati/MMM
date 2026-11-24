import time

import pytest

from src.api.flask_app import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app()
    app.testing = True
    return app.test_client()


def _wait(client, cond, timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = client.get("/api/status").get_json()
        if st.get("error"):
            pytest.fail(st["error"])
        if cond(st):
            return st
        time.sleep(1)
    pytest.fail("timeout waiting for state")


def test_index_and_health(client):
    r = client.get("/")
    assert r.status_code == 200 and b"Control Portal" in r.data
    assert client.get("/health").get_json() == {"ok": True}
    assert client.get("/static/style.css").status_code == 200


def test_run_hitl_flow_and_endpoints(client):
    assert client.post("/api/hitl/approve", json={}).status_code == 409   # nothing pending yet
    r = client.post("/api/run", json={"auto_approve": False})
    assert r.status_code == 202
    rid = r.get_json()["run_id"]
    st = _wait(client, lambda s: s.get("waiting_for_human"))
    assert st["run_id"] == rid and st["step"] == 5
    pend = client.get("/api/hitl/pending").get_json()
    assert pend["waiting"] and pend["pending"]["reasons"]
    ok = client.post("/api/hitl/approve", json={"reviewer": "pytest"})
    assert ok.status_code == 200 and ok.get_json()["hitl"]["status"] == "approved"
    st = _wait(client, lambda s: s["step"] == 6)
    assert all(g["passed"] for g in st["gates"].values())
    v = client.get("/api/validation").get_json()
    assert v["summary"]["all_passed"] is True
    assert client.get("/api/step/4").get_json()["passed"] is True
    assert client.get("/api/step/9").status_code == 404
    assert client.get("/api/metrics").get_json()["yield"]["target_met"] is True
    assert "device_name" in client.get("/api/telemetry").get_json()
    assert len(client.get("/api/audit").get_json()) > 5
    assert "response_curves.png" in client.get("/api/charts").get_json()
    assert client.get("/charts/response_curves.png").status_code == 200


def test_access_log_written(client):
    from src.utils.logger import read_jsonl
    client.get("/health")
    assert any(r["path"] == "/health" for r in read_jsonl("flask_access.jsonl"))
