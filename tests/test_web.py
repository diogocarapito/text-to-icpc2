"""Web app API with a fake predictor (no torch needed)."""

import importlib

import pytest
from fastapi.testclient import TestClient


class FakePredictor:
    type = "classifier"
    version = "v-test"
    meta = {"base_model": "fake"}

    def __init__(self):
        from icpc2.codes import codes_table

        self.codes = {c["code"]: c for c in codes_table()}

    def predict(self, texts, k=5):
        return [[("K86", 0.9), ("K87", 0.05), ("K85", 0.03), ("T90", 0.01), ("A01", 0.01)]
                for _ in texts]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/t.sqlite3")
    monkeypatch.setenv("ICPC2_ADMIN_TOKEN", "secret")
    import icpc2.web.main as m

    m = importlib.reload(m)
    monkeypatch.setattr(m, "load_model", lambda: None)
    m.S.predictor = FakePredictor()
    with TestClient(m.app) as c:
        yield c


def test_predict_feedback_export(client, monkeypatch):
    r = client.post("/api/predict", json={"text": "HTA mal controlada, utente 123456789", "lang": "pt"})
    assert r.status_code == 200
    body = r.json()
    assert body["predictions"][0]["code"] == "K86"
    rid = body["request_id"]

    assert client.post("/api/feedback", json={"request_id": rid, "kind": "correct", "code": "K86"}).json()["rank"] == 1
    assert client.post("/api/feedback", json={"request_id": rid, "kind": "corrected", "code": "T90"}).json()["rank"] == 4
    assert client.post("/api/feedback", json={"request_id": rid, "kind": "copy", "code": "Z99"}).status_code == 400

    assert client.get("/api/export").status_code == 401
    from datetime import timedelta

    import icpc2.web.main as m

    real_dt = m.datetime

    class Later(real_dt):
        @classmethod
        def now(cls, tz=None):
            return real_dt.now(tz) + timedelta(hours=1)

    monkeypatch.setattr(m, "datetime", Later)
    ev = client.get("/api/export", headers={"Authorization": "Bearer secret"}).json()["events"]
    assert len(ev) == 1
    assert "<NUM>" in ev[0]["text"]  # SNS number scrubbed before storage
    assert {f["kind"] for f in ev[0]["feedback"]} == {"correct", "corrected"}


def test_code_lookup_and_stats(client):
    r = client.post("/api/predict", json={"text": "t90"})
    assert r.json()["predictions"][0]["code"] == "T90" and r.json()["model_version"] is None
    assert client.post("/api/predict", json={"text": "Q99"}).status_code == 404
    assert client.get("/stats?token=secret").status_code == 200
    assert client.get("/").status_code == 200
