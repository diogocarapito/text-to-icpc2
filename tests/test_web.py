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
    page = client.get("/")
    assert page.status_code == 200
    assert "Sobre o projeto" in page.text and "/api/predict" in page.text  # sections from content/*.md


def test_painel_login_and_dashboard(client):
    rid = client.post("/api/predict", json={"text": "tensão alta", "lang": "pt"}).json()["request_id"]
    client.post("/api/feedback", json={"request_id": rid, "kind": "corrected", "code": "K87"})
    client.cookies.set("sid", "abc")
    client.post("/api/predict", json={"text": "hipertensão", "lang": "en"})

    # without the code: only the login form, no data
    r = client.get("/painel")
    assert r.status_code == 200 and 'name="codigo"' in r.text and "tensão alta" not in r.text
    assert r.headers["x-robots-tag"].startswith("noindex")
    assert client.post("/painel/entrar", data={"codigo": "nope"}).status_code == 401
    assert client.get("/api/export").status_code == 401

    r = client.post("/painel/entrar", data={"codigo": "secret"})
    assert r.status_code == 200 and "Desempenho" in r.text  # redirected to the dashboard, cookie set
    assert "tensão alta" in r.text and "K87" in r.text       # recent requests + corrections
    for rng in ("7", "90", "all", "bogus"):
        assert client.get(f"/painel?range={rng}").status_code == 200
    assert client.get("/api/export").status_code == 200       # the cookie also opens the admin API

    client.post("/painel/sair")
    assert 'name="codigo"' in client.get("/painel").text
    assert "Desempenho" in client.get("/stats?token=secret").text  # old links still work


def test_dashboard_numbers(client):
    import icpc2.web.main as m
    from icpc2.web import dashboard

    for kind, code in [("correct", "K86"), ("corrected", "K87"), ("copy", "T90")]:
        rid = client.post("/api/predict", json={"text": f"x {kind}"}).json()["request_id"]
        client.post("/api/feedback", json={"request_id": rid, "kind": kind, "code": code})
    client.post("/api/predict", json={"text": "sem feedback"})
    d = dashboard.compute(m.engine, "30", m._codes())
    k = d["kpi"]
    assert k["requests"] == 4 and k["api"] == 4 and k["labelled"] == 3
    assert k["top1"] == 1 / 3 and k["top5"] == 1.0  # K86 rank 1, K87 rank 2, T90 rank 4
    assert d["fixes"][0]["shown"] == "K86" and len(d["per_day"]) == 30


def test_rate_limit_and_hidden_admin_routes(client, monkeypatch):
    import icpc2.web.main as m

    monkeypatch.setattr(m, "RATE_PER_MIN", 2)
    m._hits.clear()
    codes = [client.post("/api/predict", json={"text": "tosse", "lang": "en"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    assert "Too many requests" in client.post("/api/predict", json={"text": "tosse", "lang": "en"}).json()["detail"]
    paths = client.get("/openapi.json").json()["paths"]
    assert "/api/predict" in paths and not any(p.startswith(("/painel", "/stats", "/api/export", "/api/admin")) for p in paths)
