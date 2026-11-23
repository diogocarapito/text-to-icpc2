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


class FakeRetriever(FakePredictor):
    type = "retriever"

    def __init__(self):
        super().__init__()
        self.added = []

    def add_examples(self, texts, codes):
        self.added += list(zip(texts, codes))
        return len(texts)


def fb(client, pred: dict, kind: str, code: str):
    return client.post("/api/feedback", json={"request_id": pred["request_id"], "feedback_key": pred["feedback_key"],
                                              "kind": kind, "code": code})


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/t.sqlite3")
    monkeypatch.setenv("ICPC2_ADMIN_TOKEN", "secret")
    import icpc2.web.main as m

    m = importlib.reload(m)
    m.real_load_model = m.load_model
    monkeypatch.setattr(m, "load_model", lambda: None)
    m.S.predictor = FakePredictor()
    with TestClient(m.app) as c:
        yield c


def test_predict_feedback_export(client, monkeypatch):
    r = client.post("/api/predict", json={"text": "HTA mal controlada, utente 123456789", "lang": "pt"})
    assert r.status_code == 200
    body = r.json()
    assert body["predictions"][0]["code"] == "K86"
    assert fb(client, body, "correct", "K86").json()["rank"] == 1
    assert fb(client, body, "corrected", "T90").json()["rank"] == 4
    assert fb(client, body, "copy", "Z99").status_code == 400

    assert client.get("/api/export").status_code == 401
    from datetime import timedelta

    import icpc2.web.main as m

    real_dt = m.datetime

    class Later(real_dt):
        @classmethod
        def now(cls, tz=None):
            return real_dt.now(tz) + timedelta(hours=1)

    monkeypatch.setattr(m, "datetime", Later)
    out = client.get("/api/export", headers={"Authorization": "Bearer secret"}).json()
    ev = out["events"]
    assert out["reviews"] == []  # the T90 correction is a deviation: pending, not exported as allowed
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
    fb(client, client.post("/api/predict", json={"text": "tensão alta", "lang": "pt"}).json(), "corrected", "K87")
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
    # the token is never accepted in the URL (browser history / logs)
    assert 'name="codigo"' in client.get("/stats?token=secret").text
    assert client.get("/api/export?token=secret").status_code == 401


def test_dashboard_numbers(client):
    import icpc2.web.main as m
    from icpc2.web import dashboard

    for kind, code in [("correct", "K86"), ("corrected", "K87"), ("copy", "T90")]:
        fb(client, client.post("/api/predict", json={"text": f"x {kind}"}).json(), kind, code)
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


def test_feedback_needs_key_and_is_deduplicated(client):
    pred = client.post("/api/predict", json={"text": "hta"}).json()
    other = client.post("/api/predict", json={"text": "hta"}).json()
    # someone else's request id with a key that is not its own
    forged = {"request_id": pred["request_id"], "feedback_key": other["feedback_key"], "kind": "corrected", "code": "T90"}
    assert client.post("/api/feedback", json=forged).status_code == 404
    assert client.post("/api/feedback", json={**forged, "feedback_key": "x"}).status_code == 422
    assert fb(client, pred, "corrected", "T90").status_code == 200
    assert fb(client, pred, "corrected", "T89").status_code == 409  # one per kind per request


def test_feedback_rate_limited(client, monkeypatch):
    import icpc2.web.main as m

    pred = client.post("/api/predict", json={"text": "hta"}).json()
    monkeypatch.setattr(m, "RATE_PER_MIN", 2)
    m._hits.clear()
    assert [fb(client, pred, k, "K86").status_code for k in ("copy", "correct", "wrong")] == [200, 200, 429]


@pytest.fixture
def retr(client):
    import icpc2.web.main as m

    m.S.predictor = FakeRetriever()
    return m.S.predictor


def test_deviation_waits_for_admin_approval(client, retr):
    """The attack from the review: one correction must not change answers for everyone."""
    for sid in ("a", "b", "c", "d"):  # even several sessions agreeing on a deviation
        client.cookies.set("sid", sid)
        assert fb(client, client.post("/api/predict", json={"text": "diabetes"}).json(), "corrected", "A01").status_code == 200
    assert retr.added == []

    client.post("/painel/entrar", data={"codigo": "secret"})
    page = client.get("/painel").text
    assert "Correções por rever" in page and "diabetes" in page and "forte" in page  # A01: other chapter
    assert client.post("/painel/rever", data={"text_key": "diabetes", "code": "A01", "text": "diabetes",
                                              "action": "approve"}).status_code == 200
    assert retr.added == [("diabetes", "A01")]
    assert "Nada por rever" in client.get("/painel").text
    reviews = client.get("/api/export").json()["reviews"]
    assert [(r["code"], r["status"]) for r in reviews] == [("A01", "approved")]


def test_rejected_and_unauthenticated_review(client, retr):
    fb(client, client.post("/api/predict", json={"text": "tosse"}).json(), "corrected", "R05")
    assert client.post("/painel/rever", data={"text_key": "tosse", "code": "R05", "action": "approve"}).status_code == 401
    client.post("/painel/entrar", data={"codigo": "secret"})
    client.post("/painel/rever", data={"text_key": "tosse", "code": "R05", "text": "tosse", "action": "reject"})
    client.post("/painel/sair")
    client.cookies.set("sid", "z")
    fb(client, client.post("/api/predict", json={"text": "tosse"}).json(), "corrected", "R05")
    assert retr.added == []


def test_agreement_needs_consensus_of_sessions(client, retr, monkeypatch):
    monkeypatch.setenv("ICPC2_CONSENSUS", "3")
    for i, sid in enumerate(("s1", "s1", "s2", "s3")):  # s1 twice counts once
        client.cookies.set("sid", sid)
        fb(client, client.post("/api/predict", json={"text": "HTA"}).json(), "correct", "K86")
        assert retr.added == ([] if i < 3 else [("HTA", "K86")])
    # a 5th agreement does not add it again
    client.cookies.set("sid", "s4")
    fb(client, client.post("/api/predict", json={"text": "hta"}).json(), "correct", "K86")
    assert len(retr.added) == 1


def test_admin_feedback_counts_as_approval(client, retr):
    client.post("/painel/entrar", data={"codigo": "secret"})
    fb(client, client.post("/api/predict", json={"text": "dm2 insulinotratada"}).json(), "corrected", "T89")
    assert retr.added == [("dm2 insulinotratada", "T89")]


def test_reviewed_labels_reapplied_on_model_load(client, monkeypatch):
    import icpc2.web.main as m
    from icpc2.web import learning

    with m.engine.begin() as c:
        learning.record(c, "dor lombar", "dor lombar", "L03", "approved", "painel")
        learning.record(c, "tosse", "tosse", "R05", "rejected", "painel")
    fake = FakeRetriever()
    monkeypatch.setattr("icpc2.predict.load_predictor", lambda *a, **k: fake)
    m.real_load_model()
    assert m.S.predictor is fake and m.S.load_error is None
    assert fake.added == [("dor lombar", "L03")]  # approved re-applied, rejected not
