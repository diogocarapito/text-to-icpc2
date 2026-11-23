import pandas as pd

from icpc2.codes import label_list, load_codes
from icpc2.loop.retrain import _gates
from icpc2.text import as_icpc2_code, hash_fraction, norm_key, scrub


def test_label_space():
    codes = load_codes()
    assert len(codes) == 686
    assert "K86" in codes and "T90" in codes
    assert not any(c.startswith("-") for c in codes)  # process codes excluded
    assert label_list() == sorted(label_list())


def test_text_helpers():
    assert norm_key("  Hipertensão  ARTERIAL! ") == "hipertensao arterial"
    assert as_icpc2_code("k 86") == "K86" and as_icpc2_code("K86 hta") is None
    assert scrub("utente 123456789 com HTA") == "utente <NUM> com HTA"
    assert scrub("tel 912 345 678, nasc. 14/03/1952") == "tel <NUM>, nasc. <DATA>"
    assert scrub("ver joao.silva@mail.pt") == "ver <EMAIL>"
    for clinical in ("TA 140/90 FC 72", "HbA1c 7.5% em 2024", "PA 140 90 72", "38.5ºC há 3 dias"):
        assert scrub(clinical) == clinical  # clinical values are kept
    assert hash_fraction("HTA", "s") == hash_fraction("hta!", "s")


def test_built_dataset_has_no_leakage():
    from icpc2.data.build import build

    df = build()
    keys = df["text"].map(norm_key)
    train = set(keys[df["split"] == "train"])
    evals = set(keys[df["split"] != "train"])
    assert not (train & evals)
    assert set(df["lang"]) <= {"pt", "en"}
    assert set(df["code"]) <= set(load_codes())


def _rep(top1, pt=None, en=None, fb=None):
    by_lang = {"pt": {"top1": pt if pt is not None else top1}, "en": {"top1": en if en is not None else top1}}
    splits = {"val": {"n": 500, "top1": top1, "by_lang": by_lang}}
    if fb is not None:
        splits["feedback_test"] = {"n": 50, "top1": fb, "by_lang": by_lang}
    return {"splits": splits}


def test_promotion_gates():
    from icpc2.config import load_config

    cfg = load_config("loop")
    assert _gates(_rep(0.80), None, cfg)[0]                           # first model always promoted
    assert _gates(_rep(0.82), _rep(0.80), cfg)[0]                     # better
    assert not _gates(_rep(0.79), _rep(0.80), cfg)[0]                 # worse
    assert not _gates(_rep(0.82, en=0.70), _rep(0.80, en=0.78), cfg)[0]   # EN regression
    assert not _gates(_rep(0.82, fb=0.60), _rep(0.80, fb=0.70), cfg)[0]   # real-user regression


def test_feedback_label_derivation(tmp_path, monkeypatch):
    import json

    from icpc2.data import feedback

    events = [
        {"id": 1, "text": "hta", "lang": "pt", "feedback": [{"kind": "copy", "code": "K86"}]},
        {"id": 2, "text": "HTA", "lang": "pt", "feedback": [{"kind": "corrected", "code": "K86"}]},
        {"id": 3, "text": "tosse", "lang": "pt", "feedback": [{"kind": "wrong", "code": "R05"}]},
        {"id": 4, "text": "K86", "lang": "pt", "feedback": [{"kind": "copy", "code": "K86"}]},
        {"id": 5, "text": "dor", "lang": "pt", "feedback": [{"kind": "correct", "code": "A01"}]},
        {"id": 6, "text": "dor!", "lang": "pt", "feedback": [{"kind": "correct", "code": "L18"}]},
    ]
    ev = tmp_path / "events.jsonl"
    ev.write_text("\n".join(json.dumps(e) for e in events))
    out = tmp_path / "labels.csv"
    rv = tmp_path / "reviews.json"  # every pair here is reviewed, so only the vote rules matter
    rv.write_text(json.dumps([{"text_key": k, "code": c, "status": "approved"}
                              for k, c in [("hta", "K86"), ("tosse", "R05"), ("k86", "K86"),
                                           ("dor", "A01"), ("dor", "L18")]]))
    monkeypatch.setattr(feedback, "path", lambda p: ev if "events" in str(p) else rv if "reviews" in str(p) else out)
    df = feedback.derive_labels()
    assert dict(zip(df["text"], df["code"])) == {"hta": "K86"}  # wrong-only, code-lookups, ties dropped
    assert isinstance(pd.read_csv(out), pd.DataFrame)



def test_promotion_gates_gold_primary():
    from icpc2.config import load_config

    cfg = load_config("loop")
    cand, champ = _rep(0.70), _rep(0.90)  # worse on val...
    for r, top1 in ((cand, 0.86), (champ, 0.85)):  # ...but better on a big enough gold set
        r["splits"]["gold_test"] = {"n": 200, "top1": top1, "by_lang": r["splits"]["val"]["by_lang"]}
    ok, reasons = _gates(cand, champ, cfg)
    assert reasons[0].startswith("gold_test") and not ok  # PT/EN slices are read from gold too: 0.70 vs 0.90
    assert _gates(_rep(0.80), _rep(0.80), cfg)[0]         # a tie passes (min_delta 0)

def test_feedback_labels_only_reviewed_one_vote_per_session(tmp_path, monkeypatch):
    import json

    from icpc2.data import feedback

    monkeypatch.setattr(feedback, "path", lambda p: tmp_path / p)
    (tmp_path / "data/feedback").mkdir(parents=True)
    ev = lambda i, text, sess, kind, code: {"id": i, "text": text, "lang": "pt", "session": sess,
                                             "feedback": [{"kind": kind, "code": code}]}
    events = [
        ev(1, "diabetes", "s1", "corrected", "A01"),   # unreviewed deviation (an attack): dropped
        ev(2, "diabetes", "s1", "corrected", "A01"),
        ev(3, "dm2 insulinotratada", "s2", "corrected", "T89"),  # approved
        ev(4, "HTA", "s3", "correct", "K86"),          # consensus
        ev(5, "HTA", "s3", "correct", "K86"),          # same session: one vote
        ev(6, "tosse", "s4", "corrected", "R05"),      # rejected
    ]
    (tmp_path / feedback.EVENTS).write_text("".join(json.dumps(e) + "\n" for e in events))
    (tmp_path / feedback.REVIEWS).write_text(json.dumps([
        {"text_key": "dm2 insulinotratada", "code": "T89", "status": "approved"},
        {"text_key": "hta", "code": "K86", "status": "consensus"},
        {"text_key": "tosse", "code": "R05", "status": "rejected"},
    ]))
    df = feedback.derive_labels().set_index("text")
    assert set(df.index) == {"dm2 insulinotratada", "HTA"}
    assert df.loc["HTA", "votes"] == 2  # one session, weight 2 for "correct"


def test_retriever_index_swap_is_atomic():
    """predict() and add_examples() from many threads must never see arrays of different sizes."""
    import threading

    import numpy as np

    from icpc2.predict import RetrieverPredictor

    rng = np.random.default_rng(0)
    r = object.__new__(RetrieverPredictor)
    r.cfg = {"knn_k": 5, "query_prefix": ""}
    r._encode = lambda texts: rng.standard_normal((len(texts), 8)).astype(np.float32)
    r._index = (rng.standard_normal((50, 8)).astype(np.float32), np.array(["K86"] * 50), np.ones(50, np.float32))
    r._base_n, r._lock, r.max_added = 50, threading.Lock(), 300
    errors = []

    def reader():
        try:
            for _ in range(300):
                r.predict(["x"], k=3)
        except Exception as e:  # pragma: no cover - would be the race
            errors.append(e)

    ts = [threading.Thread(target=reader) for _ in range(4)]
    for t in ts:
        t.start()
    added = sum(r.add_examples(["y"], ["T90"]) for _ in range(400))
    for t in ts:
        t.join()
    assert not errors
    assert added == 300 and len(r.idx_codes) == len(r.weights) == len(r.emb) == 350  # capped
