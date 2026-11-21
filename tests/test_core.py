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
    monkeypatch.setattr(feedback, "path", lambda p: ev if "events" in str(p) else out)
    df = feedback.derive_labels()
    assert dict(zip(df["text"], df["code"])) == {"hta": "K86"}  # wrong-only, code-lookups, ties dropped
    assert isinstance(pd.read_csv(out), pd.DataFrame)
