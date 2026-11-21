"""Automatic retrain loop.

  check   : pull new feedback from the web app, derive labels, decide if a retrain is due
  train   : build data -> train candidate -> evaluate candidate AND current champion on the
            same held-out data -> promote automatically if the gates pass
  promote : push to HF (tag vN) and tell the web app to reload

Backends
  local : everything in this process (your Mac, MPS) — used for the first trainings
  modal : `check` runs wherever you call it (or on Modal's cron), training runs on a Modal GPU
          (see icpc2/remote/modal_app.py)
"""

import json
import logging
import os
from datetime import UTC, datetime, timedelta

import httpx

from icpc2.config import load_config, path

log = logging.getLogger(__name__)


def _history() -> list[dict]:
    p = path(load_config("loop")["history"])
    return [json.loads(x) for x in open(p)] if p.exists() else []


def _append_history(rec: dict):
    p = path(load_config("loop")["history"])
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(rec) + "\n")


def check(pull: bool = True) -> dict:
    from icpc2.data import feedback

    cfg = load_config("loop")
    if pull:
        feedback.pull()
    labels = feedback.derive_labels()
    n_labels = len(labels)
    trained = [h for h in _history() if h.get("event") == "train"]
    last = trained[-1] if trained else None
    new = n_labels - (last["n_feedback_labels"] if last else 0)
    age_ok = (not last or datetime.now(UTC) - datetime.fromisoformat(last["time"])
              > timedelta(days=cfg["trigger"]["max_days_between"]))
    due = new >= cfg["trigger"]["min_new_labels"] or (last is not None and age_ok)
    res = {"n_feedback_labels": n_labels, "new_labels": new, "due": due}
    log.info("Trigger check: %s", res)
    return res


def _gates(cand: dict, champ: dict | None, cfg: dict) -> tuple[bool, list[str]]:
    p = cfg["promotion"]
    if champ is None:
        return True, ["no champion yet"]
    gold_n = cand["splits"].get("gold_test", {}).get("n", 0)
    primary_split = "gold_test" if gold_n >= p["min_gold_examples"] else "val"
    m = p["primary_metric"]
    reasons, ok = [], True
    c_val, h_val = cand["splits"][primary_split][m], champ["splits"][primary_split][m]
    if c_val < h_val + p["min_delta"]:
        ok = False
    reasons.append(f"{primary_split}.{m}: {c_val:.4f} vs champion {h_val:.4f}")
    for slice_name, tol in p["max_regression"].items():
        if slice_name.startswith("lang_"):
            lang = slice_name[5:]
            c = cand["splits"][primary_split]["by_lang"].get(lang, {}).get(m)
            h = champ["splits"][primary_split]["by_lang"].get(lang, {}).get(m)
        else:
            c = cand["splits"].get(slice_name, {}).get(m)
            h = champ["splits"].get(slice_name, {}).get(m)
        if c is not None and h is not None:
            if c < h - tol:
                ok = False
            reasons.append(f"{slice_name}: {c:.4f} vs {h:.4f} (tol {tol})")
    return ok, reasons


def train_and_promote(model_type: str | None = None, n_feedback_labels: int = 0,
                      dry_run: bool = False) -> dict:
    from icpc2 import hub
    from icpc2.data.build import build
    from icpc2.eval.evaluate import evaluate, report_dir
    from icpc2.predict import load_predictor
    from icpc2.train.common import pick_device

    cfg = load_config("loop")
    model_type = model_type or cfg["model_type"]
    df = build()

    trainer = __import__(f"icpc2.train.{model_type}", fromlist=["train"])
    cand_dir = trainer.train()
    device = pick_device("auto")
    cand_report = evaluate(load_predictor(str(cand_dir), device=device), report_dir("candidate"), df)

    champ_v = hub.champion_version()
    champ_report = None
    if champ_v:
        champ = load_predictor(cfg["hf"]["model_repo"], revision="main", device=device)
        champ_report = evaluate(champ, report_dir(f"champion-v{champ_v}"), df)

    ok, reasons = _gates(cand_report, champ_report, cfg)
    rec = {"event": "train", "time": datetime.now(UTC).isoformat(), "model_type": model_type,
           "candidate": str(cand_dir), "champion": champ_v, "promoted": False, "reasons": reasons,
           "n_feedback_labels": n_feedback_labels,
           "cand_val_top1": cand_report["splits"].get("val", {}).get("top1")}
    if ok and not dry_run:
        rec["version"] = hub.push_model(cand_dir, champ_v + 1, cand_report)
        rec["promoted"] = True
        _notify_reload()
    if cfg["hf"]["push_dataset"] and not dry_run:
        hub.push_data(f"after train {rec['time']}")
    _append_history(rec)
    log.info("Promotion %s: %s", "ACCEPTED" if rec["promoted"] else "rejected", "; ".join(reasons))
    return rec


def _notify_reload():
    cfg = load_config("loop")
    url = (os.environ.get("ICPC2_VPS_URL") or cfg["vps"]["base_url"]).rstrip("/")
    token = os.environ.get("ICPC2_ADMIN_TOKEN")
    if not token:
        log.warning("ICPC2_ADMIN_TOKEN not set — web app will pick up the model on next restart")
        return
    try:
        httpx.post(f"{url}/api/admin/reload", headers={"Authorization": f"Bearer {token}"},
                   timeout=300).raise_for_status()
        log.info("Web app reloaded the new model")
    except Exception as e:  # the model is on HF already; the app also checks on startup
        log.warning("Reload call failed: %s", e)


def run(backend: str = "local", force: bool = False, pull: bool = True, dry_run: bool = False) -> dict:
    status = check(pull=pull)
    if not (status["due"] or force):
        return {"event": "skip", **status}
    if backend == "local":
        return train_and_promote(n_feedback_labels=status["n_feedback_labels"], dry_run=dry_run)
    if backend == "modal":
        import modal

        from icpc2 import hub

        hub.push_data("pre-train sync")  # remote job reads gold/feedback from the private data repo
        fn = modal.Function.from_name("icpc2-train", "train_remote")
        rec = fn.remote(n_feedback_labels=status["n_feedback_labels"], dry_run=dry_run)
        hub.pull_data()  # bring back the updated history
        return rec
    raise ValueError(f"unknown backend {backend}")
