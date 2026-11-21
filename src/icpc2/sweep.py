"""Benchmark sweeps: train + evaluate a list of model/hyper-parameter configs, record everything.

    icpc2 sweep configs/sweeps/benchmark-01.yaml            # locally (MPS / CUDA)
    icpc2 gpu run -- icpc2 sweep configs/sweeps/benchmark-01.yaml   # on a DigitalOcean GPU

Results go to experiments/<sweep>/ (committed to git: the public lab notebook):
  <run>/metrics.json   eval report (all held-out splits, by language and chapter)
  <run>/run.json       resolved config, timings, hardware, git sha
  <run>/predictions.csv.gz  per-row predictions -> `icpc2 rescore` re-scores them on the verified
                            gold set later without the model
  summary.csv          one row per run
"""

import gzip
import json
import logging
import platform
import shutil
import time
from pathlib import Path

import pandas as pd
import yaml

from icpc2.config import path

log = logging.getLogger(__name__)

PRIMARY = ["gold_test", "draft", "val"]  # first split present is the headline


def _hardware() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        if torch.backends.mps.is_available():
            return f"Apple MPS ({platform.machine()})"
    except Exception:
        pass
    return platform.processor() or "cpu"


def _row(name: str, run: dict, rep: dict, secs: float) -> dict:
    row = {"run": name, "kind": run["kind"], "model": run.get("model", ""),
           "overrides": json.dumps(run.get("overrides", {})), "train_min": round(secs / 60, 1)}
    for split, s in rep["splits"].items():
        if not s.get("n"):
            continue
        row[f"{split}_n"] = s["n"]
        for m in ("top1", "top3", "top5", "macro_f1"):
            row[f"{split}_{m}"] = round(s[m], 4)
        for lang, ls in s["by_lang"].items():
            row[f"{split}_{lang}_top1"] = round(ls["top1"], 4)
    return row


def run_sweep(spec_path: str, only: list[str] | None = None, keep_models: int = 1) -> pd.DataFrame:
    from icpc2.eval.evaluate import evaluate, headline
    from icpc2.predict import load_predictor
    from icpc2.train.common import git_sha, pick_device

    spec = yaml.safe_load(open(path(spec_path)))
    out_root = path("experiments") / spec["name"]
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "sweep.yaml").write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True))
    device = pick_device("auto")
    rows, model_dirs = [], {}

    for run in spec["runs"]:
        name = run["name"]
        if only and name not in only:
            continue
        rdir = out_root / name
        if (rdir / "metrics.json").exists():
            log.info("skip %s (done)", name)
            rows.append(json.loads((rdir / "row.json").read_text()))
            continue
        rdir.mkdir(parents=True, exist_ok=True)
        log.info("=== %s: %s %s %s", name, run["kind"], run.get("model"), run.get("overrides", {}))
        t0 = time.time()
        try:
            if run["kind"] == "baseline":
                model_dir = run["model"]  # an existing model (HF id or dir), evaluated only
            else:
                trainer = __import__(f"icpc2.train.{run['kind']}", fromlist=["train"])
                model_dir = str(trainer.train(run.get("model"), limit=run.get("limit"),
                                              epochs=run.get("epochs"), overrides=run.get("overrides")))
            secs = time.time() - t0
            pred = load_predictor(model_dir, device=device)
            rep = evaluate(pred, rdir)
        except Exception as e:  # one failed config must not kill a paid GPU sweep
            log.exception("run %s failed", name)
            (rdir / "error.txt").write_text(f"{type(e).__name__}: {e}")
            continue
        # keep the eval artefacts small and committed
        with open(rdir / "predictions.csv", "rb") as f, gzip.open(rdir / "predictions.csv.gz", "wb") as g:
            shutil.copyfileobj(f, g)
        (rdir / "predictions.csv").unlink()
        (rdir / "errors.csv").unlink(missing_ok=True)
        (rdir / "run.json").write_text(json.dumps({
            **run, "model_dir": model_dir, "train_seconds": round(secs), "hardware": _hardware(),
            "git_sha": git_sha(), "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2))
        row = _row(name, run, rep, secs)
        (rdir / "row.json").write_text(json.dumps(row))
        rows.append(row)
        model_dirs[name] = model_dir
        log.info("%s\n%s", name, headline(rep))
        pd.DataFrame(rows).to_csv(out_root / "summary.csv", index=False)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    primary = next((f"{s}_top1" for s in PRIMARY if f"{s}_top1" in df), None)
    if primary:
        df = df.sort_values(primary, ascending=False)
    df.to_csv(out_root / "summary.csv", index=False)
    # free disk: keep only the best `keep_models` trained artefacts
    trained = [r for r in df["run"] if r in model_dirs and Path(model_dirs[r]).exists()]
    for r in trained[keep_models:]:
        shutil.rmtree(Path(model_dirs[r]).parent, ignore_errors=True)
    if trained:
        (out_root / "best_model.txt").write_text("\n".join(model_dirs[r] for r in trained[:keep_models]))
    return df


def rescore(sweep: str) -> pd.DataFrame:
    """Re-score saved predictions against the current gold.csv (after physician verification)."""
    from icpc2.eval.evaluate import _metrics

    gold = pd.read_csv(path("data/gold/gold.csv"), dtype=str).fillna("")
    gold_map = dict(zip(gold["text"], gold["code"]))
    split_map = dict(zip(gold["text"], "gold_" + gold.get("split", "test")))
    rows = []
    for rdir in sorted((path("experiments") / sweep).iterdir()):
        p = rdir / "predictions.csv.gz"
        if not p.exists():
            continue
        df = pd.read_csv(p, dtype=str).fillna("")
        df = df[df["text"].isin(gold_map)]
        row = {"run": rdir.name}
        for split, g in df.groupby(df["text"].map(split_map)):
            for lang, x in [("all", g)] + list(g.groupby("lang")):
                m = _metrics([gold_map[t] for t in x["text"]], [p.split(",") for p in x["pred"]])
                for k in ("n", "top1", "top3", "top5", "macro_f1"):
                    row[f"{split}_{lang}_{k}"] = m.get(k)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(path("experiments") / sweep / "rescore_gold.csv", index=False)
    return out
