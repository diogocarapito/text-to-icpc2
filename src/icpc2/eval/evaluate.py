"""Evaluate any predictor on every held-out split, sliced by language / source / chapter."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from icpc2.config import path
from icpc2.predict import Predictor
from icpc2.train.common import load_examples

log = logging.getLogger(__name__)

EVAL_SPLITS = ["val", "feedback_test", "gold_dev", "gold_test"]


def _metrics(gold: list[str], preds: list[list[str]]) -> dict:
    from sklearn.metrics import f1_score

    if not gold:
        return {"n": 0}
    hit = np.array([[g == p for p in ps[:5]] + [False] * (5 - len(ps)) for g, ps in zip(gold, preds)])
    return {
        "n": len(gold),
        "top1": float(hit[:, :1].any(1).mean()),
        "top3": float(hit[:, :3].any(1).mean()),
        "top5": float(hit[:, :5].any(1).mean()),
        "macro_f1": float(f1_score(gold, [p[0] for p in preds], average="macro", zero_division=0)),
    }


def evaluate(pred: Predictor, out_dir: Path | None = None, df: pd.DataFrame | None = None) -> dict:
    df = load_examples() if df is None else df
    df = df[df["split"].isin(EVAL_SPLITS)].reset_index(drop=True)
    if df.empty:
        raise ValueError("No eval rows — build data first")
    preds = pred.predict(df["text"].tolist(), k=5)
    df["pred"] = [[c for c, _ in p] for p in preds]
    df["score1"] = [p[0][1] for p in preds]
    df["correct"] = [p[0] == g for p, g in zip(df["pred"], df["code"])]

    report = {"model": pred.meta.get("base_model"), "type": pred.type,
              "evaluated": datetime.now(UTC).isoformat(), "splits": {}}
    for split, g in df.groupby("split"):
        s = _metrics(g["code"].tolist(), g["pred"].tolist())
        s["by_lang"] = {lang: _metrics(x["code"].tolist(), x["pred"].tolist())
                        for lang, x in g.groupby("lang")}
        s["by_chapter_top1"] = g.groupby(g["code"].str[0])["correct"].mean().round(3).to_dict()
        report["splits"][split] = s

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
        df.drop(columns=["pred"]).assign(pred=df["pred"].map(",".join)).to_csv(
            out_dir / "predictions.csv", index=False)
        errors = df[~df["correct"]].assign(pred=df["pred"].map(",".join))
        errors.sort_values("score1", ascending=False).to_csv(out_dir / "errors.csv", index=False)
        log.info("Eval report -> %s", out_dir)
    return report


def headline(report: dict) -> str:
    rows = []
    for split, s in report["splits"].items():
        if s.get("n"):
            langs = " ".join(f"{k}={v['top1']:.3f}" for k, v in s["by_lang"].items())
            rows.append(f"{split:14s} n={s['n']:5d} top1={s['top1']:.3f} top3={s['top3']:.3f} "
                        f"top5={s['top5']:.3f} macroF1={s['macro_f1']:.3f}  [{langs}]")
    return "\n".join(rows)


def report_dir(name: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return path("reports") / f"{stamp}-{name}"
