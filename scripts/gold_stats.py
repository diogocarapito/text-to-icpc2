"""Bootstrap CIs, paired differences and hard/normal split on gold_test for a sweep's saved predictions.

    python scripts/gold_stats.py benchmark-01 [reference_run]
Writes experiments/<sweep>/gold_stats.json and prints a table.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sweep = sys.argv[1]
ref = sys.argv[2] if len(sys.argv) > 2 else None
root = Path("experiments") / sweep
gold = pd.read_csv("data/gold/gold.csv")
gold = gold[gold.split.isin(["test", "gold_test"])].reset_index(drop=True)[["text", "code", "lang", "difficulty"]]
rng = np.random.default_rng(0)
B = 10_000
idx = rng.integers(0, len(gold), size=(B, len(gold)))

hits = {}
for d in sorted(root.iterdir()):
    f = d / "predictions.csv.gz"
    if not f.exists():
        continue
    p = pd.read_csv(f)[["text", "pred"]].drop_duplicates("text")
    m = gold.merge(p, on="text", how="left")
    top = m["pred"].fillna("").str.split(",")
    hits[d.name] = {
        "top1": (top.str[0] == m["code"]).to_numpy(dtype=bool),
        "top3": np.array([c in t[:3] for c, t in zip(m["code"], top)]),
        "missing": int(m["pred"].isna().sum()),
    }

ref = ref or max(hits, key=lambda r: hits[r]["top1"].mean())
out, hard = {}, (gold["difficulty"] == "hard").to_numpy()
for r, h in hits.items():
    t1 = h["top1"]
    boot = t1[idx].mean(1)
    row = {"n": len(t1), "missing": h["missing"], "top1": t1.mean(),
           "top1_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
           "top3": h["top3"].mean(),
           "top1_hard": t1[hard].mean(), "n_hard": int(hard.sum()), "top1_normal": t1[~hard].mean()}
    if r != ref:
        diff = (hits[ref]["top1"][idx].astype(float) - t1[idx]).mean(1)
        row[f"diff_vs_{ref}"] = float(hits[ref]["top1"].mean() - t1.mean())
        row["diff_ci95"] = [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]
        row["ref_only_correct"] = int((hits[ref]["top1"] & ~t1).sum())
        row["this_only_correct"] = int((~hits[ref]["top1"] & t1).sum())
    out[r] = {k: (round(float(v), 4) if isinstance(v, float | np.floating) else v) for k, v in row.items()}

(root / "gold_stats.json").write_text(json.dumps({"reference": ref, "bootstrap": B, "runs": out}, indent=2))
print(f"reference = {ref}; 95% CIs from {B} bootstrap resamples of gold_test")
for r, v in sorted(out.items(), key=lambda kv: -kv[1]["top1"]):
    d = f"  Δ={v[f'diff_vs_{ref}']:+.3f} [{v['diff_ci95'][0]:+.3f},{v['diff_ci95'][1]:+.3f}] (b={v['ref_only_correct']}, c={v['this_only_correct']})" if r != ref else ""
    print(f"{r:20s} top1={v['top1']:.3f} [{v['top1_ci95'][0]:.3f},{v['top1_ci95'][1]:.3f}] hard={v['top1_hard']:.3f} (n={v['n_hard']}) normal={v['top1_normal']:.3f} miss={v['missing']}{d}")
