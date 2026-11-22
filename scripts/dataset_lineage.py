"""How much of the v1 dataset (2024-25, repo text-to-icpc2) is in the v2 train/val data.

    python scripts/dataset_lineage.py [V1_REPO]      # default V1_REPO = ../text-to-icpc2

Texts are compared after `norm_key` (lowercase, accents and punctuation stripped), the same
normalisation the v2 leakage guard uses. Reads data/processed/examples.parquet (run `make data`
first) and writes docs/memoire/data/dataset_lineage.json.
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from icpc2.text import norm_key

v1_repo = Path(sys.argv[1] if len(sys.argv) > 1 else "../text-to-icpc2")
out = Path("docs/memoire/data/dataset_lineage.json")

v2 = pd.read_parquet("data/processed/examples.parquet")
v2["key"] = v2["text"].map(norm_key)
tv = v2[v2["split"].isin(["train", "val"])].copy()
keys, pairs = set(tv["key"]), set(zip(tv["key"], tv["code"]))
gold_keys = set(v2.loc[v2["split"].str.startswith("gold"), "key"])


def v1_overlap(df: pd.DataFrame) -> pd.DataFrame:
    df = df.fillna("").astype(str)
    df["key"] = df["text"].map(norm_key)
    df["same_text"] = df["key"].isin(keys)
    df["same_text_code"] = [(k, c) in pairs for k, c in zip(df["key"], df["code"])]
    df["in_gold"] = df["key"].isin(gold_keys)
    g = df.groupby("origin").agg(rows=("key", "size"), same_text=("same_text", "sum"),
                                 same_text_code=("same_text_code", "sum"), in_gold=("in_gold", "sum"))
    g.loc["TOTAL"] = g.sum()
    g["pct_same_text"] = (100 * g["same_text"] / g["rows"]).round(1)
    return g


result = {}
csv = v1_repo / "data/data_pre_train.csv"
v1 = pd.read_csv(csv, dtype=str)
g = v1_overlap(v1)
sha = subprocess.run(["git", "-C", str(v1_repo), "log", "-1", "--format=%h %ad", "--date=short", "--",
                      "data/data_pre_train.csv"], capture_output=True, text=True).stdout.strip()
result["v1_csv"] = {"file": "text-to-icpc2/data/data_pre_train.csv", "commit": sha,
                    "unique_texts": int(v1["text"].map(norm_key).nunique()),
                    "by_origin": g.reset_index().to_dict("records")}
print(f"v1 data_pre_train.csv ({sha}): {len(v1)} rows\n{g}\n")

hf_dir = v1_repo / "data/data_pre_train_hf"
if hf_dir.exists():
    from datasets import load_from_disk

    hf = load_from_disk(str(hf_dir)).to_pandas()
    g_hf = v1_overlap(hf)
    result["v1_hf_copy"] = {"file": "text-to-icpc2/data/data_pre_train_hf",
                            "by_origin": g_hf.reset_index().to_dict("records")}
    print(f"v1 HF-format copy: {len(hf)} rows\n{g_hf}\n")

v1_keys = set(v1["text"].fillna("").map(norm_key))
tv["family"] = tv["source"].str.extract(r"^(icd10|icpc2|legacy|synth|feedback)")[0]
tv["from_v1"] = tv["key"].isin(v1_keys)
s = tv.groupby("family").agg(rows=("key", "size"), text_also_in_v1=("from_v1", "sum"))
s.loc["TOTAL"] = s.sum()
s["pct_of_v2"] = (100 * s["rows"] / s.loc["TOTAL", "rows"]).round(1)
result["v2_train_val_by_family"] = s.reset_index().to_dict("records")
result["v2_legacy_by_split"] = tv[tv["family"] == "legacy"].groupby("split").size().to_dict()
print(f"v2 train+val: {len(tv)} rows\n{s}")

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=int) + "\n")
print(f"\n→ {out}")
