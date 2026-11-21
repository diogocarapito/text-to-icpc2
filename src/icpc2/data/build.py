"""Build the unified training/eval table: data/processed/examples.parquet

Columns: text, code, lang (pt|en), source, split (train|val|feedback_test|gold_dev|gold_test)

Split policy (the main fix over v1, which evaluated on its own training data):
  * official descriptions (ICPC-2 titles/inclusions, ICD-10 texts) -> always train; they define the codes
  * synthetic + legacy v1 augmentation -> train / val by a hash of the normalised text
  * user feedback -> train / feedback_test by hash (real-world proxy test that grows over time)
  * gold (hand-labelled) -> gold_dev / gold_test, never trained on
  * any train row whose normalised text also appears in an eval split is removed (leakage guard)
"""

import json
import logging
from collections import defaultdict

import pandas as pd

from icpc2.codes import load_codes
from icpc2.config import load_config, path
from icpc2.text import hash_fraction, norm_key

log = logging.getLogger(__name__)

OFFICIAL = {"icpc2_title", "icpc2_short", "icpc2_inclusion", "icpc2_title_en", "icd10_pt", "icd10_en"}
ICD_COLS = {
    "Código ICD-10-CM": "icd",
    "Válido": "valid",
    "Descrição PT_(Longa)": "pt",
    "Long_Descp ICD-10-CM": "en",
    "Ano inicio ": "year_start",
}


# ---------------------------------------------------------------- ICD-10
def build_icd10_slim(cfg: dict) -> pd.DataFrame:
    """Reduce the 54 MB ACSS ICD-10-CM file to the rows reachable from ICPC-2 mappings."""
    slim_p = path(cfg["raw"]["icd10_slim"])
    full_p = path(cfg["raw"]["icd10_full"])
    if slim_p.exists() and (not full_p.exists() or slim_p.stat().st_mtime >= full_p.stat().st_mtime):
        return pd.read_csv(slim_p, dtype=str).fillna("")
    if not full_p.exists():
        raise FileNotFoundError(f"Need {full_p} (ACSS ICD-10-CM export) or {slim_p}")

    df = pd.read_csv(full_p, dtype=str, usecols=list(ICD_COLS)).rename(columns=ICD_COLS).fillna("")
    df = df[df["valid"] == "1"].sort_values("year_start").drop_duplicates("icd", keep="last")
    prefixes = {icd for c in load_codes().values() for icd in c.icd10}
    keep = df["icd"].map(lambda x: any(x[:n] in prefixes for n in range(3, len(x) + 1)))
    out = df.loc[keep, ["icd", "pt", "en"]].reset_index(drop=True)
    out.to_csv(slim_p, index=False)
    log.info("ICD-10 slim file: %d rows -> %s", len(out), slim_p)
    return out


def icd10_examples(cfg: dict) -> pd.DataFrame:
    icfg = cfg["icd10"]
    icd_to_icpc: dict[str, set[str]] = defaultdict(set)
    for c in load_codes().values():
        for icd in c.icd10:
            icd_to_icpc[icd].add(c.code)

    rows, ambiguous = [], 0
    for r in build_icd10_slim(cfg).itertuples(index=False):
        # longest mapped prefix wins: exact code first, then (optionally) its parents
        lengths = range(len(r.icd), 2, -1) if icfg["include_children"] else [len(r.icd)]
        targets = next((icd_to_icpc[r.icd[:n]] for n in lengths if r.icd[:n] in icd_to_icpc), None)
        if not targets:
            continue
        if len(targets) > 1:
            ambiguous += 1
            if icfg["drop_ambiguous"]:
                continue
        for code in targets:
            if r.pt:
                rows.append((r.pt, code, "pt", "icd10_pt"))
            if r.en:
                rows.append((r.en, code, "en", "icd10_en"))
    log.info("ICD-10: %d ambiguous ICD codes (map to >1 ICPC-2 code)%s", ambiguous,
             " dropped" if icfg["drop_ambiguous"] else "")

    df = pd.DataFrame(rows, columns=["text", "code", "lang", "source"])
    df = df.drop_duplicates(["text", "code"])
    cap = icfg["max_per_code_per_lang"]
    # shortest texts first: they look most like what clinicians type
    df = df.assign(_len=df["text"].str.len()).sort_values("_len")
    return df.groupby(["code", "lang"]).head(cap).drop(columns="_len")


# ---------------------------------------------------------------- ICPC-2
def icpc2_examples() -> pd.DataFrame:
    rows = []
    for c in load_codes().values():
        rows.append((c.title_pt, c.code, "pt", "icpc2_title"))
        if c.short_pt and c.short_pt != c.title_pt:
            rows.append((c.short_pt, c.code, "pt", "icpc2_short"))
        rows += [(t, c.code, "pt", "icpc2_inclusion") for t in c.inclusion]
        if c.title_en:
            rows.append((c.title_en, c.code, "en", "icpc2_title_en"))
    return pd.DataFrame(rows, columns=["text", "code", "lang", "source"])


# ---------------------------------------------------------------- other sources
def legacy_examples(cfg: dict) -> pd.DataFrame:
    p = path(cfg["sources"]["legacy_v1"])
    if not p.exists():
        return pd.DataFrame(columns=["text", "code", "lang", "source"])
    df = pd.read_csv(p, dtype=str).fillna("")
    df = df[df["include"].isin(["1", "1.0", "True", "true"])]
    return pd.DataFrame({"text": df["text"], "code": df["code"], "lang": "pt",
                         "source": "legacy_" + df["origin"].str.replace("-", "_")})


def synthetic_examples(cfg: dict) -> pd.DataFrame:
    p = path(cfg["sources"]["synthetic"])
    if not p.exists():
        return pd.DataFrame(columns=["text", "code", "lang", "source"])
    recs = [json.loads(line) for line in open(p) if line.strip()]
    df = pd.DataFrame(recs)
    df["source"] = "synth_" + df.get("style", "generic")
    return df[["text", "code", "lang", "source"]]


def feedback_examples(cfg: dict) -> pd.DataFrame:
    p = path(cfg["sources"]["feedback"])
    if not p.exists():
        return pd.DataFrame(columns=["text", "code", "lang", "source"])
    df = pd.read_csv(p, dtype=str).fillna("")
    if "lang" not in df:
        df["lang"] = "pt"
    df["lang"] = df["lang"].replace("", "pt")
    return pd.DataFrame({"text": df["text"], "code": df["code"], "lang": df["lang"],
                         "source": "feedback_" + df.get("kind", "label")})


def gold_examples(cfg: dict) -> pd.DataFrame:
    p = path(cfg["sources"]["gold"])
    if not p.exists():
        return pd.DataFrame(columns=["text", "code", "lang", "source", "split"])
    df = pd.read_csv(p, dtype=str).fillna("")
    df = df[(df["text"] != "") & (df["code"] != "")]
    split = df.get("split", pd.Series("test", index=df.index)).replace("", "test")
    return pd.DataFrame({
        "text": df["text"], "code": df["code"].str.upper().str.strip(),
        "lang": df.get("lang", "pt").replace("", "pt") if "lang" in df else "pt",
        "source": "gold_" + df.get("source", pd.Series("manual", index=df.index)).replace("", "manual"),
        "split": "gold_" + split,
    })


# ---------------------------------------------------------------- assemble
def build(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config("data")
    salt, scfg = cfg["split"]["seed"], cfg["split"]
    valid_codes = set(load_codes())

    parts = []
    off = pd.concat([icpc2_examples(), icd10_examples(cfg)])
    parts.append(off.assign(split="train"))

    aug = pd.concat([legacy_examples(cfg), synthetic_examples(cfg)])
    aug["split"] = ["val" if hash_fraction(t, salt) < scfg["val_fraction"] else "train"
                    for t in aug["text"]]
    parts.append(aug)

    fb = feedback_examples(cfg)
    fb["split"] = ["feedback_test" if hash_fraction(t, salt + "fb") < scfg["feedback_test_fraction"]
                   else "train" for t in fb["text"]]
    parts.append(fb)
    parts.append(gold_examples(cfg))

    df = pd.concat(parts, ignore_index=True)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[(df["text"].str.len() >= 2) & df["code"].isin(valid_codes)]
    df["key"] = df["text"].map(norm_key)

    # leakage guard: drop train rows that appear (normalised) in any eval split
    eval_keys = set(df.loc[df["split"] != "train", "key"])
    leak = (df["split"] == "train") & df["key"].isin(eval_keys)
    if leak.any():
        log.info("Leakage guard: removed %d train rows that appear in eval splits", leak.sum())
    df = df[~leak]

    # gold wins over feedback_test wins over val if the same text sits in several eval splits
    prio = {"gold_test": 0, "gold_dev": 1, "feedback_test": 2, "val": 3, "train": 4}
    df = (df.assign(_p=df["split"].map(prio)).sort_values("_p")
            .drop_duplicates(["key", "code"]).drop(columns="_p"))
    df = df.drop(columns="key").sort_values(["split", "code"]).reset_index(drop=True)

    out = path(cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df


def summarize(df: pd.DataFrame) -> str:
    lines = [f"{len(df):,} examples, {df['code'].nunique()} codes"]
    lines.append(df.groupby(["split", "lang"]).size().unstack(fill_value=0).to_string())
    lines.append(df.groupby("source").size().sort_values(ascending=False).to_string())
    per_code = df[df["split"] == "train"].groupby("code").size()
    lines.append(f"train examples per code: min={per_code.min()} median={int(per_code.median())} "
                 f"max={per_code.max()}  codes<15: {(per_code < 15).sum()}")
    return "\n\n".join(lines)
