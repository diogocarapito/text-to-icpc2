"""Hugging Face Hub I/O.

* Model repo (public): production model on `main`, every promoted version tagged vN.
* Feedback repo (PRIVATE): user interaction logs — shared between the VPS scheduler, your Mac and
  the GPU droplet. User-submitted text never goes to a public repo.
* Dataset repo (PUBLIC): the built training/eval table without user text (`push_public_dataset`).
"""

import json
import logging
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from icpc2.config import load_config, path

log = logging.getLogger(__name__)

# PRIVATE repo: only what contains user-submitted text (+ the loop history that counts it).
# Everything else (synthetic data, gold set, drafts) is public in git and the public dataset repo.
DATA_FILES = [
    "data/feedback/events.jsonl",
    "data/feedback/state.json",
    "reports/loop_history.jsonl",
]


def push_data(message: str = "sync data"):
    cfg = load_config("loop")["hf"]
    api = HfApi()
    api.create_repo(cfg["dataset_repo"], repo_type="dataset", private=True, exist_ok=True)
    files = [f for f in DATA_FILES if path(f).exists()]
    for f in files:
        api.upload_file(path_or_fileobj=str(path(f)), path_in_repo=f,
                        repo_id=cfg["dataset_repo"], repo_type="dataset", commit_message=message)
    log.info("Pushed %d data files to %s", len(files), cfg["dataset_repo"])


def pull_data():
    cfg = load_config("loop")["hf"]
    local = snapshot_download(cfg["dataset_repo"], repo_type="dataset", allow_patterns=DATA_FILES)
    for f in DATA_FILES:
        src = Path(local) / f
        if src.exists():
            dst = path(f)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
    log.info("Pulled data from %s", cfg["dataset_repo"])


def champion_version() -> int:
    cfg = load_config("loop")["hf"]
    try:
        refs = HfApi().list_repo_refs(cfg["model_repo"])
    except Exception:
        return 0
    vs = [int(t.name[1:]) for t in refs.tags if t.name.startswith("v") and t.name[1:].isdigit()]
    return max(vs, default=0)


def push_model(model_dir: Path, version: int, metrics: dict) -> str:
    cfg = load_config("loop")["hf"]
    api = HfApi()
    api.create_repo(cfg["model_repo"], private=cfg["private"], exist_ok=True)
    card = json.loads((model_dir / "icpc2_model.json").read_text())
    card["version"] = f"v{version}"
    card["eval"] = metrics
    (model_dir / "icpc2_model.json").write_text(json.dumps(card, indent=2, ensure_ascii=False))
    _write_readme(model_dir, card)
    api.upload_folder(folder_path=str(model_dir), repo_id=cfg["model_repo"],
                      commit_message=f"v{version}: {card['type']} {card['base_model']}",
                      ignore_patterns=["checkpoints/*"])
    api.create_tag(cfg["model_repo"], tag=f"v{version}")
    log.info("Promoted v%d to %s", version, cfg["model_repo"])
    return f"v{version}"


def _write_readme(d: Path, card: dict):
    gold = card["eval"].get("splits", {})
    lines = [
        "---", "language: [pt, en]", "license: apache-2.0", "pipeline_tag: text-classification",
        "tags: [icpc-2, medical, primary-care, portuguese]", f"base_model: {card['base_model']}", "---",
        f"# text-to-icpc2 {card['version']}",
        "Maps free-text clinical problems/diagnoses (PT-PT and English) to ICPC-2 codes "
        "(686 symptom + diagnosis codes). Research demo — not a medical device.",
        f"\nModel type: **{card['type']}**, base: `{card['base_model']}`.\n",
        "| split | n | top-1 | top-3 | top-5 | macro-F1 |", "|---|---|---|---|---|---|",
    ]
    for s, m in gold.items():
        if m.get("n"):
            lines.append(f"| {s} | {m['n']} | {m['top1']:.3f} | {m['top3']:.3f} | {m['top5']:.3f} "
                         f"| {m['macro_f1']:.3f} |")
    (d / "README.md").write_text("\n".join(lines) + "\n")


def push_public_dataset():
    """Publish the training/eval table WITHOUT user feedback rows, with a dataset card."""
    import pandas as pd

    cfg = load_config("loop")["hf"]
    df = pd.read_parquet(path(load_config("data")["output"]))
    df = df[~df["source"].str.startswith("feedback_")]
    out = path("data/processed/public")
    out.mkdir(parents=True, exist_ok=True)
    for split, g in df.groupby("split"):
        g.drop(columns="split").to_parquet(out / f"{split}.parquet", index=False)
    (out / "README.md").write_text(DATASET_CARD.format(n=len(df), codes=df["code"].nunique(),
                                   splits=df["split"].value_counts().to_string()))
    api = HfApi()
    api.create_repo(cfg["public_dataset_repo"], repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=cfg["public_dataset_repo"], repo_type="dataset",
                      commit_message="update dataset")
    log.info("Published %d rows to %s", len(df), cfg["public_dataset_repo"])


DATASET_CARD = """---
language: [pt, en]
license: apache-2.0
task_categories: [text-classification]
tags: [icpc-2, medical, primary-care, portuguese, synthetic]
---
# text-to-icpc2 v2 dataset

Free-text clinical problems/diagnoses (European Portuguese and English) labelled with ICPC-2 codes
(686 symptom + diagnosis codes). Built by https://github.com/diogocarapito/text-to-icpc2-v2.

Sources: ACSS ICPC-2 (PT-PT) titles and inclusion terms; ACSS ICD-10-CM PT/EN descriptions mapped
through the official ICPC-2 -> ICD-10 table (ambiguous mappings removed); LLM-generated variants
(v1: gpt-4o-mini; v2: Claude, styles clinical/shorthand/lay/qualified/typo); gold test set drafted by
Claude and verified by a family physician. Contains no patient data and no user-submitted text.

{n} rows, {codes} codes. Splits:

```
{splits}
```
"""
