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


LICENSE_NOTE = (
    "**Licensing.** Code, model weights and synthetic data: MIT. ICPC-2 is (c) WONCA International Classification "
    "Committee; the Portuguese rubrics come from the ACSS translation (ICPC-2e v4.4). They are reproduced here "
    "for research and non-commercial use. Check WONCA's terms before any commercial use.")


def _write_readme(d: Path, card: dict):
    cfg = load_config("loop")["hf"]
    splits = card["eval"].get("splits", {})
    retriever = card["type"] == "retriever"
    lines = [
        "---", "language: [pt, en]", "license: mit",
        f"library_name: {'sentence-transformers' if retriever else 'transformers'}",
        f"pipeline_tag: {'sentence-similarity' if retriever else 'text-classification'}",
        "tags: [icpc-2, medical, primary-care, portuguese, clinical-coding]",
        f"base_model: {card['base_model']}",
        f"datasets: [{cfg['public_dataset_repo']}]", "---",
        f"# text-to-icpc2 {card['version']}", "",
        "Maps a free-text clinical problem or diagnosis, as a family doctor writes it in a problem list "
        "(European Portuguese or English), to an **ICPC-2** code: 686 symptom (component 1) and diagnosis "
        "(component 7) codes. Live demo: https://text-to-icpc2.com",
        "",
        "> Research software, **not a medical device**. Every suggestion must be checked by the clinician.",
        "",
        f"- **Type:** {card['type']}"
        + (": a fine-tuned embedding model plus a kNN index over the code descriptions and every labelled example, "
           "aggregated per code. Confirmed feedback can be added to the index without retraining." if retriever
           else ": a fine-tuned encoder with a softmax head over the 686 codes."),
        f"- **Base model:** [`{card['base_model']}`](https://huggingface.co/{card['base_model']})",
        f"- **Training rows:** {card.get('n_train', 0):,} from "
        f"[{cfg['public_dataset_repo']}](https://huggingface.co/datasets/{cfg['public_dataset_repo']})",
        "- **Code, data pipeline, experiments, lab notebook:** https://github.com/diogocarapito/text-to-icpc2 "
        f"(commit `{card.get('git_sha', '?')}`)",
        "", "## Evaluation", "",
        "`gold_test` and `gold_dev` are Claude-drafted entries, each verified (or recoded or discarded) by a family "
        "physician. Neither was used for training. The model type was chosen on `gold_dev` and `val`, where it also "
        "leads. `val` is a hash split of the training distribution.", "",
        "| split | n | top-1 | top-3 | top-5 | macro-F1 | PT top-1 | EN top-1 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s, m in splits.items():
        if m.get("n"):
            bl = m.get("by_lang", {})
            pt, en = (f"{bl[k]['top1']:.3f}" if k in bl else "–" for k in ("pt", "en"))
            lines.append(f"| {s} | {m['n']} | {m['top1']:.3f} | {m['top3']:.3f} | {m['top5']:.3f} "
                         f"| {m['macro_f1']:.3f} | {pt} | {en} |")
    lines += [
        "", "For reference, the 2024 v1 model "
        "([`text-to-icpc2-bert-base-portuguese-cased`](https://huggingface.co/diogocarapito/text-to-icpc2-bert-base-portuguese-cased)) "
        "scores **0.346** top-1 on the same `gold_test` (PT 0.417, EN 0.035).", "",
        "**Known limitations:** the gold set was drafted by the same model family that generated the synthetic "
        "training data, so scores on it are likely optimistic. It was verified by a single rater. Inputs that "
        "mention several problems at once (\"HTA e DM2\") get one code. Process codes (-30 to -69) are out "
        "of scope.", "",
        "## Usage", "", "```bash",
        "pip install 'icpc2[infer] @ git+https://github.com/diogocarapito/text-to-icpc2'", "```", "```python",
        "from icpc2.predict import load_predictor",
        f"p = load_predictor(\"{cfg['model_repo']}\", revision=\"{card['version']}\")",
        "p.predict([\"HTA não controlada\", \"type 2 diabetes\"], k=3)  # [[(code, score), ...], ...]",
        "```", "",
        LICENSE_NOTE, "",
    ]
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
                                   splits=df["split"].value_counts().to_string(),
                                   license=LICENSE_NOTE))
    api = HfApi()
    api.create_repo(cfg["public_dataset_repo"], repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=cfg["public_dataset_repo"], repo_type="dataset",
                      commit_message="update dataset")
    log.info("Published %d rows to %s", len(df), cfg["public_dataset_repo"])


DATASET_CARD = """---
language: [pt, en]
license: mit
task_categories: [text-classification]
tags: [icpc-2, medical, primary-care, portuguese, synthetic]
---
# text-to-icpc2 v2 dataset

Free-text clinical problems/diagnoses (European Portuguese and English) labelled with ICPC-2 codes
(686 symptom + diagnosis codes). Built by https://github.com/diogocarapito/text-to-icpc2.

Sources: ACSS ICPC-2 (PT-PT) titles and inclusion terms; ACSS ICD-10-CM PT/EN descriptions mapped
through the official ICPC-2 -> ICD-10 table (ambiguous mappings removed); LLM-generated variants
(v1: gpt-4o-mini; v2: Claude, styles clinical/shorthand/lay/qualified/typo); gold test set drafted by
Claude and verified by a family physician. Contains no patient data and no user-submitted text.

Columns: `text`, `code` (ICPC-2), `lang` (pt/en), `source` (provenance), plus metadata. `gold_test` and
`gold_dev` are the physician-verified evaluation sets. Never train on them.

{n} rows, {codes} codes. Splits:

```
{splits}
```

{license}
"""
