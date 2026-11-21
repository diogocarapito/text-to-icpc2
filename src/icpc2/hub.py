"""Hugging Face Hub I/O.

* Model repo (public): production model on `main`, every promoted version tagged vN.
* Data repo (PRIVATE): gold set, feedback events, synthetic data — the shared source of truth
  between your Mac and the remote GPU job. User-submitted text must never go to a public repo.
"""

import json
import logging
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from icpc2.config import load_config, path

log = logging.getLogger(__name__)

DATA_FILES = [
    "data/gold/gold.csv",
    "data/feedback/events.jsonl",
    "data/feedback/state.json",
    "data/processed/synthetic.jsonl",
    "data/processed/synth_batches.json",
    "data/raw/icpc2_en_titles.csv",
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
