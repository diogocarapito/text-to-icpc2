import json
import logging
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from icpc2.codes import codes_table
from icpc2.config import load_config, path

log = logging.getLogger(__name__)

MODEL_CARD_FILE = "icpc2_model.json"


def pick_device(pref: str = "auto") -> str:
    import torch

    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_examples() -> pd.DataFrame:
    p = path(load_config("data")["output"])
    if not p.exists():
        raise FileNotFoundError(f"{p} missing — run `icpc2 build-data` first")
    return pd.read_parquet(p)


def run_dir(root: str, model_name: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    d = path(root) / f"{stamp}-{model_name.split('/')[-1]}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return os.environ.get("ICPC2_GIT_SHA", "")[:7] or "unknown"


def write_model_card(out: Path, model_type: str, base_model: str, cfg: dict, extra: dict | None = None):
    """Metadata that makes an artifact self-describing for eval + serving."""
    card = {
        "type": model_type,
        "base_model": base_model,
        "created": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
        "config": cfg,
        **(extra or {}),
    }
    (out / MODEL_CARD_FILE).write_text(json.dumps(card, indent=2, ensure_ascii=False))
    (out / "codes.json").write_text(json.dumps(codes_table(), ensure_ascii=False))
