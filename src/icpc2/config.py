from pathlib import Path
from typing import Any

import yaml

from icpc2 import ROOT


def load_config(name: str) -> dict[str, Any]:
    """Load configs/<name>.yaml relative to the repo root."""
    path = ROOT / "configs" / f"{name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def path(p: str | Path) -> Path:
    """Resolve a repo-relative path."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p
