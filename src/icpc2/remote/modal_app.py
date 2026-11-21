"""Remote GPU training on Modal (https://modal.com) — used once the demo is live.

One-time setup:
    pip install modal && modal setup
    modal secret create icpc2 HF_TOKEN=hf_xxx ICPC2_ADMIN_TOKEN=xxx ICPC2_VPS_URL=https://text-to-icpc2.com
    modal deploy src/icpc2/remote/modal_app.py

After deploy:
  * `nightly` runs on Modal's cron on a CPU container: pulls feedback from the web app and checks
    the trigger; if due, it calls `train_remote` on a GPU. Your Mac doesn't need to be on.
  * `icpc2 loop --backend modal` triggers the same thing from your terminal.
Cost: an L4 is roughly $0.80/h; a BERT-base run on ~60k short texts is ~15-30 min.
"""

from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[3]
REMOTE_ROOT = "/root/app"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.4", "transformers>=4.48", "sentence-transformers>=3.3", "datasets>=3.0",
        "accelerate>=1.0", "scikit-learn>=1.5", "pandas>=2.2", "pyarrow>=16", "pyyaml>=6",
        "typer>=0.12", "huggingface_hub>=0.25", "httpx>=0.27", "python-dotenv>=1.0",
    )
    .env({"PYTHONPATH": f"{REMOTE_ROOT}/src", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_dir(
        REPO, remote_path=REMOTE_ROOT,
        ignore=[".git", ".venv", "models", "reports", "data/processed/*.parquet",
                "data/raw/icd10cm_pt_en.csv", "**/__pycache__"],
    )
)

app = modal.App("icpc2-train", image=image)
secrets = [modal.Secret.from_name("icpc2")]


@app.function(gpu="L4", timeout=4 * 3600, secrets=secrets)
def train_remote(n_feedback_labels: int = 0, dry_run: bool = False) -> dict:
    import logging

    from icpc2 import hub
    from icpc2.loop.retrain import train_and_promote

    logging.basicConfig(level=logging.INFO)
    hub.pull_data()
    return train_and_promote(n_feedback_labels=n_feedback_labels, dry_run=dry_run)


@app.function(schedule=modal.Cron("0 3 * * *"), timeout=1800, secrets=secrets)
def nightly() -> dict:
    import logging

    from icpc2 import hub
    from icpc2.loop.retrain import check

    logging.basicConfig(level=logging.INFO)
    hub.pull_data()
    status = check(pull=True)
    hub.push_data("nightly feedback sync")
    if status["due"]:
        return train_remote.remote(n_feedback_labels=status["n_feedback_labels"])
    return {"event": "skip", **status}
