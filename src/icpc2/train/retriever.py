"""Contrastive fine-tune of a multilingual embedding model.

Training pairs: (example text, canonical description of its code). MultipleNegativesRankingLoss
uses the other codes in the batch as negatives; the NO_DUPLICATES sampler keeps two examples of the
same code out of one batch (otherwise they'd be false negatives).

The saved artifact contains the model + an index (embeddings of code anchors and every train
example), so serving = embed query -> cosine kNN -> aggregate per code.
"""

import json
import logging
from pathlib import Path

import numpy as np
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

from icpc2.codes import load_codes
from icpc2.config import load_config
from icpc2.train.common import load_examples, pick_device, run_dir, write_model_card

log = logging.getLogger(__name__)


def anchor_text(code) -> str:
    """Canonical passage describing a code (PT title + inclusions + EN title)."""
    parts = [code.title_pt]
    if code.inclusion:
        parts.append("; ".join(code.inclusion[:6]))
    if code.title_en:
        parts.append(code.title_en)
    return " | ".join(parts)


def build_index(model: SentenceTransformer, cfg: dict, examples) -> dict:
    codes = load_codes()
    a_texts = [cfg["doc_prefix"] + anchor_text(c) for c in codes.values()]
    a_codes = list(codes.keys())
    e_texts = [cfg["query_prefix"] + t for t in examples["text"]]
    e_codes = examples["code"].tolist()
    emb = model.encode(a_texts + e_texts, batch_size=256, normalize_embeddings=True,
                       show_progress_bar=True, convert_to_numpy=True).astype(np.float16)
    weights = [cfg["anchor_weight"]] * len(a_codes) + [1.0] * len(e_codes)
    return {"emb": emb, "codes": np.array(a_codes + e_codes), "weights": np.array(weights, np.float16)}


def train(model_name: str | None = None, epochs: int | None = None, limit: int | None = None) -> Path:
    cfg = load_config("retriever")
    model_name = model_name or cfg["model"]
    if epochs:
        cfg["epochs"] = epochs
    device = pick_device(cfg["device"])
    log.info("Retriever %s on %s", model_name, device)

    codes = load_codes()
    df = load_examples()
    train_df = df[df["split"] == "train"]
    if limit:
        train_df = train_df.sample(min(limit, len(train_df)), random_state=0)

    pairs = Dataset.from_dict({
        "anchor": [cfg["query_prefix"] + t for t in train_df["text"]],
        "positive": [cfg["doc_prefix"] + anchor_text(codes[c]) for c in train_df["code"]],
    })

    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = cfg["max_length"]
    out = run_dir(cfg["output_root"], model_name)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        learning_rate=cfg["lr"],
        warmup_ratio=cfg["warmup_ratio"],
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="no",
        logging_steps=50,
        seed=cfg["seed"],
        report_to="none",
    )
    SentenceTransformerTrainer(model=model, args=args, train_dataset=pairs,
                               loss=MultipleNegativesRankingLoss(model)).train()

    final = out / "model"
    model.save(str(final))
    idx = build_index(model, cfg, df[df["split"] == "train"])
    np.savez_compressed(final / "index.npz", **idx)
    write_model_card(final, "retriever", model_name, cfg, {"n_train": len(train_df)})
    (final / "retriever_cfg.json").write_text(json.dumps(cfg))
    log.info("Saved retriever -> %s", final)
    return final
