"""Fine-tune an encoder with a softmax head over all ICPC-2 codes."""

import logging
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from icpc2.codes import label_list
from icpc2.config import load_config
from icpc2.train.common import load_examples, pick_device, run_dir, write_model_card

log = logging.getLogger(__name__)


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, label_smoothing=0.0, **kw):
        super().__init__(**kw)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        w = self.class_weights.to(outputs.logits.device) if self.class_weights is not None else None
        loss = torch.nn.functional.cross_entropy(
            outputs.logits, labels, weight=w, label_smoothing=self.label_smoothing)
        return (loss, outputs) if return_outputs else loss


def class_weights(labels: np.ndarray, n: int, mode: str) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = np.bincount(labels, minlength=n).astype(float)
    counts[counts == 0] = 1.0
    w = (counts.sum() / (n * counts))
    if mode == "sqrt":
        w = np.sqrt(w)
    return torch.tensor(w / w.mean(), dtype=torch.float)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    order = np.argsort(-logits, axis=-1)
    hit = order == labels[:, None]
    return {
        "top1": hit[:, :1].any(1).mean(),
        "top3": hit[:, :3].any(1).mean(),
        "top5": hit[:, :5].any(1).mean(),
        "macro_f1": f1_score(labels, order[:, 0], average="macro", zero_division=0),
    }


def train(model_name: str | None = None, epochs: int | None = None, limit: int | None = None,
          overrides: dict | None = None) -> Path:
    cfg = load_config("classifier")
    model_name = model_name or cfg["model"]
    cfg.update(overrides or {})
    if epochs:
        cfg["epochs"] = epochs
    device = pick_device(cfg["device"])
    log.info("Classifier %s on %s", model_name, device)

    labels = label_list()
    l2i = {c: i for i, c in enumerate(labels)}
    df = load_examples()
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    if limit:  # quick smoke runs
        train_df = train_df.sample(min(limit, len(train_df)), random_state=0)
        val_df = val_df.sample(min(limit // 5 + 1, len(val_df)), random_state=0)

    tok = AutoTokenizer.from_pretrained(model_name)

    def to_ds(d):
        ds = Dataset.from_dict({"text": d["text"].tolist(), "label": d["code"].map(l2i).tolist()})
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=cfg["max_length"]),
                      batched=True, remove_columns=["text"])

    train_ds, val_ds = to_ds(train_df), to_ds(val_df)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels),
        id2label=dict(enumerate(labels)), label2id=l2i, ignore_mismatched_sizes=True,
        dtype=torch.float32)  # transformers>=5 keeps the checkpoint dtype (Albertina ships bf16)

    # same effective batch everywhere; small devices accumulate gradients instead of OOMing
    per_device = min(cfg["batch_size"], (cfg.get("max_device_batch") or {}).get(device, cfg["batch_size"]))
    accum = max(1, cfg["batch_size"] // per_device)
    out = run_dir(cfg["output_root"], model_name)
    args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=accum,
        per_device_eval_batch_size=per_device * 2,
        learning_rate=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        warmup_steps=cfg["warmup_ratio"],  # transformers>=5: float in [0,1) = ratio
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="top1",
        logging_steps=50,
        report_to=cfg["report_to"],
        seed=cfg["seed"],
        bf16=device == "cuda" and torch.cuda.is_bf16_supported(),
        use_cpu=device == "cpu",
        dataloader_pin_memory=device == "cuda",
    )
    trainer = WeightedTrainer(
        class_weights=class_weights(np.array(train_ds["label"]), len(labels), cfg["class_weighting"]),
        label_smoothing=cfg["label_smoothing"],
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics,
    )
    trainer.train()
    val_metrics = trainer.evaluate()

    final = out / "model"
    trainer.save_model(str(final))
    tok.save_pretrained(str(final))
    import shutil

    shutil.rmtree(out / "checkpoints", ignore_errors=True)  # best weights are already in final/
    write_model_card(final, "classifier", model_name, cfg,
                     {"val_metrics": {k: float(v) for k, v in val_metrics.items()},
                      "n_train": len(train_ds)})
    log.info("Saved classifier -> %s  val=%s", final, val_metrics)
    return final
