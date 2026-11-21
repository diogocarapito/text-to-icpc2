"""One inference interface for both model types — used by evaluation AND the web app,
so the numbers you evaluate are produced by exactly the code you serve."""

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

Prediction = list[tuple[str, float]]  # [(code, score), ...] best first


class Predictor:
    type: str
    meta: dict
    codes: dict[str, dict]
    version: str

    def predict(self, texts: list[str], k: int = 5) -> list[Prediction]:
        raise NotImplementedError


class ClassifierPredictor(Predictor):
    type = "classifier"

    def __init__(self, d: Path, device: str = "cpu", max_length: int = 64, tokenizer: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(tokenizer or d)
        self.model = AutoModelForSequenceClassification.from_pretrained(d).to(device).eval()
        self.id2label = self.model.config.id2label
        self.max_length = max_length

    def predict(self, texts, k=5, batch_size=128):
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok(texts[i:i + batch_size], truncation=True, max_length=self.max_length,
                           padding=True, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                probs = self.model(**enc).logits.float().softmax(-1)
            top = probs.topk(k, dim=-1)
            for vals, idxs in zip(top.values.tolist(), top.indices.tolist()):
                out.append([(self.id2label[j], float(v)) for v, j in zip(vals, idxs)])
        return out


class RetrieverPredictor(Predictor):
    type = "retriever"
    temperature = 0.05

    def __init__(self, d: Path, device: str = "cpu", **_):
        from sentence_transformers import SentenceTransformer

        self.cfg = json.loads((d / "retriever_cfg.json").read_text())
        self.model = SentenceTransformer(str(d), device=device)
        idx = np.load(d / "index.npz")
        self.emb = idx["emb"].astype(np.float32)
        self.idx_codes = idx["codes"]
        self.weights = idx["weights"].astype(np.float32)

    def add_examples(self, texts: list[str], codes: list[str]):
        """Make confirmed user labels effective immediately, without retraining."""
        e = self._encode(texts)
        self.emb = np.vstack([self.emb, e])
        self.idx_codes = np.concatenate([self.idx_codes, np.array(codes)])
        self.weights = np.concatenate([self.weights, np.ones(len(codes), np.float32)])

    def _encode(self, texts):
        return self.model.encode([self.cfg["query_prefix"] + t for t in texts],
                                 normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    def predict(self, texts, k=5):
        q = self._encode(texts)
        sims = (q @ self.emb.T) * self.weights
        kk = min(self.cfg["knn_k"], sims.shape[1])
        out = []
        for row in sims:
            nn = np.argpartition(-row, kk - 1)[:kk]
            best: dict[str, float] = {}
            for j in nn:  # per-code max similarity
                c = str(self.idx_codes[j])
                best[c] = max(best.get(c, -1.0), float(row[j]))
            items = sorted(best.items(), key=lambda x: -x[1])
            s = np.array([v for _, v in items]) / self.temperature
            p = np.exp(s - s.max())
            p /= p.sum()
            out.append([(c, float(pi)) for (c, _), pi in list(zip(items, p))[:k]])
        return out


def resolve(model: str, revision: str | None = None) -> Path:
    """Local directory, or a Hugging Face repo id (downloaded/cached)."""
    p = Path(model)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, revision=revision))


def load_predictor(model: str, revision: str | None = None, device: str = "cpu") -> Predictor:
    d = resolve(model, revision)
    card = d / "icpc2_model.json"
    if card.exists():
        meta = json.loads(card.read_text())
    else:  # legacy v1 checkpoint: plain HF classifier whose id2label are ICPC-2 codes
        cfgj = json.loads((d / "config.json").read_text())
        has_tok = any((d / f).exists() for f in ("tokenizer.json", "vocab.txt", "tokenizer_config.json"))
        base = cfgj.get("_name_or_path") or "neuralmind/bert-base-portuguese-cased"
        if "/" not in base or base.startswith("/"):  # a local training path, not a hub id
            base = "neuralmind/bert-base-portuguese-cased"
        meta = {"type": "classifier", "base_model": model, "config": {"max_length": 128},
                "version": "legacy", "tokenizer": None if has_tok else base}
    cls = {"classifier": ClassifierPredictor, "retriever": RetrieverPredictor}[meta["type"]]
    kw = {"tokenizer": meta["tokenizer"]} if meta.get("tokenizer") else {}
    pred = cls(d, device=device, max_length=meta["config"].get("max_length", 64), **kw)
    pred.meta = meta
    codes_file = d / "codes.json"
    if codes_file.exists():
        pred.codes = {c["code"]: c for c in json.loads(codes_file.read_text())}
    else:
        from icpc2.codes import codes_table

        pred.codes = {c["code"]: c for c in codes_table()}
    pred.version = meta.get("version") or meta.get("created", "unknown")
    log.info("Loaded %s predictor (%s) from %s", meta["type"], meta["base_model"], d)
    return pred
