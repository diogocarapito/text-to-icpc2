"""Synthetic training data with the Claude API (Message Batches = 50% price).

For every ICPC-2 code Claude writes realistic PT-PT and English phrasings in several styles
(clinician problem-list wording, abbreviations, patient words, qualified forms, typos) plus an
English rubric title. Generation respects ICPC-2 inclusion/exclusion criteria so examples
don't drift into neighbouring codes.

  icpc2 synth preview --code K86        # one synchronous call, prints the result (check quality)
  icpc2 synth submit [--codes K86,T90]  # submit a batch for all/selected codes
  icpc2 synth status                    # poll batches
  icpc2 synth collect                   # parse finished batches -> synthetic.jsonl + EN titles
"""

import json
import logging
import time
from datetime import UTC, datetime

import anthropic
import pandas as pd
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from icpc2.codes import CHAPTERS_PT, Code, load_codes
from icpc2.config import load_config, path
from icpc2.text import norm_key

log = logging.getLogger(__name__)

SYSTEM = """\
You generate training data for a classifier that maps free-text clinical problems/diagnoses, \
as written in Portuguese primary care (Medicina Geral e Familiar), to ICPC-2 codes.

Rules:
- Portuguese must be European Portuguese (PT-PT), never Brazilian. Use PT-PT clinical vocabulary \
and spelling as Portuguese GPs actually write it (e.g. "rastreio" not "rastreamento", "constipação" \
means common cold, "seguimento", "HTA", "DM tipo 2"). Both pre- and post-AO90 spellings are realistic \
("infecção"/"infeção").
- Every text must unambiguously belong to the given ICPC-2 code. Respect the inclusion terms, \
exclusion notes and criteria: if a phrasing would be coded elsewhere according to the exclusions, \
do not produce it.
- Vary vocabulary, length (1-12 words) and structure. No numbering, no quotes, no explanations, \
no code letters/numbers inside the texts. No patient identifiers.
- Styles:
  clinical  = how a GP writes it in the problem list (concise, technical)
  shorthand = abbreviations, acronyms and telegraphic notes commonly used by Portuguese GPs
  lay       = the patient's own words / reason for encounter
  qualified = clinical wording plus qualifiers that keep the same code (laterality, severity, \
chronicity, "agudizada", "controlada", "de novo", follow-up context)
  typo      = realistic typing errors, missing accents, lowercase, swapped letters
- Also give the official-style English ICPC-2 rubric title for the code (title_en).
"""

STYLES = ["clinical", "shorthand", "lay", "qualified", "typo"]

SCHEMA = {
    "type": "object",
    "properties": {
        "title_en": {"type": "string"},
        "pt": {"type": "array", "items": {
            "type": "object",
            "properties": {"text": {"type": "string"}, "style": {"type": "string", "enum": STYLES}},
            "required": ["text", "style"], "additionalProperties": False}},
        "en": {"type": "array", "items": {
            "type": "object",
            "properties": {"text": {"type": "string"}, "style": {"type": "string", "enum": STYLES}},
            "required": ["text", "style"], "additionalProperties": False}},
    },
    "required": ["title_en", "pt", "en"],
    "additionalProperties": False,
}


def _icd_hints(code: Code, n: int = 8) -> list[str]:
    p = path(load_config("data")["raw"]["icd10_slim"])
    if not p.exists() or not code.icd10:
        return []
    df = pd.read_csv(p, dtype=str).fillna("")
    hits = df[df["icd"].str.startswith(tuple(code.icd10))]
    return hits["pt"].head(n).tolist()


def build_prompt(code: Code, counts: dict) -> str:
    def fmt(lang):
        return ", ".join(f"{k}: {v}" for k, v in counts[lang].items())

    parts = [
        f"ICPC-2 code: {code.code}",
        f"Chapter: {code.chapter} ({CHAPTERS_PT.get(code.chapter, '')})",
        f"Component: {'symptom/complaint' if code.component == '1' else 'diagnosis/disease'}",
        f"Title (PT): {code.title_pt}",
    ]
    if code.inclusion:
        parts.append("Inclusion terms: " + "; ".join(code.inclusion))
    if code.exclusion:
        parts.append("Exclusions (code these elsewhere): " + code.exclusion)
    if code.criteria:
        parts.append("Criteria: " + code.criteria)
    hints = _icd_hints(code)
    if hints:
        parts.append("Related ICD-10 descriptions: " + "; ".join(hints))
    parts.append(f"\nGenerate PT-PT texts — {fmt('pt')}.\nGenerate English texts — {fmt('en')}.")
    return "\n".join(parts)


def _params(code: Code, cfg: dict) -> dict:
    return dict(
        model=cfg["model"],
        max_tokens=cfg["max_tokens"],
        system=SYSTEM,
        output_config={"effort": cfg["effort"],
                       "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": build_prompt(code, cfg["per_code"])}],
    )


def _parse(msg) -> dict | None:
    if msg.stop_reason == "refusal":
        return None
    text = next((b.text for b in msg.content if b.type == "text"), "")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------- commands
def preview(code_str: str) -> dict:
    cfg = load_config("synth")
    code = load_codes()[code_str]
    client = anthropic.Anthropic()
    with client.messages.stream(**_params(code, cfg)) as stream:
        msg = stream.get_final_message()
    log.info("usage: in=%s out=%s", msg.usage.input_tokens, msg.usage.output_tokens)
    return _parse(msg) or {"error": msg.stop_reason}


def _state(cfg) -> list[dict]:
    p = path(cfg["state_file"])
    return json.loads(p.read_text()) if p.exists() else []


def _save_state(cfg, state):
    p = path(cfg["state_file"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2))


def submit(codes: list[str] | None = None) -> str:
    cfg = load_config("synth")
    all_codes = load_codes()
    selected = [all_codes[c] for c in (codes or all_codes)]
    run = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    requests = [
        Request(custom_id=f"{c.code}-{run}", params=MessageCreateParamsNonStreaming(**_params(c, cfg)))
        for c in selected
    ]
    batch = anthropic.Anthropic().messages.batches.create(requests=requests)
    state = _state(cfg)
    state.append({"id": batch.id, "created": run, "n": len(requests), "model": cfg["model"],
                  "collected": False})
    _save_state(cfg, state)
    log.info("Submitted batch %s with %d requests", batch.id, len(requests))
    return batch.id


def status() -> list[dict]:
    cfg = load_config("synth")
    client = anthropic.Anthropic()
    out = []
    for s in _state(cfg):
        b = client.messages.batches.retrieve(s["id"])
        out.append({**s, "status": b.processing_status, "counts": b.request_counts.to_dict()})
    return out


def collect(wait: bool = False) -> int:
    cfg = load_config("synth")
    client = anthropic.Anthropic()
    codes = load_codes()
    title_keys = {norm_key(c.title_pt): c.code for c in codes.values()}
    state = _state(cfg)
    out_p, en_p = path(cfg["output"]), path(cfg["en_titles_output"])
    out_p.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    if out_p.exists():
        for line in open(out_p):
            r = json.loads(line)
            seen.add((r["code"], norm_key(r["text"])))
    en_titles = (dict(pd.read_csv(en_p, dtype=str).fillna("").values) if en_p.exists() else {})

    written = 0
    with open(out_p, "a") as f:
        for s in state:
            if s["collected"]:
                continue
            while True:
                b = client.messages.batches.retrieve(s["id"])
                if b.processing_status == "ended" or not wait:
                    break
                time.sleep(60)
            if b.processing_status != "ended":
                log.info("Batch %s still %s", s["id"], b.processing_status)
                continue
            failed = []
            for res in client.messages.batches.results(s["id"]):
                code = res.custom_id.rsplit("-", 1)[0]
                data = _parse(res.result.message) if res.result.type == "succeeded" else None
                if not data:
                    failed.append(code)
                    continue
                if data.get("title_en"):
                    en_titles[code] = data["title_en"].strip()
                for lang in ("pt", "en"):
                    for item in data.get(lang, []):
                        text = item["text"].strip().strip('"').strip()
                        k = norm_key(text)
                        # drop: too short/long, dupes, or literally another code's title
                        if not (2 <= len(text) <= 120) or (code, k) in seen:
                            continue
                        if title_keys.get(k, code) != code:
                            continue
                        seen.add((code, k))
                        f.write(json.dumps({"code": code, "text": text, "lang": lang,
                                            "style": item["style"], "model": s["model"],
                                            "batch": s["id"]}, ensure_ascii=False) + "\n")
                        written += 1
            s["collected"] = True
            s["failed"] = failed
            if failed:
                log.warning("Batch %s: %d codes failed/refused: %s", s["id"], len(failed), failed[:20])
    _save_state(cfg, state)
    pd.DataFrame(sorted(en_titles.items()), columns=["code", "title_en"]).to_csv(en_p, index=False)
    log.info("Wrote %d new synthetic examples -> %s", written, out_p)
    return written
