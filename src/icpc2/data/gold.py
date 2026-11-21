"""Gold test set drafting: Claude drafts realistic entries, a physician verifies every one.

Deliberately different from the training generator (synth.py):
  * one request per ICPC-2 chapter with the whole chapter's code list (model must *choose* codes,
    not paraphrase a given one)
  * messy, context-rich entries as they appear in real records
  * ~30% deliberately hard cases between confusable codes (symptom vs diagnosis, fear-of vs disease,
    complicated vs uncomplicated, cross-chapter look-alikes)
Drafts carry a rationale to speed up review. Nothing enters gold.csv until a human verifies it.
In the paper: "LLM-drafted, physician-verified test set".
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor

import anthropic

from icpc2.codes import CHAPTERS_PT, load_codes
from icpc2.config import load_config, path

log = logging.getLogger(__name__)

DRAFTS = "data/gold/drafts.jsonl"

SYSTEM = """\
You help build a TEST set for a model that maps free-text clinical problems to ICPC-2 codes, in \
Portuguese primary care (Medicina Geral e Familiar). Write entries exactly as they appear in real \
records written by different Portuguese GPs (European Portuguese, never Brazilian) and, for the English \
part, by English-speaking GPs: problem-list items, reasons for encounter, assessment lines. Real entries \
are often terse, abbreviated, have context words, missing accents, typos or unusual word order, and are \
rarely textbook titles.

Rules:
- Each entry has exactly one correct ICPC-2 code from the list provided, chosen by strict ICPC-2 rules \
(inclusion/exclusion criteria; symptom codes only when no diagnosis is stated; "medo de" codes only for \
worry without the disease).
- Avoid copying the official titles or inclusion terms verbatim.
- About 30% must be HARD: easily confused with another code (state which one in `confusable_with`).
- Spread entries over many different codes of the chapter, favouring problems common in primary care.
- No patient identifiers. One problem per entry.
"""
RATIONALE_RULE = "- `rationale`: one short sentence explaining the code choice (Portuguese or English).\n"

SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "code": {"type": "string"},
            "lang": {"type": "string", "enum": ["pt", "en"]},
            "difficulty": {"type": "string", "enum": ["normal", "hard"]},
            "confusable_with": {"type": "string"},
            "rationale": {"type": "string"},
        },
        "required": ["text", "code", "lang", "difficulty", "confusable_with", "rationale"],
        "additionalProperties": False,
    }}},
    "required": ["items"],
    "additionalProperties": False,
}


def _chapter_prompt(chapter: str, n: int, en_share: float) -> str:
    codes = [c for c in load_codes().values() if c.chapter == chapter]
    lines = []
    for c in codes:
        extra = f" (inclui: {'; '.join(c.inclusion[:4])})" if c.inclusion else ""
        lines.append(f"{c.code} {c.title_pt}{extra}")
    n_en = round(n * en_share)
    return (f"Chapter {chapter} — {CHAPTERS_PT[chapter]}. Codes:\n" + "\n".join(lines) +
            f"\n\nWrite {n} entries: {n - n_en} in PT-PT and {n_en} in English. Use `confusable_with` "
            f"= \"\" for normal entries.")


def _schema(with_rationale: bool) -> dict:
    if with_rationale:
        return SCHEMA
    import copy

    sch = copy.deepcopy(SCHEMA)
    item = sch["properties"]["items"]["items"]
    item["properties"].pop("rationale")
    item["required"].remove("rationale")
    return sch


def draft(n_per_chapter: int = 36, en_share: float = 0.2, workers: int = 6,
          with_rationale: bool = True) -> int:
    cfg = load_config("synth")
    client = anthropic.Anthropic()
    valid = set(load_codes())

    def one(ch):
        for attempt in range(3):
            # server-side fallback: if a safety classifier declines (false positives on medical
            # text happen), the API re-runs the request on a fallback model inside the same call
            with client.beta.messages.stream(
                model=cfg["model"], max_tokens=32000,
                system=SYSTEM + (RATIONALE_RULE if with_rationale else ""),
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                output_config={"effort": "high", "format": {"type": "json_schema", "schema": _schema(with_rationale)}},
                messages=[{"role": "user", "content": _chapter_prompt(ch, n_per_chapter, en_share)}],
            ) as s:
                msg = s.get_final_message()
            if msg.stop_reason != "refusal":
                break
            log.warning("chapter %s refused (attempt %d): %s", ch, attempt + 1, msg.stop_details)
        else:
            return []
        text = next(b.text for b in msg.content if b.type == "text")
        items = json.loads(text)["items"]
        ok = [dict(i, chapter=ch, code=i["code"].strip().upper(), drafted_by=msg.model)
              for i in items if i["code"].strip().upper() in valid]
        log.info("chapter %s: %d drafts by %s (%d invalid codes dropped)",
                 ch, len(ok), msg.model, len(items) - len(ok))
        return ok

    out = path(DRAFTS)
    existing = [json.loads(x) for x in open(out)] if out.exists() else []
    done = {r["chapter"] for r in existing}
    todo = [ch for ch in CHAPTERS_PT if ch not in done]
    log.info("drafting chapters: %s", "".join(todo) or "none (all done)")
    with ThreadPoolExecutor(workers) as ex:
        new_items = [x for chunk in ex.map(one, todo) for x in chunk]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        for i, r in enumerate(new_items, start=len(existing)):
            f.write(json.dumps({"id": f"g{i:04d}", **r}, ensure_ascii=False) + "\n")
    return len(existing) + len(new_items)
