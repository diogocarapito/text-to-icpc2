"""Build the gold-set review page (a claude.ai artifact with a `db` capability).

    python scripts/build_gold_review.py OUT.html

Embeds data/gold/drafts.jsonl + the ICPC-2 code list. Verdicts are stored in the artifact's
database (collection `verdicts`, one doc per draft id) and exported with `icpc2 gold-import`.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from icpc2.codes import CHAPTERS_PT, codes_table
from icpc2.config import path

drafts = [json.loads(x) for x in open(path("data/gold/drafts.jsonl"))]
items = [{k: d.get(k, "") for k in ("id", "text", "code", "lang", "difficulty", "confusable_with",
                                     "rationale", "chapter")} for d in drafts]
codes = {c["code"]: [c["title_pt"], c["title_en"], "; ".join(c["inclusion"][:5])] for c in codes_table()}

TEMPLATE = Path(__file__).with_name("gold_review_template.html").read_text()
html = (TEMPLATE
        .replace("/*ITEMS*/[]", json.dumps(items, ensure_ascii=False))
        .replace("/*CODES*/{}", json.dumps(codes, ensure_ascii=False))
        .replace("/*CHAPTERS*/{}", json.dumps(CHAPTERS_PT, ensure_ascii=False)))
Path(sys.argv[1]).write_text(html)
print(f"{len(items)} items -> {sys.argv[1]} ({len(html) // 1024} KB)")
