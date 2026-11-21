"""Pull interaction logs from the web app and turn them into training labels.

Label rules (one event = one submitted text):
  corrected  user said the prediction was wrong and picked the right code   -> label (strong)
  correct    user confirmed one of the top-5 predictions                      -> label
  copy       user copied a code to the clipboard (used it in the record)      -> label (weak)
  wrong      prediction marked wrong without a correction                     -> no label
Conflicting labels for the same normalised text are resolved by majority (strong > weak); ties dropped.
"""

import json
import logging
import os
from collections import Counter, defaultdict

import httpx
import pandas as pd

from icpc2.config import load_config, path
from icpc2.text import as_icpc2_code, norm_key

log = logging.getLogger(__name__)

EVENTS = "data/feedback/events.jsonl"
STATE = "data/feedback/state.json"
WEIGHT = {"corrected": 3, "correct": 2, "copy": 1}


def _state() -> dict:
    p = path(STATE)
    return json.loads(p.read_text()) if p.exists() else {"last_id": 0}


def pull(base_url: str | None = None, token: str | None = None) -> int:
    cfg = load_config("loop")
    base_url = (base_url or os.environ.get("ICPC2_VPS_URL") or cfg["vps"]["base_url"]).rstrip("/")
    token = token or os.environ["ICPC2_ADMIN_TOKEN"]
    st = _state()
    n = 0
    path(EVENTS).parent.mkdir(parents=True, exist_ok=True)
    with open(path(EVENTS), "a") as f:
        while True:
            r = httpx.get(f"{base_url}/api/export", params={"since_id": st["last_id"], "limit": 1000},
                          headers={"Authorization": f"Bearer {token}"}, timeout=60)
            r.raise_for_status()
            events = r.json()["events"]
            if not events:
                break
            f.writelines(json.dumps(e, ensure_ascii=False) + "\n" for e in events)
            st["last_id"] = events[-1]["id"]
            n += len(events)
    path(STATE).write_text(json.dumps(st))
    log.info("Pulled %d new events (last_id=%s)", n, st["last_id"])
    return n


def derive_labels() -> pd.DataFrame:
    p = path(EVENTS)
    out_p = path(load_config("data")["sources"]["feedback"])
    if not p.exists():
        return pd.DataFrame()
    votes: dict[str, Counter] = defaultdict(Counter)
    info: dict[str, dict] = {}
    for line in open(p):
        e = json.loads(line)
        text = (e.get("text") or "").strip()
        if not text or as_icpc2_code(text):
            continue
        # strongest signal within an event
        best = None
        for fb in e.get("feedback", []):
            kind, code = fb.get("kind"), fb.get("code")
            if kind in WEIGHT and code and (best is None or WEIGHT[kind] > WEIGHT[best[0]]):
                best = (kind, code)
        if best:
            k = norm_key(text)
            votes[k][best[1]] += WEIGHT[best[0]]
            info.setdefault(k, {"text": text, "lang": e.get("lang", "pt"), "kind": best[0],
                                "event_id": e["id"]})
    rows = []
    for k, c in votes.items():
        (code, w), *rest = c.most_common()
        if rest and rest[0][1] == w:
            continue  # tie -> ambiguous
        rows.append({**info[k], "code": code, "votes": w})
    df = pd.DataFrame(rows)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_p, index=False)
    log.info("Derived %d feedback labels -> %s", len(df), out_p)
    return df
