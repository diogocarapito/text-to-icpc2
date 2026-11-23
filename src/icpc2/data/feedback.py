"""Pull interaction logs from the web app and turn them into training labels.

Label rules (one event = one submitted text):
  corrected  user said the prediction was wrong and picked the right code   -> label (strong)
  correct    user confirmed one of the top-5 predictions                      -> label
  copy       user copied a code to the clipboard (used it in the record)      -> label (weak)
  wrong      prediction marked wrong without a correction                     -> no label

Only (text, code) pairs the web app lists as reviewed are eligible (see icpc2/web/learning.py):
`approved` by the admin, or `consensus` (several sessions confirming the model's own 1st code).
Rejected and still-pending corrections never become labels, so unreviewed feedback can reach
neither `train` nor `feedback_test`.
Each browser session votes once per text (strongest signal); conflicts are resolved by weighted
majority, ties dropped.
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
REVIEWS = "data/feedback/reviews.json"
ELIGIBLE = ("approved", "consensus")
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
            # the full decision table comes with every page; keep the latest copy
            path(REVIEWS).write_text(json.dumps(r.json().get("reviews", []), ensure_ascii=False))
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
    rp = path(REVIEWS)
    eligible = {(r["text_key"], r["code"]) for r in (json.loads(rp.read_text()) if rp.exists() else [])
                if r["status"] in ELIGIBLE}
    # (text, session) -> strongest (kind, code); events without a session vote on their own
    ballots: dict[tuple[str, str], tuple[str, str]] = {}
    info: dict[str, dict] = {}
    for line in open(p):
        e = json.loads(line)
        text = (e.get("text") or "").strip()
        if not text or as_icpc2_code(text):
            continue
        k = norm_key(text)
        voter = (k, e.get("session") or f"event-{e['id']}")
        for fb in e.get("feedback", []):
            kind, code = fb.get("kind"), fb.get("code")
            if kind not in WEIGHT or not code or (k[:300], code) not in eligible:
                continue
            if voter not in ballots or WEIGHT[kind] > WEIGHT[ballots[voter][0]]:
                ballots[voter] = (kind, code)
                info.setdefault(k, {"text": text, "lang": e.get("lang", "pt"), "kind": kind,
                                    "event_id": e["id"]})
    votes: dict[str, Counter] = defaultdict(Counter)
    for (k, _), (kind, code) in ballots.items():
        votes[k][code] += WEIGHT[kind]
    rows = []
    for k, c in votes.items():
        (code, w), *rest = c.most_common()
        if rest and rest[0][1] == w:
            continue  # tie -> ambiguous
        rows.append({**info[k], "code": code, "votes": w})
    df = pd.DataFrame(rows, columns=["text", "lang", "kind", "event_id", "code", "votes"])
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_p, index=False)
    log.info("Derived %d feedback labels -> %s", len(df), out_p)
    return df
