"""Which user labels may change the model, and when.

Feedback is logged unconditionally, but a (text, code) pair only reaches the live retriever index
(and, through the export, the training labels) once it is listed in `label_reviews` as:

  consensus  the code is the model's own 1st suggestion and >= ICPC2_CONSENSUS distinct browser
             sessions confirmed it. Session ids are client-chosen, so this is NOT a defence on its
             own; it is safe because agreeing with the 1st suggestion cannot change the 1st
             suggestion. It only reinforces it.
  approved   any other code (a *deviation*) — only after the admin approves it in /painel, or when
             the admin gives that feedback while signed in.
  rejected   the admin said no; ignored from then on.

Deviations are "strong" when the chosen code was not in the top-5 shown, or is in a different
chapter from the 1st suggestion; /painel lists those first.
"""

import os
from collections import defaultdict

from sqlalchemy import delete, func, select

from icpc2.text import norm_key
from icpc2.web.db import feedback_t, requests_t, reviews_t

LABEL_KINDS = ("correct", "corrected")
LIVE = ("approved", "consensus")


def consensus_min() -> int:
    return int(os.environ.get("ICPC2_CONSENSUS", "3"))


def text_key(req) -> str:
    return req.text_key or norm_key(req.text or "")


def top1(req) -> str | None:
    preds = req.predictions or []
    return preds[0]["code"] if preds else None


def is_strong(shown: list[str], code: str) -> bool:
    return code not in shown or (bool(shown) and code[0] != shown[0][0])


def decision(conn, key: str, code: str) -> str | None:
    return conn.execute(select(reviews_t.c.status).where(
        reviews_t.c.text_key == key, reviews_t.c.code == code)).scalar()


def record(conn, key: str, text: str, code: str, status: str, source: str):
    conn.execute(delete(reviews_t).where(reviews_t.c.text_key == key, reviews_t.c.code == code))
    conn.execute(reviews_t.insert().values(text_key=key, text=text, code=code, status=status, source=source))


def undo(conn, key: str, code: str):
    conn.execute(delete(reviews_t).where(reviews_t.c.text_key == key, reviews_t.c.code == code))


def sessions_agreeing(conn, key: str, code: str) -> int:
    return conn.execute(
        select(func.count(func.distinct(requests_t.c.session_id)))
        .select_from(feedback_t.join(requests_t, feedback_t.c.request_id == requests_t.c.id))
        .where(requests_t.c.text_key == key, requests_t.c.session_id.is_not(None),
               feedback_t.c.kind.in_(LABEL_KINDS), feedback_t.c.code == code)).scalar() or 0


def on_label(conn, req, code: str, admin: bool) -> tuple[str, str] | None:
    """Called after a correct/corrected feedback row is stored. Returns (text, code) to add to the
    live index now, or None (not allowed yet, already live, or rejected)."""
    key = text_key(req)
    current = decision(conn, key, code)
    if current in LIVE:
        return None
    if admin:  # the reviewer's own feedback is an approval (also overrides an earlier rejection)
        record(conn, key, req.text, code, "approved", "admin_feedback")
        return req.text, code
    if current == "rejected":
        return None
    if code == top1(req) and sessions_agreeing(conn, key, code) >= consensus_min():
        record(conn, key, req.text, code, "consensus", "sessions")
        return req.text, code
    return None  # a deviation: waits for the admin in /painel


def live_pairs(engine) -> list[tuple[str, str]]:
    with engine.connect() as c:
        rows = c.execute(select(reviews_t.c.text, reviews_t.c.code)
                         .where(reviews_t.c.status.in_(LIVE)).order_by(reviews_t.c.id)).all()
    return [(r.text, r.code) for r in rows if r.text]


def all_reviews(conn) -> list[dict]:
    rows = conn.execute(select(reviews_t).order_by(reviews_t.c.id)).all()
    return [{"text_key": r.text_key, "text": r.text, "code": r.code, "status": r.status,
             "source": r.source} for r in rows]


def queue(engine, limit: int = 100) -> dict:
    """Deviations from the 1st suggestion that nobody has decided on yet, plus recent decisions."""
    with engine.connect() as c:
        rows = c.execute(
            select(feedback_t.c.code, feedback_t.c.created_at, requests_t.c.text, requests_t.c.text_key,
                   requests_t.c.session_id, requests_t.c.predictions, requests_t.c.id)
            .select_from(feedback_t.join(requests_t, feedback_t.c.request_id == requests_t.c.id))
            .where(feedback_t.c.kind.in_(LABEL_KINDS), requests_t.c.kind == "model")).all()
        decided = {(r["text_key"], r["code"]): r for r in all_reviews(c)}
    groups: dict[tuple[str, str], dict] = {}
    sessions: dict[tuple[str, str], set] = defaultdict(set)
    for r in rows:
        shown = [p["code"] for p in (r.predictions or [])]
        if shown and r.code == shown[0]:
            continue
        key = (r.text_key or norm_key(r.text or ""), r.code)
        if key in decided:
            continue
        g = groups.setdefault(key, {"text_key": key[0], "text": r.text, "code": r.code,
                                    "shown": shown[0] if shown else None, "events": 0,
                                    "strong": is_strong(shown, r.code), "last": r.created_at})
        g["events"] += 1
        g["last"] = max(g["last"], r.created_at)
        sessions[key].add(r.session_id or f"api-{r.id}")
    for key, g in groups.items():
        g["sessions"] = len(sessions[key])
    pending = sorted(groups.values(), key=lambda g: (not g["strong"], -g["sessions"], -g["last"].timestamp()))
    recent = list(reversed(list(decided.values())))[:20]
    return {"pending": pending[:limit], "n_pending": len(pending), "recent": recent}
