"""Usage and performance numbers for the admin dashboard (/painel).

Everything is computed in Python from the interaction log, so it works the same on SQLite and
Postgres. The log is small (one row per request), so a full scan of the chosen range is fine.

Feedback → outcome per request: the strongest label wins (corrected > correct > copy; latest
first). Its `rank` is the position of the confirmed code in the top-5 shown (None = outside).
Requests without a session cookie came from the API, not the web page.
"""

from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import select

from icpc2.web.db import feedback_t, model_events_t, requests_t

TZ = ZoneInfo("Europe/Lisbon")
RANGES = {"7": 7, "30": 30, "90": 90, "all": None}
LABEL_PRIORITY = {"copy": 1, "correct": 2, "corrected": 3}
CONF_BUCKETS = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _rate(a: int, b: int) -> float | None:
    return a / b if b else None


def _q(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(q * (len(xs) - 1)))]


def _perf(rows: list[dict]) -> dict:
    """Shared block for any slice of model requests."""
    lab = [r for r in rows if r["label"]]
    lat = [r["latency"] for r in rows if r["latency"] is not None]
    return {
        "n": len(rows), "labelled": len(lab),
        "top1": _rate(sum(r["rank"] == 1 for r in lab), len(lab)),
        "top5": _rate(sum(r["rank"] is not None for r in lab), len(lab)),
        "p50": _q(lat, 0.5), "p95": _q(lat, 0.95),
    }


def compute(engine, range_key: str, codes: dict) -> dict:
    days = RANGES.get(range_key, 30)
    now = datetime.now(UTC)
    since = now - timedelta(days=days) if days else None

    with engine.connect() as c:
        q = select(requests_t).order_by(requests_t.c.id)
        if since:
            q = q.where(requests_t.c.created_at >= since)
        reqs = c.execute(q).all()
        ids = [r.id for r in reqs]
        fbs = []
        for i in range(0, len(ids), 5000):
            fbs += c.execute(select(feedback_t).where(feedback_t.c.request_id.in_(ids[i:i + 5000]))
                             .order_by(feedback_t.c.id)).all()
        models = c.execute(select(model_events_t).order_by(model_events_t.c.id.desc()).limit(15)).all()

    fb_by: dict[int, list] = defaultdict(list)
    for f in fbs:
        fb_by[f.request_id].append(f)

    rows = []
    for r in reqs:
        fl = fb_by.get(r.id, [])
        labels = [f for f in fl if f.kind in LABEL_PRIORITY]
        best = max(labels, key=lambda f: (LABEL_PRIORITY[f.kind], f.id)) if labels else None
        preds = r.predictions or []
        rows.append({
            "id": r.id, "at": _aware(r.created_at).astimezone(TZ), "text": r.text, "lang": r.lang,
            "kind": r.kind, "model": r.model_version, "latency": r.latency_ms,
            "client": "web" if r.session_id else "api", "session": r.session_id,
            "top1": preds[0]["code"] if preds else None, "score": preds[0]["score"] if preds else None,
            "fb": [f.kind for f in fl], "label": best.code if best else None,
            "label_kind": best.kind if best else None, "rank": best.rank if best else None,
            "wrong": any(f.kind == "wrong" for f in fl),
        })
    model_rows = [r for r in rows if r["kind"] == "model"]
    title = lambda code: (codes.get(code) or {}).get("title_pt", "")

    # ---- per day (local), split web / api; every day in the range, zeros included
    today = now.astimezone(TZ).date()
    first = (today - timedelta(days=days - 1)) if days else (rows[0]["at"].date() if rows else today)
    per_day = {first + timedelta(d): {"web": 0, "api": 0} for d in range((today - first).days + 1)}
    for r in rows:
        if r["at"].date() in per_day:
            per_day[r["at"].date()][r["client"]] += 1
    hours = [0] * 24
    for r in rows:
        hours[r["at"].hour] += 1

    # ---- performance slices
    by_model = defaultdict(list)
    by_lang = defaultdict(list)
    by_week = defaultdict(list)
    for r in model_rows:
        by_model[r["model"] or "—"].append(r)
        by_lang[r["lang"] or "—"].append(r)
        wk = r["at"].date() - timedelta(days=r["at"].weekday())
        by_week[wk].append(r)
    conf = []
    for lo, hi in CONF_BUCKETS:
        sl = [r for r in model_rows if r["score"] is not None and lo <= r["score"] < hi]
        p = _perf(sl)
        conf.append({"label": f"{lo:.1f}–{min(hi, 1):.1f}", **p})

    # ---- codes and corrections
    top_codes = Counter(r["top1"] for r in model_rows if r["top1"]).most_common(12)
    chapters = Counter(r["top1"][0] for r in model_rows if r["top1"]).most_common()
    fixes = Counter((r["top1"], r["label"]) for r in model_rows
                    if r["label"] and r["label"] != r["top1"]).most_common(15)
    errors = [r for r in reversed(model_rows) if r["wrong"] or (r["label"] and r["rank"] != 1)][:40]

    fb_kinds = Counter(f.kind for f in fbs)
    with_fb = sum(1 for r in model_rows if r["fb"])
    lab = [r for r in model_rows if r["label"]]

    return {
        "range": range_key if range_key in RANGES else "30",
        "since": first, "today": today,
        "kpi": {
            "requests": len(rows), "model_requests": len(model_rows),
            "lookups": sum(r["kind"] == "code_lookup" for r in rows),
            "sessions": len({r["session"] for r in rows if r["session"]}),
            "api": sum(r["client"] == "api" for r in rows),
            "per_day": len(rows) / max(1, len(per_day)),
            "fb_rate": _rate(with_fb, len(model_rows)), "with_fb": with_fb,
            "en": _rate(sum(r["lang"] == "en" for r in rows), len(rows)),
            **{k: v for k, v in _perf(model_rows).items() if k != "n"},
            "outside_top5": sum(r["rank"] is None for r in lab),
            "wrong_no_fix": sum(r["wrong"] and not r["label"] for r in model_rows),
        },
        "fb_kinds": [(k, fb_kinds.get(k, 0)) for k in ("correct", "copy", "corrected", "wrong")],
        "per_day": [{"label": d.isoformat(), "web": v["web"], "api": v["api"]} for d, v in per_day.items()],
        "hours": hours,
        "by_model": sorted(({"name": k, **_perf(v)} for k, v in by_model.items()), key=lambda x: -x["n"]),
        "by_lang": sorted(({"name": k, **_perf(v)} for k, v in by_lang.items()), key=lambda x: -x["n"]),
        "by_week": [{"week": w, **_perf(v)} for w, v in sorted(by_week.items(), reverse=True)][:12],
        "conf": conf,
        "top_codes": [{"code": c, "title": title(c), "n": n} for c, n in top_codes],
        "chapters": chapters,
        "fixes": [{"shown": a, "shown_title": title(a), "right": b, "right_title": title(b), "n": n}
                  for (a, b), n in fixes],
        "errors": errors,
        "recent": list(reversed(rows))[:40],
        "models": [{"at": _aware(m.created_at).astimezone(TZ), "version": m.version, "detail": m.detail}
                   for m in models],
    }
