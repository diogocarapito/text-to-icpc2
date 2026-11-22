"""Demo web app: classify, collect feedback, expose logs for retraining.

Env:
  MODEL_PATH          local model dir (dev)       — or —
  MODEL_REPO          HF repo id (default diogocarapito/text-to-icpc2-v2), MODEL_REVISION (main)
  DATABASE_URL        sqlite:///... or postgresql+psycopg://...
  ICPC2_ADMIN_TOKEN   bearer token for /api/export, /api/admin/reload, /stats
"""

import hmac
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy import func, insert, select

from icpc2.text import as_icpc2_code, scrub
from icpc2.web import content
from icpc2.web.db import feedback_t, make_engine, model_events_t, requests_t

log = logging.getLogger("icpc2.web")
HERE = Path(__file__).parent
templates = Jinja2Templates(directory=HERE / "templates")


class State:
    predictor = None
    load_error: str | None = None
    lock = threading.Lock()


S = State()
engine = make_engine()


def load_model():
    from icpc2.predict import load_predictor

    src = os.environ.get("MODEL_PATH") or os.environ.get("MODEL_REPO", "diogocarapito/text-to-icpc2-v2")
    rev = None if os.environ.get("MODEL_PATH") else os.environ.get("MODEL_REVISION", "main")
    try:
        p = load_predictor(src, revision=rev)
        p.predict(["aquecimento"], k=1)  # warm-up
        with S.lock:
            S.predictor, S.load_error = p, None
        with engine.begin() as c:
            c.execute(insert(model_events_t).values(version=p.version, detail=f"{p.type} {src}"))
        log.info("Model %s ready", p.version)
    except Exception as e:  # app still serves code lookups + stats without a model
        S.load_error = f"{type(e).__name__}: {e}"
        log.exception("Model load failed")


@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=load_model, daemon=True).start()
    yield


app = FastAPI(title="text-to-icpc2", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


def require_admin(request: Request):
    token = os.environ.get("ICPC2_ADMIN_TOKEN", "")
    given = request.headers.get("authorization", "").removeprefix("Bearer ").strip() or \
        request.query_params.get("token", "")
    if not token or not hmac.compare_digest(given, token):
        raise HTTPException(401, "unauthorized")


def _codes() -> dict:
    if S.predictor:
        return S.predictor.codes
    from icpc2.codes import codes_table

    return {c["code"]: c for c in codes_table()}


# ---------------------------------------------------------------- pages
@app.get("/", response_class=HTMLResponse)
def index(request: Request, response: Response):
    ctx = {"t": content.ui("pt"), "i18n": {l: content.ui(l) for l in content.LANGS},
           "about": {l: content.load(l)["about_html"] for l in content.LANGS}}
    resp = templates.TemplateResponse(request, "index.html", ctx)
    if not request.cookies.get("sid"):
        resp.set_cookie("sid", str(uuid.uuid4()), max_age=3600 * 24 * 365, samesite="lax")
    return resp


@app.get("/health")
def health():
    return {"model": S.predictor.version if S.predictor else None, "error": S.load_error}


# ---------------------------------------------------------------- API
class PredictIn(BaseModel):
    text: str = Field(min_length=1, max_length=300)
    lang: str = Field(default="pt", pattern="^(pt|en)$")


@app.post("/api/predict")
def predict(body: PredictIn, request: Request):
    text = body.text.strip()
    codes = _codes()
    t0 = time.perf_counter()
    direct = as_icpc2_code(text)
    if direct:
        if direct not in codes:
            raise HTTPException(404, content.text(body.lang, "error_unknown_code", code=direct))
        preds, kind, version = [(direct, 1.0)], "code_lookup", None
    else:
        if not S.predictor:
            raise HTTPException(503, content.text(body.lang, "error_model_loading"))
        preds = S.predictor.predict([text], k=5)[0]
        kind, version = "model", S.predictor.version
    latency = (time.perf_counter() - t0) * 1000
    out = [{"code": c, "score": round(s, 4), "title_pt": codes[c]["title_pt"],
            "title_en": codes[c].get("title_en", ""), "inclusion": codes[c].get("inclusion", [])[:6]}
           for c, s in preds]
    with engine.begin() as conn:
        rid = conn.execute(insert(requests_t).values(
            session_id=request.cookies.get("sid"), text=scrub(text), lang=body.lang, kind=kind,
            model_version=version, latency_ms=latency,
            predictions=[{"code": p["code"], "score": p["score"]} for p in out],
        )).inserted_primary_key[0]
    return {"request_id": rid, "model_version": version, "latency_ms": round(latency, 1),
            "predictions": out}


class FeedbackIn(BaseModel):
    request_id: int
    kind: str = Field(pattern="^(correct|wrong|corrected|copy)$")
    code: str = Field(pattern="^[A-Z][0-9]{2}$")


@app.post("/api/feedback")
def feedback(body: FeedbackIn):
    if body.code not in _codes():
        raise HTTPException(400, "unknown code")
    with engine.begin() as conn:
        req = conn.execute(select(requests_t).where(requests_t.c.id == body.request_id)).first()
        if not req:
            raise HTTPException(404, "unknown request")
        shown = [p["code"] for p in (req.predictions or [])]
        rank = shown.index(body.code) + 1 if body.code in shown else None
        conn.execute(insert(feedback_t).values(request_id=body.request_id, kind=body.kind,
                                               code=body.code, rank=rank))
    # retriever models learn confirmed labels immediately (until the next retrain)
    if body.kind in ("correct", "corrected") and S.predictor and hasattr(S.predictor, "add_examples"):
        S.predictor.add_examples([req.text], [body.code])
    return {"ok": True, "rank": rank}


@app.get("/api/codes")
def codes():
    return [{"code": c["code"], "title_pt": c["title_pt"], "title_en": c.get("title_en", "")}
            for c in _codes().values()]


@app.get("/api/export", dependencies=[Depends(require_admin)])
def export(since_id: int = 0, limit: int = 1000):
    with engine.connect() as conn:
        reqs = conn.execute(select(requests_t).where(requests_t.c.id > since_id, requests_t.c.kind == "model")
                            .order_by(requests_t.c.id).limit(min(limit, 5000))).all()
        ids = [r.id for r in reqs]
        fbs = conn.execute(select(feedback_t).where(feedback_t.c.request_id.in_(ids))).all() if ids else []
    by_req: dict[int, list] = {}
    for f in fbs:
        by_req.setdefault(f.request_id, []).append({"kind": f.kind, "code": f.code, "rank": f.rank})
    # only export requests at least 10 min old, so late feedback isn't missed
    cutoff = datetime.now(UTC) - timedelta(minutes=10)
    events = []
    for r in reqs:
        created = r.created_at if r.created_at.tzinfo else r.created_at.replace(tzinfo=UTC)
        if created > cutoff:
            break
        events.append({"id": r.id, "created_at": created.isoformat(), "text": r.text, "lang": r.lang,
                       "model_version": r.model_version, "predictions": r.predictions,
                       "feedback": by_req.get(r.id, [])})
    return {"events": events}


@app.post("/api/admin/reload", dependencies=[Depends(require_admin)])
def reload():
    load_model()
    if S.load_error:
        raise HTTPException(500, S.load_error)
    return {"model": S.predictor.version}


@app.get("/stats", response_class=HTMLResponse, dependencies=[Depends(require_admin)])
def stats(request: Request):
    since = datetime.now(UTC) - timedelta(days=30)
    with engine.connect() as c:
        total = c.execute(select(func.count()).select_from(requests_t)).scalar()
        sessions = c.execute(select(func.count(func.distinct(requests_t.c.session_id)))).scalar()
        day = func.date(requests_t.c.created_at)
        per_day = c.execute(select(day, func.count()).where(requests_t.c.created_at >= since)
                            .group_by(day).order_by(day)).all()
        fb_kinds = dict(c.execute(select(feedback_t.c.kind, func.count()).group_by(feedback_t.c.kind)).all())
        ranks = dict(c.execute(select(feedback_t.c.rank, func.count())
                               .where(feedback_t.c.kind.in_(["correct", "copy", "corrected"]))
                               .group_by(feedback_t.c.rank)).all())
        with_fb = c.execute(select(func.count(func.distinct(feedback_t.c.request_id)))).scalar()
        latency = c.execute(select(func.avg(requests_t.c.latency_ms)).where(requests_t.c.kind == "model")).scalar()
        models = c.execute(select(model_events_t).order_by(model_events_t.c.id.desc()).limit(10)).all()
        recent = c.execute(select(requests_t).order_by(requests_t.c.id.desc()).limit(30)).all()
    labelled = sum(ranks.values()) or 1
    ctx = {
        "total": total, "sessions": sessions, "with_fb": with_fb,
        "fb_rate": with_fb / total if total else 0, "fb_kinds": fb_kinds,
        "top1_acc": ranks.get(1, 0) / labelled, "top5_acc": sum(v for k, v in ranks.items() if k) / labelled,
        "n_labelled": sum(ranks.values()), "latency": latency or 0,
        "per_day": [(str(d), n) for d, n in per_day], "max_day": max([n for _, n in per_day], default=1),
        "models": models, "recent": recent, "current": S.predictor.version if S.predictor else None,
    }
    return templates.TemplateResponse(request, "stats.html", ctx)
