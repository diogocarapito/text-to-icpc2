"""Demo web app: classify, collect feedback, expose logs for retraining.

Env:
  MODEL_PATH          local model dir (dev)       — or —
  MODEL_REPO          HF repo id (default diogocarapito/text-to-icpc2-v2), MODEL_REVISION (main)
  DATABASE_URL        sqlite:///... or postgresql+psycopg://...
  ICPC2_ADMIN_TOKEN   admin code: Bearer token for /api/export and /api/admin/reload;
                      typed once on /painel (the hidden dashboard), which then sets a cookie
  ICPC2_RATE_PER_MIN  /api/predict requests per minute per client address (default 120)
"""

import asyncio
import hashlib
import hmac
import logging
import os
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy import insert, select

from icpc2.text import as_icpc2_code, scrub
from icpc2.web import content, dashboard
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


ADMIN_COOKIE = "icpc2_admin"


def _admin_token() -> str:
    return os.environ.get("ICPC2_ADMIN_TOKEN", "")


def _cookie_value(token: str) -> str:
    # derived from the token, so changing ICPC2_ADMIN_TOKEN logs every browser out
    return hmac.new(token.encode(), b"icpc2-admin-session-v1", hashlib.sha256).hexdigest()


def is_admin(request: Request) -> bool:
    token = _admin_token()
    if not token:
        return False
    given = request.headers.get("authorization", "").removeprefix("Bearer ").strip() or \
        request.query_params.get("token", "")
    if given and hmac.compare_digest(given, token):
        return True
    cookie = request.cookies.get(ADMIN_COOKIE, "")
    return bool(cookie) and hmac.compare_digest(cookie, _cookie_value(token))


def require_admin(request: Request):
    if not is_admin(request):
        raise HTTPException(401, "unauthorized")


# in-memory, per client address; nothing is written anywhere (the app stores no IPs)
RATE_PER_MIN = int(os.environ.get("ICPC2_RATE_PER_MIN", "120"))
_hits: dict[str, deque] = {}
_hits_lock = threading.Lock()


def _rate_ok(client: str) -> bool:
    now = time.monotonic()
    with _hits_lock:
        if len(_hits) > 10_000:  # drop idle clients
            for k in [k for k, q in _hits.items() if not q or now - q[-1] > 60]:
                del _hits[k]
        q = _hits.setdefault(client, deque())
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_PER_MIN:
            return False
        q.append(now)
        return True


def _codes() -> dict:
    if S.predictor:
        return S.predictor.codes
    from icpc2.codes import codes_table

    return {c["code"]: c for c in codes_table()}


# ---------------------------------------------------------------- pages
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index(request: Request, response: Response):
    ctx = {"t": content.ui("pt"), "i18n": {l: content.ui(l) for l in content.LANGS},
           "sections": {l: content.load(l)["sections"] for l in content.LANGS}}
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
    if not _rate_ok(request.client.host if request.client else "?"):
        raise HTTPException(429, content.text(body.lang, "error_rate_limit"))
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


@app.get("/api/export", dependencies=[Depends(require_admin)], include_in_schema=False)
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


@app.post("/api/admin/reload", dependencies=[Depends(require_admin)], include_in_schema=False)
def reload():
    load_model()
    if S.load_error:
        raise HTTPException(500, S.load_error)
    return {"model": S.predictor.version}


# ---------------------------------------------------------------- admin dashboard (hidden)
# Not linked anywhere. GET /painel shows a code form until the admin code is entered once;
# then an HttpOnly cookie keeps the browser signed in for 30 days.
def _set_admin_cookie(resp: Response, request: Request) -> Response:
    resp.set_cookie(ADMIN_COOKIE, _cookie_value(_admin_token()), max_age=3600 * 24 * 30, httponly=True,
                    secure=request.url.scheme == "https", samesite="strict", path="/")
    return resp


def _noindex(resp: Response) -> Response:
    resp.headers["X-Robots-Tag"] = "noindex, nofollow"
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/painel", response_class=HTMLResponse, include_in_schema=False)
def painel(request: Request, range: str = "30"):
    if not is_admin(request):
        return _noindex(templates.TemplateResponse(request, "painel_login.html", {"error": False}))
    if request.query_params.get("token"):  # old ?token= links: swap for the cookie, clean the URL
        return _noindex(_set_admin_cookie(RedirectResponse("/painel", 303), request))
    ctx = dashboard.compute(engine, range, _codes())
    ctx["current"] = S.predictor.version if S.predictor else None
    ctx["load_error"] = S.load_error
    return _noindex(templates.TemplateResponse(request, "painel.html", ctx))


@app.post("/painel/entrar", include_in_schema=False)
async def painel_login(request: Request):
    code = (parse_qs((await request.body()).decode(errors="replace")).get("codigo") or [""])[0].strip()
    token = _admin_token()
    if token and code and hmac.compare_digest(code, token):
        return _noindex(_set_admin_cookie(RedirectResponse("/painel", 303), request))
    await asyncio.sleep(1)  # slow down guessing (without blocking other requests)
    return _noindex(templates.TemplateResponse(request, "painel_login.html", {"error": True}, status_code=401))


@app.post("/painel/sair", include_in_schema=False)
def painel_logout():
    resp = RedirectResponse("/painel", 303)
    resp.delete_cookie(ADMIN_COOKIE, path="/")
    return _noindex(resp)


@app.get("/stats", include_in_schema=False)
def stats_redirect(request: Request):
    q = f"?{request.url.query}" if request.url.query else ""
    return RedirectResponse(f"/painel{q}", 307)
