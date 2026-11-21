"""Interaction log. SQLite locally (default), Postgres in docker-compose (DATABASE_URL)."""

import os
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
)

metadata = MetaData()


def _now():
    return datetime.now(UTC)


requests_t = Table(
    "requests", metadata,
    Column("id", Integer, primary_key=True),
    Column("created_at", DateTime(timezone=True), default=_now, index=True),
    Column("session_id", String(36), index=True),
    Column("text", Text),                 # scrubbed input
    Column("lang", String(4)),
    Column("kind", String(16)),           # model | code_lookup
    Column("model_version", String(64)),
    Column("latency_ms", Float),
    Column("predictions", JSON),          # [{"code":..,"score":..}, ...]
)

feedback_t = Table(
    "feedback", metadata,
    Column("id", Integer, primary_key=True),
    Column("request_id", Integer, ForeignKey("requests.id"), index=True),
    Column("created_at", DateTime(timezone=True), default=_now),
    Column("kind", String(16)),           # correct | wrong | corrected | copy
    Column("code", String(8)),
    Column("rank", Integer, nullable=True),  # 1-based position in the shown top-5, null if not shown
)

model_events_t = Table(
    "model_events", metadata,
    Column("id", Integer, primary_key=True),
    Column("created_at", DateTime(timezone=True), default=_now),
    Column("version", String(64)),
    Column("detail", Text),
)


def make_engine():
    url = os.environ.get("DATABASE_URL", "sqlite:///data/web.sqlite3")
    eng = create_engine(url, pool_pre_ping=True)
    metadata.create_all(eng)
    return eng
