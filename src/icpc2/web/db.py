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
    UniqueConstraint,
    create_engine,
    inspect,
    text,
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
    Column("text_key", String(300), index=True),  # norm_key(text): groups the same text across sessions
    Column("feedback_key", String(32)),   # random; returned only to the requester, required for feedback
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

# Decisions on (text, code) pairs from user feedback. Only pairs listed here as approved/consensus
# reach the live retriever index or the training labels; everything else waits in /painel.
reviews_t = Table(
    "label_reviews", metadata,
    Column("id", Integer, primary_key=True),
    Column("created_at", DateTime(timezone=True), default=_now),
    Column("text_key", String(300), nullable=False),
    Column("text", Text),
    Column("code", String(8), nullable=False),
    Column("status", String(16)),         # approved | consensus | rejected
    Column("source", String(16)),         # painel | admin_feedback | sessions
    UniqueConstraint("text_key", "code", name="uq_review_text_code"),
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
    _add_missing_columns(eng)
    return eng


def _add_missing_columns(eng):
    """create_all() never alters existing tables: add new nullable columns (and their indexes)."""
    insp = inspect(eng)
    with eng.begin() as c:
        for table in metadata.sorted_tables:
            have = {col["name"] for col in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name in have:
                    continue
                c.execute(text(f'ALTER TABLE {table.name} ADD COLUMN {col.name} '
                               f'{col.type.compile(dialect=eng.dialect)}'))
                if col.index:
                    c.execute(text(f"CREATE INDEX IF NOT EXISTS ix_{table.name}_{col.name} "
                                   f"ON {table.name} ({col.name})"))
