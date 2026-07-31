"""SQLite storage and a small, sport-agnostic caching layer.

Cached rows hold normalized JSON plus an ``updated_at`` timestamp. A read is
only a cache hit when the row is younger than the caller's freshness window,
which is how we "check last_updated before re-fetching".

Nothing here imports from ``sports``. ``cached_fetch`` receives the
sport-specific fetch function as an argument (dependency injection), so the
same cache serves any sport.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

DEFAULT_DB_PATH = os.environ.get("CHECKTHEBALL_DB", "checktheball.sqlite")

# Whitelisted cache tables. Table names can't be parameterized in SQL, so we
# validate against this set to keep the API injection-safe.
_TABLES = ("games", "plays", "player_stats_cache")

# Freshness windows only decide whether a row is worth reading; nothing deletes
# it. Most keys are never requested twice (a finished game's box score is fixed
# and asked for once), so without a sweep the file grows without bound.
CACHE_RETENTION_DAYS = 7


def _connect(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _json_default(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Create the cache tables and the query log if they don't exist."""
    with closing(_connect(db_path)) as conn:
        for table in _TABLES:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    key        TEXT PRIMARY KEY,
                    sport      TEXT NOT NULL,
                    data       TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS queries (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at      TEXT NOT NULL,
                question        TEXT NOT NULL,
                answer          TEXT NOT NULL,
                tool_calls      TEXT NOT NULL,
                grounding_score REAL
            )
            """
        )
        conn.commit()


def log_query(
    question: str,
    answer: str,
    tool_calls: Any,
    grounding_score: float | None,
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    """Record one answered question and its grounding score."""
    with closing(_connect(db_path)) as conn:
        conn.execute(
            """
            INSERT INTO queries (created_at, question, answer, tool_calls, grounding_score)
            VALUES (?, ?, ?, ?, ?)
            """,
            (_now(), question, answer, json.dumps(tool_calls, default=_json_default), grounding_score),
        )
        conn.commit()


def read_cache(
    table: str,
    key: str,
    max_age_seconds: float | None = None,
    db_path: str = DEFAULT_DB_PATH,
) -> Any | None:
    """Return cached data for ``key`` if present and fresh, else None.

    ``max_age_seconds=None`` means "any age is acceptable".
    """
    if table not in _TABLES:
        raise ValueError(f"Unknown cache table: {table}")

    with closing(_connect(db_path)) as conn:
        row = conn.execute(
            f"SELECT data, updated_at FROM {table} WHERE key = ?", (key,)
        ).fetchone()

    if row is None:
        return None

    if max_age_seconds is not None:
        updated = datetime.fromisoformat(row["updated_at"])
        age = (datetime.now(timezone.utc) - updated).total_seconds()
        if age > max_age_seconds:
            return None

    return json.loads(row["data"])


def write_cache(
    table: str,
    key: str,
    sport: str,
    payload: Any,
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    """Insert or replace a cache row, stamping it with the current time."""
    if table not in _TABLES:
        raise ValueError(f"Unknown cache table: {table}")

    data = json.dumps(payload, default=_json_default)
    with closing(_connect(db_path)) as conn:
        conn.execute(
            f"""
            INSERT INTO {table} (key, sport, data, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                sport = excluded.sport,
                data = excluded.data,
                updated_at = excluded.updated_at
            """,
            (key, sport, data, _now()),
        )
        conn.commit()


def sweep_cache(
    max_age_days: float = CACHE_RETENTION_DAYS,
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """Delete cache rows older than ``max_age_days``; return how many went.

    Cache tables only. The query log is real data, not a disposable copy, so it
    is never swept. Deleting a row that is still wanted costs one refetch.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()
    removed = 0
    with closing(_connect(db_path)) as conn:
        for table in _TABLES:
            removed += conn.execute(
                f"DELETE FROM {table} WHERE updated_at < ?", (cutoff,)
            ).rowcount
        conn.commit()
    return removed


def cached_fetch(
    table: str,
    key: str,
    sport: str,
    fetch_fn: Callable[[], Any],
    max_age_seconds: float | None,
    db_path: str = DEFAULT_DB_PATH,
) -> Any:
    """Return fresh cached data, or call ``fetch_fn``, store it, and return it.

    ``fetch_fn`` is provided by the sport module, so this stays sport-agnostic.
    """
    cached = read_cache(table, key, max_age_seconds, db_path)
    if cached is not None:
        return cached

    fresh = fetch_fn()
    write_cache(table, key, sport, fresh, db_path)
    return fresh
