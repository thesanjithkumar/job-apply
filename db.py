"""
Turso (libSQL) integration via HTTP API.
No extra packages — uses `requests` which is already in requirements.txt.

Required env vars:
  TURSO_DB_URL      e.g. libsql://your-db.turso.io
  TURSO_AUTH_TOKEN  JWT token from the Turso dashboard
"""

import os
import requests
from datetime import datetime, timezone

_URL   = os.environ.get("TURSO_DB_URL", "").replace("libsql://", "https://")
_TOKEN = os.environ.get("TURSO_AUTH_TOKEN", "")


def enabled() -> bool:
    return bool(_URL and _TOKEN)


def _val(v) -> dict:
    if v is None:
        return {"type": "null"}
    if isinstance(v, int):
        return {"type": "integer", "value": str(v)}
    if isinstance(v, float):
        return {"type": "float", "value": str(v)}
    return {"type": "text", "value": str(v)}


def _run(stmts: list[dict]) -> list:
    if not enabled():
        return []
    headers = {
        "Authorization": f"Bearer {_TOKEN}",
        "Content-Type":  "application/json",
    }
    body = {
        "requests": [{"type": "execute", "stmt": s} for s in stmts] + [{"type": "close"}]
    }
    try:
        r = requests.post(f"{_URL}/v2/pipeline", json=body, headers=headers, timeout=20)
        r.raise_for_status()
        return r.json().get("results", [])
    except Exception as e:
        print(f"  [DB] Error: {e}")
        return []


# ── Schema ─────────────────────────────────────────────────────────────────

def init_db():
    """Create tables if they don't exist. Safe to call on every run."""
    _run([
        {
            "sql": """
                CREATE TABLE IF NOT EXISTS scraped_jobs (
                    url        TEXT PRIMARY KEY,
                    title      TEXT,
                    company    TEXT,
                    location   TEXT,
                    score      INTEGER,
                    rank       INTEGER,
                    reason     TEXT,
                    scraped_at TEXT
                )
            """
        },
        {
            "sql": """
                CREATE TABLE IF NOT EXISTS applied_jobs (
                    url        TEXT PRIMARY KEY,
                    title      TEXT,
                    company    TEXT,
                    score      INTEGER,
                    applied_at TEXT
                )
            """
        },
        {
            "sql": """
                CREATE TABLE IF NOT EXISTS seen_jobs (
                    url      TEXT PRIMARY KEY,
                    title    TEXT,
                    company  TEXT,
                    score    INTEGER,
                    seen_at  TEXT
                )
            """
        },
    ])
    if enabled():
        print("  [DB] Connected — tables ready")
    else:
        print("  [DB] TURSO_DB_URL / TURSO_AUTH_TOKEN not set — skipping DB")


# ── Write ──────────────────────────────────────────────────────────────────

def save_scraped_jobs(jobs: list[dict]):
    """Upsert all ranked jobs. Updates score/rank if URL already exists."""
    if not jobs or not enabled():
        return
    now = datetime.now(timezone.utc).isoformat()
    stmts = []
    for j in jobs:
        stmts.append({
            "sql": """
                INSERT INTO scraped_jobs (url, title, company, location, score, rank, reason, scraped_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    score      = excluded.score,
                    rank       = excluded.rank,
                    reason     = excluded.reason,
                    scraped_at = excluded.scraped_at
            """,
            "args": [
                _val(j.get("url")),      _val(j.get("title")),
                _val(j.get("company")),  _val(j.get("location")),
                _val(j.get("score")),    _val(j.get("rank")),
                _val(j.get("reason")),   _val(now),
            ],
        })
    _run(stmts)
    print(f"  [DB] Saved {len(jobs)} scraped jobs")


def mark_applied(job: dict):
    """Record a successfully applied job. Skips if URL already recorded."""
    if not enabled():
        return
    now = datetime.now(timezone.utc).isoformat()
    _run([{
        "sql": """
            INSERT OR IGNORE INTO applied_jobs (url, title, company, score, applied_at)
            VALUES (?, ?, ?, ?, ?)
        """,
        "args": [
            _val(job.get("url")),     _val(job.get("title")),
            _val(job.get("company")), _val(job.get("score")),
            _val(now),
        ],
    }])


def mark_seen(jobs: list[dict]):
    """Record jobs that were included in the report email (emailed but not auto-applied)."""
    if not jobs or not enabled():
        return
    now = datetime.now(timezone.utc).isoformat()
    stmts = [{
        "sql": "INSERT OR IGNORE INTO seen_jobs (url, title, company, score, seen_at) VALUES (?, ?, ?, ?, ?)",
        "args": [_val(j.get("url")), _val(j.get("title")), _val(j.get("company")), _val(j.get("score")), _val(now)],
    } for j in jobs]
    _run(stmts)
    print(f"  [DB] Marked {len(jobs)} jobs as seen (will skip next run)")


# ── Read ───────────────────────────────────────────────────────────────────

def get_seen_urls() -> set[str]:
    """Return URLs of all jobs already applied to OR emailed to the user."""
    if not enabled():
        return set()
    results = _run([
        {"sql": "SELECT url FROM applied_jobs"},
        {"sql": "SELECT url FROM seen_jobs"},
    ])
    urls = set()
    for res in results:
        try:
            rows = res["response"]["result"]["rows"]
            urls |= {row[0]["value"] for row in rows if row}
        except Exception:
            pass
    print(f"  [DB] {len(urls)} previously seen/applied jobs — will skip")
    return urls


def get_applied_urls() -> set[str]:
    """Return URLs of every job ever successfully applied to."""
    if not enabled():
        return set()
    results = _run([{"sql": "SELECT url FROM applied_jobs"}])
    try:
        rows = results[0]["response"]["result"]["rows"]
        urls = {row[0]["value"] for row in rows if row}
        print(f"  [DB] {len(urls)} previously applied jobs")
        return urls
    except Exception:
        return set()
