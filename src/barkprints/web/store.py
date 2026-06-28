"""Lightweight persistence for users and saved barkprints.

Uses the Python stdlib ``sqlite3`` (no ORM, no extra deps) plus ``bcrypt`` for
password hashing. Image bytes live on disk under ``<data_dir>/uploads``; only a
filename is stored in the database. A single SQLite file holds users and the
saved entries.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import bcrypt

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entries (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id        INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at     TEXT NOT NULL,
    corpus         TEXT NOT NULL,
    alpha          REAL,
    max_words      INTEGER,
    text           TEXT NOT NULL,
    image_filename TEXT NOT NULL,
    lat            REAL,
    lon            REAL,
    accuracy       REAL
);

CREATE INDEX IF NOT EXISTS idx_entries_user ON entries(user_id, created_at);
"""


@dataclass(frozen=True)
class User:
    id: int
    username: str


@dataclass(frozen=True)
class Entry:
    id: int
    user_id: int
    created_at: str
    corpus: str
    alpha: Optional[float]
    max_words: Optional[int]
    text: str
    image_filename: str
    lat: Optional[float]
    lon: Optional[float]
    accuracy: Optional[float]

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "corpus": self.corpus,
            "alpha": self.alpha,
            "max_words": self.max_words,
            "text": self.text,
            "lat": self.lat,
            "lon": self.lon,
            "accuracy": self.accuracy,
            "image_url": f"/api/entries/{self.id}/image",
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    """SQLite-backed store. A fresh connection is opened per operation so the
    store is safe to use from FastAPI's threadpool and from multiple worker
    processes (WAL mode allows concurrent readers with a single writer)."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.uploads_dir = self.data_dir / "uploads"
        self.db_path = self.data_dir / "barkprints.db"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -- users -----------------------------------------------------------

    def create_user(self, username: str, password: str) -> User:
        username = username.strip()
        if not username:
            raise ValueError("Username must not be empty")
        if not password:
            raise ValueError("Password must not be empty")
        pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        with self._connect() as conn:
            try:
                cur = conn.execute(
                    "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                    (username, pw_hash, _now()),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"User {username!r} already exists") from exc
            return User(id=int(cur.lastrowid), username=username)

    def set_password(self, username: str, password: str) -> bool:
        if not password:
            raise ValueError("Password must not be empty")
        pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE users SET password_hash = ? WHERE username = ?",
                (pw_hash, username.strip()),
            )
            return cur.rowcount > 0

    def verify_login(self, username: str, password: str) -> Optional[User]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, username, password_hash FROM users WHERE username = ?",
                (username.strip(),),
            ).fetchone()
        if row is None:
            # Hash a dummy value anyway to keep timing roughly constant.
            bcrypt.checkpw(b"x", bcrypt.hashpw(b"x", bcrypt.gensalt()))
            return None
        if bcrypt.checkpw(password.encode("utf-8"), row["password_hash"].encode("utf-8")):
            return User(id=row["id"], username=row["username"])
        return None

    def get_user(self, user_id: int) -> Optional[User]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, username FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        return User(id=row["id"], username=row["username"]) if row else None

    def list_users(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT username FROM users ORDER BY username").fetchall()
        return [r["username"] for r in rows]

    # -- entries ---------------------------------------------------------

    def add_entry(
        self,
        *,
        user_id: int,
        corpus: str,
        alpha: Optional[float],
        max_words: Optional[int],
        text: str,
        image_filename: str,
        lat: Optional[float],
        lon: Optional[float],
        accuracy: Optional[float],
    ) -> Entry:
        created_at = _now()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO entries
                    (user_id, created_at, corpus, alpha, max_words, text,
                     image_filename, lat, lon, accuracy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, created_at, corpus, alpha, max_words, text,
                 image_filename, lat, lon, accuracy),
            )
            entry_id = int(cur.lastrowid)
        return Entry(
            id=entry_id, user_id=user_id, created_at=created_at, corpus=corpus,
            alpha=alpha, max_words=max_words, text=text, image_filename=image_filename,
            lat=lat, lon=lon, accuracy=accuracy,
        )

    def list_entries(self, user_id: int) -> list[Entry]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM entries WHERE user_id = ? ORDER BY created_at DESC, id DESC",
                (user_id,),
            ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def get_entry(self, entry_id: int, user_id: int) -> Optional[Entry]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM entries WHERE id = ? AND user_id = ?",
                (entry_id, user_id),
            ).fetchone()
        return self._row_to_entry(row) if row else None

    def delete_entry(self, entry_id: int, user_id: int) -> Optional[str]:
        """Delete an entry, returning its image filename so the caller can
        remove the file on disk. Returns None if no such entry for this user."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT image_filename FROM entries WHERE id = ? AND user_id = ?",
                (entry_id, user_id),
            ).fetchone()
            if row is None:
                return None
            conn.execute("DELETE FROM entries WHERE id = ? AND user_id = ?", (entry_id, user_id))
            return row["image_filename"]

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> Entry:
        return Entry(
            id=row["id"], user_id=row["user_id"], created_at=row["created_at"],
            corpus=row["corpus"], alpha=row["alpha"], max_words=row["max_words"],
            text=row["text"], image_filename=row["image_filename"],
            lat=row["lat"], lon=row["lon"], accuracy=row["accuracy"],
        )
