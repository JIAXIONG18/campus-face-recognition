"""SQLite 数据访问层。"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

DB_PATH = Path(os.environ.get("FACE_APP_DB", Path(__file__).resolve().parents[2] / "data" / "campus_face.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS user_info (
    user_id   TEXT PRIMARY KEY,
    name      TEXT NOT NULL,
    category  TEXT NOT NULL DEFAULT '学生',
    status    INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS face_feature (
    feat_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL,
    feature    BLOB NOT NULL,
    image_path TEXT,
    reg_time   TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES user_info(user_id)
);
CREATE TABLE IF NOT EXISTS attendance_record (
    rec_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id  TEXT,
    name     TEXT,
    score    REAL NOT NULL,
    passed   INTEGER NOT NULL,
    scene    TEXT NOT NULL DEFAULT '考勤',
    liveness REAL,
    rec_time TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_record_time ON attendance_record(rec_time);
"""


@contextmanager
def connect(path: Path = DB_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(path: Path = DB_PATH):
    with connect(path) as conn:
        conn.executescript(SCHEMA)
        columns = [r[1] for r in conn.execute("PRAGMA table_info(attendance_record)")]
        if "liveness" not in columns:
            conn.execute("ALTER TABLE attendance_record ADD COLUMN liveness REAL")


def upsert_user(user_id: str, name: str, category: str, path: Path = DB_PATH):
    with connect(path) as conn:
        conn.execute(
            "INSERT INTO user_info(user_id,name,category,status,created_at) VALUES(?,?,?,1,?) "
            "ON CONFLICT(user_id) DO UPDATE SET name=excluded.name, category=excluded.category",
            (user_id, name, category, datetime.now().isoformat(timespec="seconds")),
        )


def add_feature(user_id: str, feature: bytes, image_path: str, path: Path = DB_PATH):
    with connect(path) as conn:
        conn.execute(
            "INSERT INTO face_feature(user_id,feature,image_path,reg_time) VALUES(?,?,?,?)",
            (user_id, feature, image_path, datetime.now().isoformat(timespec="seconds")),
        )


def load_gallery(path: Path = DB_PATH):
    with connect(path) as conn:
        rows = conn.execute(
            "SELECT f.user_id, u.name, u.category, f.feature FROM face_feature f "
            "JOIN user_info u ON u.user_id = f.user_id WHERE u.status = 1"
        ).fetchall()
    return rows


def add_record(user_id, name, score, passed, scene, liveness=None, path: Path = DB_PATH):
    with connect(path) as conn:
        conn.execute(
            "INSERT INTO attendance_record(user_id,name,score,passed,scene,liveness,rec_time) VALUES(?,?,?,?,?,?,?)",
            (user_id, name, float(score), int(passed), scene, float(liveness) if liveness is not None else None,
             datetime.now().isoformat(timespec="seconds")),
        )


def list_records(limit: int = 50, path: Path = DB_PATH):
    with connect(path) as conn:
        rows = conn.execute(
            "SELECT * FROM attendance_record ORDER BY rec_id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def list_users(path: Path = DB_PATH):
    with connect(path) as conn:
        rows = conn.execute(
            "SELECT u.user_id, u.name, u.category, u.status, u.created_at, COUNT(f.feat_id) AS photos "
            "FROM user_info u LEFT JOIN face_feature f ON f.user_id = u.user_id "
            "GROUP BY u.user_id ORDER BY u.created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def delete_user(user_id: str, path: Path = DB_PATH):
    with connect(path) as conn:
        conn.execute("DELETE FROM face_feature WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM user_info WHERE user_id = ?", (user_id,))


def stats(path: Path = DB_PATH):
    with connect(path) as conn:
        users = conn.execute("SELECT COUNT(*) FROM user_info").fetchone()[0]
        today = datetime.now().strftime("%Y-%m-%d")
        today_records = conn.execute(
            "SELECT COUNT(*) FROM attendance_record WHERE rec_time LIKE ?", (today + "%",)
        ).fetchone()[0]
        passed = conn.execute("SELECT COUNT(*) FROM attendance_record WHERE passed = 1").fetchone()[0]
        total = conn.execute("SELECT COUNT(*) FROM attendance_record").fetchone()[0]
    return {"users": users, "today": today_records, "passed": passed, "records": total}
