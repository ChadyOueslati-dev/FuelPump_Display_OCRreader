"""Keeps every scan: the picture as a file, the result as a row in SQLite."""

import json
import secrets
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    id           TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,   -- UTC, ISO 8601
    status       TEXT NOT NULL,   -- accepted / retake
    liters       TEXT,            -- as displayed, e.g. "22.93"; NULL on a retake
    confidence   REAL,
    flags        TEXT NOT NULL,   -- JSON list of retake reasons
    scan_seconds REAL NOT NULL,
    image_file   TEXT NOT NULL,   -- file name inside the scans folder
    user         TEXT,            -- whose access code was used; NULL if none
    feedback     TEXT,            -- what the person said: correct / wrong; NULL if nothing
    true_liters  TEXT             -- the liters the pump really showed, per the person
)
"""


class ScanStore:
    def __init__(self, data_dir: Path):
        self.image_dir = data_dir / "scans"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = data_dir / "scans.db"
        with closing(self._connect()) as db, db:
            db.execute(_SCHEMA)
            # Older databases lack the columns that were added later.
            columns = {row["name"] for row in db.execute("PRAGMA table_info(scans)")}
            for column in ("user", "feedback", "true_liters"):
                if column not in columns:
                    db.execute(f"ALTER TABLE scans ADD COLUMN {column} TEXT")

    def _connect(self) -> sqlite3.Connection:
        # One short-lived connection per call: requests run on several threads.
        db = sqlite3.connect(self.db_path)
        db.row_factory = sqlite3.Row
        return db

    def save(
        self,
        image: np.ndarray,
        status: str,
        liters: str | None,
        confidence: float | None,
        flags: list[str],
        scan_seconds: float,
        user: str | None = None,
    ) -> str:
        """Store one scan and return its id. The picture is saved as JPEG."""
        now = datetime.now(timezone.utc)
        scan_id = f"{now:%Y%m%dT%H%M%S}_{secrets.token_hex(4)}"
        image_file = f"{scan_id}.jpg"
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise ValueError("could not encode the scan picture")
        # The picture and its record are stored as one unit: if the record
        # cannot be written, the picture is removed again, so there is never
        # a picture without a value or a value without a picture.
        image_path = self.image_dir / image_file
        image_path.write_bytes(encoded.tobytes())
        try:
            with closing(self._connect()) as db, db:
                db.execute(
                    "INSERT INTO scans (id, created_at, status, liters, confidence, flags,"
                    " scan_seconds, image_file, user) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        scan_id,
                        now.isoformat(timespec="seconds"),
                        status,
                        liters,
                        confidence,
                        json.dumps(flags),
                        scan_seconds,
                        image_file,
                        user,
                    ),
                )
        except Exception:
            image_path.unlink(missing_ok=True)
            raise
        return scan_id

    def get(self, scan_id: str) -> dict | None:
        with closing(self._connect()) as db:
            row = db.execute("SELECT * FROM scans WHERE id = ?", (scan_id,)).fetchone()
        return None if row is None else {**dict(row), "flags": json.loads(row["flags"])}

    def set_feedback(self, scan_id: str, feedback: str, true_liters: str | None) -> bool:
        """Record what the person said about a scan. False if there is no such scan."""
        with closing(self._connect()) as db, db:
            changed = db.execute(
                "UPDATE scans SET feedback = ?, true_liters = ? WHERE id = ?",
                (feedback, true_liters, scan_id),
            ).rowcount
        return changed == 1

    def delete(self, scan_id: str) -> bool:
        """Remove a scan: its record and its picture. False if there is no such scan."""
        with closing(self._connect()) as db, db:
            row = db.execute("SELECT image_file FROM scans WHERE id = ?", (scan_id,)).fetchone()
            if row is None:
                return False
            db.execute("DELETE FROM scans WHERE id = ?", (scan_id,))
        (self.image_dir / row["image_file"]).unlink(missing_ok=True)
        return True

    def check(self) -> tuple[list[str], list[str]]:
        """Compare the records with the picture files.

        Returns (records whose picture is missing, pictures with no record).
        Both lists are empty when storage is consistent.
        """
        with closing(self._connect()) as db:
            recorded = {row["image_file"] for row in db.execute("SELECT image_file FROM scans")}
        on_disk = {p.name for p in self.image_dir.glob("*.jpg")}
        return sorted(recorded - on_disk), sorted(on_disk - recorded)

    def recent(self, limit: int = 200) -> list[dict]:
        """The newest scans first."""
        with closing(self._connect()) as db:
            rows = db.execute(
                "SELECT * FROM scans ORDER BY created_at DESC, id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [{**dict(row), "flags": json.loads(row["flags"])} for row in rows]

    def image_path(self, scan_id: str) -> Path | None:
        """The picture for a scan, or None if there is no such scan."""
        # Looked up in the database rather than built from the id, so a
        # crafted id cannot point outside the scans folder.
        with closing(self._connect()) as db:
            row = db.execute("SELECT image_file FROM scans WHERE id = ?", (scan_id,)).fetchone()
        if row is None:
            return None
        path = self.image_dir / row["image_file"]
        return path if path.is_file() else None
