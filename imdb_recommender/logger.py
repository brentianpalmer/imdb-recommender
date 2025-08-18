from __future__ import annotations

import csv
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .schemas import IMDB_CONST_RE, ActionLogRow


class ActionLogger:
    def __init__(self, data_dir: str = "data", batch_id: str | None = None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.data_dir / "imdb_actions_log.csv"
        self.batch_id = batch_id or str(uuid.uuid4())
        self._seen: set[tuple[str, str, int | None, str | None]] = set()
        self._ensure_header()

    def _ensure_header(self):
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    [
                        "timestamp_iso",
                        "imdb_const",
                        "action",
                        "rating",
                        "notes",
                        "source",
                        "batch_id",
                    ]
                )

    def _append(self, row: ActionLogRow):
        key = (row.imdb_const, row.action, row.rating, row.notes)
        if key in self._seen:
            return
        self._seen.add(key)
        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    row.timestamp_iso,
                    row.imdb_const,
                    row.action,
                    row.rating or "",
                    row.notes or "",
                    row.source,
                    row.batch_id,
                ]
            )

    def log_rate(self, imdb_const: str, rating: int, notes: str | None = None, source: str = "cli"):
        if not IMDB_CONST_RE.match(imdb_const):
            raise ValueError(f"Invalid IMDb constant: {imdb_const}")
        if not (1 <= int(rating) <= 10):
            raise ValueError("Rating must be 1–10")
        row = ActionLogRow(
            timestamp_iso=datetime.now(timezone.utc).isoformat(),
            imdb_const=imdb_const,
            action="rate",
            rating=int(rating),
            notes=notes,
            source=source,
            batch_id=self.batch_id,
        )
        self._append(row)

    def log_watchlist(
        self, imdb_const: str, add: bool, notes: str | None = None, source: str = "cli"
    ):
        if not IMDB_CONST_RE.match(imdb_const):
            raise ValueError(f"Invalid IMDb constant: {imdb_const}")
        row = ActionLogRow(
            timestamp_iso=datetime.now(timezone.utc).isoformat(),
            imdb_const=imdb_const,
            action="watchlist_add" if add else "watchlist_remove",
            rating=None,
            notes=notes,
            source=source,
            batch_id=self.batch_id,
        )
        self._append(row)

    def export(self, out_path: str | None = None) -> str:
        out = Path(out_path) if out_path else self.path
        return str(out)
