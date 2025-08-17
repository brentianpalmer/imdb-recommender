
from __future__ import annotations
import csv
from pathlib import Path
from typing import Optional, Set, Tuple
from datetime import datetime, timezone
import uuid
from .schemas import ActionLogRow, IMDB_CONST_RE

class ActionLogger:
    def __init__(self, data_dir: str = "data", batch_id: Optional[str] = None):
        self.data_dir = Path(data_dir); self.data_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.data_dir / "imdb_actions_log.csv"
        self.batch_id = batch_id or str(uuid.uuid4())
        self._seen: Set[Tuple[str, str, Optional[int], Optional[str]]] = set()
        self._ensure_header()

    def _ensure_header(self):
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["timestamp_iso","imdb_const","action","rating","notes","source","batch_id"])

    def _append(self, row: ActionLogRow):
        key = (row.imdb_const, row.action, row.rating, row.notes)
        if key in self._seen: return
        self._seen.add(key)
        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([row.timestamp_iso, row.imdb_const, row.action, row.rating or "", row.notes or "", row.source, row.batch_id])

    def log_rate(self, imdb_const: str, rating: int, notes: Optional[str] = None, source: str = "cli"):
        if not IMDB_CONST_RE.match(imdb_const): raise ValueError(f"Invalid IMDb constant: {imdb_const}")
        if not (1 <= int(rating) <= 10): raise ValueError("Rating must be 1–10")
        row = ActionLogRow(timestamp_iso=datetime.now(timezone.utc).isoformat(), imdb_const=imdb_const, action="rate", rating=int(rating), notes=notes, source=source, batch_id=self.batch_id)
        self._append(row)

    def log_watchlist(self, imdb_const: str, add: bool, notes: Optional[str] = None, source: str = "cli"):
        if not IMDB_CONST_RE.match(imdb_const): raise ValueError(f"Invalid IMDb constant: {imdb_const}")
        row = ActionLogRow(timestamp_iso=datetime.now(timezone.utc).isoformat(), imdb_const=imdb_const, action="watchlist_add" if add else "watchlist_remove", rating=None, notes=notes, source=source, batch_id=self.batch_id)
        self._append(row)

    def export(self, out_path: Optional[str] = None) -> str:
        out = Path(out_path) if out_path else self.path
        return str(out)
