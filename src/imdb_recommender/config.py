from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


@dataclass
class AppConfig:
    ratings_csv_path: str
    watchlist_path: str
    data_dir: str = "data"
    random_seed: int = 42

    @classmethod
    def from_file(cls, path: str) -> AppConfig:
        p = Path(path)
        with p.open("rb") as f:
            cfg = tomllib.load(f)

        # Support both 'data' and 'paths' sections for backward compatibility
        data_section = cfg.get("data", {}) or cfg.get("paths", {})
        recommendation_section = cfg.get("recommendation", {})

        # Read environment variables if already present (no implicit .env loading)
        ratings = os.getenv("RATINGS_CSV_PATH") or data_section.get(
            "ratings_csv_path", "data/raw/ratings.csv"
        )
        watch = os.getenv("WATCHLIST_PATH") or data_section.get(
            "watchlist_path", "data/raw/watchlist.xlsx"
        )
        data_dir = os.getenv("DATA_DIR") or data_section.get("data_dir", "data")
        seed = int(os.getenv("RANDOM_SEED") or recommendation_section.get("random_seed", 42))

        return cls(
            ratings_csv_path=ratings, watchlist_path=watch, data_dir=data_dir, random_seed=seed
        )
