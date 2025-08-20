from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from dotenv import load_dotenv


@dataclass
class AppConfig:
    ratings_csv_path: str
    watchlist_path: str
    data_dir: str = "data"
    random_seed: int = 42

    @classmethod
    def from_file(cls, path: str) -> AppConfig:
        load_dotenv(override=False)
        p = Path(path)
        with p.open("rb") as f:
            cfg = tomllib.load(f)
        ratings = os.getenv("RATINGS_CSV_PATH", cfg.get("paths", {}).get("ratings_csv_path", ""))
        watch = os.getenv("WATCHLIST_PATH", cfg.get("paths", {}).get("watchlist_path", ""))
        data_dir = os.getenv("DATA_DIR", cfg.get("paths", {}).get("data_dir", "data"))
        seed = int(os.getenv("RANDOM_SEED", cfg.get("runtime", {}).get("random_seed", 42)))
        return cls(
            ratings_csv_path=ratings, watchlist_path=watch, data_dir=data_dir, random_seed=seed
        )
