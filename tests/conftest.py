import random
import shutil
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures" / "data"


@pytest.fixture
def tmp_data_dir(tmp_path, fixtures_dir):
    dst = tmp_path / "data"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("ratings_sample.csv", "watchlist_sample.csv"):
        shutil.copy2(fixtures_dir / name, dst / name)
    return dst


@pytest.fixture
def sample_ratings_path(tmp_data_dir) -> Path:
    return tmp_data_dir / "ratings_sample.csv"


@pytest.fixture
def sample_watchlist_path(tmp_data_dir) -> Path:
    return tmp_data_dir / "watchlist_sample.csv"


# Optional: deterministic seed fixture for ML paths
@pytest.fixture(autouse=True)
def _force_seed(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    try:
        np.random.seed(1234)
        random.seed(1234)
    except Exception:
        pass
    yield
