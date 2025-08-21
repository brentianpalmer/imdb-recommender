import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.pipeline import make_pipeline

from imdb_recommender.data_io import Dataset
from imdb_recommender.sklearn_integration import _DatasetWrapper, RecommenderSplitter


# Helpers ---------------------------------------------------------------------

def _make_dataset() -> Dataset:
    """Create a tiny deterministic ratings dataset."""
    np.random.seed(0)
    n_samples = 20
    titles = [f"tt{i:07d}" for i in range(5)]
    ratings = pd.DataFrame(
        {
            "imdb_const": [titles[i % 5] for i in range(n_samples)],
            "my_rating": np.random.randint(1, 11, size=n_samples),
            "title_type": ["movie"] * n_samples,
            "date_rated": pd.date_range("2020-01-01", periods=n_samples),
        }
    )
    watchlist = pd.DataFrame({"imdb_const": titles, "title_type": ["movie"] * 5})
    return Dataset(ratings=ratings, watchlist=watchlist)


def _build_pipeline(random_state: int = 0):
    """Small ElasticNet pipeline for tests."""
    model = ElasticNet(
        alpha=0.1,
        l1_ratio=0.5,
        random_state=random_state,
        max_iter=10000,
    )
    return make_pipeline(model)


class _KFoldStub:
    """Minimal stand-in for StratifiedKFoldCV returning index splits."""

    def __init__(self, n_splits=5, random_state=None):  # pragma: no cover - simple
        self.kf = KFold(n_splits=n_splits)

    def split(self, dataset):  # pragma: no cover - simple
        for train_idx, test_idx in self.kf.split(dataset.ratings):
            yield train_idx, test_idx


class _TemporalStub:
    """Minimal stand-in for TemporalKFoldCV returning index splits."""

    def __init__(self, n_splits=5, random_state=None):  # pragma: no cover - simple
        self.tscv = TimeSeriesSplit(n_splits=n_splits)

    def split(self, dataset):  # pragma: no cover - simple
        ratings = dataset.ratings.sort_values("date_rated").reset_index(drop=True)
        for train_idx, test_idx in self.tscv.split(ratings):
            yield train_idx, test_idx


# Tests -----------------------------------------------------------------------


def test_kfold_splitter(monkeypatch):
    dataset = _make_dataset()
    wrapper = _DatasetWrapper(dataset)
    y = dataset.ratings["my_rating"].to_numpy()

    monkeypatch.setattr(
        "imdb_recommender.sklearn_integration.StratifiedKFoldCV", _KFoldStub
    )

    splitter = RecommenderSplitter(n_splits=3, strategy="stratified")

    for train_idx, test_idx in splitter.split(wrapper, y):
        pipe = _build_pipeline()
        pipe.fit(wrapper[train_idx], y[train_idx])
        preds = pipe.predict(wrapper[test_idx])
        assert preds.shape == (len(test_idx),)


def test_temporal_splitter(monkeypatch):
    dataset = _make_dataset()
    wrapper = _DatasetWrapper(dataset)
    y = dataset.ratings["my_rating"].to_numpy()

    monkeypatch.setattr(
        "imdb_recommender.sklearn_integration.TemporalKFoldCV", _TemporalStub
    )

    splitter = RecommenderSplitter(n_splits=3, strategy="temporal")

    for train_idx, test_idx in splitter.split(wrapper, y):
        pipe = _build_pipeline()
        pipe.fit(wrapper[train_idx], y[train_idx])
        preds = pipe.predict(wrapper[test_idx])
        assert preds.shape == (len(test_idx),)


def test_pipeline_random_state_reproducible():
    dataset = _make_dataset()
    wrapper = _DatasetWrapper(dataset)
    y = dataset.ratings["my_rating"].to_numpy()
    X = wrapper[np.arange(len(wrapper))]

    pipe1 = _build_pipeline(random_state=123)
    pipe2 = _build_pipeline(random_state=123)

    pipe1.fit(X, y)
    pipe2.fit(X, y)

    preds1 = pipe1.predict(X)
    preds2 = pipe2.predict(X)

    assert np.allclose(preds1, preds2)
