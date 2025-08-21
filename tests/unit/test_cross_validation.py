import pandas as pd
import pytest

pytest.importorskip("sklearn")

from imdb_recommender.cross_validation import (
    CrossValidationEvaluator,
    StratifiedKFoldCV,
    TemporalKFoldCV,
)
from imdb_recommender.data_io import Dataset
from imdb_recommender.recommender_base import RecommenderAlgo


# Helper to construct a tiny deterministic dataset

def _small_dataset() -> Dataset:
    ratings = pd.DataFrame(
        {
            "imdb_const": ["tt1", "tt2", "tt3", "tt4"],
            "my_rating": [1, 2, 9, 10],
            "date_rated": pd.date_range("2021-01-01", periods=4),
        }
    )
    watchlist = pd.DataFrame({"imdb_const": ratings["imdb_const"]})
    return Dataset(ratings=ratings, watchlist=watchlist)


class _MeanRecommender(RecommenderAlgo):
    """Stub recommender predicting the mean training rating."""

    def fit(self, **_):  # pragma: no cover - trivial
        self._score = (self.dataset.ratings["my_rating"].mean() - 5.5) / 3

    def score(
        self,
        seeds,  # pragma: no cover - interface requirement
        user_weight,
        global_weight,
        recency=0.0,
        exclude_rated=False,
    ):
        items = self.dataset.watchlist["imdb_const"].tolist()
        return {i: self._score for i in items}, {}


def test_kfold_splits_cover_all_and_no_leakage():
    ds = _small_dataset()
    splitter = StratifiedKFoldCV(n_splits=2, random_state=42)
    splits = list(splitter.split_dataset(ds))
    assert len(splits) == 2

    seen = set()
    for train_ds, val_ds in splits:
        train_ids = set(train_ds.ratings["imdb_const"])
        val_ids = set(val_ds.ratings["imdb_const"])
        assert train_ids.isdisjoint(val_ids)
        seen |= val_ids
    assert seen == set(ds.ratings["imdb_const"])


def test_temporal_split_respects_chronology():
    ds = _small_dataset()
    splitter = TemporalKFoldCV(n_splits=2)
    splits = list(splitter.split_dataset(ds))
    assert len(splits) == 2
    for train_ds, val_ds in splits:
        assert train_ds.ratings["date_rated"].max() < val_ds.ratings["date_rated"].min()


def test_evaluate_aggregates_metrics():
    ds = _small_dataset()
    evaluator = CrossValidationEvaluator("stratified", n_splits=2, random_state=42)
    result = evaluator.evaluate_recommender(
        _MeanRecommender,
        ds,
        recommender_params={},
        score_params={"user_weight": 0.7, "global_weight": 0.3},
    )

    assert result.mean_rmse == pytest.approx(4.0)
    assert result.mean_mae == pytest.approx(4.0)
    assert result.std_rmse == pytest.approx(0.7071, rel=1e-3)
    assert result.std_mae == pytest.approx(0.7071, rel=1e-3)
