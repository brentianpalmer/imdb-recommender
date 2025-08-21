import pandas as pd
import pytest

from imdb_recommender.utils import filter_by_content_type


def _build_mixed_df():
    return pd.DataFrame(
        {
            "imdb_const": ["m1", "t1", "s1"],
            "titleType": ["movie", "tvSeries", "short"],
            "score": [0.9, 0.8, 0.7],
        }
    )


def test_only_movies_returns_movies():
    df = _build_mixed_df()
    out = filter_by_content_type(df, "movies")
    assert set(out["titleType"]) <= {"movie", "short"}


def test_only_tv_returns_tv():
    df = _build_mixed_df()
    out = filter_by_content_type(df, "tv")
    assert set(out["titleType"]) <= {"tvSeries", "tvMiniSeries", "tvMovie", "tvSpecial"}


def test_all_returns_all():
    df = _build_mixed_df()
    out = filter_by_content_type(df, "all")
    assert len(out) == len(df)


def test_filter_applied_after_ranking():
    df = pd.DataFrame(
        {
            "imdb_const": ["tv1", "mov1"],
            "titleType": ["tvSeries", "movie"],
            "score": [1.0, 0.5],
        }
    )
    df = df.sort_values("score", ascending=False)
    filtered = filter_by_content_type(df, "movies").head(1)
    assert filtered.iloc[0]["imdb_const"] == "mov1"


def test_missing_metadata_raises():
    df = pd.DataFrame({"imdb_const": ["m1"], "score": [0.9]})
    with pytest.raises(ValueError):
        filter_by_content_type(df, "movies")
