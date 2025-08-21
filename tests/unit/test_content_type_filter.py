import pandas as pd
import pytest

from imdb_recommender.data_io import load_ratings_csv, load_watchlist
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


def test_filtering_with_fixture_watchlist(sample_watchlist_path):
    watch = load_watchlist(str(sample_watchlist_path))
    df = watch.rename(columns={"title_type": "titleType"})
    movies = filter_by_content_type(df, "movies")
    assert set(movies["titleType"]) <= {"movie", "short"}
    tv = filter_by_content_type(df, "tv")
    assert set(tv["titleType"]) <= {"tvSeries", "tvMiniSeries", "tvMovie", "tvSpecial"}


def test_load_watchlist_requires_title_type(tmp_path):
    p = tmp_path / "w.csv"
    p.write_text("titleId,added_at\nxx,2020-01-01T00:00:00Z\n")
    with pytest.raises(ValueError):
        load_watchlist(str(p))


def test_load_ratings_requires_title_type(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("userId,titleId,rating,timestamp\nu1,tt1,5,2020-01-01T00:00:00Z\n")
    with pytest.raises(ValueError):
        load_ratings_csv(str(p))
