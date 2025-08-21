import pandas as pd
from typer.testing import CliRunner

from imdb_recommender.cli import app
from imdb_recommender.data_io import Dataset, IngestResult, ingest_sources
from imdb_recommender.schemas import Recommendation

runner = CliRunner()


def _patched_ingest(monkeypatch, include_title_type: bool = True):
    data = {
        "imdb_const": ["m1", "t1"],
        "title": ["Movie1", "TV1"],
        "year": [2000, 2001],
        "genres": ["Drama", "Sport"],
    }
    if include_title_type:
        data["title_type"] = ["movie", "tvSeries"]
    watch = pd.DataFrame(data)
    ratings = watch.iloc[0:0].copy()
    ds = Dataset(ratings=ratings, watchlist=watch)
    ingest_res = IngestResult(dataset=ds, warnings=[])
    monkeypatch.setattr("imdb_recommender.cli.ingest_sources", lambda *a, **k: ingest_res)
    monkeypatch.setattr("imdb_recommender.cli._validate_paths", lambda *a, **k: None)

    def fake_score(self, seeds, uw, gw, recency, exclude_rated):
        return {"m1": 0.1, "t1": 0.2}, {}

    monkeypatch.setattr("imdb_recommender.cli.SVDAutoRecommender.score", fake_score)

    def fake_top_n(self, blended, dataset, topk, explanations=None, exclude_rated=True):
        items = sorted(blended.items(), key=lambda x: x[1], reverse=True)
        cat = dataset.catalog.set_index("imdb_const")
        out = []
        for cid, score in items[:topk]:
            row = cat.loc[cid]
            title = row.get("title")
            year = row.get("year")
            genres = row.get("genres")
            if pd.isna(title):
                title = cid
            out.append(
                Recommendation(
                    imdb_const=cid,
                    title=title if pd.notna(title) else None,
                    year=int(year) if pd.notna(year) else None,
                    genres=genres if pd.notna(genres) else None,
                    score=score,
                    why_explainer="",
                )
            )
        return out

    monkeypatch.setattr("imdb_recommender.cli.Ranker.top_n", fake_top_n)


def test_cli_recommend_movies_only_returns_no_tv(monkeypatch):
    _patched_ingest(monkeypatch)
    result = runner.invoke(
        app,
        [
            "recommend",
            "--ratings-file",
            "r.csv",
            "--watchlist-file",
            "w.csv",
            "--content-type",
            "movies",
            "--no-exclude-rated",
        ],
    )
    assert result.exit_code == 0
    assert "Movie1" in result.stdout
    assert "TV1" not in result.stdout


def test_cli_recommend_tv_only_returns_no_movies(monkeypatch):
    _patched_ingest(monkeypatch)
    result = runner.invoke(
        app,
        [
            "recommend",
            "--ratings-file",
            "r.csv",
            "--watchlist-file",
            "w.csv",
            "--content-type",
            "tv",
            "--no-exclude-rated",
        ],
    )
    assert result.exit_code == 0
    assert "TV1" in result.stdout
    assert "Movie1" not in result.stdout


def test_cli_default_all_includes_both_when_present(monkeypatch):
    _patched_ingest(monkeypatch)
    result = runner.invoke(
        app,
        [
            "recommend",
            "--ratings-file",
            "r.csv",
            "--watchlist-file",
            "w.csv",
            "--no-exclude-rated",
        ],
    )
    assert result.exit_code == 0
    assert "TV1" in result.stdout and "Movie1" in result.stdout


def test_graceful_when_metadata_missing(monkeypatch):
    _patched_ingest(monkeypatch, include_title_type=False)
    result = runner.invoke(
        app,
        [
            "recommend",
            "--ratings-file",
            "r.csv",
            "--watchlist-file",
            "w.csv",
            "--content-type",
            "movies",
            "--no-exclude-rated",
        ],
    )
    assert result.exit_code != 0
    assert "titleType metadata required" in result.stderr


def _patch_algorithms(monkeypatch):
    def fake_score(self, seeds, uw, gw, recency, exclude_rated):
        return {
            "tt0133093": 0.6,
            "tt0120737": 0.5,
            "tt0120815": 0.4,
            "tt0167260": 0.3,
            "tt4154796": 0.2,
            "tt4154756": 0.1,
        }, {}

    monkeypatch.setattr("imdb_recommender.cli.SVDAutoRecommender.score", fake_score)

    def fake_top_n(self, blended, dataset, topk, explanations=None, exclude_rated=True):
        items = sorted(blended.items(), key=lambda x: x[1], reverse=True)
        cat = dataset.catalog.set_index("imdb_const")
        out = []
        for cid, score in items[:topk]:
            row = cat.loc[cid]
            title = row.get("title")
            year = row.get("year")
            genres = row.get("genres")
            if pd.isna(title):
                title = cid
            out.append(
                Recommendation(
                    imdb_const=cid,
                    title=title if pd.notna(title) else None,
                    year=int(year) if pd.notna(year) else None,
                    genres=genres if pd.notna(genres) else None,
                    score=score,
                    why_explainer="",
                )
            )
        return out

    monkeypatch.setattr("imdb_recommender.cli.Ranker.top_n", fake_top_n)


def test_cli_with_fixtures_movies(
    monkeypatch, sample_ratings_path, sample_watchlist_path, tmp_path
):
    _patch_algorithms(monkeypatch)

    def patched_ingest(ratings_csv, watchlist_path, data_dir):
        return ingest_sources(ratings_csv, watchlist_path, data_dir=str(tmp_path))

    monkeypatch.setattr("imdb_recommender.cli.ingest_sources", patched_ingest)

    result = runner.invoke(
        app,
        [
            "recommend",
            "--ratings-file",
            str(sample_ratings_path),
            "--watchlist-file",
            str(sample_watchlist_path),
            "--content-type",
            "movies",
            "--no-exclude-rated",
        ],
    )

    assert result.exit_code == 0
    out = result.stdout
    assert "tt0120815" not in out and "tt0167260" not in out and "tt4154756" not in out
    assert "tt0133093" in out and "tt0120737" in out and "tt4154796" in out


def test_cli_with_fixtures_tv(monkeypatch, sample_ratings_path, sample_watchlist_path, tmp_path):
    _patch_algorithms(monkeypatch)

    def patched_ingest(ratings_csv, watchlist_path, data_dir):
        return ingest_sources(ratings_csv, watchlist_path, data_dir=str(tmp_path))

    monkeypatch.setattr("imdb_recommender.cli.ingest_sources", patched_ingest)

    result = runner.invoke(
        app,
        [
            "recommend",
            "--ratings-file",
            str(sample_ratings_path),
            "--watchlist-file",
            str(sample_watchlist_path),
            "--content-type",
            "tv",
            "--no-exclude-rated",
        ],
    )

    assert result.exit_code == 0
    out = result.stdout
    assert "tt0133093" not in out and "tt0120737" not in out and "tt4154796" not in out
    assert "tt0120815" in out and "tt0167260" in out and "tt4154756" in out
