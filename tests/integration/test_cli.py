import pandas as pd
from typer.testing import CliRunner

from imdb_recommender.cli import app
from imdb_recommender.data_io import Dataset, IngestResult
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

    def fake_score(self, seeds, uw, gw, recency, exclude_rated):
        return {"m1": 0.1, "t1": 0.2}, {}

    monkeypatch.setattr("imdb_recommender.cli.SVDAutoRecommender.score", fake_score)

    def fake_top_n(self, blended, dataset, topk, explanations=None, exclude_rated=True):
        items = sorted(blended.items(), key=lambda x: x[1], reverse=True)
        cat = dataset.catalog.set_index("imdb_const")
        out = []
        for cid, score in items[:topk]:
            row = cat.loc[cid]
            out.append(
                Recommendation(
                    imdb_const=cid,
                    title=row.get("title"),
                    year=row.get("year"),
                    genres=row.get("genres"),
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
            "--ratings",
            "r.csv",
            "--watchlist",
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
            "--ratings",
            "r.csv",
            "--watchlist",
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
            "--ratings",
            "r.csv",
            "--watchlist",
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
            "--ratings",
            "r.csv",
            "--watchlist",
            "w.csv",
            "--content-type",
            "movies",
            "--no-exclude-rated",
        ],
    )
    assert result.exit_code != 0
    assert "titleType metadata required" in result.stderr
