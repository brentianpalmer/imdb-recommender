import pandas as pd

from imdb_recommender.pipeline import run_pipeline


def test_run_pipeline_smoke(sample_ratings_path, sample_watchlist_path, tmp_path):
    df = run_pipeline(
        ratings_csv=sample_ratings_path,
        watchlist_path=sample_watchlist_path,
        topk=3,
        content_type="all",
        output_dir=tmp_path,
    )

    assert isinstance(df, pd.DataFrame)
    assert {"titleId", "score"}.issubset(df.columns)
    assert len(df) <= 3
    artifact_parquet = tmp_path / "recommendations.parquet"
    artifact_csv = tmp_path / "recommendations.csv"
    assert artifact_parquet.exists() or artifact_csv.exists()
