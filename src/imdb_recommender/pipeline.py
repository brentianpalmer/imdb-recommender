"""High level pipeline helpers.

This module provides a thin programmatic wrapper around the pieces used by the
CLI commands so that integration tests (and other consumers) can exercise the
full recommendation flow without having to invoke the CLI.  Only the small
subset of functionality required for smoke tests is implemented here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data_io import ingest_sources
from .ranker import Ranker
from .recommender_svd import SVDAutoRecommender
from .utils import filter_by_content_type


def run_pipeline(
    ratings_csv: str | Path,
    watchlist_path: str | Path,
    topk: int = 25,
    content_type: str = "all",
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Execute the minimal ingestion → recommendation flow.

    Parameters
    ----------
    ratings_csv, watchlist_path:
        File paths for the user's ratings and watchlist exports.
    topk:
        Number of recommendations to return.
    content_type:
        One of ``{"all", "movies", "tv"}`` used to optionally filter the
        output.  Filtering is delegated to :func:`filter_by_content_type`.
    output_dir:
        Optional directory where intermediate artefacts and the resulting
        recommendations are written.  If provided, either a
        ``recommendations.parquet`` or ``recommendations.csv`` file is created
        inside this directory depending on available dependencies.

    Returns
    -------
    pd.DataFrame
        DataFrame containing up to ``topk`` rows with at least ``titleId`` and
        ``score`` columns.  If metadata is available a ``titleType`` column is
        also included.
    """

    data_dir = Path(output_dir) if output_dir else Path("data")
    res = ingest_sources(str(ratings_csv), str(watchlist_path), data_dir=str(data_dir))

    svd = SVDAutoRecommender(res.dataset, random_seed=42)
    svd_scores, svd_explanations = svd.score(
        seeds=[],
        user_weight=0.5,
        global_weight=0.1,
        recency=0.0,
        exclude_rated=True,
    )

    ranker = Ranker(random_seed=42)
    recommendations = ranker.top_n(
        svd_scores,
        res.dataset,
        topk=topk,
        explanations={"svd": svd_explanations},
        exclude_rated=True,
    )

    catalog = res.dataset.catalog.set_index("imdb_const")
    rows: list[dict[str, str | float | None]] = []
    for rec in recommendations:
        title_type = (
            catalog.loc[rec.imdb_const].get("title_type")
            if rec.imdb_const in catalog.index
            else None
        )
        rows.append({"titleId": rec.imdb_const, "score": rec.score, "titleType": title_type})

    df = pd.DataFrame(rows)
    df = filter_by_content_type(df, content_type)

    if output_dir:
        data_dir.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(data_dir / "recommendations.parquet", index=False)
        except Exception:
            df.to_csv(data_dir / "recommendations.csv", index=False)

    return df


__all__ = ["filter_by_content_type", "run_pipeline"]

