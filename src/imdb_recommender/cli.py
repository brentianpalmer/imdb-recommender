from __future__ import annotations

import enum
from pathlib import Path

import pandas as pd
import typer

from .config import AppConfig
from .data_io import ingest_sources
from .ranker import Ranker
from .recommender_svd import SVDAutoRecommender
from .recommender_elasticnet import ElasticNetRecommender
from .utils import filter_by_content_type

# from .hyperparameter_tuning import HyperparameterTuningPipeline  # Temporarily disabled

app = typer.Typer(help="Multi-Model IMDb Movie Recommender (SVD + ElasticNet)")


def _deprecated_ratings(ctx: typer.Context, param: typer.CallbackParam, value: str | None) -> None:
    """Handle deprecated --ratings option."""
    if value is not None:
        typer.echo("⚠️ '--ratings' is deprecated; use '--ratings-file'", err=True)
        if ctx.params.get("ratings") is None:
            ctx.params["ratings"] = value


def _deprecated_watchlist(
    ctx: typer.Context, param: typer.CallbackParam, value: str | None
) -> None:
    """Handle deprecated --watchlist option."""
    if value is not None:
        typer.echo("⚠️ '--watchlist' is deprecated; use '--watchlist-file'", err=True)
        if ctx.params.get("watchlist") is None:
            ctx.params["watchlist"] = value


def _validate_paths(ratings: str, watchlist: str) -> None:
    """Ensure ratings and watchlist files exist and are readable."""
    for name, path in {"ratings": ratings, "watchlist": watchlist}.items():
        file_path = Path(path)
        if not file_path.is_file():
            typer.echo(f"❌ {name} file not found: {path}", err=True)
            raise typer.Exit(1)
        try:
            with file_path.open("r"):
                pass
        except OSError as exc:  # pragma: no cover - defensive
            typer.echo(f"❌ cannot read {name} file: {exc}", err=True)
            raise typer.Exit(1) from exc


@app.command()
def ingest(
    ratings: str = typer.Option(..., "--ratings-file", help="Path to ratings CSV file"),
    watchlist: str = typer.Option(..., "--watchlist-file", help="Path to watchlist CSV file"),
    data_dir: str = typer.Option("data", help="Data directory for processed files"),
    _ratings_depr: str | None = typer.Option(
        None,
        "--ratings",
        help="Deprecated alias for --ratings-file",
        hidden=True,
        callback=_deprecated_ratings,
        expose_value=False,
    ),
    _watchlist_depr: str | None = typer.Option(
        None,
        "--watchlist",
        help="Deprecated alias for --watchlist-file",
        hidden=True,
        callback=_deprecated_watchlist,
        expose_value=False,
    ),
):
    """Ingest ratings and watchlist data."""
    _validate_paths(ratings, watchlist)
    res = ingest_sources(ratings_csv=ratings, watchlist_path=watchlist, data_dir=data_dir)
    typer.echo(
        f"✅ Ingested ratings: {len(res.dataset.ratings)}, watchlist: {len(res.dataset.watchlist)}"
    )


class ContentType(str, enum.Enum):
    all = "all"
    movies = "movies"
    tv = "tv"


class ModelType(str, enum.Enum):
    svd = "svd"
    elasticnet = "elasticnet"


@app.command()
def recommend(
    seeds: str = typer.Option("", help="Comma-separated IMDb IDs to base recommendations on"),
    topk: int = typer.Option(25, help="Number of recommendations to return"),
    model: ModelType = typer.Option(  # noqa: B008
        ModelType.svd,
        "--model",
        case_sensitive=False,
        help="Choose recommendation model: SVD or ElasticNet",
    ),
    user_weight: float = typer.Option(
        0.5, help="Weight for personal preferences (0.0-1.0) [OPTIMAL: 0.5] (SVD only)"
    ),
    global_weight: float = typer.Option(
        0.1, help="Weight for global popularity (0.0-1.0) [OPTIMAL: 0.1] (SVD only)"
    ),
    recency: float = typer.Option(0.0, help="Recency bias factor (SVD only)"),
    exclude_rated: bool = typer.Option(True, help="Exclude already rated items"),
    content_type: ContentType = typer.Option(  # noqa: B008
        ContentType.all,
        "--content-type",
        case_sensitive=False,
        help="Filter recommendations by content type",
    ),
    ratings: str | None = typer.Option(None, "--ratings-file", help="Path to ratings CSV file"),
    watchlist: str | None = typer.Option(
        None, "--watchlist-file", help="Path to watchlist CSV file"
    ),
    config: str | None = typer.Option(None, help="Path to config TOML file"),
    export_csv: str | None = typer.Option(None, help="Export recommendations to CSV file"),
    _ratings_depr: str | None = typer.Option(
        None,
        "--ratings",
        help="Deprecated alias for --ratings-file",
        hidden=True,
        callback=_deprecated_ratings,
        expose_value=False,
    ),
    _watchlist_depr: str | None = typer.Option(
        None,
        "--watchlist",
        help="Deprecated alias for --watchlist-file",
        hidden=True,
        callback=_deprecated_watchlist,
        expose_value=False,
    ),
):
    """Get movie recommendations using SVD or ElasticNet models."""

    # Load data
    if config:
        cfg = AppConfig.from_file(config)
        ratings = cfg.ratings_csv_path
        watchlist = cfg.watchlist_path
        data_dir = cfg.data_dir
    else:
        if not ratings or not watchlist:
            typer.echo(
                "❌ Provide --config or both --ratings-file and --watchlist-file",
                err=True,
            )
            raise typer.Exit(1)
        data_dir = "data"

    _validate_paths(ratings, watchlist)
    res = ingest_sources(ratings_csv=ratings, watchlist_path=watchlist, data_dir=data_dir)

    seeds_list = [s.strip() for s in seeds.split(",") if s.strip()]

    # Create and run recommender based on selected model
    if model == ModelType.svd:
        typer.echo("🎯 Using optimal SVD hyperparameters (discovered through rigorous testing)")

        # Create SVD recommender with built-in optimal hyperparameters
        recommender = SVDAutoRecommender(res.dataset, random_seed=42)

        # Get SVD recommendations
        scores, explanations = recommender.score(
            seeds_list, user_weight, global_weight, recency, exclude_rated
        )
        model_name = "SVD"

    else:  # model == ModelType.elasticnet
        typer.echo("🔬 Using ElasticNet with feature engineering (optimal hyperparameters)")

        # Create ElasticNet recommender with optimal hyperparameters
        recommender = ElasticNetRecommender(res.dataset, alpha=0.1, l1_ratio=0.1, random_seed=42)

        # Get ElasticNet recommendations
        scores, explanations = recommender.score(
            seeds_list, user_weight, global_weight, recency, exclude_rated
        )
        model_name = "ElasticNet"

    if not scores:
        typer.echo("❌ No recommendations found")
        return

    # Rank without content-type filtering to get global order
    ranker = Ranker(random_seed=42)
    all_recs = ranker.top_n(
        scores,
        res.dataset,
        topk=len(scores),
        explanations={model_name.lower(): explanations},
        exclude_rated=exclude_rated,
    )

    catalog = res.dataset.catalog
    if content_type is not ContentType.all:
        if "title_type" not in catalog.columns or catalog["title_type"].isna().all():
            typer.echo("❌ titleType metadata required for content-type filtering.", err=True)
            raise typer.Exit(1)

    title_map = dict(zip(catalog.get("imdb_const"), catalog.get("title_type"), strict=False))
    df = pd.DataFrame(
        {
            "imdb_const": [r.imdb_const for r in all_recs],
            "titleType": [title_map.get(r.imdb_const) for r in all_recs],
        }
    )

    try:
        df = filter_by_content_type(df, content_type.value)
    except ValueError as exc:
        typer.echo(f"❌ {exc}", err=True)
        raise typer.Exit(1) from exc

    allowed = set(df["imdb_const"].tolist())
    recommendations = [r for r in all_recs if r.imdb_const in allowed][:topk]

    if content_type is not ContentType.all:
        typer.echo(f"🎬 Filtered for content type: {content_type.value}")

    if not recommendations:
        typer.echo("❌ No recommendations after filtering")
        return

    # Display recommendations
    typer.echo(f"\n🎬 Top {len(recommendations)} {model_name} Recommendations:")
    typer.echo("=" * 80)

    for i, rec in enumerate(recommendations, 1):
        score = rec.score  # Direct attribute access
        title = rec.title or "Unknown"
        year = rec.year or ""
        genres = rec.genres or ""
        explanation = rec.why_explainer or ""

        typer.echo(f"{i:2d}. {title} ({year})")
        typer.echo(f"    🎯 Score: {score:.3f}  🎬 {genres}")

        # Show explanation
        if explanation:
            typer.echo(f"    💡 {explanation}")
        typer.echo()

    # Export to CSV if requested
    if export_csv:
        export_recommendations_csv(recommendations, export_csv, topk, model_name)
        typer.echo(f"💾 Exported {len(recommendations)} recommendations to {export_csv}")


@app.command()
def top_watchlist_movies(
    topk: int = typer.Option(10, help="Number of movie recommendations to return"),
    model: ModelType = typer.Option(  # noqa: B008
        ModelType.svd,
        "--model",
        case_sensitive=False,
        help="Choose recommendation model: SVD or ElasticNet",
    ),
    config: str | None = typer.Option("config.toml", help="Path to config TOML file"),
):
    """Get top movie recommendations from your watchlist."""

    if not config:
        typer.echo("❌ Config file required", err=True)
        raise typer.Exit(1)

    cfg = AppConfig.from_file(config)
    res = ingest_sources(cfg.ratings_csv_path, cfg.watchlist_path, cfg.data_dir)

    # Create recommender based on selected model
    if model == ModelType.svd:
        typer.echo("🎯 Using optimal SVD hyperparameters")
        recommender = SVDAutoRecommender(res.dataset, random_seed=42)
        model_name = "SVD"
    else:
        typer.echo("🔬 Using ElasticNet with feature engineering")
        recommender = ElasticNetRecommender(res.dataset, alpha=0.1, l1_ratio=0.1, random_seed=42)
        model_name = "ElasticNet"

    # Get recommendations using optimal weights
    scores, explanations = recommender.score(
        seeds=[], user_weight=0.5, global_weight=0.1, recency=0.0, exclude_rated=True
    )

    # Filter for movies only
    catalog_df = res.dataset.catalog.set_index("imdb_const")
    movie_scores = {}
    for imdb_id, score in scores.items():
        if imdb_id in catalog_df.index:
            if catalog_df.loc[imdb_id, "title_type"] == "Movie":
                movie_scores[imdb_id] = score

    # Get top movie recommendations
    ranker = Ranker(random_seed=42)
    recommendations = ranker.top_n(
        movie_scores,
        res.dataset,
        topk=topk,
        explanations={model_name.lower(): explanations},
        exclude_rated=True,
    )

    typer.echo(f"\n🎬 Top {len(recommendations)} Movie Recommendations from Watchlist:")
    typer.echo("=" * 80)

    for i, rec in enumerate(recommendations, 1):
        score = rec.score
        title = rec.title or "Unknown"
        year = rec.year or ""
        explanation = rec.why_explainer or ""

        typer.echo(f"{i:2d}. {title} ({year})")
        typer.echo(f"    🎯 {model_name} Score: {score:.3f}")
        if explanation:
            typer.echo(f"    💡 {explanation}")
        typer.echo()


@app.command()
def top_watchlist_tv(
    topk: int = typer.Option(10, help="Number of TV recommendations to return"),
    model: ModelType = typer.Option(  # noqa: B008
        ModelType.svd,
        "--model",
        case_sensitive=False,
        help="Choose recommendation model: SVD or ElasticNet",
    ),
    config: str | None = typer.Option("config.toml", help="Path to config TOML file"),
):
    """Get top TV series recommendations from your watchlist."""

    if not config:
        typer.echo("❌ Config file required", err=True)
        raise typer.Exit(1)

    cfg = AppConfig.from_file(config)
    res = ingest_sources(cfg.ratings_csv_path, cfg.watchlist_path, cfg.data_dir)

    # Create recommender based on selected model
    if model == ModelType.svd:
        typer.echo("🎯 Using optimal SVD hyperparameters")
        recommender = SVDAutoRecommender(res.dataset, random_seed=42)
        model_name = "SVD"
    else:
        typer.echo("🔬 Using ElasticNet with feature engineering")
        recommender = ElasticNetRecommender(res.dataset, alpha=0.1, l1_ratio=0.1, random_seed=42)
        model_name = "ElasticNet"

    # Get recommendations using optimal weights
    scores, explanations = recommender.score(
        seeds=[], user_weight=0.5, global_weight=0.1, recency=0.0, exclude_rated=True
    )

    # Filter for TV content (TV Series, TV Mini Series, TV Movie, TV Special)
    catalog_df = res.dataset.catalog.set_index("imdb_const")
    tv_scores = {}
    tv_types = ["TV Series", "TV Mini Series", "TV Movie", "TV Special"]

    for imdb_id, score in scores.items():
        if imdb_id in catalog_df.index:
            if catalog_df.loc[imdb_id, "title_type"] in tv_types:
                tv_scores[imdb_id] = score

    # Get top TV recommendations
    ranker = Ranker(random_seed=42)
    recommendations = ranker.top_n(
        tv_scores,
        res.dataset,
        topk=topk,
        explanations={model_name.lower(): explanations},
        exclude_rated=True,
    )

    typer.echo(f"\n📺 Top {len(recommendations)} TV Recommendations from Watchlist:")
    typer.echo("=" * 80)

    for i, rec in enumerate(recommendations, 1):
        score = rec.score
        title = rec.title or "Unknown"
        year = rec.year or ""
        genres = rec.genres or ""
        explanation = rec.why_explainer or ""

        typer.echo(f"{i:2d}. {title} ({year})")
        typer.echo(f"    🎯 {model_name} Score: {score:.3f}  🎬 {genres}")
        if explanation:
            typer.echo(f"    💡 {explanation}")
        typer.echo()


def export_recommendations_csv(recommendations, filename: str, topk: int, model_name: str = "SVD"):
    """Export recommendations to CSV file."""
    import pandas as pd

    # Prepare data for CSV export
    csv_data = []
    for i, rec in enumerate(recommendations[:topk], 1):
        csv_data.append(
            {
                "rank": i,
                "imdb_const": rec.imdb_const,
                "title": rec.title,
                "year": rec.year,
                "genres": rec.genres,
                f"{model_name.lower()}_score": rec.score,
                "explanation": rec.why_explainer,
            }
        )

    df = pd.DataFrame(csv_data)
    df.to_csv(filename, index=False)


if __name__ == "__main__":
    app()
