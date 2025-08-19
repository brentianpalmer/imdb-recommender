from __future__ import annotations

import typer

from .config import AppConfig
from .data_io import ingest_sources
from .hyperparameter_tuning import HyperparameterTuningPipeline
from .logger import ActionLogger
from .ranker import Ranker
from .recommender_all_in_one import AllInOneRecommender
from .recommender_pop import PopSimRecommender
from .recommender_svd import SVDAutoRecommender

app = typer.Typer(help="IMDb Recommender CLI")


@app.command()
def ingest(
    ratings: str = typer.Option(...),
    watchlist: str = typer.Option(...),
    data_dir: str = typer.Option("data"),
):
    res = ingest_sources(ratings_csv=ratings, watchlist_path=watchlist, data_dir=data_dir)
    typer.echo(
        f"Ingested ratings: {len(res.dataset.ratings)}, watchlist: {len(res.dataset.watchlist)}"
    )


@app.command()
def recommend(
    seeds: str = typer.Option(""),
    topk: int = typer.Option(25),
    user_weight: float = typer.Option(0.7),
    global_weight: float = typer.Option(0.3),
    recency: float = typer.Option(0.0),
    exclude_rated: bool = typer.Option(True),
    ratings: str | None = typer.Option(None),
    watchlist: str | None = typer.Option(None),
    config: str | None = typer.Option(None),
):
    if config:
        cfg = AppConfig.from_file(config)
        res = ingest_sources(cfg.ratings_csv_path, cfg.watchlist_path, cfg.data_dir)
    else:
        if not ratings or not watchlist:
            typer.echo("Provide --config or both --ratings and --watchlist", err=True)
            raise typer.Exit(1)
        res = ingest_sources(ratings_csv=ratings, watchlist_path=watchlist, data_dir="data")
    seeds_list = [s.strip() for s in seeds.split(",") if s.strip()]
    pop = PopSimRecommender(res.dataset, random_seed=42)
    svd = SVDAutoRecommender(res.dataset, random_seed=42)
    pop_scores, pop_expl = pop.score(seeds_list, user_weight, global_weight, recency, exclude_rated)
    svd_scores, svd_expl = svd.score(seeds_list, user_weight, global_weight, recency, exclude_rated)
    blended = Ranker().blend({"pop": pop_scores, "svd": svd_scores})
    recs = Ranker().top_n(
        blended,
        res.dataset,
        topk=topk,
        explanations={"pop": pop_expl, "svd": svd_expl},
        exclude_rated=exclude_rated,
    )
    for r in recs:
        typer.echo(
            f"{r.imdb_const}    {r.title or ''} ({r.year or ''})    "
            f"score={r.score:.3f} {r.why_explainer}"
        )


@app.command()
def rate(
    ttid: str,
    rating: int = typer.Argument(...),
    notes: str | None = typer.Option(None, "--notes"),
):
    logger = ActionLogger()
    logger.log_rate(imdb_const=ttid, rating=rating, notes=notes, source="cli")
    typer.echo(f"Logged rating {rating} for {ttid} (batch {logger.batch_id})")


@app.command("watchlist")
def watchlist_cmd(action: str = typer.Argument(...), ttid: str = typer.Argument(...)):
    logger = ActionLogger()
    if action not in {"add", "remove"}:
        typer.echo("Action must be 'add' or 'remove'", err=True)
        raise typer.Exit(1)
    logger.log_watchlist(imdb_const=ttid, add=(action == "add"), source="cli")
    typer.echo(f"Logged {action} for {ttid} (batch {logger.batch_id})")


@app.command("quick-review")
def quick_review(
    ttid: str,
    rating: int | None = typer.Option(None, "--rating"),
    wl: str | None = typer.Option(None, "--watchlist"),
    notes: str | None = typer.Option(None, "--notes"),
):
    logger = ActionLogger()
    if rating is not None:
        logger.log_rate(imdb_const=ttid, rating=int(rating), notes=notes, source="cli")
    if wl is not None:
        if wl not in {"add", "remove"}:
            typer.echo("--watchlist must be 'add' or 'remove'", err=True)
            raise typer.Exit(1)
        logger.log_watchlist(imdb_const=ttid, add=(wl == "add"), notes=notes, source="cli")
    typer.echo(f"Logged quick-review for {ttid} (batch {logger.batch_id})")


@app.command("all-in-one")
def all_in_one_recommend(
    topk: int = typer.Option(25, help="Number of recommendations to return"),
    user_weight: float = typer.Option(0.7, help="Weight for personal preferences"),
    global_weight: float = typer.Option(0.3, help="Weight for popularity"),
    exclude_rated: bool = typer.Option(True, help="Exclude already rated items"),
    watchlist_only: bool = typer.Option(False, help="Only recommend from unrated watchlist items"),
    content_type: str | None = typer.Option(
        None, help="Filter by content type (Movie, TV Series, TV Mini Series, etc.)"
    ),
    candidates: int = typer.Option(
        500, help="Size of candidate pool (ignored if --watchlist-only)"
    ),
    save_model: str | None = typer.Option(None, help="Path to save trained model"),
    export_csv: str | None = typer.Option(None, help="Path to export recommendations CSV"),
    evaluate: bool = typer.Option(False, help="Run evaluation with temporal split"),
    ratings: str | None = typer.Option(None, help="Ratings CSV path"),
    watchlist: str | None = typer.Option(None, help="Watchlist path"),
    config: str | None = typer.Option(None, help="Config file path"),
):
    """Run the All-in-One Four-Stage IMDb Recommender."""

    # Load data
    if config:
        cfg = AppConfig.from_file(config)
        res = ingest_sources(cfg.ratings_csv_path, cfg.watchlist_path, cfg.data_dir)
    else:
        if not ratings or not watchlist:
            typer.echo("Provide --config or both --ratings and --watchlist", err=True)
            raise typer.Exit(1)
        res = ingest_sources(ratings_csv=ratings, watchlist_path=watchlist, data_dir="data")

    typer.echo("🚀 Starting All-in-One Four-Stage Recommender...")

    # Initialize recommender
    recommender = AllInOneRecommender(res.dataset, random_seed=42)

    # Try to load cached hyperparameters
    cached_params = HyperparameterTuningPipeline.load_best_hyperparameters("all-in-one")
    if cached_params:
        typer.echo("🔧 Using cached optimal hyperparameters...")
        recommender.apply_hyperparameters(cached_params)
    else:
        typer.echo("⚠️ No cached hyperparameters found, using defaults...")
        typer.echo(
            "   💡 Run 'imdbrec hyperparameter-tune --models all-in-one' to optimize parameters"
        )

    # Determine candidates based on options
    candidate_list = None
    if watchlist_only:
        # Get unrated watchlist items
        rated_items = (
            set(res.dataset.ratings["imdb_const"].values) if len(res.dataset.ratings) > 0 else set()
        )
        unrated_watchlist = [
            item for item in res.dataset.watchlist["imdb_const"].tolist() if item not in rated_items
        ]

        if not unrated_watchlist:
            typer.echo("🤷 No unrated items in your watchlist!")
            raise typer.Exit(0)

        candidate_list = unrated_watchlist
        typer.echo(f"🎯 Analyzing {len(candidate_list)} unrated watchlist items...")
    else:
        # Use intelligent candidate building
        candidate_list = recommender.build_candidates(max_candidates=candidates)
        typer.echo(f"🧠 Built intelligent candidate pool of {len(candidate_list)} items")

    # Apply content type filter if specified
    if content_type:
        catalog = res.dataset.catalog
        if "title_type" not in catalog.columns:
            typer.echo(
                "⚠️ Content type filtering not available - title_type column missing", err=True
            )
        else:
            # Validate content type
            available_types = catalog["title_type"].value_counts()
            if content_type not in available_types.index:
                typer.echo(f"❌ Content type '{content_type}' not found.", err=True)
                typer.echo(f"Available types: {', '.join(available_types.index.tolist())}")
                raise typer.Exit(1)

            # Filter candidates by content type
            type_filtered_catalog = catalog[catalog["title_type"] == content_type]
            original_count = len(candidate_list)
            candidate_list = [
                item
                for item in candidate_list
                if item in type_filtered_catalog["imdb_const"].values
            ]

            if not candidate_list:
                typer.echo(f"❌ No {content_type} items found in candidate pool!")
                raise typer.Exit(1)

            typer.echo(
                f"🎬 Filtered to {len(candidate_list)} {content_type} items "
                f"(from {original_count} total)"
            )

    # Generate recommendations
    scores, explanations = recommender.score(
        seeds=[],  # All-in-one doesn't use seeds
        user_weight=user_weight,
        global_weight=global_weight,
        exclude_rated=exclude_rated,
        candidates=candidate_list,
    )

    # Get top recommendations
    sorted_recs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]

    # Display recommendations
    header = ""
    if watchlist_only and content_type:
        header = f"🎯 Top {len(sorted_recs)} {content_type} Watchlist Recommendations:"
    elif watchlist_only:
        header = f"🎯 Top {len(sorted_recs)} Watchlist Recommendations:"
    elif content_type:
        header = f"🎬 Top {len(sorted_recs)} {content_type} Recommendations:"
    else:
        header = f"📊 Top {len(sorted_recs)} Recommendations:"

    typer.echo(f"\n{header}")
    typer.echo("=" * 80)

    for i, (const, score) in enumerate(sorted_recs, 1):
        # Find item details
        item_row = res.dataset.catalog[res.dataset.catalog["imdb_const"] == const]
        if len(item_row) > 0:
            item = item_row.iloc[0]
            title = item.get("title", "Unknown")
            year = item.get("year", "Unknown")
            rating = item.get("imdb_rating", "N/A")
            explanation = explanations.get(const, "AI-powered recommendation")

            typer.echo(f"{i:2d}. {const} | {title} ({year}) | IMDb: {rating} | Score: {score:.3f}")
            typer.echo(f"    💡 {explanation}")
            typer.echo()

    # Export CSV if requested
    if export_csv:
        recommender.export_recommendations_csv(scores, export_csv, topk)
        typer.echo(f"📝 Exported recommendations to {export_csv}")

    # Save model if requested
    if save_model:
        recommender.save_model(save_model)
        typer.echo(f"💾 Model saved to {save_model}")

    # Run evaluation if requested
    if evaluate and len(res.dataset.ratings) >= 10:
        typer.echo("📊 Running evaluation with temporal split...")
        metrics = recommender.evaluate_temporal_split()

        typer.echo("\n📈 Evaluation Results:")
        typer.echo("-" * 40)
        for metric, value in metrics.items():
            typer.echo(f"{metric}: {value:.4f}")
    elif evaluate:
        typer.echo("⚠️ Need at least 10 ratings for evaluation")

    typer.echo("✅ All-in-One recommendation complete!")


@app.command("export-log")
def export_log(out: str = typer.Option("data/imdb_actions_log.csv", "--out")):
    logger = ActionLogger()
    path = logger.export(out_path=out)
    typer.echo(path)


@app.command()
def explain(ttid: str, ratings: str = typer.Option(...), watchlist: str = typer.Option(...)):
    res = ingest_sources(ratings_csv=ratings, watchlist_path=watchlist, data_dir="data")
    pop = PopSimRecommender(res.dataset)
    scores, expl = pop.score(
        seeds=[ttid], user_weight=0.7, global_weight=0.3, recency=0.5, exclude_rated=False
    )
    typer.echo(expl.get(ttid) or "blend of your taste and popularity")


@app.command("hyperparameter-tune")
def hyperparameter_tune(
    ratings: str | None = typer.Option(None, help="Ratings CSV path"),
    watchlist: str | None = typer.Option(None, help="Watchlist path"),
    config: str | None = typer.Option(None, help="Config file path"),
    models: str = typer.Option(
        "all", help="Models to tune: 'all', 'all-in-one', 'pop-sim', 'svd', or comma-separated list"
    ),
    test_size: float = typer.Option(0.2, help="Test set proportion (0.1-0.4)"),
    cv_folds: int = typer.Option(3, help="Cross-validation folds (2-5)"),
    random_state: int = typer.Option(42, help="Random seed for reproducibility"),
):
    """
    Run comprehensive hyperparameter tuning for recommender systems.

    This command performs stratified train/test splits on your ratings data,
    runs cross-validation hyperparameter tuning, and evaluates models on
    multiple metrics including RMSE, MAE, Precision@K, Recall@K, and NDCG@K.

    The goal is to find models that can accurately predict your ratings and
    provide high-quality recommendations.
    """

    # Validate parameters
    if not (0.1 <= test_size <= 0.4):
        typer.echo("❌ Test size must be between 0.1 and 0.4", err=True)
        raise typer.Exit(1)

    if not (2 <= cv_folds <= 5):
        typer.echo("❌ CV folds must be between 2 and 5", err=True)
        raise typer.Exit(1)

    # Load data
    if config:
        cfg = AppConfig.from_file(config)
        res = ingest_sources(cfg.ratings_csv_path, cfg.watchlist_path, cfg.data_dir)
    else:
        if not ratings or not watchlist:
            typer.echo("❌ Provide --config or both --ratings and --watchlist", err=True)
            raise typer.Exit(1)
        res = ingest_sources(ratings_csv=ratings, watchlist_path=watchlist, data_dir="data")

    # Check minimum data requirements
    if len(res.dataset.ratings) < 50:
        typer.echo("❌ Need at least 50 ratings for meaningful hyperparameter tuning", err=True)
        raise typer.Exit(1)

    # Parse models
    if models.lower() == "all":
        model_list = ["all-in-one", "pop-sim", "svd"]
    else:
        model_list = [m.strip() for m in models.split(",")]
        valid_models = {"all-in-one", "pop-sim", "svd"}
        invalid_models = set(model_list) - valid_models
        if invalid_models:
            typer.echo(f"❌ Invalid models: {invalid_models}. Valid: {valid_models}", err=True)
            raise typer.Exit(1)

    typer.echo("🚀 Starting hyperparameter tuning pipeline...")
    typer.echo(
        f"   Dataset: {len(res.dataset.ratings)} ratings, "
        f"{len(res.dataset.watchlist)} watchlist items"
    )
    typer.echo(f"   Models: {model_list}")
    typer.echo(f"   Test size: {test_size*100:.1f}%")
    typer.echo(f"   CV folds: {cv_folds}")
    typer.echo(f"   Random state: {random_state}")
    typer.echo()

    # Initialize tuning pipeline
    tuner = HyperparameterTuningPipeline(
        dataset=res.dataset, test_size=test_size, random_state=random_state
    )

    try:
        # Run tuning
        tuner.run_full_tuning(cv_folds=cv_folds, models=model_list)

        typer.echo("🎉 Hyperparameter tuning completed successfully!")
        typer.echo("💾 Results saved to: hyperparameter_results/")
        typer.echo()
        typer.echo("📋 Next Steps:")
        typer.echo("   1. Review the results in the hyperparameter_results/ directory")
        typer.echo("   2. Use the best parameters in your recommendation commands")
        typer.echo("   3. Apply the tuned models to your watchlist for recommendations")

    except Exception as e:
        typer.echo(f"❌ Hyperparameter tuning failed: {e}", err=True)
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
