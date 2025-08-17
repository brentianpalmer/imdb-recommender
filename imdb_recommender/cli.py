
from __future__ import annotations
import typer
from typing import Optional
from .config import AppConfig
from .data_io import ingest_sources
from .logger import ActionLogger
from .recommender_pop import PopSimRecommender
from .recommender_svd import SVDAutoRecommender
from .ranker import Ranker

app = typer.Typer(help="IMDb Recommender CLI")

@app.command()
def ingest(ratings: str = typer.Option(...), watchlist: str = typer.Option(...), data_dir: str = typer.Option("data")):
    res = ingest_sources(ratings_csv=ratings, watchlist_path=watchlist, data_dir=data_dir)
    typer.echo(f"Ingested ratings: {len(res.dataset.ratings)}, watchlist: {len(res.dataset.watchlist)}")

@app.command()
def recommend(seeds: str = typer.Option(""), topk: int = typer.Option(25), user_weight: float = typer.Option(0.7), global_weight: float = typer.Option(0.3), recency: float = typer.Option(0.0), exclude_rated: bool = typer.Option(True), ratings: Optional[str] = typer.Option(None), watchlist: Optional[str] = typer.Option(None), config: Optional[str] = typer.Option(None)):
    if config:
        cfg = AppConfig.from_file(config); res = ingest_sources(cfg.ratings_csv_path, cfg.watchlist_path, cfg.data_dir)
    else:
        if not ratings or not watchlist:
            typer.echo("Provide --config or both --ratings and --watchlist", err=True); raise typer.Exit(1)
        res = ingest_sources(ratings_csv=ratings, watchlist_path=watchlist, data_dir="data")
    seeds_list = [s.strip() for s in seeds.split(",") if s.strip()]
    pop = PopSimRecommender(res.dataset, random_seed=42); svd = SVDAutoRecommender(res.dataset, random_seed=42)
    pop_scores, pop_expl = pop.score(seeds_list, user_weight, global_weight, recency, exclude_rated)
    svd_scores, svd_expl = svd.score(seeds_list, user_weight, global_weight, recency, exclude_rated)
    blended = Ranker().blend({"pop": pop_scores, "svd": svd_scores})
    recs = Ranker().top_n(blended, res.dataset, topk=topk, explanations={"pop": pop_expl, "svd": svd_expl}, exclude_rated=exclude_rated)
    for r in recs:
        typer.echo(f"{r.imdb_const}	{r.title or ''} ({r.year or ''})	score={r.score:.3f}	{r.why_explainer}")

@app.command()
def rate(ttid: str, rating: int = typer.Argument(...), notes: Optional[str] = typer.Option(None, "--notes")):
    logger = ActionLogger(); logger.log_rate(imdb_const=ttid, rating=rating, notes=notes, source="cli")
    typer.echo(f"Logged rating {rating} for {ttid} (batch {logger.batch_id})")

@app.command("watchlist")
def watchlist_cmd(action: str = typer.Argument(...), ttid: str = typer.Argument(...)):
    logger = ActionLogger()
    if action not in {"add", "remove"}:
        typer.echo("Action must be 'add' or 'remove'", err=True); raise typer.Exit(1)
    logger.log_watchlist(imdb_const=ttid, add=(action == "add"), source="cli")
    typer.echo(f"Logged {action} for {ttid} (batch {logger.batch_id})")

@app.command("quick-review")
def quick_review(ttid: str, rating: Optional[int] = typer.Option(None, "--rating"), wl: Optional[str] = typer.Option(None, "--watchlist"), notes: Optional[str] = typer.Option(None, "--notes")):
    logger = ActionLogger()
    if rating is not None: logger.log_rate(imdb_const=ttid, rating=int(rating), notes=notes, source="cli")
    if wl is not None:
        if wl not in {"add", "remove"}:
            typer.echo("--watchlist must be 'add' or 'remove'", err=True); raise typer.Exit(1)
        logger.log_watchlist(imdb_const=ttid, add=(wl == "add"), notes=notes, source="cli")
    typer.echo(f"Logged quick-review for {ttid} (batch {logger.batch_id})")

@app.command("export-log")
def export_log(out: str = typer.Option("data/imdb_actions_log.csv", "--out")):
    logger = ActionLogger(); path = logger.export(out_path=out); typer.echo(path)

@app.command()
def explain(ttid: str, ratings: str = typer.Option(...), watchlist: str = typer.Option(...)):
    res = ingest_sources(ratings_csv=ratings, watchlist_path=watchlist, data_dir="data")
    pop = PopSimRecommender(res.dataset)
    scores, expl = pop.score(seeds=[ttid], user_weight=0.7, global_weight=0.3, recency=0.5, exclude_rated=False)
    typer.echo(expl.get(ttid) or "blend of your taste and popularity")
