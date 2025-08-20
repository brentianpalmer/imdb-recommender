from .config import AppConfig
from .data_io import Dataset, ingest_sources
from .logger import ActionLogger
from .ranker import Ranker
from .recommender_base import Registry
from .recommender_svd import SVDAutoRecommender


class Recommender:
    def __init__(self, config: AppConfig, dataset: Dataset | None = None):
        self.config = config
        self.dataset = dataset
        self.logger = ActionLogger(self.config.data_dir)
        Registry.register("svd", SVDAutoRecommender)
        self.ranker = Ranker(random_seed=self.config.random_seed)

    @classmethod
    def from_config(cls, config_path: str) -> "Recommender":
        config = AppConfig.from_file(config_path)
        ds = ingest_sources(
            ratings_csv=config.ratings_csv_path,
            watchlist_path=config.watchlist_path,
            data_dir=config.data_dir,
        ).dataset
        return cls(config=config, dataset=ds)

    def recommend(
        self,
        seeds=None,
        topk=25,
        user_weight=0.7,
        global_weight=0.3,
        recency=0.0,
        exclude_rated=True,
        algos=None,
    ):
        assert self.dataset is not None, "Dataset not loaded. Run ingest or from_config."
        algos = algos or ["svd"]
        algo_scores, algo_explain = {}, {}
        for name in algos:
            algo_cls = Registry.get(name)
            algo = algo_cls(self.dataset, random_seed=self.config.random_seed)
            scores, explain = algo.score(
                seeds=seeds or [],
                user_weight=user_weight,
                global_weight=global_weight,
                recency=recency,
                exclude_rated=exclude_rated,
            )
            algo_scores[name] = scores
            algo_explain[name] = explain
        blended = self.ranker.blend(algo_scores)
        return self.ranker.top_n(
            blended, self.dataset, topk=topk, explanations=algo_explain, exclude_rated=exclude_rated
        )

    def rate(self, imdb_const: str, rating: int, notes: str | None = None, source: str = "api"):
        self.logger.log_rate(imdb_const=imdb_const, rating=rating, notes=notes, source=source)

    def watchlist_add(self, imdb_const: str, notes: str | None = None, source: str = "api"):
        self.logger.log_watchlist(imdb_const=imdb_const, add=True, notes=notes, source=source)

    def watchlist_remove(self, imdb_const: str, notes: str | None = None, source: str = "api"):
        self.logger.log_watchlist(imdb_const=imdb_const, add=False, notes=notes, source=source)

    def export_log(self, out_path: str | None = None) -> str:
        return self.logger.export(out_path=out_path)
