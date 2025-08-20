"""Performance and benchmarking tests for the SVD Recommender package."""

import tempfile
import time

from imdb_recommender.data_io import ingest_sources
from imdb_recommender.ranker import Ranker
from imdb_recommender.recommender_svd import SVDAutoRecommender


class TestSVDPerformance:
    """Test SVD performance and scalability."""

    def test_svd_recommendation_performance(self):
        """Test SVD recommendation generation speed."""
        ratings_csv = "tests/fixtures_ratings.csv"
        watchlist_csv = "tests/fixtures_watchlist.csv"

        with tempfile.TemporaryDirectory() as temp_dir:
            result = ingest_sources(ratings_csv, watchlist_csv, temp_dir)
            dataset = result.dataset
            seed_ids = dataset.ratings["imdb_const"].head(3).tolist()

            # Test SVD performance with optimal hyperparameters
            svd_rec = SVDAutoRecommender(dataset, random_seed=42)
            start_time = time.time()
            svd_scores, svd_expl = svd_rec.score(seed_ids, 0.5, 0.1, 0.0, exclude_rated=True)
            svd_time = time.time() - start_time

            print(f"SVD recommendation completed in {svd_time:.3f}s")
            print(f"Generated {len(svd_scores)} recommendations")

            # Performance assertions
            assert svd_time < 10.0, f"SVD took too long: {svd_time:.3f}s"
            assert len(svd_scores) > 0
            assert len(svd_expl) > 0
            assert len(svd_scores) == len(svd_expl)

    def test_ranking_performance(self):
        """Test recommendation ranking and formatting performance."""
        ratings_csv = "tests/fixtures_ratings.csv"
        watchlist_csv = "tests/fixtures_watchlist.csv"

        with tempfile.TemporaryDirectory() as temp_dir:
            result = ingest_sources(ratings_csv, watchlist_csv, temp_dir)
            dataset = result.dataset

            # Generate SVD scores
            svd_rec = SVDAutoRecommender(dataset, random_seed=42)
            scores, explanations = svd_rec.score([], 0.5, 0.1, 0.0, exclude_rated=True)

            # Test ranking performance
            ranker = Ranker(random_seed=42)
            start_time = time.time()

            top_recs = ranker.top_n(
                scores,
                dataset,
                topk=25,
                explanations={"svd": explanations},
                exclude_rated=True,
            )

            ranking_time = time.time() - start_time

            print(f"Ranking completed in {ranking_time:.3f}s")
            print(f"Generated {len(top_recs)} top recommendations")

            # Performance assertions
            assert ranking_time < 2.0, f"Ranking took too long: {ranking_time:.3f}s"
            assert len(top_recs) <= 25
            assert len(top_recs) > 0

            # Verify sorted by score
            if len(top_recs) > 1:
                scores_list = [rec.score for rec in top_recs]
                assert scores_list == sorted(scores_list, reverse=True)
