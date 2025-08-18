"""Performance and benchmarking tests for the IMDb Recommender package."""

import tempfile
import time

from imdb_recommender.data_io import ingest_sources
from imdb_recommender.ranker import Ranker
from imdb_recommender.recommender_pop import PopSimRecommender
from imdb_recommender.recommender_svd import SVDAutoRecommender


class TestPerformance:
    """Test performance and scalability."""

    def test_ingest_performance(self):
        """Test data ingestion performance."""
        start_time = time.time()

        with tempfile.TemporaryDirectory() as temp_dir:
            result = ingest_sources(
                "tests/fixtures_ratings.csv", "tests/fixtures_watchlist.csv", temp_dir
            )

        elapsed = time.time() - start_time

        # Should complete quickly for small datasets
        assert elapsed < 5.0, f"Ingestion took too long: {elapsed:.2f}s"
        assert len(result.dataset.ratings) > 0
        assert len(result.dataset.watchlist) > 0

    def test_recommendation_performance(self):
        """Test recommendation generation performance."""
        # Load data
        with tempfile.TemporaryDirectory() as temp_dir:
            result = ingest_sources(
                "tests/fixtures_ratings.csv", "tests/fixtures_watchlist.csv", temp_dir
            )
            dataset = result.dataset

        # Test PopSim performance
        start_time = time.time()
        pop_rec = PopSimRecommender(dataset, random_seed=42)
        seed_ids = dataset.ratings["imdb_const"].head(1).tolist()
        pop_scores, pop_expl = pop_rec.score(seed_ids, 0.7, 0.3, 0.0, exclude_rated=True)
        pop_time = time.time() - start_time

        # Test SVD performance
        start_time = time.time()
        svd_rec = SVDAutoRecommender(dataset, random_seed=42)
        svd_scores, svd_expl = svd_rec.score(seed_ids, 0.7, 0.3, 0.0, exclude_rated=True)
        svd_time = time.time() - start_time

        # Test blending performance
        start_time = time.time()
        ranker = Ranker()
        blended = ranker.blend({"pop": pop_scores, "svd": svd_scores})
        recs = ranker.top_n(
            blended,
            dataset,
            topk=10,
            explanations={"pop": pop_expl, "svd": svd_expl},
            exclude_rated=True,
        )
        blend_time = time.time() - start_time

        # Performance assertions (should be fast for small datasets)
        assert pop_time < 2.0, f"PopSim too slow: {pop_time:.2f}s"
        assert svd_time < 3.0, f"SVD too slow: {svd_time:.2f}s"
        assert blend_time < 1.0, f"Blending too slow: {blend_time:.2f}s"

        # Verify results are valid
        assert len(recs) <= 10
        assert all(rec.score > 0 for rec in recs)

        print("Performance results:")
        print(f"  PopSim: {pop_time:.3f}s")
        print(f"  SVD: {svd_time:.3f}s")
        print(f"  Blending: {blend_time:.3f}s")
        print(f"  Total recommendations: {len(recs)}")


class TestScalabilityLimits:
    """Test behavior with edge cases and limits."""

    def test_large_seed_list(self):
        """Test with many seed titles."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = ingest_sources(
                "tests/fixtures_ratings.csv", "tests/fixtures_watchlist.csv", temp_dir
            )
            dataset = result.dataset

        # Use all available movie IDs as seeds
        all_ids = dataset.ratings["imdb_const"].tolist()

        pop_rec = PopSimRecommender(dataset, random_seed=42)
        scores, explanations = pop_rec.score(all_ids, 0.7, 0.3, 0.0, exclude_rated=True)

        # Should handle gracefully
        assert isinstance(scores, dict)
        assert isinstance(explanations, dict)

    def test_extreme_parameters(self):
        """Test with extreme parameter values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = ingest_sources(
                "tests/fixtures_ratings.csv", "tests/fixtures_watchlist.csv", temp_dir
            )
            dataset = result.dataset

        seed_ids = dataset.ratings["imdb_const"].head(1).tolist()
        pop_rec = PopSimRecommender(dataset, random_seed=42)

        # Test extreme weights
        scores1, _ = pop_rec.score(seed_ids, 1.0, 0.0, 0.0, exclude_rated=True)  # Pure user
        scores2, _ = pop_rec.score(seed_ids, 0.0, 1.0, 0.0, exclude_rated=True)  # Pure global
        scores3, _ = pop_rec.score(seed_ids, 0.5, 0.5, 1.0, exclude_rated=True)  # Max recency

        # Should all return valid results
        assert isinstance(scores1, dict)
        assert isinstance(scores2, dict)
        assert isinstance(scores3, dict)

    def test_object_creation_stress(self):
        """Test creating multiple recommender instances."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = ingest_sources(
                "tests/fixtures_ratings.csv", "tests/fixtures_watchlist.csv", temp_dir
            )
            dataset = result.dataset

        # Create multiple instances
        recommenders = []
        for i in range(5):
            pop_rec = PopSimRecommender(dataset, random_seed=42 + i)
            svd_rec = SVDAutoRecommender(dataset, random_seed=42 + i)
            recommenders.extend([pop_rec, svd_rec])

        # Test they all work
        seed_ids = dataset.ratings["imdb_const"].head(1).tolist()

        results = []
        for rec in recommenders:
            scores, explanations = rec.score(seed_ids, 0.7, 0.3, 0.0, exclude_rated=True)
            results.append((scores, explanations))

        # All should return valid results
        assert len(results) == 10
        for scores, explanations in results:
            assert isinstance(scores, dict)
            assert isinstance(explanations, dict)
            assert len(scores) > 0

        print(f"Successfully created and tested {len(recommenders)} recommender instances")
