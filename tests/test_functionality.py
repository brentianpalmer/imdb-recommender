"""Comprehensive functionality tests for IMDb Recommender package."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from imdb_recommender.config import AppConfig
from imdb_recommender.data_io import Dataset, ingest_sources
from imdb_recommender.features import (
    content_vector,
    cosine,
    genres_to_vec,
    recency_weight,
    year_to_bucket,
)
from imdb_recommender.logger import ActionLogger
from imdb_recommender.ranker import Ranker
from imdb_recommender.recommender_pop import PopSimRecommender
from imdb_recommender.recommender_svd import SVDAutoRecommender
from imdb_recommender.schemas import Recommendation


class TestFeatureEngineering:
    """Test feature engineering functions."""

    def test_genres_to_vec(self):
        """Test genre vectorization."""
        # Test valid genres
        vec = genres_to_vec("Action,Drama")
        assert vec.shape[0] == 22  # Number of genres
        assert vec[0] > 0  # Action should be present
        assert vec[7] > 0  # Drama should be present
        assert np.sum(vec > 0) == 2  # Only 2 genres should be non-zero

        # Test empty/invalid input
        vec_empty = genres_to_vec("")
        assert np.sum(vec_empty) == 0

        vec_none = genres_to_vec(None)
        assert np.sum(vec_none) == 0

    def test_year_to_bucket(self):
        """Test year bucketing."""
        # Test normal years
        vec_1985 = year_to_bucket(1985)
        assert vec_1985[1] == 1.0  # Should be in 1980-1990 bucket

        vec_2010 = year_to_bucket(2010)
        assert vec_2010[3] == 1.0  # Should be in 2010 bucket (index 3)

        vec_2025 = year_to_bucket(2025)
        assert vec_2025[5] == 1.0  # Should be in the last bucket (> 2020)

        # Test edge cases
        vec_none = year_to_bucket(None)
        assert np.sum(vec_none) == 0

        vec_na = year_to_bucket(pd.NA)
        assert np.sum(vec_na) == 0

    def test_content_vector(self):
        """Test content vector generation."""
        row = pd.Series({"genres": "Action,Drama", "year": 2010})
        vec = content_vector(row)
        assert len(vec) == 28  # 22 genres + 6 year buckets

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        a = np.array([1, 0, 1])
        b = np.array([1, 1, 0])
        sim = cosine(a, b)
        assert 0 <= sim <= 1

        # Test identical vectors
        sim_identical = cosine(a, a)
        assert abs(sim_identical - 1.0) < 1e-6

        # Test zero vectors
        zero = np.zeros(3)
        sim_zero = cosine(a, zero)
        assert sim_zero == 0.0

    def test_recency_weight(self):
        """Test recency weighting."""
        # Test normal year
        weight = recency_weight(2010, 0.5)
        assert isinstance(weight, float)
        assert weight > 0

        # Test NA values
        weight_na = recency_weight(pd.NA, 0.5)
        assert weight_na == 1.0

        # Test alpha = 0
        weight_zero = recency_weight(2010, 0.0)
        assert weight_zero == 1.0


class TestDataIngestion:
    """Test data ingestion functionality."""

    def test_ingest_sources_with_fixtures(self):
        """Test data ingestion with fixture files."""
        ratings_path = "tests/fixtures_ratings.csv"
        watchlist_path = "tests/fixtures_watchlist.csv"

        with tempfile.TemporaryDirectory() as temp_dir:
            result = ingest_sources(ratings_path, watchlist_path, temp_dir)

            # Check that dataset was created
            assert hasattr(result, "dataset")
            assert isinstance(result.dataset, Dataset)

            # Check ratings data
            assert len(result.dataset.ratings) > 0
            required_cols = ["imdb_const", "my_rating", "title", "year", "genres"]
            for col in required_cols:
                assert col in result.dataset.ratings.columns

            # Check watchlist data
            assert len(result.dataset.watchlist) > 0
            assert "imdb_const" in result.dataset.watchlist.columns

            # Check catalog was created
            assert len(result.dataset.catalog) > 0


class TestRecommenders:
    """Test recommendation algorithms."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        ratings_data = {
            "imdb_const": ["tt1", "tt2", "tt3", "tt4", "tt5"],
            "my_rating": [8, 9, 7, 6, 10],
            "title": ["Movie 1", "Movie 2", "Movie 3", "Movie 4", "Movie 5"],
            "year": [2010, 2015, 2020, 2005, 2018],
            "genres": ["Action", "Drama", "Comedy", "Action,Drama", "Sci-Fi"],
            "imdb_rating": [7.5, 8.2, 6.8, 7.1, 8.9],
            "num_votes": [100000, 150000, 80000, 90000, 200000],
        }

        watchlist_data = {
            "imdb_const": ["tt6", "tt7", "tt8"],
            "in_watchlist": [True, True, True],
            "title": ["Movie 6", "Movie 7", "Movie 8"],
            "year": [2021, 2019, 2022],
            "genres": ["Horror", "Romance", "Thriller"],
            "imdb_rating": [6.5, 7.8, 7.2],
            "num_votes": [75000, 110000, 95000],
        }

        ratings_df = pd.DataFrame(ratings_data)
        watchlist_df = pd.DataFrame(watchlist_data)

        # Create dataset with proper dataclass constructor
        return Dataset(ratings=ratings_df, watchlist=watchlist_df)

    def test_pop_recommender(self, sample_dataset):
        """Test popularity-based recommender."""
        recommender = PopSimRecommender(sample_dataset, random_seed=42)

        # Test scoring
        scores, explanations = recommender.score(["tt1"], 0.7, 0.3, 0.0, exclude_rated=True)

        assert isinstance(scores, dict)
        assert isinstance(explanations, dict)
        assert len(scores) > 0
        assert len(explanations) > 0

        # Should exclude rated items
        for rated_id in ["tt1", "tt2", "tt3", "tt4", "tt5"]:
            assert rated_id not in scores

    def test_svd_recommender(self, sample_dataset):
        """Test SVD recommender."""
        recommender = SVDAutoRecommender(sample_dataset, random_seed=42)

        # Test scoring
        scores, explanations = recommender.score(["tt1"], 0.7, 0.3, 0.0, exclude_rated=True)

        assert isinstance(scores, dict)
        assert isinstance(explanations, dict)
        assert len(scores) > 0
        assert len(explanations) > 0


class TestRanker:
    """Test recommendation ranking and blending."""

    def test_blend_scores(self):
        """Test score blending."""
        ranker = Ranker()

        algo_scores = {
            "pop": {"tt1": 0.8, "tt2": 0.6, "tt3": 0.7},
            "svd": {"tt1": 0.9, "tt2": 0.5, "tt3": 0.8},
        }

        blended = ranker.blend(algo_scores)

        assert isinstance(blended, dict)
        assert len(blended) == 3
        assert all(0 <= score <= 1 for score in blended.values())

        # tt1 should have highest score (both algorithms rate it highly)
        assert blended["tt1"] > blended["tt2"]

    def test_top_n_recommendations(self, sample_dataset=None):
        """Test top-N recommendation generation."""
        # Create minimal dataset for testing with proper watchlist structure
        ratings_df = pd.DataFrame(
            {
                "imdb_const": ["tt1", "tt2"],
                "my_rating": [8, 9],
                "title": ["Movie 1", "Movie 2"],
                "year": [2010, 2015],
                "genres": ["Action", "Drama"],
                "imdb_rating": [7.5, 8.2],
                "num_votes": [100000, 150000],
            }
        )

        # Create watchlist with tt3 so it appears in catalog
        watchlist_df = pd.DataFrame(
            {
                "imdb_const": ["tt3"],
                "in_watchlist": [True],
                "title": ["Movie 3"],
                "year": [2020],
                "genres": ["Comedy"],
                "imdb_rating": [6.8],
                "num_votes": [80000],
            }
        )

        sample_dataset = Dataset(ratings=ratings_df, watchlist=watchlist_df)

        ranker = Ranker()
        blended_scores = {"tt3": 0.8, "tt1": 0.6}  # tt1 should be excluded if exclude_rated=True

        recs = ranker.top_n(blended_scores, sample_dataset, topk=5, exclude_rated=True)

        assert isinstance(recs, list)
        assert len(recs) <= 5
        assert all(isinstance(rec, Recommendation) for rec in recs)

        # Should not include rated items
        rec_ids = [rec.imdb_const for rec in recs]
        assert "tt1" not in rec_ids  # Rated item should be excluded
        assert "tt2" not in rec_ids  # Rated item should be excluded
        assert "tt3" in rec_ids  # Watchlist item should be included


class TestCLIIntegration:
    """Test CLI functionality integration."""

    def test_config_loading(self):
        """Test configuration loading."""
        config = AppConfig.from_file("config.toml")

        assert hasattr(config, "ratings_csv_path")
        assert hasattr(config, "watchlist_path")
        assert hasattr(config, "data_dir")

    def test_logger_functionality(self):
        """Test action logger."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = ActionLogger(data_dir=temp_dir, batch_id="test")

            # Test rating log
            logger.log_rate("tt1234567", 8, notes="Great movie")

            # Test watchlist log
            logger.log_watchlist("tt2345678", add=True)

            # Export logs
            log_path = logger.export()
            assert Path(log_path).exists()

            # Verify log contents
            df = pd.read_csv(log_path)
            assert len(df) == 2
            assert "imdb_const" in df.columns
            assert "action" in df.columns


class TestEndToEndWorkflow:
    """Test complete end-to-end workflow."""

    def test_complete_recommendation_pipeline(self):
        """Test the complete recommendation pipeline."""
        # Use fixture data
        ratings_path = "tests/fixtures_ratings.csv"
        watchlist_path = "tests/fixtures_watchlist.csv"

        with tempfile.TemporaryDirectory() as temp_dir:
            # 1. Ingest data
            result = ingest_sources(ratings_path, watchlist_path, temp_dir)
            dataset = result.dataset

            # 2. Initialize recommenders
            pop_rec = PopSimRecommender(dataset, random_seed=42)
            svd_rec = SVDAutoRecommender(dataset, random_seed=42)

            # Get a seed from the ratings
            seed_ids = dataset.ratings["imdb_const"].head(1).tolist()

            # 3. Generate recommendations
            pop_scores, pop_expl = pop_rec.score(seed_ids, 0.7, 0.3, 0.0, exclude_rated=True)
            svd_scores, svd_expl = svd_rec.score(seed_ids, 0.7, 0.3, 0.0, exclude_rated=True)

            # 4. Blend recommendations
            ranker = Ranker()
            blended = ranker.blend({"pop": pop_scores, "svd": svd_scores})

            # 5. Get top recommendations
            recs = ranker.top_n(
                blended,
                dataset,
                topk=5,
                explanations={"pop": pop_expl, "svd": svd_expl},
                exclude_rated=True,
            )

            # Verify results
            assert len(recs) <= 5
            assert all(isinstance(rec, Recommendation) for rec in recs)
            assert all(rec.score > 0 for rec in recs)
            assert all(rec.imdb_const for rec in recs)

            # Verify recommendations are sorted by score (descending)
            scores = [rec.score for rec in recs]
            assert scores == sorted(scores, reverse=True)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_data_handling(self):
        """Test handling of empty datasets."""
        # Should handle empty dataframes gracefully
        try:
            # This should not crash
            vec = genres_to_vec("")
            assert np.sum(vec) == 0
        except Exception as e:
            pytest.fail(f"Should handle empty data gracefully, but got: {e}")

    def test_missing_values(self):
        """Test handling of missing/NaN values."""
        # Test NA year handling
        weight = recency_weight(pd.NA, 0.5)
        assert weight == 1.0

        # Test NA year in bucket
        vec = year_to_bucket(pd.NA)
        assert np.sum(vec) == 0

    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        # Test invalid genre strings
        vec = genres_to_vec("InvalidGenre,AnotherInvalid")
        assert np.sum(vec) == 0

        # Test cosine with zero vectors
        zero_vec = np.zeros(5)
        other_vec = np.ones(5)
        sim = cosine(zero_vec, other_vec)
        assert sim == 0.0
