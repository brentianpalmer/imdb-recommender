"""
Test the All-in-One Four-Stage IMDb Recommender

This test validates all the key functionality of the AllInOneRecommender:
1. Feature engineering
2. Exposure modeling
3. Preference modeling
4. MMR diversity optimization
5. Model persistence
6. Evaluation metrics

Author: IMDb Recommender Team
Date: August 2025
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from imdb_recommender.data_io import Dataset
from imdb_recommender.recommender_all_in_one import AllInOneRecommender


class TestAllInOneRecommender:
    """Test suite for the AllInOneRecommender."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        # Create sample ratings
        ratings = pd.DataFrame(
            {
                "imdb_const": ["tt0000001", "tt0000002", "tt0000003", "tt0000004", "tt0000005"],
                "my_rating": [9, 8, 7, 6, 5],
                "rated_at": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"],
                "title": ["Movie A", "Movie B", "Movie C", "Movie D", "Movie E"],
                "year": [2020, 2019, 2018, 2017, 2016],
                "genres": ["Drama", "Action", "Comedy", "Horror", "Romance"],
                "imdb_rating": [8.5, 7.8, 6.9, 6.2, 5.5],
                "num_votes": [100000, 80000, 60000, 40000, 20000],
            }
        )

        # Create sample watchlist
        watchlist = pd.DataFrame(
            {
                "imdb_const": ["tt0000006", "tt0000007", "tt0000008"],
                "in_watchlist": [True, True, True],
                "title": ["Movie F", "Movie G", "Movie H"],
                "year": [2021, 2022, 2023],
                "genres": ["Sci-Fi", "Thriller", "Documentary"],
                "imdb_rating": [8.0, 7.5, 7.0],
                "num_votes": [50000, 30000, 10000],
            }
        )

        return Dataset(ratings=ratings, watchlist=watchlist)

    def test_initialization(self, sample_dataset):
        """Test that the recommender initializes correctly."""
        recommender = AllInOneRecommender(sample_dataset, random_seed=42)

        assert recommender.dataset == sample_dataset
        assert recommender.random_seed == 42
        assert recommender.personal_weight == 0.7
        assert recommender.popularity_weight == 0.3
        assert recommender.mmr_lambda == 0.8
        assert recommender.svd_components == 64
        assert recommender.recency_lambda == 0.03

    def test_build_features(self, sample_dataset):
        """Test feature engineering."""
        recommender = AllInOneRecommender(sample_dataset, random_seed=42)
        features_df = recommender.build_features()

        # Check that all expected features are created
        expected_features = [
            "runtime_filled",
            "runtime_bin",
            "year_filled",
            "recency",
            "recency_decay",
            "decade",
            "imdb_rating_norm",
            "log_votes",
            "popularity_raw",
            "has_runtime",
            "has_year",
            "has_rating",
            "has_votes",
            "in_ratings",
            "in_watchlist",
            "exposed",
        ]

        for feature in expected_features:
            assert feature in features_df.columns

        # Check that exposure indicators work correctly
        assert features_df["in_ratings"].sum() == 5  # 5 rated movies
        assert features_df["in_watchlist"].sum() == 3  # 3 watchlisted movies
        assert features_df["exposed"].sum() == 8  # All 8 movies are exposed

        # Check feature ranges
        assert (
            0 <= features_df["imdb_rating_norm"].min() <= features_df["imdb_rating_norm"].max() <= 1
        )
        assert features_df["recency"].min() >= 0
        assert 0 <= features_df["recency_decay"].min() <= features_df["recency_decay"].max() <= 1

    def test_build_feature_matrix(self, sample_dataset):
        """Test numerical feature matrix construction."""
        recommender = AllInOneRecommender(sample_dataset, random_seed=42)
        features_df = recommender.build_features()
        feature_matrix = recommender.build_feature_matrix(features_df)

        # Check matrix shape
        assert feature_matrix.shape[0] == len(features_df)
        assert feature_matrix.shape[1] > 0

        # Check that there are no NaN values
        assert not np.isnan(feature_matrix).any()

        # Check that the matrix contains reasonable values
        assert np.isfinite(feature_matrix).all()

    def test_train_exposure_model(self, sample_dataset):
        """Test exposure model training."""
        recommender = AllInOneRecommender(sample_dataset, random_seed=42)
        features_df = recommender.build_features()
        feature_matrix = recommender.build_feature_matrix(features_df)

        exposure_probs = recommender.train_exposure_model(features_df, feature_matrix)

        # Check exposure probabilities
        assert len(exposure_probs) == len(features_df)
        assert 0.1 <= exposure_probs.min() <= exposure_probs.max() <= 0.9
        assert recommender.feature_scaler is not None

    def test_build_pairwise_data(self, sample_dataset):
        """Test pairwise preference data construction."""
        recommender = AllInOneRecommender(sample_dataset, random_seed=42)
        features_df = recommender.build_features()
        feature_matrix = recommender.build_feature_matrix(features_df)

        X_pairs, y_pairs, weights = recommender.build_pairwise_data(features_df, feature_matrix)

        # Check that pairwise data has correct structure
        if len(X_pairs) > 0:
            assert X_pairs.shape[0] == len(y_pairs)
            assert X_pairs.shape[1] == feature_matrix.shape[1]
            assert len(weights) == len(y_pairs)
            # Should have both positive and negative examples (y=1 and y=0)
            unique_labels = set(y_pairs)
            assert 1 in unique_labels  # Should have positive examples
            assert 0 in unique_labels  # Should have negative examples
            assert len(unique_labels) == 2  # Should have exactly 2 classes

        # Non-default min_gap should reduce pair count
        X_gap, y_gap, w_gap = recommender.build_pairwise_data(
            features_df, feature_matrix, min_gap=3
        )
        assert len(X_gap) <= len(X_pairs)

        # Hard negative path produces pairs even with large min_gap
        X_none, y_none, w_none = recommender.build_pairwise_data(
            features_df, feature_matrix, min_gap=5
        )
        assert len(X_none) == 0

        X_hard, y_hard, w_hard = recommender.build_pairwise_data(
            features_df, feature_matrix, min_gap=5, hard_negative=True
        )
        assert len(X_hard) > 0
        assert len(w_hard) == len(y_hard)

    def test_calculate_popularity_prior(self, sample_dataset):
        """Test popularity prior calculation."""
        recommender = AllInOneRecommender(sample_dataset, random_seed=42)
        features_df = recommender.build_features()

        popularity_scores = recommender.calculate_popularity_prior(features_df)

        # Check that popularity scores are z-normalized
        assert len(popularity_scores) == len(features_df)
        assert abs(popularity_scores.mean()) < 0.1  # Should be close to 0
        assert abs(popularity_scores.std() - 1.0) < 0.1  # Should be close to 1

    def test_build_latent_space(self, sample_dataset):
        """Test latent space construction."""
        recommender = AllInOneRecommender(sample_dataset, random_seed=42)
        features_df = recommender.build_features()
        feature_matrix = recommender.build_feature_matrix(features_df)

        recommender.build_latent_space(feature_matrix)

        # Check that latent features are created
        assert recommender.latent_features is not None
        assert recommender.latent_features.shape[0] == feature_matrix.shape[0]
        assert recommender.latent_features.shape[1] <= min(
            recommender.svd_components, feature_matrix.shape[1] - 1
        )
        assert recommender.svd_model is not None

    def test_mmr_rerank(self, sample_dataset):
        """Test MMR re-ranking."""
        recommender = AllInOneRecommender(sample_dataset, random_seed=42)
        features_df = recommender.build_features()
        feature_matrix = recommender.build_feature_matrix(features_df)
        recommender.build_latent_space(feature_matrix)

        # Create dummy scores
        scores = np.random.rand(len(features_df))

        # Test MMR re-ranking
        ranked_indices = recommender.mmr_rerank(scores, top_k=5)

        assert len(ranked_indices) <= 5
        assert len(set(ranked_indices)) == len(ranked_indices)  # No duplicates
        assert all(0 <= idx < len(scores) for idx in ranked_indices)

    def test_score_method(self, sample_dataset):
        """Test the main score method."""
        recommender = AllInOneRecommender(sample_dataset, random_seed=42)

        scores, explanations = recommender.score(
            seeds=[], user_weight=0.7, global_weight=0.3, exclude_rated=True
        )

        # Check that scores are returned
        assert isinstance(scores, dict)
        assert isinstance(explanations, dict)
        assert len(scores) > 0
        assert len(explanations) == len(scores)

        # Check that scores are reasonable
        for score in scores.values():
            assert isinstance(score, int | float)
            assert np.isfinite(score)

        # Check that explanations are strings
        for explanation in explanations.values():
            assert isinstance(explanation, str)
            assert len(explanation) > 0

    def test_evaluate_temporal_split(self, sample_dataset):
        """Test temporal split evaluation."""
        recommender = AllInOneRecommender(sample_dataset, random_seed=42)

        # Test evaluation (might not work with small dataset)
        metrics = recommender.evaluate_temporal_split(test_size=0.2)

        # Check that metrics are returned (might be empty for small dataset)
        assert isinstance(metrics, dict)

        if len(metrics) > 0:
            expected_metrics = ["hits_at_10", "ndcg_at_10", "diversity"]
            for metric in expected_metrics:
                if metric in metrics:
                    assert isinstance(metrics[metric], int | float)
                    assert np.isfinite(metrics[metric])

    def test_model_persistence(self, sample_dataset):
        """Test model saving and loading."""
        recommender = AllInOneRecommender(sample_dataset, random_seed=42)

        # Train the model
        scores, explanations = recommender.score(seeds=[], user_weight=0.7, global_weight=0.3)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.pkl"

            # Save model
            recommender.save_model(model_path)
            assert model_path.exists()

            # Load model into new instance
            new_recommender = AllInOneRecommender(sample_dataset, random_seed=42)
            new_recommender.load_model(model_path)

            # Check that parameters are preserved
            assert new_recommender.personal_weight == recommender.personal_weight
            assert new_recommender.popularity_weight == recommender.popularity_weight
            assert new_recommender.recency_lambda == recommender.recency_lambda
            assert new_recommender.mmr_lambda == recommender.mmr_lambda
            assert new_recommender.svd_components == recommender.svd_components

    def test_export_recommendations_csv(self, sample_dataset):
        """Test CSV export functionality."""
        recommender = AllInOneRecommender(sample_dataset, random_seed=42)

        scores, explanations = recommender.score(seeds=[], user_weight=0.7, global_weight=0.3)

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "test_recommendations.csv"

            # Export recommendations
            recommendations = recommender.export_recommendations_csv(scores, csv_path, top_k=5)

            # Check that file exists
            assert csv_path.exists()

            # Check that recommendations have correct structure
            assert isinstance(recommendations, list)
            assert len(recommendations) <= 5

            if len(recommendations) > 0:
                expected_columns = [
                    "tconst",
                    "title",
                    "year",
                    "genres",
                    "title_type",
                    "imdb_rating",
                    "num_votes",
                    "runtime",
                    "score_personal",
                    "score_pop",
                    "score_final",
                ]

                for rec in recommendations:
                    assert isinstance(rec, dict)
                    for col in expected_columns:
                        assert col in rec

            # Check CSV content
            df = pd.read_csv(csv_path)
            assert len(df) == len(recommendations)
            assert "tconst" in df.columns
            assert "score_final" in df.columns


if __name__ == "__main__":
    # Run a simple test
    import sys

    sys.path.append(".")

    # Create a simple test dataset
    ratings = pd.DataFrame(
        {
            "imdb_const": ["tt0000001", "tt0000002", "tt0000003"],
            "my_rating": [9, 7, 5],
            "rated_at": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "title": ["Movie A", "Movie B", "Movie C"],
            "year": [2020, 2019, 2018],
            "genres": ["Drama", "Action", "Comedy"],
            "imdb_rating": [8.5, 7.8, 6.9],
            "num_votes": [100000, 80000, 60000],
        }
    )

    watchlist = pd.DataFrame(
        {
            "imdb_const": ["tt0000004", "tt0000005"],
            "in_watchlist": [True, True],
            "title": ["Movie D", "Movie E"],
            "year": [2021, 2022],
            "genres": ["Sci-Fi", "Thriller"],
            "imdb_rating": [8.0, 7.5],
            "num_votes": [50000, 30000],
        }
    )

    dataset = Dataset(ratings=ratings, watchlist=watchlist)
    recommender = AllInOneRecommender(dataset, random_seed=42)

    # Test basic functionality
    print("🧪 Testing AllInOneRecommender...")

    scores, explanations = recommender.score(
        seeds=[], user_weight=0.7, global_weight=0.3, exclude_rated=True
    )

    print(f"✅ Generated {len(scores)} recommendations")
    print(f"✅ Sample explanation: {list(explanations.values())[0] if explanations else 'None'}")

    # Test evaluation
    metrics = recommender.evaluate_temporal_split()
    print(f"✅ Evaluation metrics: {metrics}")

    print("🎉 All basic tests passed!")
