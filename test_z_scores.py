#!/usr/bin/env python3
"""
Quick test script to verify z-score normalization in the AllInOneRecommender.
"""

import numpy as np
import pandas as pd

from imdb_recommender.data_io import Dataset
from imdb_recommender.recommender_all_in_one import AllInOneRecommender


def test_z_score_normalization():
    """Test that final scores properly use z-score normalization."""

    # Create a minimal test dataset
    ratings = pd.DataFrame(
        {
            "imdb_const": ["tt0000001", "tt0000002", "tt0000003"],
            "my_rating": [8, 6, 9],
            "title": ["Movie A", "Movie B", "Movie C"],
            "year": [2020, 2019, 2021],
            "imdb_rating": [7.5, 6.0, 8.5],
            "num_votes": [10000, 5000, 15000],
            "genres": ["Drama", "Action", "Comedy"],
        }
    )

    watchlist = pd.DataFrame(
        {
            "imdb_const": ["tt0000004", "tt0000005"],
            "title": ["Movie D", "Movie E"],
            "year": [2018, 2022],
            "imdb_rating": [7.0, 8.0],
            "num_votes": [8000, 12000],
            "genres": ["Horror", "Romance"],
        }
    )

    dataset = Dataset(ratings=ratings, watchlist=watchlist)
    recommender = AllInOneRecommender(dataset, random_seed=42)

    # Generate recommendations
    scores, explanations = recommender.score(
        seeds=[], user_weight=0.7, global_weight=0.3, exclude_rated=True
    )

    print(f"Generated {len(scores)} recommendations")
    print("Final scores (z-score normalized):")
    for item_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {item_id}: {score:.3f}")

    # Verify scores can be negative/positive (z-scores)
    score_values = list(scores.values())
    has_negative = any(s < 0 for s in score_values)
    has_positive = any(s > 0 for s in score_values)

    print("\nZ-score verification:")
    print(f"  Has negative scores: {has_negative}")
    print(f"  Has positive scores: {has_positive}")
    print(f"  Score range: [{min(score_values):.3f}, {max(score_values):.3f}]")
    print(f"  Score mean: {np.mean(score_values):.3f} (should be close to 0)")

    return True


if __name__ == "__main__":
    test_z_score_normalization()
    print("✅ Z-score normalization test completed!")
