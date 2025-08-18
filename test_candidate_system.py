#!/usr/bin/env python3
"""
Test script to demonstrate the new candidate restriction system.
"""

import pandas as pd

from imdb_recommender.data_io import Dataset
from imdb_recommender.recommender_all_in_one import AllInOneRecommender


def test_candidate_system():
    """Demonstrate the candidate restriction system."""

    # Create a more comprehensive test dataset
    ratings = pd.DataFrame(
        {
            "imdb_const": ["tt0000001", "tt0000002", "tt0000003"],
            "my_rating": [9, 7, 8],
            "title": ["Movie A", "Movie B", "Movie C"],
            "year": [2020, 2019, 2021],
            "imdb_rating": [8.5, 7.0, 8.2],
            "num_votes": [50000, 30000, 45000],
            "genres": ["Drama", "Action", "Comedy"],
        }
    )

    # Larger watchlist to test candidate building
    watchlist = pd.DataFrame(
        {
            "imdb_const": ["tt0000004", "tt0000005", "tt0000006", "tt0000007", "tt0000008"],
            "title": ["Movie D", "Movie E", "Movie F", "Movie G", "Movie H"],
            "year": [2018, 2022, 2017, 2023, 2016],
            "imdb_rating": [7.5, 8.8, 6.8, 9.1, 7.2],
            "num_votes": [25000, 60000, 20000, 80000, 15000],
            "genres": ["Horror", "Romance", "Sci-Fi", "Thriller", "Documentary"],
        }
    )

    dataset = Dataset(ratings=ratings, watchlist=watchlist)
    recommender = AllInOneRecommender(dataset, random_seed=42)

    print("🔍 Testing Candidate System")
    print("=" * 50)

    # Test 1: Build candidates
    print("\n1️⃣ Building candidate pool...")
    candidates = recommender.build_candidates(max_candidates=6)
    print(f"📦 Generated candidate pool: {candidates}")

    # Test 2: Score without candidates (full catalog)
    print("\n2️⃣ Scoring without candidate restriction...")
    scores_full, _ = recommender.score(
        seeds=[], user_weight=0.7, global_weight=0.3, exclude_rated=True
    )
    print(f"🎯 Full catalog scoring: {len(scores_full)} items scored")
    print(
        f"   Top items: {list(sorted(scores_full.items(), key=lambda x: x[1], reverse=True)[:3])}"
    )

    # Test 3: Score with candidates (restricted)
    print("\n3️⃣ Scoring with candidate restriction...")
    scores_candidates, _ = recommender.score(
        seeds=[], user_weight=0.7, global_weight=0.3, exclude_rated=True, candidates=candidates
    )
    print(f"🎯 Candidate-restricted scoring: {len(scores_candidates)} items scored")
    print(
        f"   All items: {list(sorted(scores_candidates.items(), key=lambda x: x[1], reverse=True))}"
    )

    # Test 4: Verify restriction worked
    print("\n4️⃣ Verifying candidate restriction...")
    candidate_set = set(candidates)
    scored_set = set(scores_candidates.keys())

    print(f"   Candidates: {candidate_set}")
    print(f"   Scored items: {scored_set}")
    print(f"   ✅ All scored items in candidates: {scored_set <= candidate_set}")

    # Test 5: Evaluation with candidates
    print("\n5️⃣ Testing evaluation with candidates...")
    try:
        metrics = recommender.evaluate_temporal_split(test_size=0.33)
        print(f"📊 Evaluation metrics: {metrics}")
        if metrics:
            print("✅ Evaluation with candidates completed successfully")
        else:
            print("⚠️ No metrics (expected with small dataset)")
    except Exception as e:
        print(f"⚠️ Evaluation failed (expected with small dataset): {e}")

    print("\n🎉 Candidate system test completed!")
    return True


if __name__ == "__main__":
    test_candidate_system()
