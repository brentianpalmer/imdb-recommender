#!/usr/bin/env python3
# Canonical location: scripts/diagnostics/diagnose_elasticnet.py
# Expected inputs: data/raw/ratings.csv and data/raw/watchlist.xlsx

"""
Diagnostic script to understand ElasticNet prediction issues
"""

import numpy as np
import pandas as pd


def check_data_issues():
    print("🔍 ELASTICNET DIAGNOSTIC ANALYSIS")
    print("=" * 50)

    # Load ratings data
    ratings_df = pd.read_csv("data/raw/ratings.csv")
    print(f"📊 Ratings data: {len(ratings_df)} movies")
    print(
        "   Your Rating range: "
        f"{ratings_df['Your Rating'].min()} - {ratings_df['Your Rating'].max()}"
    )
    print(
        "   IMDb Rating range: "
        f"{ratings_df['IMDb Rating'].min():.1f} - {ratings_df['IMDb Rating'].max():.1f}"
    )
    print(f"   IMDb Rating nulls: {ratings_df['IMDb Rating'].isna().sum()}")

    # Load watchlist data
    watchlist_df = pd.read_excel("data/raw/watchlist.xlsx")
    print(f"\n📋 Watchlist data: {len(watchlist_df)} movies")
    print(
        "   IMDb Rating range: "
        f"{watchlist_df['IMDb Rating'].min():.1f} - {watchlist_df['IMDb Rating'].max():.1f}"
    )
    print(f"   IMDb Rating nulls: {watchlist_df['IMDb Rating'].isna().sum()}")

    # Check for extreme values that could cause issues
    print("\n⚠️  POTENTIAL ISSUES:")

    # Check for very high IMDb ratings
    high_ratings = ratings_df[ratings_df["IMDb Rating"] > 9.5]
    if len(high_ratings) > 0:
        print(f"   Very high IMDb ratings (>9.5): {len(high_ratings)}")
        print(f"   Max IMDb rating: {ratings_df['IMDb Rating'].max()}")

    # Check for very high vote counts (could affect log_votes feature)
    high_votes = ratings_df[ratings_df["Num Votes"] > 1000000]
    if len(high_votes) > 0:
        print(f"   Very high vote counts (>1M): {len(high_votes)}")
        print(f"   Max vote count: {ratings_df['Num Votes'].max():,}")
        print(f"   Max log(votes): {np.log1p(ratings_df['Num Votes'].max()):.2f}")

    # Check year ranges
    print(f"   Year range (ratings): {ratings_df['Year'].min()} - {ratings_df['Year'].max()}")
    print(f"   Year range (watchlist): {watchlist_df['Year'].min()} - {watchlist_df['Year'].max()}")

    # Check for future movies in watchlist (could cause issues)
    future_movies = watchlist_df[watchlist_df["Year"] > 2025]
    if len(future_movies) > 0:
        print(f"   Future movies (>2025): {len(future_movies)}")
        print(f"   Latest year: {watchlist_df['Year'].max()}")

    print("\n🎯 RECOMMENDATION: ")
    print("   1. Clip predictions to 1-10 range")
    print("   2. Handle null IMDb ratings properly")
    print("   3. Check feature scaling issues")


if __name__ == "__main__":
    check_data_issues()
